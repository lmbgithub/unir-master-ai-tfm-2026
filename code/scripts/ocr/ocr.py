#!/usr/bin/env python3
"""Benchmark de modelos OCR (imagen/PDF → texto) en CPU para UrgeNurse.

Mismo espíritu que ``asr.py`` y ``llm_performance.py`` pero para reconocimiento
óptico de caracteres: carga motores OCR pequeños (≤2 GB RAM, solo CPU,
cuantizados/ligeros), procesa el dataset sintético de documentos clínicos
(urgencias, exámenes de laboratorio, radiología, diagnósticos, órdenes de
medicación) y mide las métricas estándar de OCR:

  * **CER**  Character Error Rate (métrica principal de OCR).
  * **WER**  Word Error Rate.
  * **Word P/R/F1**  precisión, exhaustividad y F1 a nivel de palabra
    (a partir de la alineación de edición hipótesis vs referencia).
  * **term-WER**  error sobre terminología clínica (medicamentos, valores
    numéricos, abreviaturas) — equivocar una dosis es más grave que un "the".
  * **field recall**  fracción de entidades clínicas de la referencia recuperadas.
  * **latencia (ms/página)**  tiempo medio de proceso por documento.
  * **throughput (páginas/s)**  velocidad de inferencia en CPU.
  * **RAM Peak**  pico de RSS del proceso durante el OCR.
  * **load time**  tiempo de carga del modelo.

Diseño (idéntico al benchmark ASR):
  1. ``run_benchmark()`` recorre el catálogo de modelos (uno a uno, ≤2 GB), por
     cada documento corre el OCR, escribe ``predictions/<modelo>/{n}.txt`` y
     calcula las métricas. Devuelve un ``pandas.DataFrame`` (una fila por modelo).
  2. Cada backend (Tesseract, PaddleOCR, RapidOCR, EasyOCR, docTR, Docling) es
     OPCIONAL: si la librería o el binario no están, el modelo se OMITE con un
     aviso claro (no rompe el benchmark).

REQUISITO: el dataset (imágenes + referencias verbatim) debe existir ya en
``assets/images/{n}.png`` y ``assets/references/{n}.txt``. Se genera con el
comando aparte ``prepare_dataset.py`` (ReportLab → PDF → PNG + ground-truth).
Este módulo NO lo genera: solo lo lee.

Uso:
    # 0) una sola vez, deja assets/ listo:
    python prepare_dataset.py
    # 1) desde un notebook:
    import ocr
    df = ocr.run_benchmark()
    # o por CLI:
    python ocr.py

Variables de entorno opcionales:
    OCR_MODELS_DIR     carpeta de modelos/caches (default: ../models)
    OCR_IMAGE_DIR      carpeta de imágenes PNG (default: assets/images)
    OCR_PDF_DIR        carpeta de PDFs (default: assets/docs)
    OCR_REF_DIR        carpeta de referencias txt (default: assets/references)
    OCR_OUT_DIR        carpeta de predicciones (default: predictions)
    OCR_INPUT          "image" o "pdf": qué se le da al OCR (default: image)
    OCR_MAX_RAM_GB     presupuesto de RAM (default: 2.0)
    OCR_LIMIT          nº de documentos a procesar (default: 0 = todos)
    OCR_DOWNLOAD       "0" para no descargar modelos faltantes (default: "1")
    OCR_THREADS        nº de hilos (default: núcleos físicos)
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Reutilizamos los helpers de sistema del benchmark de LLMs (DRY): información del
# host, RSS en MB. Import por ruta para no depender de la mecánica de paquetes
# (todos viven bajo code/scripts/).
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
_LP_PATH = SCRIPT_DIR.parent / "llm_performance" / "llm_performance.py"


def _load_lp():
    import sys

    if "llm_performance" in sys.modules:
        return sys.modules["llm_performance"]
    spec = importlib.util.spec_from_file_location("llm_performance", _LP_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["llm_performance"] = mod
    spec.loader.exec_module(mod)
    return mod


_lp = _load_lp()
get_rss_mb = _lp.get_rss_mb
human_mb = _lp.human_mb
collect_system_info = _lp.collect_system_info
print_system_info = _lp.print_system_info


# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = Path(os.environ.get("OCR_MODELS_DIR", SCRIPT_DIR.parent / "models"))
IMAGE_DIR = Path(os.environ.get("OCR_IMAGE_DIR", SCRIPT_DIR / "assets" / "images"))
PDF_DIR = Path(os.environ.get("OCR_PDF_DIR", SCRIPT_DIR / "assets" / "docs"))
REF_DIR = Path(os.environ.get("OCR_REF_DIR", SCRIPT_DIR / "assets" / "references"))
OUT_DIR = Path(os.environ.get("OCR_OUT_DIR", SCRIPT_DIR / "predictions"))
OUTPUT_DIR = Path(os.environ.get("OCR_OUTPUT_DIR", SCRIPT_DIR))  # error__*.json

INPUT_KIND = os.environ.get("OCR_INPUT", "image").lower()  # "image" | "pdf"
MAX_RAM_GB = float(os.environ.get("OCR_MAX_RAM_GB", 2.0))
DOC_LIMIT = int(os.environ.get("OCR_LIMIT", 0))  # 0 = todos
DOWNLOAD_MISSING = os.environ.get("OCR_DOWNLOAD", "1") != "0"
N_THREADS = int(os.environ.get("OCR_THREADS", _lp._physical_cores()))

# UrgeNurse es CPU-only por diseño: forzamos que ninguna librería
# (PyTorch, Paddle, ONNX Runtime) descubra una GPU/acelerador. CUDA_VISIBLE_DEVICES
# vacío oculta CUDA a torch/paddle; las flags por backend (use_gpu=False, gpu=False,
# proveedor CPU) cierran el resto, incluido CoreML/ANE en macOS.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# PaddlePaddle 3.0 + ejecutor PIR + oneDNN(MKL-DNN) revienta en CPU al convertir
# atributos ("ConvertPirAttribute2RuntimeAttribute not support"). Desactivamos
# oneDNN para que Paddle no tome esa ruta de instrucción. Debe fijarse ANTES de
# importar paddle.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
# Limitar hilos de las librerías nativas (OpenMP / ONNX Runtime) para una medida
# de RAM/latencia comparable entre motores.
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
if not DOWNLOAD_MISSING:
    # Evita que HuggingFace/transformers intenten descargar pesos faltantes.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Referencias (ground-truth verbatim) e inputs — solo lectura
# ─────────────────────────────────────────────────────────────────────────────


def load_references() -> dict[int, str]:
    """Carga la ground-truth verbatim de ``assets/references/{n}.txt``.

    NO la genera: si falta, indica ejecutar ``prepare_dataset.py`` primero.
    """
    if not REF_DIR.exists() or not any(REF_DIR.glob("*.txt")):
        raise FileNotFoundError(
            f"No hay referencias en {REF_DIR}. Genera el dataset antes de "
            f"correr el benchmark:\n    python prepare_dataset.py"
        )
    refs: dict[int, str] = {}
    for p in REF_DIR.glob("*.txt"):
        if p.stem.isdigit():
            refs[int(p.stem)] = p.read_text(encoding="utf-8")
    return dict(sorted(refs.items()))


def list_inputs() -> list[tuple[int, Path]]:
    """Lista (n, ruta) de los documentos a procesar (imagen o PDF), ordenados."""
    src_dir, ext = (IMAGE_DIR, "*.png") if INPUT_KIND == "image" else (PDF_DIR, "*.pdf")
    items: list[tuple[int, Path]] = []
    for p in src_dir.glob(ext):
        if p.stem.isdigit():
            items.append((int(p.stem), p))
    items.sort(key=lambda x: x[0])
    if DOC_LIMIT > 0:
        items = items[:DOC_LIMIT]
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Terminología clínica para métricas específicas (compartida con el dataset)
# ─────────────────────────────────────────────────────────────────────────────
# El CER/WER globales tratan todo el texto por igual. En un documento clínico,
# equivocar un medicamento, una dosis o un valor de laboratorio es mucho más
# grave. Medimos error y recall solo sobre esas "entidades clínicas".

_MED_NAMES = {
    "nitroglycerin",
    "nitro",
    "aspirin",
    "warfarin",
    "heparin",
    "metoprolol",
    "bisoprolol",
    "atenolol",
    "carvedilol",
    "digoxin",
    "amiodarone",
    "furosemide",
    "spironolactone",
    "ramipril",
    "enalapril",
    "lisinopril",
    "amlodipine",
    "simvastatin",
    "atorvastatin",
    "clopidogrel",
    "insulin",
    "metformin",
    "morphine",
    "fentanyl",
    "paracetamol",
    "acetaminophen",
    "ibuprofen",
    "codeine",
    "tramadol",
    "salbutamol",
    "albuterol",
    "prednisone",
    "prednisolone",
    "dexamethasone",
    "amoxicillin",
    "ceftriaxone",
    "gentamicin",
    "vancomycin",
    "diazepam",
    "lorazepam",
    "midazolam",
    "phenytoin",
    "levetiracetam",
    "mannitol",
    "labetalol",
    "noradrenaline",
    "adrenaline",
    "epinephrine",
    "dopamine",
    "potassium",
    "magnesium",
    "calcium",
    "oxygen",
    "saline",
    "dextrose",
    "lasix",
    "tamsulosin",
    "allopurinol",
    "omeprazole",
    "pantoprazole",
    "ondansetron",
}
_MED_SUFFIXES = (
    "olol",
    "pril",
    "sartan",
    "statin",
    "azepam",
    "cillin",
    "mycin",
    "floxacin",
    "dipine",
    "prazole",
    "tidine",
)
_ABBREVIATIONS = {
    "bp",
    "hr",
    "rr",
    "spo2",
    "sao2",
    "ecg",
    "ekg",
    "mi",
    "ami",
    "stemi",
    "copd",
    "icu",
    "ccu",
    "iv",
    "im",
    "po",
    "prn",
    "bpm",
    "mmhg",
    "mg",
    "ml",
    "mcg",
    "kg",
    "gcs",
    "dnr",
    "cpr",
    "abg",
    "cabg",
    "htn",
    "dm",
    "ckd",
    "aki",
    "tia",
    "cva",
    "uti",
    "sob",
    "nbm",
    "obs",
    "sats",
    "tds",
    "bd",
    "od",
    "ng",
    "egfr",
    "bnp",
    "crp",
    "wbc",
    "inr",
    "hb",
    "esi",
    "mrn",
    "ct",
    "mri",
    "cxr",
    "wcc",
    "rbc",
    "plt",
    "na",
    "cl",
    "bun",
    "ldl",
    "hdl",
}


def _is_number(tok: str) -> bool:
    return bool(re.fullmatch(r"\d+([.,/]\d+)*", tok))


def _is_medication(tok: str) -> bool:
    return tok in _MED_NAMES or tok.endswith(_MED_SUFFIXES)


def _is_abbrev(tok: str) -> bool:
    return tok in _ABBREVIATIONS


def classify_term(tok: str) -> str | None:
    """Devuelve 'med' | 'num' | 'abbr' si el token es entidad clínica, si no None."""
    if _is_number(tok):
        return "num"
    if _is_medication(tok):
        return "med"
    if _is_abbrev(tok):
        return "abbr"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Normalización + métricas de OCR (CER / WER / P-R-F1 / alineación)
# ─────────────────────────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s./]")


def normalize_text(text: str) -> str:
    """Normaliza para comparar: minúsculas, sin acentos/puntuación, espacios simples.

    Mantiene '/' y '.' SOLO entre dígitos (150/95, 39.2) porque son clínicamente
    significativos. A diferencia del ASR, aquí se conservan las etiquetas de
    campo ('name:', 'bp:') porque el OCR sí debe leerlas en el documento.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = text.replace("%", " percent ")
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"(?<!\d)[./]|[./](?!\d)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def _levenshtein_ops(seq_ref: list, seq_hyp: list) -> tuple[int, int, int, list[tuple]]:
    """Distancia de edición con backtrace (sirve a nivel de palabra y de carácter).

    Devuelve (sub, del, ins, alineación) donde alineación es una lista de tuplas
    ('ok'|'sub'|'del'|'ins', ref_token_or_None, hyp_token_or_None).
    """
    n, m = len(seq_ref), len(seq_hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_ref[i - 1] == seq_hyp[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # deletion
                d[i][j - 1] + 1,  # insertion
                d[i - 1][j - 1] + cost,  # match/substitution
            )
    i, j = n, m
    align: list[tuple] = []
    sub = dele = ins = 0
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and seq_ref[i - 1] == seq_hyp[j - 1]
            and d[i][j] == d[i - 1][j - 1]
        ):
            align.append(("ok", seq_ref[i - 1], seq_hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            align.append(("sub", seq_ref[i - 1], seq_hyp[j - 1]))
            sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            align.append(("del", seq_ref[i - 1], None))
            dele += 1
            i -= 1
        else:
            align.append(("ins", None, seq_hyp[j - 1]))
            ins += 1
            j -= 1
    align.reverse()
    return sub, dele, ins, align


@dataclass
class OcrScore:
    cer: float = 0.0
    wer: float = 0.0
    char_acc: float = 0.0  # 1 - CER (acotado a [0,1])
    word_precision: float = 0.0
    word_recall: float = 0.0
    word_f1: float = 0.0
    term_wer: float = 0.0  # error sobre terminología clínica
    field_recall: float = 0.0  # entidades clínicas recuperadas / esperadas
    n_ref_words: int = 0
    n_ref_chars: int = 0
    n_terms: int = 0


def score_ocr(reference: str, hypothesis: str) -> OcrScore:
    """Compara hipótesis vs referencia: CER, WER, P/R/F1, term-WER y field recall."""
    ref_tok = tokenize(reference)
    hyp_tok = tokenize(hypothesis)
    sub, dele, ins, align = _levenshtein_ops(ref_tok, hyp_tok)
    n = max(1, len(ref_tok))
    wer = (sub + dele + ins) / n

    hits = sum(1 for op, _r, _h in align if op == "ok")
    precision = hits / max(1, len(hyp_tok))  # correctas / total predichas
    recall = hits / n  # correctas / total esperadas
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # CER sobre texto normalizado sin espacios.
    ref_c = list(normalize_text(reference).replace(" ", ""))
    hyp_c = list(normalize_text(hypothesis).replace(" ", ""))
    cs, cd, ci, _ = _levenshtein_ops(ref_c, hyp_c)
    cer = (cs + cd + ci) / max(1, len(ref_c))

    # Métricas restringidas a terminología clínica de la referencia.
    n_terms = term_err = term_hit = 0
    for op, r_tok, _h in align:
        if r_tok is None or classify_term(r_tok) is None:
            continue
        n_terms += 1
        if op == "ok":
            term_hit += 1
        else:  # sub o del sobre un término clínico
            term_err += 1
    term_wer = term_err / n_terms if n_terms else 0.0
    field_recall = term_hit / n_terms if n_terms else 1.0

    return OcrScore(
        cer=round(cer, 4),
        wer=round(wer, 4),
        char_acc=round(max(0.0, 1.0 - cer), 4),
        word_precision=round(precision, 4),
        word_recall=round(recall, 4),
        word_f1=round(f1, 4),
        term_wer=round(term_wer, 4),
        field_recall=round(field_recall, 4),
        n_ref_words=len(ref_tok),
        n_ref_chars=len(ref_c),
        n_terms=n_terms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Catálogo de modelos OCR
# ─────────────────────────────────────────────────────────────────────────────
# Modelos pequeños, CPU, ligeros, ≤2 GB de RAM. Cada modelo declara su `backend`;
# la carga/inferencia concreta vive en BACKENDS.
#
#   tesseract  : Tesseract 5 (LSTM) vía pytesseract — baseline del proyecto.
#   paddleocr  : PaddleOCR PP-OCR (det+rec ligeros, CPU). Stack del agente OCR.
#   rapidocr   : RapidOCR (modelos PP-OCR en ONNX Runtime; ultraligero, ~8 MB).
#   easyocr    : EasyOCR (CRAFT + CRNN, PyTorch CPU). Muy popular.
#   doctr      : docTR de Mindee (det+rec, backend PyTorch/ONNX). Alta precisión.
#   docling    : Docling de IBM (pipeline de documentos con OCR integrado).


@dataclass(frozen=True)
class OcrModelSpec:
    name: str  # etiqueta única en tablas/figuras
    backend: str  # clave en BACKENDS
    family: str  # agrupación (tesseract / paddle / rapid / easy / doctr / docling)
    is_baseline: bool = False  # motor de referencia (Tesseract, stack del proyecto)
    ram_est_mb: float = 600.0  # footprint estimado para el presupuesto de RAM
    params: dict = field(default_factory=dict)


MODELS: list[OcrModelSpec] = [
    # ── Tesseract 5 (baseline del proyecto) ──────────────────────────────────
    OcrModelSpec(
        "tesseract-5-eng",
        "tesseract",
        "tesseract",
        is_baseline=True,
        ram_est_mb=200,
        params={"lang": "eng", "psm": 6},
    ),
    # ── PaddleOCR PP-OCR (stack del agente OCR) ──────────────────────────────
    OcrModelSpec(
        "paddleocr-ppocr-en",
        "paddleocr",
        "paddle",
        ram_est_mb=900,
        params={"lang": "en"},
    ),
    # ── RapidOCR (PP-OCR en ONNX Runtime, ultraligero) ───────────────────────
    OcrModelSpec(
        "rapidocr-onnx-en",
        "rapidocr",
        "rapid",
        ram_est_mb=500,
        params={},
    ),
    # ── EasyOCR (CRAFT + CRNN, PyTorch CPU) ──────────────────────────────────
    OcrModelSpec(
        "easyocr-en",
        "easyocr",
        "easy",
        ram_est_mb=1300,
        params={"lang": "en"},
    ),
    # ── docTR (Mindee, variante MobileNet ligera para CPU) ───────────────────
    OcrModelSpec(
        "doctr-mobilenet",
        "doctr",
        "doctr",
        ram_est_mb=900,
        params={"det": "db_mobilenet_v3_large", "reco": "crnn_mobilenet_v3_small"},
    ),
    # ── Docling (IBM, pipeline de documentos con OCR) ────────────────────────
    OcrModelSpec(
        "docling-pipeline",
        "docling",
        "docling",
        ram_est_mb=1900,
        params={},
    ),
]


def _have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Backends de OCR
# ─────────────────────────────────────────────────────────────────────────────
# Cada backend implementa load(spec) -> objeto modelo, y ocr(model, path) ->
# texto. Lanza RuntimeError/ImportError si no está disponible; el runner lo
# captura y OMITE el modelo con un aviso.


class Backend:
    name = "base"

    def available(self, spec: OcrModelSpec) -> str:
        """'' si está disponible; si no, motivo legible para omitir el modelo."""
        return ""

    def load(self, spec: OcrModelSpec):
        raise NotImplementedError

    def ocr(self, model, path: Path) -> str:
        raise NotImplementedError

    def unload(self, model) -> None:
        del model


class TesseractBackend(Backend):
    name = "tesseract"

    def available(self, spec: OcrModelSpec) -> str:
        import shutil

        if not _have_module("pytesseract"):
            return "falta 'pytesseract' (pip install pytesseract)"
        if not _have_module("PIL"):
            return "falta 'Pillow' (pip install pillow)"
        if shutil.which("tesseract") is None:
            return "falta el binario 'tesseract' (brew install tesseract)"
        return ""

    def load(self, spec: OcrModelSpec):
        return spec.params  # config; Tesseract no mantiene estado en memoria

    def ocr(self, model, path: Path) -> str:
        import pytesseract
        from PIL import Image

        cfg = f"--psm {model.get('psm', 6)}"
        return pytesseract.image_to_string(
            Image.open(path), lang=model.get("lang", "eng"), config=cfg
        ).strip()


class PaddleOCRBackend(Backend):
    name = "paddleocr"

    def available(self, spec: OcrModelSpec) -> str:
        if not _have_module("paddleocr"):
            return "falta 'paddleocr' (pip install paddleocr paddlepaddle)"
        if not _have_module("paddle"):
            return "falta 'paddlepaddle' (pip install paddlepaddle)"
        return ""

    def load(self, spec: OcrModelSpec):
        from paddleocr import PaddleOCR

        lang = spec.params.get("lang", "en")
        # La firma del constructor cambió entre PaddleOCR 2.x y 3.x: probamos la
        # configuración ligera de 3.x primero y caemos a 2.x / mínima si algún
        # kwarg no existe. Claves de bajo consumo:
        #   - detector MOBILE (no el server, mucho más pesado),
        #   - sin clasificación de orientación de página ni "unwarping" (UVDoc),
        #   - sin orientación de línea de texto,
        #   - enable_mkldnn=False evita el bug PIR/oneDNN de paddle 3.0 en CPU.
        for kwargs in (
            {
                "lang": lang,
                "text_detection_model_name": "PP-OCRv5_mobile_det",
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "enable_mkldnn": False,
            },
            {
                "lang": lang,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "enable_mkldnn": False,
            },
            {
                "lang": lang,
                "use_angle_cls": False,
                "show_log": False,
                "use_gpu": False,
                "enable_mkldnn": False,
            },
            {"lang": lang},
        ):
            try:
                return PaddleOCR(**kwargs)
            except (TypeError, ValueError):
                continue
        return PaddleOCR()

    def ocr(self, model, path: Path) -> str:
        try:
            res = model.ocr(str(path), cls=False)
        except (TypeError, ValueError):
            res = model.ocr(str(path))
        return _paddle_text(res)


def _paddle_text(res) -> str:
    """Extrae el texto de la salida de PaddleOCR (formato 2.x o 3.x)."""
    texts: list[str] = []
    for page in res or []:
        if page is None:
            continue
        if isinstance(page, dict):  # PaddleOCR 3.x predict()
            texts.extend(page.get("rec_texts", []))
            continue
        for line in page:  # PaddleOCR 2.x: [box, (text, conf)]
            try:
                texts.append(line[1][0])
            except (IndexError, TypeError):
                if (
                    isinstance(line, (list, tuple))
                    and line
                    and isinstance(line[-1], str)
                ):
                    texts.append(line[-1])
    return "\n".join(texts).strip()


class RapidOCRBackend(Backend):
    name = "rapidocr"

    def available(self, spec: OcrModelSpec) -> str:
        return (
            ""
            if _have_module("rapidocr_onnxruntime")
            else "falta 'rapidocr-onnxruntime' (pip install rapidocr-onnxruntime)"
        )

    def load(self, spec: OcrModelSpec):
        # rapidocr_onnxruntime usa por defecto el CPUExecutionProvider de ONNX
        # Runtime (no CUDA ni CoreML/DirectML salvo que se pidan explícitamente).
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR()

    def ocr(self, model, path: Path) -> str:
        result, _elapsed = model(str(path))
        if not result:
            return ""
        return "\n".join(line[1] for line in result).strip()


class EasyOCRBackend(Backend):
    name = "easyocr"

    def available(self, spec: OcrModelSpec) -> str:
        return (
            "" if _have_module("easyocr") else "falta 'easyocr' (pip install easyocr)"
        )

    def load(self, spec: OcrModelSpec):
        import easyocr

        store = MODELS_DIR / "easyocr"
        store.mkdir(parents=True, exist_ok=True)
        return easyocr.Reader(
            [spec.params.get("lang", "en")],
            gpu=False,
            verbose=False,
            download_enabled=DOWNLOAD_MISSING,
            model_storage_directory=str(store),
        )

    def ocr(self, model, path: Path) -> str:
        lines = model.readtext(str(path), detail=0, paragraph=True)
        return "\n".join(lines).strip()


class DocTRBackend(Backend):
    name = "doctr"

    def available(self, spec: OcrModelSpec) -> str:
        if not _have_module("doctr"):
            return "falta 'python-doctr' (pip install 'python-doctr[torch]')"
        if not (_have_module("torch") or _have_module("tensorflow")):
            return "docTR necesita backend torch o tensorflow"
        return ""

    def load(self, spec: OcrModelSpec):
        from doctr.models import ocr_predictor

        return ocr_predictor(
            det_arch=spec.params.get("det", "db_resnet50"),
            reco_arch=spec.params.get("reco", "crnn_vgg16_bn"),
            pretrained=True,
        )

    def ocr(self, model, path: Path) -> str:
        from doctr.io import DocumentFile

        if path.suffix.lower() == ".pdf":
            doc = DocumentFile.from_pdf(str(path))
        else:
            doc = DocumentFile.from_images(str(path))
        return model(doc).render().strip()


class DoclingBackend(Backend):
    name = "docling"

    def available(self, spec: OcrModelSpec) -> str:
        return (
            "" if _have_module("docling") else "falta 'docling' (pip install docling)"
        )

    def load(self, spec: OcrModelSpec):
        from docling.document_converter import DocumentConverter

        # Docling es el pipeline más pesado; lo forzamos a CPU explícitamente.
        # La API de opciones cambia entre versiones, así que es best-effort: si no
        # está disponible, CUDA_VISIBLE_DEVICES="" ya garantiza el modo CPU.
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                AcceleratorDevice,
                AcceleratorOptions,
                PdfPipelineOptions,
            )
            from docling.document_converter import PdfFormatOption

            opts = PdfPipelineOptions()
            opts.accelerator_options = AcceleratorOptions(
                num_threads=N_THREADS, device=AcceleratorDevice.CPU
            )
            return DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
        except Exception:  # noqa: BLE001
            return DocumentConverter()

    def ocr(self, model, path: Path) -> str:
        result = model.convert(str(path))
        return result.document.export_to_text().strip()


BACKENDS: dict[str, Backend] = {
    "tesseract": TesseractBackend(),
    "paddleocr": PaddleOCRBackend(),
    "rapidocr": RapidOCRBackend(),
    "easyocr": EasyOCRBackend(),
    "doctr": DocTRBackend(),
    "docling": DoclingBackend(),
}


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ModelRow:
    model: str
    family: str
    backend: str
    is_baseline: bool
    load_time_ms: float = 0.0
    mem_after_load_mb: float = 0.0
    ram_peak_mb: float = 0.0
    n_docs: int = 0
    # Agregados de calidad (media sobre documentos)
    cer: float = 0.0
    wer: float = 0.0
    char_acc: float = 0.0
    word_precision: float = 0.0
    word_recall: float = 0.0
    word_f1: float = 0.0
    term_wer: float = 0.0
    field_recall: float = 0.0
    # Rendimiento
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    pages_per_sec: float = 0.0
    load_error: str = ""


RESULT_COLUMNS = [
    "model",
    "family",
    "backend",
    "is_baseline",
    "load_time_ms",
    "mem_after_load_mb",
    "ram_peak_mb",
    "n_docs",
    "cer",
    "wer",
    "char_acc",
    "word_precision",
    "word_recall",
    "word_f1",
    "term_wer",
    "field_recall",
    "mean_latency_ms",
    "p95_latency_ms",
    "pages_per_sec",
    "load_error",
]


def _write_prediction(
    model_name: str, n: int, text: str, score: OcrScore, latency_ms: float
) -> None:
    """Escribe predictions/<modelo>/{n}.txt con la hipótesis + métricas del doc."""
    d = OUT_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    header = (
        f"# doc {n} · modelo {model_name}\n"
        f"# CER={score.cer:.3f} WER={score.wer:.3f} F1={score.word_f1:.3f} "
        f"term_WER={score.term_wer:.3f} field_recall={score.field_recall:.3f}\n"
        f"# latency_ms={latency_ms:.0f}\n\n"
    )
    (d / f"{n}.txt").write_text(header + text + "\n", encoding="utf-8")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def benchmark_model(
    spec: OcrModelSpec, refs: dict[int, str], docs: list[tuple[int, Path]]
) -> ModelRow:
    print(f"\n=== {spec.name}  [{spec.backend}] ===")
    row = ModelRow(
        model=spec.name,
        family=spec.family,
        backend=spec.backend,
        is_baseline=spec.is_baseline,
    )

    backend = BACKENDS[spec.backend]
    why = backend.available(spec)
    if why:
        row.load_error = f"skipped: {why}"
        print(f"  ⨯ omitido — {why}")
        return row

    budget_mb = MAX_RAM_GB * 1024
    if spec.ram_est_mb > budget_mb:
        row.load_error = (
            f"skipped: footprint estimado {spec.ram_est_mb} MB > "
            f"presupuesto {budget_mb:.0f} MB"
        )
        print(f"  ⨯ omitido — {row.load_error}")
        return row

    mem_before = get_rss_mb()
    t0 = time.perf_counter()
    try:
        model = backend.load(spec)
    except Exception as exc:  # noqa: BLE001
        row.load_error = f"{type(exc).__name__}: {exc}"[:200]
        print(f"  ⨯ error al cargar: {row.load_error}")
        return row
    row.load_time_ms = round((time.perf_counter() - t0) * 1000, 1)
    row.mem_after_load_mb = round(get_rss_mb(), 1)
    row.ram_peak_mb = row.mem_after_load_mb
    print(
        f"  cargado en {row.load_time_ms:.0f} ms · RAM {row.mem_after_load_mb:.0f} MB "
        f"(Δ {row.mem_after_load_mb - mem_before:.0f})"
    )

    cer_s = wer_s = cacc_s = wp_s = wr_s = wf1_s = twer_s = fr_s = 0.0
    proc_total = 0.0
    latencies: list[float] = []
    n_ok = 0
    for n, path in docs:
        ref = refs.get(n)
        if ref is None:
            continue
        t = time.perf_counter()
        try:
            text = backend.ocr(model, path)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⨯ doc {n}: {type(exc).__name__}: {exc}")
            continue
        latency_ms = (time.perf_counter() - t) * 1000
        row.ram_peak_mb = max(row.ram_peak_mb, get_rss_mb())

        score = score_ocr(ref, text)
        _write_prediction(spec.name, n, text, score, latency_ms)

        cer_s += score.cer
        wer_s += score.wer
        cacc_s += score.char_acc
        wp_s += score.word_precision
        wr_s += score.word_recall
        wf1_s += score.word_f1
        twer_s += score.term_wer
        fr_s += score.field_recall
        proc_total += latency_ms
        latencies.append(latency_ms)
        n_ok += 1

    if n_ok:
        row.n_docs = n_ok
        row.cer = round(cer_s / n_ok, 4)
        row.wer = round(wer_s / n_ok, 4)
        row.char_acc = round(cacc_s / n_ok, 4)
        row.word_precision = round(wp_s / n_ok, 4)
        row.word_recall = round(wr_s / n_ok, 4)
        row.word_f1 = round(wf1_s / n_ok, 4)
        row.term_wer = round(twer_s / n_ok, 4)
        row.field_recall = round(fr_s / n_ok, 4)
        row.mean_latency_ms = round(proc_total / n_ok, 1)
        row.p95_latency_ms = round(_percentile(latencies, 0.95), 1)
        row.pages_per_sec = round(1000.0 * n_ok / proc_total, 3) if proc_total else 0.0
        row.ram_peak_mb = round(row.ram_peak_mb, 1)
        print(
            f"  ✓ {n_ok} docs · CER {row.cer:.3f} · WER {row.wer:.3f} · "
            f"F1 {row.word_f1:.3f} · term_WER {row.term_wer:.3f} · "
            f"field_recall {row.field_recall:.3f} · "
            f"latencia {row.mean_latency_ms:.0f} ms · "
            f"{row.pages_per_sec:.2f} pág/s · RAM_peak {row.ram_peak_mb:.0f} MB"
        )

    try:
        backend.unload(model)
    except Exception:  # noqa: BLE001
        pass
    del model
    gc.collect()
    return row


def _row_to_dict(row: ModelRow) -> dict:
    return {c: getattr(row, c) for c in RESULT_COLUMNS}


def write_error_report(rows: list[ModelRow]) -> Path:
    entries = [
        {
            "model": r.model,
            "backend": r.backend,
            "status": "ok" if not r.load_error and r.n_docs else "con_problemas",
            "load_error": r.load_error or None,
            "n_docs": r.n_docs,
        }
        for r in rows
    ]
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "host": collect_system_info(),
        "config": {
            "max_ram_gb": MAX_RAM_GB,
            "threads": N_THREADS,
            "input_kind": INPUT_KIND,
            "doc_limit": DOC_LIMIT,
        },
        "summary": {
            "total": len(rows),
            "con_problemas": sum(1 for e in entries if e["status"] != "ok"),
        },
        "models": entries,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"error__{datetime.now().strftime('%d_%m_%Y__%H_%M')}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_benchmark(models: list[OcrModelSpec] = MODELS):
    """Ejecuta el benchmark OCR completo y devuelve un ``pandas.DataFrame``.

    Una fila por modelo con CER / WER / P-R-F1 / term-WER / field-recall /
    latencia / throughput / RAM-peak. Escribe ``predictions/<modelo>/{n}.txt``
    por cada documento y un ``error__*.json`` con los modelos omitidos.
    """
    import pandas as pd

    print_system_info()
    refs = load_references()
    docs = list_inputs()
    src = IMAGE_DIR if INPUT_KIND == "image" else PDF_DIR
    print(f"Documentos: {len(docs)} ({INPUT_KIND}) en {src} · referencias: {len(refs)}")
    print(
        f"Modelos: {len(models)} · presupuesto RAM: {MAX_RAM_GB} GB · "
        f"hilos: {N_THREADS}"
    )

    rows: list[ModelRow] = []
    t_run = time.perf_counter()
    for spec in models:
        rows.append(benchmark_model(spec, refs, docs))
        gc.collect()

    error_path = write_error_report(rows)
    print(f"\n✓ Benchmark completo en {(time.perf_counter() - t_run) / 60:.1f} min")
    print(f"✓ Predicciones en: {OUT_DIR}")
    print(f"✓ Reporte de errores: {error_path}")
    return pd.DataFrame([_row_to_dict(r) for r in rows], columns=RESULT_COLUMNS)


def main() -> int:
    try:
        df = run_benchmark()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    except ImportError as exc:
        print(f"ERROR: falta una dependencia ({exc.name}). pip install pandas")
        return 1
    print("\n" + df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
