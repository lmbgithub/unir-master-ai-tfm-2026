#!/usr/bin/env python3
"""Benchmark de modelos ASR (speech-to-text) en CPU para UrgeNurse.

Mismo espíritu que ``llm_performance.py`` pero para reconocimiento de voz: carga
modelos de ASR pequeños (≤2 GB RAM, solo CPU, cuantizados int8/fp16/Q4/Q5),
transcribe el dataset *Synthetic Nursing Handoff* (100 audios WAV 16 kHz mono
16-bit con su transcripción de referencia) y mide las métricas que importan en
ASR clínico:

  * **WER**  Word Error Rate global.
  * **WER-nursing**  WER restringido a terminología de enfermería (nombres de
    medicamentos, valores numéricos, abreviaturas clínicas).
  * **CER**  Character Error Rate.
  * **RTF**  Real-Time Factor = tiempo_de_proceso / duración_del_audio.
  * **RAM Peak**  pico de RSS del proceso durante la transcripción.
  * **NER Recall**  fracción de entidades clínicas de la referencia recuperadas.
  * **latencia / duración (ms)**  por audio.

Alucinación de Whisper (riesgo documentado por el MIT en silencios y ruido):
cada modelo Whisper se ejecuta con **filtro VAD** (Silero, integrado en
faster-whisper) + **umbral de confianza por palabra** + **detección de
repeticiones**, y se reporta una métrica de alucinación. Ver MITIGATION abajo.

Diseño:
  1. ``run_benchmark()`` recorre el catálogo de modelos (uno a uno, ≤2 GB), por
     cada audio transcribe, escribe ``transcriptions/<modelo>/{n}.txt`` y calcula
     las métricas. Devuelve un ``pandas.DataFrame`` (una fila por modelo).
  2. Cada backend (faster-whisper, whisper.cpp, sherpa-onnx Parakeet/Moonshine,
     Vosk) es OPCIONAL: si la librería o el modelo no están disponibles, el
     modelo se OMITE con un aviso claro (no rompe el benchmark).

REQUISITO: las referencias (ground-truth verbatim) deben existir ya en
``assets/references/{n}.txt``. Se generan con el comando aparte
``prepare_references.py`` (perfiles .docx → .txt + transcripción large-v3). Este
módulo NO las genera: solo las lee.

Uso:
    # 0) una sola vez, deja assets/references/ listo:
    python prepare_references.py
    # 1) desde un notebook:
    import asr
    df = asr.run_benchmark()
    # o por CLI:
    python asr.py

Variables de entorno opcionales:
    ASR_MODELS_DIR     carpeta de modelos (default: ../models junto al script)
    ASR_AUDIO_DIR      carpeta de audios WAV (default: assets/audio)
    ASR_REF_DIR        carpeta de referencias txt (default: assets/references)
    ASR_OUT_DIR        carpeta de transcripciones (default: transcriptions)
    ASR_MAX_RAM_GB     presupuesto de RAM (default: 2.0)
    ASR_LIMIT          nº de audios a procesar (default: 0 = todos)
    ASR_DOWNLOAD       "0" para no descargar modelos faltantes (default: "1")
    ASR_THREADS        nº de hilos (default: núcleos físicos)
    ASR_VAD            "0" para desactivar el filtro VAD anti-alucinación (default "1")
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import json
import os
import re
import time
import unicodedata
import urllib.request
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Reutilizamos los helpers de sistema del benchmark de LLMs (DRY): información del
# host, RSS en MB, descarga con barra de progreso. Import por ruta para no
# depender de la mecánica de paquetes (ambos viven bajo code/scripts/).
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
    # Registrar en sys.modules ANTES de ejecutar: @dataclass lo necesita.
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

MODELS_DIR = Path(os.environ.get("ASR_MODELS_DIR", SCRIPT_DIR.parent / "models"))
AUDIO_DIR = Path(os.environ.get("ASR_AUDIO_DIR", SCRIPT_DIR / "assets" / "audio"))
# REF_DIR = ground-truth verbatim (la genera prepare_references.py; aquí solo se lee).
REF_DIR = Path(os.environ.get("ASR_REF_DIR", SCRIPT_DIR / "assets" / "references"))
OUT_DIR = Path(os.environ.get("ASR_OUT_DIR", SCRIPT_DIR / "transcriptions"))
OUTPUT_DIR = Path(os.environ.get("ASR_OUTPUT_DIR", SCRIPT_DIR))  # error__*.json

MAX_RAM_GB = float(os.environ.get("ASR_MAX_RAM_GB", 2.0))
AUDIO_LIMIT = int(os.environ.get("ASR_LIMIT", 0))  # 0 = todos
DOWNLOAD_MISSING = os.environ.get("ASR_DOWNLOAD", "1") != "0"
N_THREADS = int(os.environ.get("ASR_THREADS", _lp._physical_cores()))
USE_VAD = os.environ.get("ASR_VAD", "1") != "0"

# Umbral de confianza por palabra (logprob medio). Por debajo de esto, una
# palabra se considera potencialmente alucinada. faster-whisper expone avg_logprob
# por segmento y probabilidad por palabra (word_timestamps=True).
WORD_CONF_THRESHOLD = float(os.environ.get("ASR_WORD_CONF", -1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Referencias (ground-truth verbatim) — solo lectura
# ─────────────────────────────────────────────────────────────────────────────


def load_references() -> dict[int, str]:
    """Carga la ground-truth verbatim de ``assets/references/{n}.txt``.

    NO la genera: si falta, indica ejecutar ``prepare_references.py`` primero.
    """
    if not REF_DIR.exists() or not any(REF_DIR.glob("*.txt")):
        raise FileNotFoundError(
            f"No hay referencias en {REF_DIR}. Genera la ground-truth antes de "
            f"correr el benchmark:\n    python prepare_references.py"
        )
    refs: dict[int, str] = {}
    for p in REF_DIR.glob("*.txt"):
        if p.stem.isdigit():
            refs[int(p.stem)] = p.read_text(encoding="utf-8")
    return dict(sorted(refs.items()))


def list_audio() -> list[tuple[int, Path]]:
    """Lista (n, ruta) de los WAV disponibles, ordenados por número."""
    items: list[tuple[int, Path]] = []
    for p in AUDIO_DIR.glob("*.wav"):
        if p.stem.isdigit():
            items.append((int(p.stem), p))
    items.sort(key=lambda x: x[0])
    if AUDIO_LIMIT > 0:
        items = items[:AUDIO_LIMIT]
    return items


def audio_duration_s(path: Path) -> float:
    """Duración del WAV en segundos (lee la cabecera, sin cargar las muestras)."""
    with wave.open(str(path)) as w:
        return w.getnframes() / float(w.getframerate())


# ─────────────────────────────────────────────────────────────────────────────
# Terminología de enfermería para métricas específicas
# ─────────────────────────────────────────────────────────────────────────────
# WER global trata todas las palabras por igual. En enfermería, equivocar un
# medicamento o un valor numérico es mucho más grave que un "the" mal puesto.
# Definimos tres clases de "entidad clínica" y medimos WER/recall solo sobre
# esas palabras de la referencia.

# Medicamentos frecuentes en el dataset (cardio/resp/neuro/renal) + genéricos.
_MED_NAMES = {
    "nitroglycerin",
    "nitro",
    "nitros",
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
    "frusemide",
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
    "ventolin",
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
    "keppra",
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
}
# Sufijos farmacológicos comunes (capturan nombres no listados).
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
# Abreviaturas / unidades clínicas.
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
    "iabp",
    "egfr",
    "bnp",
    "crp",
    "wbc",
    "inr",
    "hb",
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
# Normalización + métricas de transcripción (WER / CER / alineación)
# ─────────────────────────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s./]")


def normalize_text(text: str) -> str:
    """Normaliza para comparar: minúsculas, sin acentos/puntuación, espacios simples.

    Mantiene '/' y '.' dentro de números (150/95, 39.2) porque son clínicamente
    significativos. Quita etiquetas del perfil ('name:', 'age:', etc.) ya que el
    audio narra el contenido, no los rótulos del formulario.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"\b(name|age|admission story|at medical ward)\s*:", " ", text)
    text = text.replace("%", " percent ")
    text = _PUNCT_RE.sub(" ", text)
    # Conserva '.' y '/' SOLO entre dígitos (39.2, 150/95); el resto fuera.
    text = re.sub(r"(?<!\d)[./]|[./](?!\d)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def _levenshtein_ops(
    ref: list[str], hyp: list[str]
) -> tuple[int, int, int, list[tuple]]:
    """Distancia de edición a nivel de palabra con backtrace.

    Devuelve (sub, dele, ins, alineación) donde alineación es una lista de tuplas
    ('ok'|'sub'|'del'|'ins', ref_token_or_None, hyp_token_or_None).
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # deletion
                d[i][j - 1] + 1,  # insertion
                d[i - 1][j - 1] + cost,  # match/substitution
            )
    # Backtrace
    i, j = n, m
    align: list[tuple] = []
    sub = dele = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            align.append(("ok", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            align.append(("sub", ref[i - 1], hyp[j - 1]))
            sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            align.append(("del", ref[i - 1], None))
            dele += 1
            i -= 1
        else:
            align.append(("ins", None, hyp[j - 1]))
            ins += 1
            j -= 1
    align.reverse()
    return sub, dele, ins, align


@dataclass
class TranscriptScore:
    wer: float = 0.0
    cer: float = 0.0
    term_wer: float = 0.0  # WER sobre terminología de enfermería
    ner_recall: float = 0.0  # entidades clínicas recuperadas / esperadas
    n_ref_words: int = 0
    n_terms: int = 0
    sub: int = 0
    dele: int = 0
    ins: int = 0


def score_transcription(reference: str, hypothesis: str) -> TranscriptScore:
    """Compara hipótesis vs referencia: WER, CER, WER-nursing y NER recall."""
    ref_tok = tokenize(reference)
    hyp_tok = tokenize(hypothesis)
    sub, dele, ins, align = _levenshtein_ops(ref_tok, hyp_tok)
    n = max(1, len(ref_tok))
    wer = (sub + dele + ins) / n

    # CER sobre texto normalizado sin espacios.
    ref_c = list(normalize_text(reference).replace(" ", ""))
    hyp_c = list(normalize_text(hypothesis).replace(" ", ""))
    cs, cd, ci, _ = _levenshtein_ops(ref_c, hyp_c)
    cer = (cs + cd + ci) / max(1, len(ref_c))

    # Métricas restringidas a terminología clínica de la referencia.
    n_terms = term_err = term_hit = 0
    for op, r_tok, _h in align:
        if r_tok is None:
            continue
        if classify_term(r_tok) is None:
            continue
        n_terms += 1
        if op == "ok":
            term_hit += 1
        else:  # sub o del sobre un término clínico
            term_err += 1
    term_wer = term_err / n_terms if n_terms else 0.0
    ner_recall = term_hit / n_terms if n_terms else 1.0

    return TranscriptScore(
        wer=round(wer, 4),
        cer=round(cer, 4),
        term_wer=round(term_wer, 4),
        ner_recall=round(ner_recall, 4),
        n_ref_words=len(ref_tok),
        n_terms=n_terms,
        sub=sub,
        dele=dele,
        ins=ins,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Detección de alucinación (riesgo de Whisper en silencios/ruido)
# ─────────────────────────────────────────────────────────────────────────────


def hallucination_signals(text: str, avg_word_conf: float | None) -> dict:
    """Heurística ligera para detectar alucinaciones típicas de Whisper.

    - Repetición: n-gramas que se repiten muchas veces (loop de decodificación).
    - Frases "fantasma": muletillas que Whisper inventa en silencio
      (p. ej. "thank you", "thanks for watching", subtítulos).
    - Confianza media por palabra por debajo del umbral.
    """
    norm = normalize_text(text)
    words = norm.split()
    flags: list[str] = []

    # Repetición: ¿algún trigrama representa >30% del texto?
    repeated = False
    if len(words) >= 6:
        from collections import Counter

        trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        top, cnt = Counter(trigrams).most_common(1)[0]
        if cnt >= max(3, len(trigrams) * 0.3):
            repeated = True
            flags.append(f"repeat:'{top}'x{cnt}")

    ghosts = (
        "thank you",
        "thanks for watching",
        "please subscribe",
        "subtitles by",
        "for watching",
    )
    ghost = any(g in norm for g in ghosts)
    if ghost:
        flags.append("ghost_phrase")

    low_conf = avg_word_conf is not None and avg_word_conf < WORD_CONF_THRESHOLD
    if low_conf:
        flags.append(f"low_conf:{avg_word_conf:.2f}")

    return {
        "hallucinated": bool(repeated or ghost or low_conf),
        "repeated": repeated,
        "ghost": ghost,
        "low_conf": low_conf,
        "flags": flags,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Catálogo de modelos ASR
# ─────────────────────────────────────────────────────────────────────────────
# Cada modelo declara su `backend`. La función de carga/transcripción concreta
# vive en BACKENDS. Modelos pequeños, CPU, cuantizados, ≤2 GB de RAM.
#
#   faster_whisper : Whisper en CTranslate2 (int8 / int8_float16 / float16).
#                    Trae VAD Silero integrado y confianza por palabra. Cubre la
#                    "variante base de Whisper" y la comparación de cuantización.
#   whispercpp     : whisper.cpp (ggml/GGUF base/small, Q5/Q8) vía pywhispercpp.
#   sherpa_parakeet: NVIDIA Parakeet TDT 0.6B int8 (transducer) vía sherpa-onnx.
#   sherpa_moonshine: Moonshine base int8 vía sherpa-onnx (popular, muy ligero).
#   vosk           : Kaldi/Vosk small en (popular, ~40 MB, RAM mínima).


@dataclass(frozen=True)
class AsrModelSpec:
    name: str  # etiqueta única en tablas/figuras
    backend: str  # clave en BACKENDS
    family: str  # agrupación (whisper / parakeet / moonshine / vosk)
    is_whisper: bool = False  # sujeto al riesgo de alucinación de Whisper
    is_baseline: bool = False  # "variante base de Whisper" de referencia
    ram_est_mb: float = 600.0  # footprint estimado para el presupuesto de RAM
    # Parámetros libres que interpreta el backend (model size, compute_type, url…).
    params: dict = field(default_factory=dict)


MODELS: list[AsrModelSpec] = [
    # ── Whisper (faster-whisper / CTranslate2) ───────────────────────────────
    # Variante BASE de referencia para justificar la elección (int8).
    AsrModelSpec(
        "faster-whisper-base.en-int8",
        "faster_whisper",
        "whisper",
        is_whisper=True,
        is_baseline=True,
        ram_est_mb=350,
        params={"size": "base.en", "compute_type": "int8"},
    ),
    AsrModelSpec(
        "faster-whisper-tiny.en-int8",
        "faster_whisper",
        "whisper",
        is_whisper=True,
        ram_est_mb=200,
        params={"size": "tiny.en", "compute_type": "int8"},
    ),
    # Small en int8 y fp16 (comparación de cuantización, ≤2 GB).
    AsrModelSpec(
        "faster-whisper-small.en-int8",
        "faster_whisper",
        "whisper",
        is_whisper=True,
        ram_est_mb=700,
        params={"size": "small.en", "compute_type": "int8"},
    ),
    AsrModelSpec(
        "faster-whisper-small.en-fp16",
        "faster_whisper",
        "whisper",
        is_whisper=True,
        ram_est_mb=1100,
        params={"size": "small.en", "compute_type": "float16"},
    ),
    # Distil-Whisper small.en (popular, ~2x más rápido, int8).
    AsrModelSpec(
        "distil-whisper-small.en-int8",
        "faster_whisper",
        "whisper",
        is_whisper=True,
        ram_est_mb=700,
        params={"size": "distil-small.en", "compute_type": "int8"},
    ),
    # ── whisper.cpp (ggml/GGUF cuantizado) ───────────────────────────────────
    AsrModelSpec(
        "whisper.cpp-base.en-q5_1",
        "whispercpp",
        "whisper",
        is_whisper=True,
        ram_est_mb=300,
        params={"model": "base.en-q5_1"},
    ),
    AsrModelSpec(
        "whisper.cpp-small.en-q5_1",
        "whispercpp",
        "whisper",
        is_whisper=True,
        ram_est_mb=600,
        params={"model": "small.en-q5_1"},
    ),
    # ── NVIDIA Parakeet TDT 0.6B (int8, transducer) vía sherpa-onnx ───────────
    AsrModelSpec(
        "parakeet-tdt-0.6b-int8",
        "sherpa_parakeet",
        "parakeet",
        ram_est_mb=1300,
        params={
            "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
            "dirname": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
        },
    ),
    # ── Moonshine base (popular, ultraligero) vía sherpa-onnx ────────────────
    AsrModelSpec(
        "moonshine-base-int8",
        "sherpa_moonshine",
        "moonshine",
        ram_est_mb=500,
        params={
            "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-moonshine-base-en-int8.tar.bz2",
            "dirname": "sherpa-onnx-moonshine-base-en-int8",
        },
    ),
    # ── Vosk small en (popular, RAM mínima) ──────────────────────────────────
    AsrModelSpec(
        "vosk-small-en-0.15",
        "vosk",
        "vosk",
        ram_est_mb=300,
        params={
            "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
            "dirname": "vosk-model-small-en-us-0.15",
        },
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Descarga de artefactos (modelos sherpa/vosk empaquetados)
# ─────────────────────────────────────────────────────────────────────────────


def _download(url: str, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  ↓ descargando {dst.name}\n    desde {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "urgenurse-asr/1.0"})
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done = last = 0
            while True:
                buf = resp.read(1024 * 1024)
                if not buf:
                    break
                out.write(buf)
                done += len(buf)
                if total and (pct := int(done * 100 / total)) >= last + 5:
                    print(f"    {pct:3d}%  ({human_mb(done)} / {human_mb(total)} MB)")
                    last = pct
        tmp.rename(dst)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ⨯ fallo al descargar {dst.name}: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def _ensure_archive(url: str, dirname: str) -> Path | None:
    """Descarga y extrae un .tar.bz2 / .zip de modelo si falta. Devuelve la carpeta."""
    target = MODELS_DIR / dirname
    if target.exists() and any(target.iterdir()):
        return target
    if not DOWNLOAD_MISSING:
        print(f"  ⨯ {dirname} no está y la descarga está desactivada — se omite")
        return None
    archive = MODELS_DIR / url.split("/")[-1]
    if not archive.exists() and not _download(url, archive):
        return None
    try:
        if archive.suffix == ".zip":
            import zipfile

            with zipfile.ZipFile(archive) as z:
                z.extractall(MODELS_DIR)
        else:
            import tarfile

            with tarfile.open(archive) as t:
                t.extractall(MODELS_DIR)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⨯ fallo al extraer {archive.name}: {exc}")
        return None
    return target if target.exists() else None


def _have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Backends de ASR
# ─────────────────────────────────────────────────────────────────────────────
# Cada backend implementa load(spec) -> objeto modelo, y transcribe(model, wav)
# -> (texto, avg_word_conf|None). Lanza RuntimeError/ImportError si no está
# disponible; el runner lo captura y OMITE el modelo con un aviso.


class Backend:
    name = "base"

    def available(self, spec: AsrModelSpec) -> str:
        """'' si está disponible; si no, motivo legible para omitir el modelo."""
        return ""

    def load(self, spec: AsrModelSpec):
        raise NotImplementedError

    def transcribe(self, model, wav: Path) -> tuple[str, float | None]:
        raise NotImplementedError

    def unload(self, model) -> None:
        del model


class FasterWhisperBackend(Backend):
    name = "faster_whisper"

    def available(self, spec: AsrModelSpec) -> str:
        return (
            ""
            if _have_module("faster_whisper")
            else "falta 'faster-whisper' (pip install faster-whisper)"
        )

    def load(self, spec: AsrModelSpec):
        from faster_whisper import WhisperModel

        size = spec.params["size"]
        compute = spec.params.get("compute_type", "int8")
        return WhisperModel(
            size,
            device="cpu",
            compute_type=compute,
            cpu_threads=N_THREADS,
            download_root=str(MODELS_DIR),
        )

    def transcribe(self, model, wav: Path) -> tuple[str, float | None]:
        # vad_filter=True → Silero VAD recorta silencios: mitiga la alucinación de
        # Whisper en silencio/ruido (riesgo documentado por el MIT).
        segments, _info = model.transcribe(
            str(wav),
            language="en",
            beam_size=5,
            vad_filter=USE_VAD,
            vad_parameters={"min_silence_duration_ms": 500} if USE_VAD else None,
            word_timestamps=True,
            condition_on_previous_text=False,  # evita arrastrar alucinaciones
        )
        parts: list[str] = []
        probs: list[float] = []
        for seg in segments:
            parts.append(seg.text)
            for w in seg.words or []:
                probs.append(w.probability)
        text = " ".join(p.strip() for p in parts).strip()
        # avg_word_conf en escala logprob aprox: log(prob_media) para comparar con umbral.
        import math

        avg_conf = math.log(sum(probs) / len(probs)) if probs else None
        return text, avg_conf


class WhisperCppBackend(Backend):
    name = "whispercpp"

    def available(self, spec: AsrModelSpec) -> str:
        return (
            ""
            if _have_module("pywhispercpp")
            else "falta 'pywhispercpp' (pip install pywhispercpp)"
        )

    def load(self, spec: AsrModelSpec):
        from pywhispercpp.model import Model

        # pywhispercpp descarga el ggml cuantizado por nombre (p. ej. 'base.en-q5_1').
        return Model(
            spec.params["model"],
            models_dir=str(MODELS_DIR),
            n_threads=N_THREADS,
            redirect_whispercpp_logs_to=None,
        )

    def transcribe(self, model, wav: Path) -> tuple[str, float | None]:
        segs = model.transcribe(str(wav), language="en")
        text = " ".join(s.text.strip() for s in segs).strip()
        return text, None


class _SherpaBase(Backend):
    """Común a modelos sherp-onnx offline (Parakeet, Moonshine, Whisper)."""

    def available(self, spec: AsrModelSpec) -> str:
        return (
            ""
            if _have_module("sherpa_onnx")
            else "falta 'sherpa-onnx' (pip install sherpa-onnx)"
        )

    def _read_wav(self, wav: Path):
        import numpy as np

        with wave.open(str(wav)) as w:
            sr = w.getframerate()
            frames = w.readframes(w.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype("float32") / 32768.0
        return samples, sr

    def transcribe(self, model, wav: Path) -> tuple[str, float | None]:
        samples, sr = self._read_wav(wav)
        stream = model.create_stream()
        stream.accept_waveform(sr, samples)
        model.decode_stream(stream)
        return stream.result.text.strip(), None


class SherpaParakeetBackend(_SherpaBase):
    name = "sherpa_parakeet"

    def load(self, spec: AsrModelSpec):
        import sherpa_onnx

        d = _ensure_archive(spec.params["url"], spec.params["dirname"])
        if d is None:
            raise RuntimeError("modelo Parakeet no disponible")
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"),
            decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"),
            tokens=str(d / "tokens.txt"),
            num_threads=N_THREADS,
            model_type="nemo_transducer",
        )


class SherpaMoonshineBackend(_SherpaBase):
    name = "sherpa_moonshine"

    def load(self, spec: AsrModelSpec):
        import sherpa_onnx

        d = _ensure_archive(spec.params["url"], spec.params["dirname"])
        if d is None:
            raise RuntimeError("modelo Moonshine no disponible")
        return sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=str(d / "preprocess.onnx"),
            encoder=str(d / "encode.int8.onnx"),
            uncached_decoder=str(d / "uncached_decode.int8.onnx"),
            cached_decoder=str(d / "cached_decode.int8.onnx"),
            tokens=str(d / "tokens.txt"),
            num_threads=N_THREADS,
        )


class VoskBackend(Backend):
    name = "vosk"

    def available(self, spec: AsrModelSpec) -> str:
        return "" if _have_module("vosk") else "falta 'vosk' (pip install vosk)"

    def load(self, spec: AsrModelSpec):
        from vosk import Model

        d = _ensure_archive(spec.params["url"], spec.params["dirname"])
        if d is None:
            raise RuntimeError("modelo Vosk no disponible")
        return Model(str(d))

    def transcribe(self, model, wav: Path) -> tuple[str, float | None]:
        from vosk import KaldiRecognizer

        with wave.open(str(wav)) as w:
            rec = KaldiRecognizer(model, w.getframerate())
            rec.SetWords(True)
            out: list[str] = []
            while True:
                data = w.readframes(4000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    out.append(json.loads(rec.Result()).get("text", ""))
            out.append(json.loads(rec.FinalResult()).get("text", ""))
        return " ".join(t for t in out if t).strip(), None


BACKENDS: dict[str, Backend] = {
    "faster_whisper": FasterWhisperBackend(),
    "whispercpp": WhisperCppBackend(),
    "sherpa_parakeet": SherpaParakeetBackend(),
    "sherpa_moonshine": SherpaMoonshineBackend(),
    "vosk": VoskBackend(),
}


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ModelRow:
    model: str
    family: str
    backend: str
    is_whisper: bool
    is_baseline: bool
    load_time_ms: float = 0.0
    mem_after_load_mb: float = 0.0
    ram_peak_mb: float = 0.0
    n_audios: int = 0
    # Agregados de métricas (media sobre audios)
    wer: float = 0.0
    cer: float = 0.0
    term_wer: float = 0.0
    ner_recall: float = 0.0
    rtf: float = 0.0
    mean_latency_ms: float = 0.0
    mean_audio_ms: float = 0.0
    hallucination_rate: float = 0.0
    load_error: str = ""


RESULT_COLUMNS = [
    "model",
    "family",
    "backend",
    "is_whisper",
    "is_baseline",
    "load_time_ms",
    "mem_after_load_mb",
    "ram_peak_mb",
    "n_audios",
    "wer",
    "cer",
    "term_wer",
    "ner_recall",
    "rtf",
    "mean_latency_ms",
    "mean_audio_ms",
    "hallucination_rate",
    "load_error",
]


def _write_transcription(
    model_name: str,
    n: int,
    text: str,
    score: TranscriptScore,
    halluc: dict,
    latency_ms: float,
    audio_ms: float,
) -> None:
    """Escribe transcriptions/<modelo>/{n}.txt con la hipótesis + métricas del audio."""
    d = OUT_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    header = (
        f"# audio {n}.wav · modelo {model_name}\n"
        f"# WER={score.wer:.3f} CER={score.cer:.3f} term_WER={score.term_wer:.3f} "
        f"NER_recall={score.ner_recall:.3f}\n"
        f"# latency_ms={latency_ms:.0f} audio_ms={audio_ms:.0f} "
        f"RTF={latency_ms / audio_ms if audio_ms else 0:.3f}\n"
        f"# hallucination={halluc['hallucinated']} flags={halluc['flags']}\n\n"
    )
    (d / f"{n}.txt").write_text(header + text + "\n", encoding="utf-8")


def benchmark_model(
    spec: AsrModelSpec, refs: dict[int, str], audios: list[tuple[int, Path]]
) -> ModelRow:
    print(f"\n=== {spec.name}  [{spec.backend}] ===")
    row = ModelRow(
        model=spec.name,
        family=spec.family,
        backend=spec.backend,
        is_whisper=spec.is_whisper,
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
        row.load_error = f"skipped: footprint estimado {spec.ram_est_mb} MB > presupuesto {budget_mb:.0f} MB"
        print(f"  ⨯ omitido — {row.load_error}")
        return row

    # Cargar modelo
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
        f"(Δ {row.mem_after_load_mb - mem_before:.0f}) · VAD={'on' if USE_VAD and spec.is_whisper else 'n/a'}"
    )

    # Transcribir cada audio
    wer_s = cer_s = twer_s = ner_s = 0.0
    proc_total = audio_total = 0.0
    halluc_count = 0
    n_ok = 0
    for n, wav in audios:
        ref = refs.get(n)
        if ref is None:
            continue
        audio_ms = audio_duration_s(wav) * 1000
        t = time.perf_counter()
        try:
            text, conf = backend.transcribe(model, wav)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⨯ audio {n}: {type(exc).__name__}: {exc}")
            continue
        latency_ms = (time.perf_counter() - t) * 1000
        row.ram_peak_mb = max(row.ram_peak_mb, get_rss_mb())

        score = score_transcription(ref, text)
        halluc = hallucination_signals(text, conf)
        _write_transcription(spec.name, n, text, score, halluc, latency_ms, audio_ms)

        wer_s += score.wer
        cer_s += score.cer
        twer_s += score.term_wer
        ner_s += score.ner_recall
        proc_total += latency_ms
        audio_total += audio_ms
        halluc_count += int(halluc["hallucinated"])
        n_ok += 1

    if n_ok:
        row.n_audios = n_ok
        row.wer = round(wer_s / n_ok, 4)
        row.cer = round(cer_s / n_ok, 4)
        row.term_wer = round(twer_s / n_ok, 4)
        row.ner_recall = round(ner_s / n_ok, 4)
        row.rtf = round(proc_total / audio_total, 4) if audio_total else 0.0
        row.mean_latency_ms = round(proc_total / n_ok, 1)
        row.mean_audio_ms = round(audio_total / n_ok, 1)
        row.hallucination_rate = round(halluc_count / n_ok, 4)
        row.ram_peak_mb = round(row.ram_peak_mb, 1)
        print(
            f"  ✓ {n_ok} audios · WER {row.wer:.3f} · CER {row.cer:.3f} · "
            f"term_WER {row.term_wer:.3f} · NER_recall {row.ner_recall:.3f} · "
            f"RTF {row.rtf:.3f} · RAM_peak {row.ram_peak_mb:.0f} MB · "
            f"halluc {row.hallucination_rate:.0%}"
        )

    # Liberar
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
            "status": "ok" if not r.load_error and r.n_audios else "con_problemas",
            "load_error": r.load_error or None,
            "n_audios": r.n_audios,
        }
        for r in rows
    ]
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "host": collect_system_info(),
        "config": {
            "max_ram_gb": MAX_RAM_GB,
            "threads": N_THREADS,
            "vad": USE_VAD,
            "audio_limit": AUDIO_LIMIT,
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


def run_benchmark(models: list[AsrModelSpec] = MODELS):
    """Ejecuta el benchmark ASR completo y devuelve un ``pandas.DataFrame``.

    Una fila por modelo con WER / CER / term-WER / NER-recall / RTF / RAM-peak /
    latencia. Escribe ``transcriptions/<modelo>/{n}.txt`` por cada audio y un
    ``error__*.json`` con los modelos omitidos o con problemas.
    """
    import pandas as pd

    print_system_info()
    refs = load_references()
    audios = list_audio()
    print(f"Audios: {len(audios)} en {AUDIO_DIR} · referencias: {len(refs)}")
    print(
        f"Modelos: {len(models)} · presupuesto RAM: {MAX_RAM_GB} GB · "
        f"hilos: {N_THREADS} · VAD anti-alucinación: {'on' if USE_VAD else 'off'}"
    )

    rows: list[ModelRow] = []
    t_run = time.perf_counter()
    for spec in models:
        rows.append(benchmark_model(spec, refs, audios))
        gc.collect()

    error_path = write_error_report(rows)
    print(f"\n✓ Benchmark completo en {(time.perf_counter() - t_run) / 60:.1f} min")
    print(f"✓ Transcripciones en: {OUT_DIR}")
    print(f"✓ Reporte de errores: {error_path}")
    return pd.DataFrame([_row_to_dict(r) for r in rows], columns=RESULT_COLUMNS)


def main() -> int:
    try:
        df = run_benchmark()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    except ImportError as exc:
        print(
            f"ERROR: falta una dependencia ({exc.name}). pip install pandas faster-whisper"
        )
        return 1
    print("\n" + df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
