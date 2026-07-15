#!/usr/bin/env python3
"""Preparación del dataset ASR para UrgeNurse (comando aparte del benchmark).

Genera los archivos que el benchmark (``asr.py`` / ``asr.ipynb``) da por hechos:

  1. **Perfiles clínicos** (`assets/profiles/{n}.txt`): convierte los 100 perfiles
     `.docx` originales (CSIRO "Synthetic nursing handover") a texto plano con
     python-docx. Son el contexto clínico del paciente, NO la transcripción del
     audio (el handoff hablado es una paráfrasis del perfil).
  2. **Ground-truth verbatim** (`assets/references/{n}.txt`): transcribe cada audio
     UNA vez con Whisper large-v3 (con VAD). Esa salida verbatim es el ground-truth
     contra el que el benchmark calcula el WER. Sesgo conocido: el GT es de la
     familia Whisper (declararlo en la memoria).

Uso:
    python prepare_references.py                 # perfiles + ground-truth
    python prepare_references.py --profiles      # solo perfiles .docx → .txt
    python prepare_references.py --ground-truth  # solo ground-truth (large-v3)
    python prepare_references.py --force         # regenera aunque ya exista

Variables de entorno (compartidas con asr.py): ASR_AUDIO_DIR, ASR_REF_DIR,
ASR_MODELS_DIR, ASR_THREADS. Además:
    ASR_PROFILE_DIR   carpeta de perfiles txt (default: assets/profiles)
    ASR_GT_MODEL      modelo de referencia (default: large-v3; o distil-large-v3)
    ASR_GT_COMPUTE    compute_type del GT (default: int8)

Tras ejecutarlo, `assets/references/` queda listo y ya se puede correr asr.ipynb.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_asr():
    """Importa asr.py por ruta para reutilizar su configuración (rutas, hilos)."""
    import sys

    if "asr" in sys.modules:
        return sys.modules["asr"]
    spec = importlib.util.spec_from_file_location("asr", SCRIPT_DIR / "asr.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["asr"] = mod
    spec.loader.exec_module(mod)
    return mod


asr = _load_asr()

# ─────────────────────────────────────────────────────────────────────────────
# Configuración propia de la preparación (no vive en asr.py)
# ─────────────────────────────────────────────────────────────────────────────

PROFILE_TXT_DIR = Path(
    os.environ.get("ASR_PROFILE_DIR", SCRIPT_DIR / "assets" / "profiles")
)
PROFILES_DIR = (
    SCRIPT_DIR
    / "assets"
    / "script-original"
    / "data"
    / "dataset 1- text files"
    / "100profiles"
)
GT_MODEL = os.environ.get("ASR_GT_MODEL", "large-v3")
GT_COMPUTE = os.environ.get("ASR_GT_COMPUTE", "int8")

# Reparto de los 100 perfiles por especialidad (numeración 1–100). El audio N
# corresponde al perfil .docx N de su categoría.
CATEGORY_RANGES: dict[str, range] = {
    "cardiovascular": range(1, 26),  # 1–25
    "respiratory": range(26, 51),  # 26–50
    "neurological": range(51, 76),  # 51–75
    "renal": range(76, 101),  # 76–100
}


def _category_of(n: int) -> str | None:
    for cat, rng in CATEGORY_RANGES.items():
        if n in rng:
            return cat
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 1) Perfiles clínicos: .docx → .txt
# ─────────────────────────────────────────────────────────────────────────────


def docx_to_text(path: Path) -> str:
    """Extrae el texto de un .docx (un párrafo por línea) con python-docx."""
    import docx  # dependencia: python-docx

    doc = docx.Document(str(path))
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(parts)
    return re.sub(r"[ \t]+", " ", text).strip()  # normaliza dobles espacios


def build_profiles(force: bool = False) -> dict[int, str]:
    """Convierte los perfiles clínicos 1–100 (.docx) a ``assets/profiles/{n}.txt``."""
    PROFILE_TXT_DIR.mkdir(parents=True, exist_ok=True)
    profiles: dict[int, str] = {}
    missing: list[int] = []
    for n in range(1, 101):
        cat = _category_of(n)
        if cat is None:
            continue
        src = PROFILES_DIR / cat / f"{n}.docx"
        dst = PROFILE_TXT_DIR / f"{n}.txt"
        if not src.exists():
            missing.append(n)
            continue
        if dst.exists() and not force:
            profiles[n] = dst.read_text(encoding="utf-8")
            continue
        text = docx_to_text(src)
        dst.write_text(text, encoding="utf-8")
        profiles[n] = text
    print(f"✓ Perfiles clínicos: {len(profiles)} txt en {PROFILE_TXT_DIR}")
    if missing:
        print(f"  ⚠ perfiles .docx no encontrados: {missing}")
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# 2) Ground-truth verbatim (Whisper large-v3)
# ─────────────────────────────────────────────────────────────────────────────


def build_ground_truth(
    model_name: str | None = None, force: bool = False
) -> dict[int, str]:
    """Genera la pseudo-ground-truth verbatim transcribiendo con Whisper large-v3.

    Cada audio se transcribe UNA vez (con VAD para evitar alucinaciones) y su
    salida se guarda en ``assets/references/{n}.txt``. Es caro pero se cachea:
    en corridas posteriores se reutiliza. Sesgo conocido: GT de familia Whisper.
    """
    from faster_whisper import WhisperModel

    model_name = model_name or GT_MODEL
    asr.REF_DIR.mkdir(parents=True, exist_ok=True)
    audios = asr.list_audio()
    pending = [
        (n, p) for n, p in audios if force or not (asr.REF_DIR / f"{n}.txt").exists()
    ]
    refs: dict[int, str] = {}

    if pending:
        print(
            f"Generando ground-truth con '{model_name}' ({GT_COMPUTE}) para "
            f"{len(pending)} audios — esto tarda (una sola vez)…"
        )
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type=GT_COMPUTE,
            cpu_threads=asr.N_THREADS,
            download_root=str(asr.MODELS_DIR),
        )
        for i, (n, wav) in enumerate(pending, 1):
            segments, _ = model.transcribe(
                str(wav),
                language="en",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            (asr.REF_DIR / f"{n}.txt").write_text(text, encoding="utf-8")
            refs[n] = text
            if i % 10 == 0 or i == len(pending):
                print(f"  {i}/{len(pending)} audios transcritos")
        del model
        gc.collect()

    for p in asr.REF_DIR.glob("*.txt"):
        if p.stem.isdigit():
            refs[int(p.stem)] = p.read_text(encoding="utf-8")
    print(f"✓ Ground-truth lista: {len(refs)} referencias en {asr.REF_DIR}")
    return dict(sorted(refs.items()))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prepara perfiles y ground-truth del dataset ASR."
    )
    ap.add_argument(
        "--profiles", action="store_true", help="solo perfiles .docx → .txt"
    )
    ap.add_argument(
        "--ground-truth", action="store_true", help="solo ground-truth (large-v3)"
    )
    ap.add_argument("--force", action="store_true", help="regenera aunque ya exista")
    args = ap.parse_args()

    do_profiles = args.profiles or not args.ground_truth
    do_gt = args.ground_truth or not args.profiles

    if do_profiles:
        build_profiles(force=args.force)
    if do_gt:
        build_ground_truth(force=args.force)
    print("\n✓ Preparación completa: assets/references/ listo para correr asr.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
