#!/usr/bin/env python3
"""Preparación del dataset de evaluación del Agente Triage ESI (MIMIC-IV-ED).

Lee las tablas `triage` y `edstays` de MIMIC-IV-ED v2.2 (acceso credencializado
PhysioNet — ver LICENSE.txt en data/mimic-iv-ed/2.2), limpia los casos sin
acuity o motivo de consulta, y calcula la discordancia entre el nivel ESI
documentado y las vitales de "danger zone" del estándar ESI v5 (Decision
Point D, regla compartida con el agente real en
packages/agent-triage/.../esi_rules.py). Deja además una muestra estratificada
por acuity lista para evaluate.py.

MIMIC-IV-ED se usa aquí como fuente de EDA y como conjunto de evaluación del
validador de reglas ya implementado en agent-triage — NO como conjunto de
entrenamiento de ningún modelo (ver document/src/results.tex, sección
"Módulo de triage ESI").

Uso:
    python prepare_dataset.py                  # EDA + muestra estratificada
    python prepare_dataset.py --eda-only        # solo el EDA, sin muestrear
    python prepare_dataset.py --sample-size 800 # tamaño de muestra (default 800)
    python prepare_dataset.py --force           # regenera aunque ya exista

Variables de entorno:
    MIMIC_ED_DIR    carpeta con las tablas ed/*.csv.gz
                    (default: data/mimic-iv-ed/2.2/ed)
    TRIAGE_OUT_DIR  carpeta de salida (default: data/processed)
    TRIAGE_SEED     semilla del muestreo estratificado (default: 42)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
_PACKAGES_DIR = SCRIPT_DIR.parent.parent / "packages"
for _pkg in ("agent", "agent-triage"):
    _src = _PACKAGES_DIR / _pkg / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from urgenurse.agents.triage.esi_rules import danger_zone_flag  # noqa: E402
MIMIC_ED_DIR = Path(
    os.environ.get("MIMIC_ED_DIR", SCRIPT_DIR / "data" / "mimic-iv-ed" / "2.2" / "ed")
)
OUT_DIR = Path(os.environ.get("TRIAGE_OUT_DIR", SCRIPT_DIR / "data" / "processed"))
SEED = int(os.environ.get("TRIAGE_SEED", 42))
SAMPLE_SIZE = 800

PATIENT_COLS = ["gender", "race", "arrival_transport", "disposition"]
VITAL_COLS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]


def load_triage() -> pd.DataFrame:
    triage = pd.read_csv(MIMIC_ED_DIR / "triage.csv.gz", compression="gzip")
    edstays = pd.read_csv(
        MIMIC_ED_DIR / "edstays.csv.gz",
        compression="gzip",
        usecols=["stay_id", "subject_id", *PATIENT_COLS],
    )
    return triage.merge(edstays, on=["stay_id", "subject_id"], how="left")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta casos sin acuity válida (1-5) o sin motivo de consulta —
    ambos son obligatorios para poder alimentar al agente y compararlo."""
    df = df.copy()
    df["acuity"] = pd.to_numeric(df["acuity"], errors="coerce")
    df = df.dropna(subset=["acuity"])
    df = df[df["acuity"].between(1, 5)]
    df = df[df["chiefcomplaint"].notna() & (df["chiefcomplaint"].str.strip() != "")]
    df["acuity"] = df["acuity"].astype(int)
    return df.reset_index(drop=True)


def is_discordant(acuity: int, danger_zone: bool | None) -> bool | None:
    """Patrón de discordancia descrito en results.tex: un nivel de baja
    prioridad (ESI 3-5) documentado mientras coexisten vitales de danger-zone.
    Concepto exclusivo del EDA offline — el agente de producción no compara
    contra ningún nivel "documentado", solo reescala con danger_zone_flag."""
    if danger_zone is None:
        return None
    return bool(danger_zone) and acuity >= 3


def annotate_danger_zone(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["danger_zone"] = [
        danger_zone_flag(hr, rr, o2)
        for hr, rr, o2 in zip(df["heartrate"], df["resprate"], df["o2sat"])
    ]
    df["discordant"] = [
        is_discordant(acuity, dz) for acuity, dz in zip(df["acuity"], df["danger_zone"])
    ]
    return df


def eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de discordancia por nivel de acuity, solo sobre casos con al menos
    una vital de danger-zone registrada (danger_zone no nulo)."""
    scored = df[df["danger_zone"].notna()]
    summary = (
        scored.groupby("acuity")
        .agg(
            n=("stay_id", "count"),
            danger_zone_rate=("danger_zone", "mean"),
            discordant_n=("discordant", "sum"),
        )
        .reset_index()
    )
    summary["discordant_rate"] = summary["discordant_n"] / summary["n"]
    return summary


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Muestra balanceada entre los 5 niveles de acuity (no proporcional a su
    frecuencia real), para que los niveles minoritarios (ESI 1 y 5, <6% del
    corpus) tengan representación suficiente y el kappa no se calcule casi sin
    datos de esas clases. Si un nivel no tiene suficientes casos disponibles,
    se toman todos los que haya."""
    target_per_level = n // 5
    parts = [
        df[df["acuity"] == level].sample(
            n=min(target_per_level, int((df["acuity"] == level).sum())),
            random_state=seed,
        )
        for level in range(1, 6)
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prepara el dataset de evaluación del Agente Triage ESI sobre MIMIC-IV-ED."
    )
    ap.add_argument("--eda-only", action="store_true", help="solo el EDA, sin muestrear")
    ap.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help=f"tamaño de la muestra estratificada (default {SAMPLE_SIZE})",
    )
    ap.add_argument("--force", action="store_true", help="regenera aunque ya exista")
    args = ap.parse_args()

    if not MIMIC_ED_DIR.exists():
        raise SystemExit(f"No se encuentra {MIMIC_ED_DIR}. ¿Está el dataset descargado ahí?")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eda_path = OUT_DIR / "eda_discordance.csv"
    sample_path = OUT_DIR / "triage_sample.csv"

    print(f"Cargando triage + edstays desde {MIMIC_ED_DIR} …")
    df = load_triage()
    print(f"  {len(df)} filas crudas")

    df = clean(df)
    print(f"  {len(df)} filas tras limpieza (acuity + motivo de consulta presentes)")

    df = annotate_danger_zone(df)

    summary = eda_summary(df)
    summary.to_csv(eda_path, index=False)
    print(f"\n✓ EDA de discordancia guardado en {eda_path}")
    print(summary.to_string(index=False))

    overall_rate = df["discordant"].mean(skipna=True)
    print(f"\nTasa global de discordancia (ESI≥3 + danger-zone vitals): {overall_rate:.1%}")

    if args.eda_only:
        return 0

    if sample_path.exists() and not args.force:
        print(f"\n{sample_path} ya existe (usa --force para regenerar). Nada más que hacer.")
        return 0

    sample = stratified_sample(df, args.sample_size, SEED)
    sample.to_csv(sample_path, index=False)
    print(f"\n✓ Muestra estratificada ({len(sample)} casos) guardada en {sample_path}")
    print(sample["acuity"].value_counts().sort_index().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
