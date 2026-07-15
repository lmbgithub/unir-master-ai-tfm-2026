#!/usr/bin/env python3
"""Evaluación del Agente Triage ESI contra MIMIC-IV-ED (kappa agente vs acuity documentado).

Corre el validador real de `packages/agent-triage` (`evaluate_triage`, sin
pasar por NATS) sobre la muestra que deja `prepare_dataset.py`, comparando el
`esi_level` que el agente asigna de forma independiente contra el `acuity`
documentado en MIMIC-IV-ED (la clasificación de la enfermera de triage). El
kappa de Cohen mide esa concordancia agente-vs-humano — ver
document/src/results.tex, sección "Módulo de triage ESI".

Reanudable: si ya existe `triage_eval_results.csv`, los `stay_id` ya evaluados
se saltan (usa --force para reevaluar desde cero). Cada caso es una llamada
real al LLM (temperature=0.0), así que con la muestra completa esto puede
tardar — usa --limit para una corrida corta de verificación primero.

Uso:
    python evaluate.py --limit 20                  # smoke test
    python evaluate.py                              # muestra completa
    python evaluate.py --force                      # reevalúa todo desde cero
    python evaluate.py --llm-url http://localhost:8080

Requiere el servicio `llm` (llama.cpp) corriendo y accesible — con
`docker compose up -d llm` desde `code/`, por defecto en localhost:8080.

Variables de entorno:
    TRIAGE_SAMPLE       CSV de entrada (default: data/processed/triage_sample.csv)
    TRIAGE_OUT_DIR       carpeta de salida (default: data/processed)
    TRIAGE_LLM_URL       URL del servicio llm (default: http://localhost:8080)
    TRIAGE_CONCURRENCY   llamadas simultáneas al LLM (default: 1 — el compose
                         de UrgeNurse fija LLAMA_ARG_N_PARALLEL=1)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGES_DIR = SCRIPT_DIR.parent.parent / "packages"
for _pkg in ("agent", "agent-triage"):
    _src = PACKAGES_DIR / _pkg / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from urgenurse.agents.triage.config import TriageAgentConfig  # noqa: E402
from urgenurse.agents.triage.llm import evaluate_triage  # noqa: E402

SAMPLE_PATH = Path(
    os.environ.get("TRIAGE_SAMPLE", SCRIPT_DIR / "data" / "processed" / "triage_sample.csv")
)
OUT_DIR = Path(os.environ.get("TRIAGE_OUT_DIR", SCRIPT_DIR / "data" / "processed"))
LLM_URL = os.environ.get("TRIAGE_LLM_URL", "http://localhost:8080")
CONCURRENCY = int(os.environ.get("TRIAGE_CONCURRENCY", 1))

ESI_LABELS = [1, 2, 3, 4, 5]


def build_patient(row: pd.Series) -> dict:
    """MIMIC-IV-ED (módulo ed) no trae edad ni antecedentes/alergias — el
    agente los tratará como no disponibles, igual que haría con un triage
    real incompleto en esos campos.

    Nombres de campo alineados con `PatientInfoCreate`
    (packages/api/src/urgenurse/api/schemas/api/case.py: pulse,
    blood_pressure_systolic/diastolic) para que la entrada se parezca a la
    de producción. Para los campos que la API no captura estructuradamente
    (temperature, respiratory rate, o2 sat, pain), se nombran con la unidad
    explícita en la clave — la primera corrida (--limit 20) mostró que
    etiquetas ambiguas como "Temperature F" hacían que el modelo inventara
    conversiones de unidad incorrectas."""
    return {
        "gender": row.get("gender"),
        "race": row.get("race"),
        "arrival_transport": row.get("arrival_transport"),
        "pulse": row.get("heartrate"),
        "blood_pressure_systolic": row.get("sbp"),
        "blood_pressure_diastolic": row.get("dbp"),
        "respiratory_rate": row.get("resprate"),
        "oxygen_saturation_percent": row.get("o2sat"),
        "temperature_fahrenheit": row.get("temperature"),
        "pain_score_0_to_10": row.get("pain"),
    }


async def evaluate_case(config: TriageAgentConfig, sem: asyncio.Semaphore, row: pd.Series) -> dict:
    patient = build_patient(row)
    async with sem:
        try:
            valid, missing_fields, esi_level, analysis = await evaluate_triage(
                config=config,
                patient=patient,
                chief_complaint=str(row["chiefcomplaint"]),
                attachments=[],
            )
        except Exception as exc:  # se reporta y se sigue con el resto de la muestra
            return {
                "stay_id": row["stay_id"],
                "acuity": row["acuity"],
                "agent_valid": None,
                "agent_esi_level": None,
                "agent_missing_fields": None,
                "agent_analysis": None,
                "error": str(exc),
            }
    return {
        "stay_id": row["stay_id"],
        "acuity": row["acuity"],
        "agent_valid": valid,
        "agent_esi_level": esi_level,
        "agent_missing_fields": "; ".join(missing_fields) if missing_fields else "",
        "agent_analysis": analysis,
        "error": None,
    }


async def run_evaluation(sample: pd.DataFrame, llm_url: str, concurrency: int) -> pd.DataFrame:
    config = TriageAgentConfig(llm_url=llm_url)
    sem = asyncio.Semaphore(concurrency)
    tasks = [evaluate_case(config, sem, row) for _, row in sample.iterrows()]

    results: list[dict] = []
    start = time.monotonic()
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        results.append(await coro)
        if i % 10 == 0 or i == len(tasks):
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed else 0
            eta = (len(tasks) - i) / rate if rate else 0
            print(f"  {i}/{len(tasks)} casos evaluados ({elapsed:.0f}s transcurridos, ETA ~{eta:.0f}s)")
    return pd.DataFrame(results)


_ESCALATION_RE = re.compile(r"Auto-escalated from ESI (\d)")


def _raw_esi_level(row: pd.Series) -> int:
    """Nivel que habría dado el LLM sin el reescalado determinista del Punto D
    (código, no juicio del modelo). La nota de escalado guarda ese valor
    original en el propio texto de 'analysis' — se recupera de ahí en vez de
    correr una segunda vez la inferencia."""
    match = _ESCALATION_RE.search(str(row.get("agent_analysis", "")))
    return int(match.group(1)) if match else int(row["agent_esi_level"])


def _score(acuity: pd.Series, predicted: pd.Series) -> dict:
    return {
        "exact_agreement": (predicted == acuity).mean(),
        "mean_abs_error": (predicted - acuity).abs().mean(),
        "kappa_unweighted": cohen_kappa_score(acuity, predicted),
        "kappa_linear": cohen_kappa_score(acuity, predicted, weights="linear"),
        "kappa_quadratic": cohen_kappa_score(acuity, predicted, weights="quadratic"),
    }


def compute_metrics(results: pd.DataFrame) -> tuple[dict, pd.DataFrame] | None:
    ok = results[results["error"].isna()].copy()
    if ok.empty:
        return None
    ok["agent_esi_level"] = ok["agent_esi_level"].astype(int)
    ok["acuity"] = ok["acuity"].astype(int)
    ok["raw_esi_level"] = ok.apply(_raw_esi_level, axis=1)
    n_escalated = int((ok["raw_esi_level"] != ok["agent_esi_level"]).sum())

    metrics = {
        "n_total": len(results),
        "n_ok": len(ok),
        "n_failed": len(results) - len(ok),
        "n_escalated_by_decision_point_d": n_escalated,
        **{f"with_d_{k}": v for k, v in _score(ok["acuity"], ok["agent_esi_level"]).items()},
        **{f"without_d_{k}": v for k, v in _score(ok["acuity"], ok["raw_esi_level"]).items()},
    }
    cm = confusion_matrix(ok["acuity"], ok["agent_esi_level"], labels=ESI_LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"acuity_{i}" for i in ESI_LABELS],
        columns=[f"agent_{i}" for i in ESI_LABELS],
    )
    return metrics, cm_df


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evalúa el Agente Triage ESI contra MIMIC-IV-ED (kappa vs acuity documentado)."
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="evalúa solo los primeros N casos (0 = todos)"
    )
    ap.add_argument(
        "--force", action="store_true", help="reevalúa desde cero, ignorando resultados previos"
    )
    ap.add_argument("--llm-url", default=LLM_URL, help=f"URL del servicio llm (default {LLM_URL})")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY,
        help=f"llamadas simultáneas al LLM (default {CONCURRENCY})",
    )
    args = ap.parse_args()

    if not SAMPLE_PATH.exists():
        raise SystemExit(f"No se encuentra {SAMPLE_PATH}. Corre antes prepare_dataset.py.")

    sample = pd.read_csv(SAMPLE_PATH)
    if args.limit:
        sample = sample.head(args.limit)

    results_path = OUT_DIR / "triage_eval_results.csv"
    already: pd.DataFrame | None = None
    if results_path.exists() and not args.force:
        already = pd.read_csv(results_path)
        done_ids = set(already["stay_id"])
        sample = sample[~sample["stay_id"].isin(done_ids)]
        print(f"Reanudando: {len(done_ids)} casos ya evaluados, {len(sample)} pendientes.")

    if len(sample):
        print(
            f"Evaluando {len(sample)} casos contra {args.llm_url} "
            f"(concurrencia={args.concurrency}) …"
        )
        new_results = asyncio.run(run_evaluation(sample, args.llm_url, args.concurrency))
        results = pd.concat([already, new_results], ignore_index=True) if already is not None else new_results
    else:
        results = already if already is not None else pd.DataFrame()

    if results.empty:
        print("Nada que evaluar.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    print(f"\n✓ Resultados por caso guardados en {results_path}")

    n_failed = int(results["error"].notna().sum())
    if n_failed:
        print(f"  ⚠ {n_failed} casos fallaron (ver columna 'error') — excluidos de las métricas")

    computed = compute_metrics(results)
    if computed is None:
        print("\n⚠ Ningún caso se evaluó con éxito — no hay métricas que calcular.")
        print("  Revisa 'error' en triage_eval_results.csv (¿está el servicio llm corriendo?).")
        return 1
    metrics, cm_df = computed
    metrics_path = OUT_DIR / "triage_eval_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    cm_path = OUT_DIR / "triage_eval_confusion.csv"
    cm_df.to_csv(cm_path)

    print(f"\n✓ Métricas guardadas en {metrics_path}")
    print(
        f"  n_ok={metrics['n_ok']}/{metrics['n_total']}  "
        f"escalados por Punto D={metrics['n_escalated_by_decision_point_d']}"
    )
    for label, prefix in (("CON Punto D", "with_d"), ("SIN Punto D (crudo LLM)", "without_d")):
        print(
            f"\n  {label}: exact_agreement={metrics[f'{prefix}_exact_agreement']:.3f}  "
            f"mean_abs_error={metrics[f'{prefix}_mean_abs_error']:.3f}"
        )
        print(f"    kappa (unweighted) = {metrics[f'{prefix}_kappa_unweighted']:.3f}")
        print(f"    kappa (linear)     = {metrics[f'{prefix}_kappa_linear']:.3f}")
        print(f"    kappa (quadratic)  = {metrics[f'{prefix}_kappa_quadratic']:.3f}")
    print(f"\n✓ Matriz de confusión guardada en {cm_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
