"""ESI v5 Decision Point D — high-risk vital signs (age-stratified danger zone).

Fuente: Emergency Severity Index (ESI) Implementation Handbook, 5th edition
(Emergency Nurses Association, 2023) — Capítulo 6 / Apéndice B, Figura 6-1.

Es la única parte del algoritmo ESI v5 que es puramente numérica (umbrales de
frecuencia cardíaca, frecuencia respiratoria y SpO2 por grupo etario); los
puntos A, B y C requieren juicio clínico sobre texto libre y se dejan al LLM,
guiado por los criterios explícitos del prompt (ver _SYSTEM_PROMPT en llm.py).

Esta regla se aplica como reescalado determinista DESPUÉS de que el LLM
propone un nivel — igual que en el algoritmo real, donde el punto D reevalúa
la decisión de C sin importar cuántos recursos se hayan estimado. Nunca se
usa para bajar un nivel, solo para subirlo a 2 si hay vitales de riesgo.
"""

from __future__ import annotations

import math
from datetime import date

# (edad_max_años_exclusiva, hr_high, rr_high) — SpO2 < 92% aplica a todos los grupos
_DANGER_ZONE_TABLE: list[tuple[float, float, float]] = [
    (1 / 12, 190, 60),  # < 1 mes
    (1, 180, 55),  # 1-12 meses
    (3, 140, 40),  # 1-3 años
    (5, 120, 35),  # 3-5 años
    (12, 120, 30),  # 5-12 años
    (18, 100, 20),  # 12-18 años
    (float("inf"), 100, 20),  # > 18 años (adulto)
]
SPO2_LOW = 92.0


def _present(value: float | None) -> bool:
    return value is not None and not math.isnan(value)


def _thresholds_for_age(age_years: float | None) -> tuple[float, float]:
    """(hr_high, rr_high) para la edad dada. Sin edad conocida, se asume
    adulto (>18 años): el grupo más frecuente en un servicio de urgencias y
    el criterio menos sensible de la tabla, para no sobre-escalar por
    defecto ante la ausencia del dato."""
    effective_age = 19.0 if age_years is None else age_years
    for age_limit, hr_high, rr_high in _DANGER_ZONE_TABLE:
        if effective_age < age_limit:
            return hr_high, rr_high
    return _DANGER_ZONE_TABLE[-1][1], _DANGER_ZONE_TABLE[-1][2]


def age_years_from_date_of_birth(date_of_birth: str | date | None) -> float | None:
    if not date_of_birth:
        return None
    if isinstance(date_of_birth, str):
        try:
            date_of_birth = date.fromisoformat(date_of_birth[:10])
        except ValueError:
            return None
    return (date.today() - date_of_birth).days / 365.25


def danger_zone_flag(
    heartrate: float | None,
    resprate: float | None,
    o2sat: float | None,
    age_years: float | None = None,
) -> bool | None:
    """True si alguna vital de danger-zone documentada excede el umbral de
    su grupo etario (Decision Point D), False si las documentadas están
    todas dentro de rango, None si no hay ninguna de las tres registrada."""
    if not (_present(heartrate) or _present(resprate) or _present(o2sat)):
        return None
    hr_high, rr_high = _thresholds_for_age(age_years)
    if _present(heartrate) and heartrate > hr_high:
        return True
    if _present(resprate) and resprate > rr_high:
        return True
    if _present(o2sat) and o2sat < SPO2_LOW:
        return True
    return False
