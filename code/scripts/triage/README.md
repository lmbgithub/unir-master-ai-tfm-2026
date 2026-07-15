# Evaluación del Agente Triage ESI (MIMIC-IV-ED)

Usa MIMIC-IV-ED como fuente de análisis exploratorio (EDA) y de evaluación del
Agente Triage ESI (`packages/agent-triage`), **no** como conjunto de
entrenamiento — el agente es un revisor documental guiado por las reglas del
estándar ESI v5 (algoritmo real de 4 puntos de decisión, Gilboy et al., ESI
Implementation Handbook v5), no un clasificador entrenado. Ver
`document/src/results.tex`, sección "Módulo de triage ESI".

El agente implementa el algoritmo como un diseño híbrido: los puntos A
(intervención salvavidas), B (situación de alto riesgo) y C (recursos
anticipados) requieren juicio clínico sobre texto libre y los razona el LLM,
guiado por los criterios explícitos del estándar en el prompt
(`packages/agent-triage/.../llm.py`). El punto D (vitales de "danger zone"
por edad) es puramente numérico y se aplica en **código determinista**
después de la respuesta del LLM, reescalando a ESI 2 si corresponde — nunca
al revés. La regla compartida vive en
`packages/agent-triage/src/urgenurse/agents/triage/esi_rules.py` (fuente
única; este directorio la importa, no la duplica).

Componentes:

- `prepare_dataset.py` — carga `triage` + `edstays`, limpia, calcula
  discordancia (nivel ESI de baja prioridad documentado + vitales de riesgo,
  usando el mismo `esi_rules.danger_zone_flag` del agente) y deja una muestra
  estratificada por acuity en `data/processed/`.
- `evaluate.py` — corre el `evaluate_triage()` real de `packages/agent-triage`
  (sin pasar por NATS) sobre la muestra, y calcula el kappa de Cohen entre el
  `esi_level` del agente y el `acuity` documentado en MIMIC-IV-ED (kappa
  agente vs clasificación humana). Reanudable; requiere el servicio `llm`
  (llama.cpp) accesible.

**Nota de alcance real**: el punto D necesita frecuencia cardíaca, frecuencia
respiratoria y SpO2. La API de producción (`PatientInfoCreate`) hoy solo
captura `pulse` y presión arterial — no frecuencia respiratoria ni SpO2 — así
que el reescalado automático solo puede activarse por completo en esta
evaluación offline con MIMIC hasta que el formulario de triage capture esos
campos. Se degrada con gracia (no revienta) si faltan.

## Dataset

MIMIC-IV-ED es de **acceso credencializado en PhysioNet** (Data Use Agreement,
curso CITI). Los CSV **no se commitean** — `code/scripts/triage/data/` está en
`.gitignore`. Estructura esperada:

```
data/mimic-iv-ed/2.2/ed/
├── triage.csv.gz
├── edstays.csv.gz
├── diagnosis.csv.gz
├── medrecon.csv.gz
├── pyxis.csv.gz
├── vitalsign.csv.gz
└── LICENSE.txt
```

## Instalación

```bash
cd code/scripts/triage
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

```bash
python prepare_dataset.py                  # EDA + muestra estratificada (800 casos)
python prepare_dataset.py --eda-only        # solo el EDA, sin muestrear
python prepare_dataset.py --sample-size 500 # otro tamaño de muestra
python prepare_dataset.py --force           # regenera aunque ya exista
```

Salida en `data/processed/`:

- `eda_discordance.csv` — tasa de discordancia por nivel de acuity documentado.
- `triage_sample.csv` — muestra balanceada entre los 5 niveles de ESI (no
  proporcional a su frecuencia real: ESI 5 es solo 0,26% del corpus), lista
  para `evaluate.py`.

Después, con el servicio `llm` corriendo (`docker compose up -d llm` desde
`code/`, expone `localhost:8080`):

```bash
python evaluate.py --limit 20   # smoke test — verifica conexión y parseo antes de la corrida larga
python evaluate.py              # evalúa toda la muestra (reanudable si se interrumpe)
python evaluate.py --force      # reevalúa todo desde cero
```

Salida adicional en `data/processed/`:

- `triage_eval_results.csv` — un resultado del agente por caso (`agent_valid`,
  `agent_esi_level`, `agent_missing_fields`, `agent_analysis`, `error`).
- `triage_eval_metrics.csv` — kappa de Cohen (no ponderado, lineal, cuadrático
  — la memoria no fija la ponderación, se reportan los tres), concordancia
  exacta y error absoluto medio en niveles ESI, reportados dos veces: **con**
  el reescalado del Punto D aplicado (`with_d_*`, lo que el sistema realmente
  devuelve) y **sin** él (`without_d_*`, el nivel crudo del LLM). El nivel
  crudo se recupera de la propia nota de reescalado en `agent_analysis` — no
  hace falta correr la evaluación dos veces para comparar el efecto del
  Punto D.
- `triage_eval_confusion.csv` — matriz de confusión 5×5 (acuity documentado ×
  esi_level del agente), útil para la figura/tabla en `results.tex`.

## Variables de entorno

| Variable            | Default                             | Descripción                                     |
| ------------------- | ------------------------------------ | ------------------------------------------------ |
| `MIMIC_ED_DIR`       | `data/mimic-iv-ed/2.2/ed`            | Carpeta con las tablas `ed/*.csv.gz`             |
| `TRIAGE_OUT_DIR`     | `data/processed`                     | Carpeta de salida                                |
| `TRIAGE_SEED`        | `42`                                  | Semilla del muestreo estratificado               |
| `TRIAGE_SAMPLE`      | `data/processed/triage_sample.csv`   | CSV de entrada de `evaluate.py`                  |
| `TRIAGE_LLM_URL`     | `http://localhost:8080`              | URL del servicio `llm`                           |
| `TRIAGE_CONCURRENCY` | `1`                                   | Llamadas simultáneas (compose fija `N_PARALLEL=1`) |
