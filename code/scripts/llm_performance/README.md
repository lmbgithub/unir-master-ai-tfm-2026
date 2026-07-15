# Benchmark de LLMs GGUF (llama.cpp en CPU)

Compara modelos GGUF locales pequeños (cuantizados Q4_K_M) corriendo en CPU.
Por cada modelo prueba dos contextos (ctx_min/ctx_max), mide tamaño de archivo,
tiempo de carga, RAM tras cargar y ejecuta las tareas de UrgeNurse
(triage / NER / SBAR) midiendo tiempo y accuracy. Pensado para una máquina tipo
Beelink S12 / N100 con 16 GB de RAM.

Componentes:

- `llm_performance.py` — el benchmark. También usable como librería.
- `llm_performance.ipynb` — el mismo benchmark con la tabla y figuras.

## Requisitos

- Python 3.12.
- Los modelos GGUF se descargan a `../models` (carpeta `code/scripts/models`)
  desde HuggingFace la primera vez. La lista completa son ~12 GB; comenta o
  descomenta entradas del catálogo en `llm_performance.py` para elegir cuáles
  correr.

## Instalación

```bash
cd code/scripts/llm_performance
python3 -m venv .venv && source .venv/bin/activate
pip install llama-cpp-python pandas psutil
pip install ipykernel          # solo para el notebook
```

`llama-cpp-python` trae llama.cpp embebido (única dependencia pesada). `psutil`
es opcional pero da una medición de RAM más precisa y cross-platform.

GPU automática: si hay GPU y el build de `llama-cpp-python` la soporta
(Metal/CUDA), se hace offload de todas las capas; si no, corre en CPU. Se fuerza
con `LLM_PERF_GPU_LAYERS` (`0` = solo CPU).

## Ejecución

```bash
python llm_performance.py      # imprime la tabla de resultados
```

Los problemas detectados por modelo quedan en `error__<fecha>.json`.

Desde un notebook o REPL (la función es async):

```python
df = await run_benchmark()     # DataFrame, una fila por modelo × contexto
```

Para el notebook: abrir `llm_performance.ipynb`.

## Variables de entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `LLM_PERF_MODELS_DIR` | `../models` | Carpeta de modelos GGUF |
| `LLM_PERF_OUTPUT_DIR` | carpeta del script | Destino del `error__*.json` |
| `LLM_PERF_DOWNLOAD` | `1` | `0` para no descargar modelos faltantes |
| `LLM_PERF_THREADS` | núcleos físicos | Nº de hilos |
| `LLM_PERF_CTX` | `4096` | Contexto por defecto si el modelo no fija `n_ctx` |
| `LLM_PERF_MAX_TOKENS` | `512` | Tope de tokens por respuesta |
| `LLM_PERF_MAX_RAM_GB` | `5.0` | Presupuesto de RAM; omite modelos mayores |
| `LLM_PERF_GPU_LAYERS` | `auto` | `0` = solo CPU, `-1` = todas las capas, `N` = N capas |
| `LLM_PERF_STREAM` | `0` | `1` activa streaming con continuación |
