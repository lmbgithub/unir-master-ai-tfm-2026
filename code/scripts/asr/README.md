# Benchmark ASR (speech-to-text en CPU)

Compara modelos de reconocimiento de voz pequeños (≤2 GB RAM, solo CPU,
cuantizados int8/fp16/Q4/Q5) sobre el dataset *Synthetic Nursing Handoff* y
reporta WER, WER-nursing, CER, RTF, RAM peak, NER recall y latencia. Cada modelo
Whisper corre con filtro VAD anti-alucinación.

Componentes:

- `prepare_references.py` — prepara el dataset (perfiles `.docx` → `.txt` y
  ground-truth verbatim con Whisper `large-v3`). Se corre **una vez**.
- `asr.py` — el benchmark. También usable como librería (`import asr`).
- `asr.ipynb` — el mismo benchmark con figuras comparativas.

## Requisitos

- Python 3.12, solo CPU.
- Los modelos se descargan a `../models` (carpeta `code/scripts/models`) la
  primera vez; reserva espacio en disco.

## Instalación

```bash
cd code/scripts/asr
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install python-docx          # solo para prepare_references.py (perfiles .docx)
```

Los backends ASR de `requirements.txt` (`pywhispercpp`, `sherpa-onnx`, `vosk`)
son opcionales: si alguno falta, su modelo se omite con un aviso y el benchmark
continúa. `faster-whisper` es el núcleo y sí es necesario.

## Ejecución

```bash
# 1) una sola vez: deja assets/references/ listo (perfiles + ground-truth large-v3)
python prepare_references.py
#    o por partes:
#    python prepare_references.py --profiles       # solo .docx → .txt
#    python prepare_references.py --ground-truth   # solo ground-truth large-v3
#    python prepare_references.py --force          # regenera aunque ya exista

# 2) correr el benchmark
python asr.py
```

Las transcripciones de cada modelo quedan en `transcriptions/<modelo>/`, y los
problemas detectados en `error__<fecha>.json`.

Desde un notebook o REPL:

```python
import asr
df = asr.run_benchmark()   # DataFrame, una fila por modelo
```

Para el notebook: `pip install ipykernel` (ya en `requirements.txt`) y abrir
`asr.ipynb`.

## Variables de entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `ASR_MODELS_DIR` | `../models` | Carpeta de modelos |
| `ASR_AUDIO_DIR` | `assets/audio` | Audios WAV 16 kHz mono |
| `ASR_REF_DIR` | `assets/references` | Ground-truth verbatim |
| `ASR_OUT_DIR` | `transcriptions` | Salida de transcripciones |
| `ASR_MAX_RAM_GB` | `2.0` | Presupuesto de RAM |
| `ASR_LIMIT` | `0` | Nº de audios a procesar (0 = todos) |
| `ASR_DOWNLOAD` | `1` | `0` para no descargar modelos faltantes |
| `ASR_THREADS` | núcleos físicos | Nº de hilos |
| `ASR_VAD` | `1` | `0` desactiva el filtro VAD anti-alucinación |

Variables propias de `prepare_references.py`: `ASR_PROFILE_DIR`
(default `assets/profiles`), `ASR_GT_MODEL` (default `large-v3`), `ASR_GT_COMPUTE`
(default `int8`).
