# Benchmark OCR (imagen/PDF → texto en CPU)

Compara motores OCR ligeros (≤2 GB RAM, solo CPU) sobre un dataset de 20
documentos clínicos sintéticos (urgencias, exámenes de laboratorio, radiología,
diagnósticos y órdenes de medicación) y reporta CER, WER, precisión/recall/F1 de
palabra, WER de terminología clínica, field recall, latencia (ms/página),
throughput (páginas/s) y RAM peak.

Modelos comparados:

| Modelo | Familia | Notas |
|--------|---------|-------|
| `tesseract-5-eng` | Tesseract | **baseline** del proyecto (stack UrgeNurse) |
| `paddleocr-ppocr-en` | PaddleOCR | stack del agente OCR (`packages/agent-ocr`) |
| `rapidocr-onnx-en` | RapidOCR | PP-OCR en ONNX Runtime, ultraligero |
| `easyocr-en` | EasyOCR | CRAFT + CRNN, PyTorch CPU, muy popular |
| `doctr-mobilenet` | docTR | Mindee, variante MobileNet ligera |
| `docling-pipeline` | Docling | IBM, pipeline de documentos con OCR |

Todos corren **solo en CPU, sin GPU** (UrgeNurse es CPU-only por diseño) y con
huella ≤2 GB. `ocr.py` fija `CUDA_VISIBLE_DEVICES=""` y pasa `use_gpu/gpu=False`
a cada motor para blindar el modo CPU; instala siempre las ruedas CPU
(`paddlepaddle`, `onnxruntime`, `torch` CPU), nunca las variantes `-gpu`.

Componentes:

- `prepare_dataset.py` — genera el dataset con **ReportLab**: 20 PDFs clínicos,
  sus imágenes PNG (rasterizadas con PyMuPDF) y la ground-truth verbatim. Se corre
  **una vez**. La ground-truth es **exacta** (es el texto que se imprimió en el
  documento), así que no hay sesgo de transcripción.
- `ocr.py` — el benchmark. También usable como librería (`import ocr`).
- `ocr.ipynb` — el mismo benchmark con figuras comparativas y selección del modelo.

## Requisitos

- Python 3.12, solo CPU.
- Tesseract necesita además el binario del sistema: `brew install tesseract`.
- Los modelos se descargan/cachean en `../models` (`code/scripts/models`) la
  primera vez; reserva espacio en disco.

## Instalación

```bash
cd code/scripts/ocr
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Los backends OCR (`pytesseract`, `paddleocr`, `rapidocr-onnxruntime`, `easyocr`,
`python-doctr`, `docling`) son opcionales: si alguno falta, su modelo se omite con
un aviso y el benchmark continúa. El núcleo (`pandas`, `matplotlib`, `numpy`) y la
generación del dataset (`reportlab`, `pymupdf`, `pillow`) sí son necesarios.

## Ejecución

```bash
# 1) una sola vez: deja assets/ listo (PDFs + imágenes + ground-truth)
python prepare_dataset.py
#    o por partes:
#    python prepare_dataset.py --pdf       # solo PDFs (+ referencias)
#    python prepare_dataset.py --images    # solo rasterizar PDFs ya existentes
#    python prepare_dataset.py --force     # regenera aunque ya exista
#    python prepare_dataset.py --n 30      # genera 30 documentos (default 20)

# 2) correr el benchmark
python ocr.py
```

Las predicciones de cada modelo quedan en `predictions/<modelo>/`, y los problemas
detectados en `error__<fecha>.json`.

Desde un notebook o REPL:

```python
import ocr
df = ocr.run_benchmark()   # DataFrame, una fila por modelo
```

Para el notebook: `pip install ipykernel` (ya en `requirements.txt`) y abrir
`ocr.ipynb`.

## Dataset (`assets/`)

```
assets/
├── docs/{n}.pdf          # documento maquetado con ReportLab
├── images/{n}.png        # PDF rasterizado (entrada del OCR)
└── references/{n}.txt    # ground-truth verbatim (orden de lectura)
```

Los 20 documentos rotan cinco plantillas clínicas (triage de urgencias, resultados
de laboratorio, informe de radiología, informe de alta y orden de medicación) con
datos sintéticos reproducibles (`OCR_SEED`).

## Métricas

| Métrica | Sentido | Descripción |
|---------|---------|-------------|
| `cer` | menor mejor | Character Error Rate (principal en OCR) |
| `wer` | menor mejor | Word Error Rate |
| `char_acc` | mayor mejor | 1 − CER |
| `word_precision` / `word_recall` / `word_f1` | mayor mejor | a nivel de palabra |
| `term_wer` | menor mejor | error sobre terminología clínica (fármacos, dosis, abreviaturas) |
| `field_recall` | mayor mejor | entidades clínicas recuperadas |
| `mean_latency_ms` / `p95_latency_ms` | menor mejor | latencia por página |
| `pages_per_sec` | mayor mejor | throughput en CPU |
| `ram_peak_mb` | menor mejor | pico de RSS durante el OCR |
| `load_time_ms` | menor mejor | tiempo de carga del modelo |

## Variables de entorno

### `ocr.py` (benchmark)

| Variable | Default | Descripción |
|----------|---------|-------------|
| `OCR_MODELS_DIR` | `../models` | Carpeta de modelos/caches |
| `OCR_IMAGE_DIR` | `assets/images` | Imágenes PNG |
| `OCR_PDF_DIR` | `assets/docs` | PDFs |
| `OCR_REF_DIR` | `assets/references` | Ground-truth verbatim |
| `OCR_OUT_DIR` | `predictions` | Salida de predicciones |
| `OCR_INPUT` | `image` | `image` (PNG) o `pdf` |
| `OCR_MAX_RAM_GB` | `2.0` | Presupuesto de RAM |
| `OCR_LIMIT` | `0` | Nº de documentos a procesar (0 = todos) |
| `OCR_DOWNLOAD` | `1` | `0` para no descargar modelos faltantes |
| `OCR_THREADS` | núcleos físicos | Nº de hilos |

### `prepare_dataset.py` (dataset)

| Variable | Default | Descripción |
|----------|---------|-------------|
| `OCR_DOC_DIR` | `assets/docs` | Salida de PDFs |
| `OCR_IMAGE_DIR` | `assets/images` | Salida de imágenes |
| `OCR_REF_DIR` | `assets/references` | Salida de ground-truth |
| `OCR_DPI` | `200` | Resolución de rasterizado |
| `OCR_SEED` | `42` | Semilla de los datos sintéticos |
