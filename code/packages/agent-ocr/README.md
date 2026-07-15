# agent-ocr

OCR/PDF worker for the UrgeNurse platform.

Subscribes to `attachment.created` on NATS JetStream, extracts text from images and PDFs using Tesseract 5 / pdfminer.six, and publishes `attachment.processed`.

## Development

```bash
pip install -e ../agent
pip install -e .
python -m pytest tests/ -v
```

## Running

```bash
python -m urgenurse.agents.ocr.main
# or via entry point:
worker
```

Requires `NATS_URL` env var (default: `nats://localhost:4222`).
