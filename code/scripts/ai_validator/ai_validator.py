import argparse

import pypdf
from transformers import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detecta texto generado por IA en un PDF con un clasificador académico."
    )
    parser.add_argument("pdf", help="ruta del PDF a analizar")
    parser.add_argument(
        "--model",
        default="andreas122001/roberta-academic-detector",
        help="modelo de clasificación de Hugging Face",
    )
    parser.add_argument(
        "--start", type=int, default=12000, help="offset inicial del fragmento"
    )
    parser.add_argument(
        "--end", type=int, default=13500, help="offset final del fragmento"
    )
    args = parser.parse_args()

    # 1. Extraer el texto real detectado
    reader = pypdf.PdfReader(args.pdf)
    full_text = "".join([page.extract_text() for page in reader.pages])

    # 2. Inicializar detector académico de Hugging Face
    # Este clasifica en 'RoBERTa-Real' (Humano) o 'RoBERTa-Fake' (IA)
    detector = pipeline("text-classification", model=args.model)

    # 3. Analizar los bloques principales (ej. la Introducción extraída)
    # Ajustado al límite de contexto del modelo de clasificación
    fragmento = full_text[args.start : args.end]
    result = detector(fragmento)[0]

    # 4. Mapear a formato 0.0 - 1.0
    score_confianza = result["score"]
    score_final_ia = (
        score_confianza if result["label"] == "fake" else (1.0 - score_confianza)
    )

    print(f"Puntaje de IA (0-1.0): {score_final_ia:.4f}")


if __name__ == "__main__":
    main()
