#!/usr/bin/env python3
"""Generación del dataset OCR para UrgeNurse (comando aparte del benchmark).

Crea, con **ReportLab**, 20 documentos clínicos sintéticos (urgencias, exámenes
de laboratorio, radiología, diagnósticos y órdenes de medicación) y, para cada
uno, deja listos los tres artefactos que el benchmark (``ocr.py`` / ``ocr.ipynb``)
da por hechos:

  1. **PDF**    `assets/docs/{n}.pdf`        — el documento maquetado con ReportLab.
  2. **Imagen** `assets/images/{n}.png`      — el PDF rasterizado (entrada del OCR).
  3. **Ground-truth** `assets/references/{n}.txt` — el texto exacto que se imprimió
     en el documento, en orden de lectura. Es el verbatim contra el que el
     benchmark calcula CER/WER.

Como el texto de referencia se construye con el MISMO contenido que se maqueta,
la ground-truth es exacta (no hay sesgo de transcripción, a diferencia del ASR).

Uso:
    python prepare_dataset.py                 # PDFs + imágenes + referencias
    python prepare_dataset.py --pdf           # solo PDFs (+ referencias)
    python prepare_dataset.py --images        # solo rasterizar PDFs ya existentes
    python prepare_dataset.py --force         # regenera aunque ya exista
    python prepare_dataset.py --n 30          # genera 30 documentos (default 20)

Variables de entorno:
    OCR_DOC_DIR     carpeta de PDFs (default: assets/docs)
    OCR_IMAGE_DIR   carpeta de imágenes (default: assets/images)
    OCR_REF_DIR     carpeta de referencias (default: assets/references)
    OCR_DPI         resolución de rasterizado (default: 200)
    OCR_SEED        semilla para datos sintéticos reproducibles (default: 42)
"""

from __future__ import annotations

import argparse
import io
import os
import random
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DOC_DIR = Path(os.environ.get("OCR_DOC_DIR", SCRIPT_DIR / "assets" / "docs"))
IMAGE_DIR = Path(os.environ.get("OCR_IMAGE_DIR", SCRIPT_DIR / "assets" / "images"))
REF_DIR = Path(os.environ.get("OCR_REF_DIR", SCRIPT_DIR / "assets" / "references"))
DPI = int(os.environ.get("OCR_DPI", 200))
SEED = int(os.environ.get("OCR_SEED", 42))
N_DOCS = 20


# ─────────────────────────────────────────────────────────────────────────────
# Modelo de documento: lista ordenada de elementos.
# Cada elemento es una tupla cuyo primer campo indica el tipo. De ahí salen, en
# el MISMO orden, tanto los flowables de ReportLab como las líneas de ground-truth.
#   ("title", texto)
#   ("heading", texto)
#   ("para", texto)
#   ("kv", etiqueta, valor)            -> "etiqueta: valor"
#   ("table", [cabeceras], [[fila...]])
# ─────────────────────────────────────────────────────────────────────────────

Element = tuple


def element_to_lines(el: Element) -> list[str]:
    """Texto en orden de lectura de un elemento (alimenta la ground-truth)."""
    kind = el[0]
    if kind in ("title", "heading", "para"):
        return [el[1]]
    if kind == "kv":
        return [f"{el[1]}: {el[2]}"]
    if kind == "table":
        headers, rows = el[1], el[2]
        out = [" ".join(str(c) for c in headers)]
        out += [" ".join(str(c) for c in row) for row in rows]
        return out
    raise ValueError(f"Elemento desconocido: {kind}")


def document_to_text(elements: list[Element]) -> str:
    lines: list[str] = []
    for el in elements:
        lines.extend(element_to_lines(el))
    text = "\n".join(lines)
    return re.sub(r"[ \t]+", " ", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Pools de datos clínicos sintéticos (inglés, para casar con los motores OCR EN)
# ─────────────────────────────────────────────────────────────────────────────

FIRST_NAMES = [
    "Vera",
    "Harold",
    "Margaret",
    "Arthur",
    "Doris",
    "Walter",
    "Edith",
    "Frank",
    "Beatrice",
    "Leonard",
    "Mabel",
    "Stanley",
    "Gloria",
    "Herbert",
    "Pauline",
    "Clifford",
    "Agnes",
    "Reginald",
    "Mavis",
    "Cyril",
]
LAST_NAMES = [
    "Abbott",
    "Whitfield",
    "Pemberton",
    "Ashworth",
    "Caldwell",
    "Hartley",
    "Brennan",
    "Lockwood",
    "Saunders",
    "Driscoll",
    "Mortimer",
    "Rowntree",
    "Fairbanks",
    "Holloway",
    "Kendrick",
    "Thornbury",
    "Underwood",
    "Vance",
    "Wexford",
    "Yarrow",
]
WARDS = [
    "Acute Medical Unit",
    "Coronary Care Unit",
    "Respiratory Ward",
    "Stroke Unit",
    "Renal Ward",
    "Emergency Department",
]
PRESCRIBERS = [
    "Dr. Liu",
    "Dr. Okafor",
    "Dr. Mensah",
    "Dr. Petrov",
    "Dr. Singh",
    "Dr. Romano",
]

COMPLAINTS = [
    "Central chest pain radiating to the left arm for two hours, not relieved "
    "by three sublingual nitroglycerin sprays.",
    "Progressive shortness of breath over four days with a productive cough and "
    "wheeze, worse on exertion.",
    "Sudden right-sided weakness and slurred speech noticed on waking, last seen "
    "well six hours ago.",
    "Reduced urine output, ankle swelling and fatigue over the past week in a "
    "known chronic kidney disease patient.",
    "Palpitations and light-headedness with an irregular pulse, no chest pain "
    "but a history of atrial fibrillation.",
]

DIAGNOSES = [
    "Acute anterior ST-elevation myocardial infarction (STEMI).",
    "Infective exacerbation of chronic obstructive pulmonary disease (COPD).",
    "Acute ischaemic stroke of the left middle cerebral artery territory.",
    "Acute kidney injury on a background of stage 4 chronic kidney disease.",
    "Fast atrial fibrillation with rapid ventricular response.",
]
SECONDARY_DX = [
    "Type 2 diabetes mellitus, hypertension and hyperlipidaemia.",
    "Chronic heart failure with reduced ejection fraction.",
    "Previous transient ischaemic attack and glaucoma.",
    "Osteoarthritis and well-controlled asthma.",
    "Benign prostatic hyperplasia and gout.",
]

RADIOLOGY = [
    (
        "Chest X-ray PA and lateral",
        "Clinical question of pulmonary oedema versus consolidation in a breathless "
        "patient.",
        "There is cardiomegaly with upper lobe venous diversion and small bilateral "
        "pleural effusions. No focal consolidation or pneumothorax is seen.",
        "Findings consistent with mild pulmonary oedema. No acute consolidation.",
    ),
    (
        "CT head without contrast",
        "Sudden focal neurological deficit, exclude haemorrhage before thrombolysis.",
        "No intracranial haemorrhage. Loss of grey-white differentiation in the left "
        "insular cortex with early sulcal effacement.",
        "Early established left middle cerebral artery infarct. No haemorrhage.",
    ),
    (
        "Chest X-ray portable",
        "Productive cough and fever, query consolidation.",
        "Patchy consolidation is noted at the right lung base with air bronchograms. "
        "The cardiac silhouette is within normal limits.",
        "Right lower lobe pneumonia.",
    ),
    (
        "Renal tract ultrasound",
        "Rising creatinine and reduced urine output.",
        "Both kidneys are normal in size with preserved cortical thickness. No "
        "hydronephrosis and no renal calculi are demonstrated.",
        "No obstruction. Appearances consistent with intrinsic acute kidney injury.",
    ),
]

MEDICATIONS = [
    ("Aspirin", "300 mg", "PO", "stat then 75 mg OD"),
    ("Atorvastatin", "80 mg", "PO", "OD at night"),
    ("Furosemide", "40 mg", "IV", "BD"),
    ("Metoprolol", "25 mg", "PO", "BD"),
    ("Clopidogrel", "75 mg", "PO", "OD"),
    ("Salbutamol", "5 mg", "NEB", "QDS PRN"),
    ("Prednisolone", "30 mg", "PO", "OD for 5 days"),
    ("Amoxicillin", "500 mg", "PO", "TDS"),
    ("Enoxaparin", "40 mg", "SC", "OD"),
    ("Ramipril", "2.5 mg", "PO", "OD"),
    ("Paracetamol", "1 g", "PO", "QDS PRN"),
    ("Insulin glargine", "12 units", "SC", "OD at night"),
]

CBC_TESTS = [
    ("Haemoglobin", "g/dL", (8.0, 16.5), (13.0, 17.0)),
    ("White cell count", "x10^9/L", (3.5, 18.0), (4.0, 11.0)),
    ("Platelets", "x10^9/L", (90, 420), (150, 400)),
    ("Neutrophils", "x10^9/L", (1.5, 14.0), (2.0, 7.5)),
]
BMP_TESTS = [
    ("Sodium", "mmol/L", (128, 146), (135, 145)),
    ("Potassium", "mmol/L", (3.0, 6.2), (3.5, 5.0)),
    ("Urea", "mmol/L", (3.0, 22.0), (2.5, 7.8)),
    ("Creatinine", "umol/L", (60, 380), (60, 110)),
    ("eGFR", "mL/min", (12, 95), (90, 120)),
    ("CRP", "mg/L", (1, 180), (0, 5)),
]


def _flag(value: float, low: float, high: float) -> str:
    if value < low:
        return "Low"
    if value > high:
        return "High"
    return "Normal"


def _lab_rows(rng: random.Random, tests) -> list[list[str]]:
    rows = []
    for name, unit, (vmin, vmax), (rlow, rhigh) in tests:
        if isinstance(vmin, int) and isinstance(vmax, int):
            value = rng.randint(vmin, vmax)
            value_s = str(value)
        else:
            value = round(rng.uniform(vmin, vmax), 1)
            value_s = f"{value:.1f}"
        rows.append([name, value_s, unit, _flag(value, rlow, rhigh)])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Plantillas de documento (una por tipo clínico)
# ─────────────────────────────────────────────────────────────────────────────


def _patient(rng: random.Random, n: int) -> dict:
    return {
        "mrn": f"MRN-{rng.randint(100000, 999999)}",
        "name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
        "age": rng.randint(58, 95),
        "sex": rng.choice(["Male", "Female"]),
        "ward": rng.choice(WARDS),
        "prescriber": rng.choice(PRESCRIBERS),
    }


def doc_triage(rng: random.Random, n: int) -> list[Element]:
    p = _patient(rng, n)
    return [
        ("title", "EMERGENCY DEPARTMENT TRIAGE NOTE"),
        ("kv", "MRN", p["mrn"]),
        ("kv", "Patient", p["name"]),
        ("kv", "Age", f"{p['age']} years"),
        ("kv", "Sex", p["sex"]),
        ("kv", "Triage time", f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"),
        ("heading", "Chief Complaint"),
        ("para", rng.choice(COMPLAINTS)),
        ("heading", "Vital Signs"),
        (
            "table",
            ["Parameter", "Value"],
            [
                ["HR", f"{rng.randint(48, 138)} bpm"],
                ["BP", f"{rng.randint(95, 185)}/{rng.randint(55, 105)} mmHg"],
                ["RR", f"{rng.randint(12, 34)} /min"],
                ["Temp", f"{round(rng.uniform(35.5, 39.4), 1)} C"],
                ["SpO2", f"{rng.randint(85, 99)} percent"],
                ["GCS", f"{rng.randint(11, 15)}/15"],
            ],
        ),
        ("kv", "ESI level", str(rng.randint(1, 4))),
        (
            "kv",
            "Allergies",
            rng.choice(["Penicillin", "No known drug allergies", "Aspirin", "Codeine"]),
        ),
        ("heading", "Triage Notes"),
        (
            "para",
            "Patient assessed on arrival, cardiac monitor attached and IV "
            "access obtained. Bloods sent and ECG performed. Referred to the "
            "medical team for urgent review.",
        ),
    ]


def doc_lab(rng: random.Random, n: int) -> list[Element]:
    p = _patient(rng, n)
    return [
        ("title", "LABORATORY RESULTS REPORT"),
        ("kv", "MRN", p["mrn"]),
        ("kv", "Patient", p["name"]),
        ("kv", "Specimen", "Venous blood"),
        ("kv", "Collected", f"{rng.randint(1, 28):02d}/0{rng.randint(1, 9)}/2026"),
        ("heading", "Complete Blood Count"),
        ("table", ["Test", "Result", "Unit", "Flag"], _lab_rows(rng, CBC_TESTS)),
        ("heading", "Renal and Inflammatory Markers"),
        ("table", ["Test", "Result", "Unit", "Flag"], _lab_rows(rng, BMP_TESTS)),
        ("heading", "Interpretation"),
        (
            "para",
            "Results reviewed against age-matched reference ranges. Abnormal "
            "values flagged for clinical correlation and repeat testing as "
            "indicated.",
        ),
    ]


def doc_radiology(rng: random.Random, n: int) -> list[Element]:
    p = _patient(rng, n)
    exam, indication, findings, impression = rng.choice(RADIOLOGY)
    return [
        ("title", "RADIOLOGY REPORT"),
        ("kv", "MRN", p["mrn"]),
        ("kv", "Patient", p["name"]),
        ("kv", "Examination", exam),
        ("kv", "Date", f"{rng.randint(1, 28):02d}/0{rng.randint(1, 9)}/2026"),
        ("heading", "Clinical Indication"),
        ("para", indication),
        ("heading", "Findings"),
        ("para", findings),
        ("heading", "Impression"),
        ("para", impression),
    ]


def doc_discharge(rng: random.Random, n: int) -> list[Element]:
    p = _patient(rng, n)
    meds = rng.sample(MEDICATIONS, 3)
    med_line = "; ".join(f"{m[0]} {m[1]} {m[2]} {m[3]}" for m in meds)
    return [
        ("title", "DISCHARGE SUMMARY"),
        ("kv", "MRN", p["mrn"]),
        ("kv", "Patient", p["name"]),
        ("kv", "Ward", p["ward"]),
        ("kv", "Consultant", p["prescriber"]),
        ("heading", "Primary Diagnosis"),
        ("para", rng.choice(DIAGNOSES)),
        ("heading", "Secondary Diagnoses"),
        ("para", rng.choice(SECONDARY_DX)),
        ("heading", "Discharge Medications"),
        ("para", med_line),
        ("heading", "Follow-up Plan"),
        (
            "para",
            "Review in the outpatient clinic in six weeks with repeat bloods. "
            "Advised to attend the emergency department if symptoms recur.",
        ),
    ]


def doc_medication(rng: random.Random, n: int) -> list[Element]:
    p = _patient(rng, n)
    meds = rng.sample(MEDICATIONS, 5)
    return [
        ("title", "MEDICATION ADMINISTRATION ORDER"),
        ("kv", "MRN", p["mrn"]),
        ("kv", "Patient", p["name"]),
        ("kv", "Ward", p["ward"]),
        ("kv", "Prescriber", p["prescriber"]),
        ("heading", "Prescribed Medications"),
        (
            "table",
            ["Medication", "Dose", "Route", "Frequency"],
            [list(m) for m in meds],
        ),
        ("heading", "Notes"),
        (
            "para",
            "Check renal function before each furosemide dose and hold if "
            "systolic blood pressure is below 100 mmHg. Monitor potassium daily.",
        ),
    ]


DOC_BUILDERS = [doc_triage, doc_lab, doc_radiology, doc_discharge, doc_medication]


def build_documents(n_docs: int = N_DOCS, seed: int = SEED) -> dict[int, list[Element]]:
    """Construye n_docs documentos rotando las plantillas con datos reproducibles."""
    rng = random.Random(seed)
    docs: dict[int, list[Element]] = {}
    for n in range(1, n_docs + 1):
        builder = DOC_BUILDERS[(n - 1) % len(DOC_BUILDERS)]
        docs[n] = builder(rng, n)
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Render: elementos → PDF (ReportLab)
# ─────────────────────────────────────────────────────────────────────────────


def render_pdf(elements: list[Element], dst: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Title"], fontSize=16, spaceAfter=10
    )
    head_style = ParagraphStyle(
        "DocHead", parent=styles["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4
    )
    body_style = ParagraphStyle(
        "DocBody", parent=styles["BodyText"], fontSize=10.5, leading=15
    )

    story = []
    for el in elements:
        kind = el[0]
        if kind == "title":
            story.append(Paragraph(el[1], title_style))
        elif kind == "heading":
            story.append(Paragraph(el[1], head_style))
        elif kind == "para":
            story.append(Paragraph(el[1], body_style))
        elif kind == "kv":
            story.append(Paragraph(f"<b>{el[1]}:</b> {el[2]}", body_style))
        elif kind == "table":
            headers, rows = el[1], el[2]
            data = [list(headers)] + [list(r) for r in rows]
            table = Table(data, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7ef")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.append(Spacer(1, 4))
            story.append(table)
            story.append(Spacer(1, 4))

    dst.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(dst),
        pagesize=letter,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title=elements[0][1] if elements else "document",
    )
    doc.build(story)


# ─────────────────────────────────────────────────────────────────────────────
# Render: PDF → PNG (PyMuPDF). Apila todas las páginas en una imagen vertical
# para que la ground-truth completa quede siempre dentro de la imagen.
# ─────────────────────────────────────────────────────────────────────────────


def rasterize(pdf_path: Path, dst: Path, dpi: int = DPI) -> None:
    import fitz  # PyMuPDF
    from PIL import Image

    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("L"))
    doc.close()

    if not pages:
        raise RuntimeError(f"PDF sin páginas: {pdf_path}")
    width = max(p.width for p in pages)
    height = sum(p.height for p in pages)
    canvas = Image.new("L", (width, height), color=255)
    y = 0
    for p in pages:
        canvas.paste(p, (0, y))
        y += p.height
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(dst))


# ─────────────────────────────────────────────────────────────────────────────
# Orquestación
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Genera el dataset OCR (PDFs + imágenes + ground-truth) con ReportLab."
    )
    ap.add_argument("--pdf", action="store_true", help="solo PDFs (+ referencias)")
    ap.add_argument(
        "--images", action="store_true", help="solo rasterizar PDFs ya existentes"
    )
    ap.add_argument("--force", action="store_true", help="regenera aunque ya exista")
    ap.add_argument(
        "--n", type=int, default=N_DOCS, help="nº de documentos (default 20)"
    )
    args = ap.parse_args()

    do_pdf = args.pdf or not args.images
    do_images = args.images or not args.pdf

    docs = build_documents(args.n)

    if do_pdf:
        DOC_DIR.mkdir(parents=True, exist_ok=True)
        REF_DIR.mkdir(parents=True, exist_ok=True)
        for n, elements in docs.items():
            pdf_path = DOC_DIR / f"{n}.pdf"
            ref_path = REF_DIR / f"{n}.txt"
            if pdf_path.exists() and ref_path.exists() and not args.force:
                continue
            render_pdf(elements, pdf_path)
            ref_path.write_text(document_to_text(elements), encoding="utf-8")
        print(f"✓ PDFs: {len(docs)} en {DOC_DIR}")
        print(f"✓ Ground-truth: {len(docs)} en {REF_DIR}")

    if do_images:
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        rasterized = 0
        for n in docs:
            pdf_path = DOC_DIR / f"{n}.pdf"
            png_path = IMAGE_DIR / f"{n}.png"
            if not pdf_path.exists():
                print(f"  ⚠ falta {pdf_path}, ejecuta primero --pdf")
                continue
            if png_path.exists() and not args.force:
                continue
            rasterize(pdf_path, png_path)
            rasterized += 1
        print(f"✓ Imágenes ({DPI} dpi): {rasterized} nuevas en {IMAGE_DIR}")

    print("\n✓ Dataset listo: assets/ preparado para correr ocr.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
