"""Project configuration values for the Entity Recognition System."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_PATH = DATA_DIR / "sample_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
SPACY_MODEL_DIR = MODELS_DIR / "spacy"
TRANSFORMER_MODEL_DIR = MODELS_DIR / "transformer"
REPORTS_DIR = PROJECT_ROOT / "reports"

LABELS = [
    "O",
    "B-PERSON",
    "I-PERSON",
    "B-ORGANIZATION",
    "I-ORGANIZATION",
    "B-LOCATION",
    "I-LOCATION",
    "B-DATE",
    "I-DATE",
    "B-DOMAIN_TERM",
    "I-DOMAIN_TERM",
]

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
