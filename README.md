# Entity Recognition System

This project builds an Entity Recognition System from scratch in Python. The final project will compare a spaCy-based NER model with a Hugging Face transformer-based NER model.

Phase 1 focuses on project setup, BIO-tagged data, configuration, dataset validation, and tests. No model training code is included yet.

## Entity Types

- `PERSON`: People, such as employees, managers, or researchers.
- `ORGANIZATION`: Companies, universities, departments, and business units.
- `LOCATION`: Cities, countries, and geographic places.
- `DATE`: Dates, months, years, and date expressions.
- `DOMAIN_TERM`: Business, technology, AI, employment, or domain-specific terms.

## BIO Tagging

BIO tagging marks each token with its role in an entity:

- `B-` means the token begins an entity.
- `I-` means the token continues the same entity.
- `O` means the token is outside any entity.

Example:

```text
Ali        B-PERSON
Khan       I-PERSON
works      O
at         O
Google     B-ORGANIZATION
```

An `I-PERSON` tag must come after `B-PERSON` or another `I-PERSON`. The same rule applies to all entity types.

## Project Data

- `data/sample_data.csv`: small Phase 1 sample dataset with at least 15 BIO-tagged sentences.
- `data/raw/ner_starter_dataset.xlsx`: original training dataset workbook.
- `data/processed/ner_starter_bio.csv`: exported BIO token dataset used for validation and future training.
- `data/processed/ner_starter_sentences.csv`: exported sentence-level dataset with train, validation, and test splits.
- `reports/Entity_Recognition_System_Technical_Document.pdf`: technical design document for the project.

The starter training dataset contains 300 sentences and 4,258 BIO-token rows split into train, validation, and test records.

## Project Structure

```text
entity-recognition-system/
|-- data/
|   |-- raw/
|   |   `-- ner_starter_dataset.xlsx
|   |-- processed/
|   |   |-- ner_starter_bio.csv
|   |   `-- ner_starter_sentences.csv
|   `-- sample_data.csv
|-- src/
|   |-- __init__.py
|   |-- config.py
|   `-- validate_dataset.py
|-- notebooks/
|-- models/
|   |-- spacy/
|   `-- transformer/
|-- reports/
|   `-- Entity_Recognition_System_Technical_Document.pdf
|-- tests/
|   `-- test_validate_dataset.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Setup

Run these commands from the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Validate Datasets

Validate the sample dataset:

```bash
python -m src.validate_dataset
```

Validate the full processed training dataset:

```bash
python -m src.validate_dataset --path data/processed/ner_starter_bio.csv
```

## Colab Training Notebook

Use this notebook when training without a local GPU:

```text
notebooks/colab_free_t4_training.ipynb
```

It is designed for Google Colab free tier and trains:

- A lightweight spaCy NER model
- A DistilBERT-based Hugging Face token-classification model

In Colab, choose `Runtime > Change runtime type > T4 GPU` when available.

The validator reports:

- Dataset path
- Number of sentences
- Number of tokens
- Label distribution
- Validation errors, if any
- A success message when the dataset is valid

## Run Tests

Run this command from the project root:

```bash
pytest
```

## Current Status

Phase 1 data setup is complete when both datasets validate successfully and all tests pass. Model training, notebooks, APIs, apps, and deployment code are intentionally left out for later phases.
