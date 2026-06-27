"""Validate BIO-tagged NER datasets."""

import argparse
from pathlib import Path

import pandas as pd

from src.config import LABELS, SAMPLE_DATA_PATH


REQUIRED_COLUMNS = {"sentence_id", "token", "label"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset validation."""
    parser = argparse.ArgumentParser(description="Validate a BIO-tagged NER CSV dataset.")
    parser.add_argument(
        "--path",
        type=Path,
        default=SAMPLE_DATA_PATH,
        help="Path to the CSV file to validate. Defaults to data/sample_data.csv.",
    )
    return parser.parse_args()


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load a CSV dataset and raise a clear error if it cannot be read."""
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset file not found: {csv_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset file is empty: {csv_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Dataset file could not be parsed as CSV: {csv_path}") from exc


def validate_required_columns(dataframe: pd.DataFrame) -> list[str]:
    """Check that all required dataset columns are present."""
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if not missing_columns:
        return []

    return [f"Missing required columns: {', '.join(sorted(missing_columns))}"]


def validate_missing_values(dataframe: pd.DataFrame) -> list[str]:
    """Check required columns for missing values."""
    errors: list[str] = []
    available_columns = REQUIRED_COLUMNS.intersection(dataframe.columns)

    for column in sorted(available_columns):
        missing_count = int(dataframe[column].isna().sum())
        if missing_count > 0:
            errors.append(f"Column '{column}' has {missing_count} missing value(s).")

    return errors


def validate_labels(dataframe: pd.DataFrame) -> list[str]:
    """Check that every label belongs to the configured label list."""
    if "label" not in dataframe.columns:
        return []

    valid_labels = set(LABELS)
    invalid_labels = sorted(set(dataframe["label"].dropna()) - valid_labels)

    if not invalid_labels:
        return []

    return [f"Unknown label(s): {', '.join(invalid_labels)}"]


def validate_sentence_ids(dataframe: pd.DataFrame) -> list[str]:
    """Check that sentence IDs are present and usable."""
    if "sentence_id" not in dataframe.columns:
        return []

    if dataframe["sentence_id"].dropna().empty:
        return ["No sentence IDs are present."]

    return []


def validate_bio_consistency(dataframe: pd.DataFrame) -> list[str]:
    """Check basic BIO tag consistency within each sentence."""
    if not REQUIRED_COLUMNS.issubset(dataframe.columns):
        return []

    errors: list[str] = []

    for sentence_id, sentence in dataframe.groupby("sentence_id", sort=False):
        previous_label = "O"

        for position, row in enumerate(sentence.itertuples(index=False), start=1):
            label = str(row.label)
            token = str(row.token)

            if label == "O" or label.startswith("B-"):
                previous_label = label
                continue

            if not label.startswith("I-"):
                previous_label = label
                continue

            entity_type = label[2:]
            valid_previous_labels = {f"B-{entity_type}", f"I-{entity_type}"}

            if position == 1:
                errors.append(
                    f"Sentence {sentence_id}, token '{token}': first token cannot use {label}."
                )
            elif previous_label not in valid_previous_labels:
                errors.append(
                    "Sentence "
                    f"{sentence_id}, token '{token}': {label} must follow "
                    f"B-{entity_type} or I-{entity_type}, but found {previous_label}."
                )

            previous_label = label

    return errors


def validate_dataset(dataframe: pd.DataFrame) -> list[str]:
    """Run all dataset validation checks and return validation errors."""
    errors: list[str] = []
    errors.extend(validate_required_columns(dataframe))
    errors.extend(validate_missing_values(dataframe))
    errors.extend(validate_sentence_ids(dataframe))
    errors.extend(validate_labels(dataframe))
    errors.extend(validate_bio_consistency(dataframe))
    return errors


def print_dataset_summary(dataframe: pd.DataFrame) -> None:
    """Print dataset size and label distribution details."""
    if "sentence_id" in dataframe.columns:
        print(f"Number of sentences: {dataframe['sentence_id'].nunique()}")
    else:
        print("Number of sentences: unavailable")

    print(f"Number of tokens: {len(dataframe)}")

    if "label" in dataframe.columns:
        print("\nLabel distribution:")
        for label, count in dataframe["label"].value_counts().sort_index().items():
            print(f"  {label}: {count}")
    else:
        print("\nLabel distribution: unavailable")


def main() -> None:
    """Load and validate a BIO-tagged dataset."""
    args = parse_args()

    try:
        dataframe = load_dataset(args.path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Validation failed: {exc}")
        raise SystemExit(1) from exc

    print(f"Dataset path: {args.path}")
    errors = validate_dataset(dataframe)
    print_dataset_summary(dataframe)

    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("\nDataset validation passed successfully.")


if __name__ == "__main__":
    main()
