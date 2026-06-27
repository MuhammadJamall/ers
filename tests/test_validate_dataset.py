"""Tests for dataset validation helpers."""

import pandas as pd

from src.validate_dataset import validate_dataset


def make_dataframe(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    """Create a validation dataframe from simple row tuples."""
    return pd.DataFrame(rows, columns=["sentence_id", "token", "label"])


def test_valid_bio_sequence() -> None:
    dataframe = make_dataframe(
        [
            (1, "Ali", "B-PERSON"),
            (1, "Khan", "I-PERSON"),
            (1, "joined", "O"),
            (1, "Acme", "B-ORGANIZATION"),
            (1, "Analytics", "I-ORGANIZATION"),
        ]
    )

    assert validate_dataset(dataframe) == []


def test_invalid_i_tag_at_start_of_sentence() -> None:
    dataframe = make_dataframe(
        [
            (1, "Khan", "I-PERSON"),
            (1, "joined", "O"),
        ]
    )

    errors = validate_dataset(dataframe)

    assert any("first token cannot use I-PERSON" in error for error in errors)


def test_i_tag_following_wrong_entity_type() -> None:
    dataframe = make_dataframe(
        [
            (1, "Google", "B-ORGANIZATION"),
            (1, "Lahore", "I-LOCATION"),
        ]
    )

    errors = validate_dataset(dataframe)

    assert any("I-LOCATION must follow B-LOCATION or I-LOCATION" in error for error in errors)


def test_unknown_label() -> None:
    dataframe = make_dataframe(
        [
            (1, "Ali", "B-PER"),
            (1, "works", "O"),
        ]
    )

    errors = validate_dataset(dataframe)

    assert any("Unknown label(s): B-PER" in error for error in errors)


def test_missing_required_columns() -> None:
    dataframe = pd.DataFrame(
        [
            {"sentence_id": 1, "token": "Ali"},
        ]
    )

    errors = validate_dataset(dataframe)

    assert any("Missing required columns: label" in error for error in errors)
