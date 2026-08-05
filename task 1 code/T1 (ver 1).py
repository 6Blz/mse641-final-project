#!/usr/bin/env python3
"""
Task 1 v1 - word + char TF-IDF into a class-balanced LinearSVC, 5-fold CV.
The baseline before any transformer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
BASE_DIR = SCRIPT_PATH.parent
VENV_DIR = BASE_DIR / ".venv"
OUTPUT_DEFAULT = BASE_DIR / "prediction_task1_improved.csv"

REQUIRED_PACKAGES = [
    "numpy",
    "scipy",
    "scikit-learn",
]

VALID_LABELS = {"phrase", "passage", "multi"}


# --- environment bootstrap: this section uses only python's standard library. ---
def venv_python_path() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def running_inside_project_venv() -> bool:
    try:
        return Path(sys.executable).resolve() == venv_python_path().resolve()
    except FileNotFoundError:
        return False


def run_command(command: list[str], description: str) -> None:
    print(f"\n{description}")
    print(" ".join(str(part) for part in command))

    result = subprocess.run(command, cwd=BASE_DIR)

    if result.returncode != 0:
        raise RuntimeError(
            f"\nCommand failed while: {description}\n"
            f"Exit code: {result.returncode}"
        )


def bootstrap_environment() -> None:
    python_in_venv = venv_python_path()

    if not python_in_venv.exists():
        run_command(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            "Creating the project virtual environment...",
        )

    run_command(
        [str(python_in_venv), "-m", "pip", "install", "--upgrade",
         "pip", "setuptools", "wheel"],
        "Upgrading Python packaging tools...",
    )

    run_command(
        [str(python_in_venv), "-m", "pip", "install", "--upgrade",
         *REQUIRED_PACKAGES],
        "Installing/upgrading model dependencies...",
    )

    print("\nEnvironment is ready. Restarting the script inside .venv...\n")

    os.execv(
        str(python_in_venv),
        [str(python_in_venv), str(SCRIPT_PATH), *sys.argv[1:]],
    )


def verify_dependencies() -> None:
    try:
        import numpy  
        import scipy 
        import sklearn 
    except ImportError as exc:
        raise RuntimeError(
            "The virtual environment exists, but a required package could "
            f"not be imported: {exc}"
        ) from exc


# --- arguments and file discovery ---
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete Task 1 training, validation, and submission runner."
    )
    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help="Optional explicit path to train.jsonl.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Optional explicit path to test.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DEFAULT),
        help=f"Output CSV path. Default: {OUTPUT_DEFAULT.name}",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation and train the final model immediately.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified CV folds. Default: 5.",
    )
    parser.add_argument(
        "--max-paragraphs",
        type=int,
        default=8,
        help="Maximum number of article paragraphs used per record. Default: 8.",
    )
    parser.add_argument(
        "--max-article-chars",
        type=int,
        default=6000,
        help="Maximum article characters used per record. Default: 6000.",
    )
    return parser.parse_args()


def find_jsonl_files() -> list[Path]:
    ignored_parts = {".venv", "__pycache__", ".git"}
    files: list[Path] = []

    for path in BASE_DIR.rglob("*.jsonl"):
        if any(part in ignored_parts for part in path.parts):
            continue
        files.append(path.resolve())

    return sorted(files)


def score_filename(path: Path, kind: str) -> tuple[int, int]:
    name = path.name.lower()
    score = 0

    if name == f"{kind}.jsonl":
        score += 100
    if kind in name:
        score += 50
    if "task1" in name or "task-1" in name:
        score += 10
    if kind == "train" and "test" in name:
        score -= 100
    if kind == "test" and "train" in name:
        score -= 100

    depth_penalty = len(path.relative_to(BASE_DIR).parts)
    return score, -depth_penalty


def resolve_data_file(explicit_path: str | None, kind: str) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()

        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"{kind.title()} file not found: {path}")

        return path

    candidates = find_jsonl_files()

    if not candidates:
        raise FileNotFoundError(
            "\nNo .jsonl files were found inside:\n"
            f"  {BASE_DIR}\n\n"
            "Download the Task 1 competition data from Kaggle, unzip it, "
            "and place the JSONL files anywhere inside this folder."
        )

    ranked = sorted(
        candidates,
        key=lambda path: score_filename(path, kind),
        reverse=True,
    )

    best = ranked[0]
    best_score = score_filename(best, kind)[0]

    if best_score <= 0:
        listed = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            f"\nCould not automatically identify the {kind} JSONL file.\n"
            f"Found:\n{listed}\n\n"
            f"Run again with --{kind} followed by the correct path."
        )

    return best


# --- data parsing and feature construction ---
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}: {exc}"
                ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {path}, "
                    f"but found {type(value).__name__}."
                )

            records.append(value)

    if not records:
        raise ValueError(f"No usable records found in {path}")

    return records


def get_id(record: dict[str, Any]) -> str:
    for key in ("id", "postId", "uuid"):
        value = record.get(key)

        if value not in (None, ""):
            return str(value)

    raise KeyError(
        "No submission ID field found. Available keys: "
        + ", ".join(sorted(record.keys()))
    )


def flatten_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (list, tuple, set)):
        return " ".join(
            flatten_text(item)
            for item in value
            if item is not None
        )

    if isinstance(value, dict):
        return " ".join(
            f"{flatten_text(key)} {flatten_text(item)}"
            for key, item in value.items()
        )

    return str(value)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_label(record: dict[str, Any]) -> str:
    candidate_keys = (
        "spoilerType",
        "spoiler_type",
        "label",
        "target",
        "tags",
    )

    for key in candidate_keys:
        if key not in record:
            continue

        value = record[key]

        if isinstance(value, (list, tuple, set)):
            normalized = {
                str(item).strip().lower()
                for item in value
            }
            matches = normalized & VALID_LABELS

            if len(matches) == 1:
                return next(iter(matches))

        else:
            label = str(value).strip().lower()

            if label in VALID_LABELS:
                return label

    raise KeyError(
        "Could not find a valid Task 1 label. Expected phrase, passage, "
        "or multi in a field such as tags. Available keys: "
        + ", ".join(sorted(record.keys()))
    )


def build_signal_tokens(post_text: str, title: str) -> str:
    combined = f"{post_text} {title}".lower()
    tokens: list[str] = []

    question_marks = combined.count("?")
    exclamation_marks = combined.count("!")
    digit_count = sum(character.isdigit() for character in combined)
    word_count = len(combined.split())

    tokens.extend(
        [
            f"POST_WORD_BUCKET_{min(word_count // 5, 20)}",
            f"QUESTION_MARK_BUCKET_{min(question_marks, 3)}",
            f"EXCLAMATION_MARK_BUCKET_{min(exclamation_marks, 3)}",
            f"DIGIT_BUCKET_{min(digit_count, 5)}",
        ]
    )

    patterns = {
        "WHO_SIGNAL": r"\bwho\b",
        "WHERE_SIGNAL": r"\bwhere\b",
        "WHEN_SIGNAL": r"\bwhen\b",
        "WHAT_SIGNAL": r"\bwhat\b",
        "WHICH_SIGNAL": r"\bwhich\b",
        "HOW_MANY_SIGNAL": r"\bhow many\b",
        "HOW_MUCH_SIGNAL": r"\bhow much\b",
        "NAME_SIGNAL": r"\b(name|called|identity)\b",
        "WHY_SIGNAL": r"\bwhy\b",
        "HOW_SIGNAL": r"\bhow\b",
        "WHAT_HAPPENED_SIGNAL": r"\bwhat happened\b",
        "REASON_SIGNAL": r"\b(reason|because|explains?|explanation)\b",
        "LIST_SIGNAL": r"\b(list|ways|reasons|things|facts|tips|steps|signs)\b",
        "NUMBERED_SIGNAL": (
            r"\b(two|three|four|five|six|seven|eight|nine|ten|\d+)\b"
        ),
        "AND_SIGNAL": r"\band\b",
        "VERSUS_SIGNAL": (
            r"\b(vs\.?|versus|compared with|difference between)\b"
        ),
    }

    for token, pattern in patterns.items():
        if re.search(pattern, combined):
            tokens.append(token)

    return " ".join(tokens)


def build_document(
    record: dict[str, Any],
    max_paragraphs: int,
    max_article_chars: int,
) -> str:
    post_text = flatten_text(
        record.get("postText")
        or record.get("post_text")
        or record.get("post")
        or record.get("text")
    )

    title = flatten_text(
        record.get("targetTitle")
        or record.get("target_title")
        or record.get("title")
    )

    description = flatten_text(
        record.get("targetDescription")
        or record.get("target_description")
        or record.get("description")
    )

    keywords = flatten_text(
        record.get("targetKeywords")
        or record.get("target_keywords")
        or record.get("keywords")
    )

    platform = flatten_text(
        record.get("postPlatform")
        or record.get("post_platform")
        or record.get("platform")
    )

    paragraphs_value = (
        record.get("targetParagraphs")
        or record.get("target_paragraphs")
        or record.get("paragraphs")
        or record.get("article")
        or ""
    )

    if isinstance(paragraphs_value, list):
        selected = paragraphs_value[:max_paragraphs]
        article = " ".join(flatten_text(item) for item in selected)
    else:
        article = flatten_text(paragraphs_value)

    article = article[:max_article_chars]
    signals = build_signal_tokens(post_text, title)

    # Post and title are repeated because they tend to carry the strongest
    # spoiler-type cues. Article context is included but capped.
    document = " ".join(
        [
            "[POST]",
            post_text,
            post_text,
            "[TITLE]",
            title,
            title,
            "[DESCRIPTION]",
            description,
            "[KEYWORDS]",
            keywords,
            "[PLATFORM]",
            platform,
            "[ARTICLE]",
            article,
            "[SIGNALS]",
            signals,
        ]
    )

    return normalize_whitespace(document)


# --- model, validation, training, and output ---
def import_ml_dependencies():
    import numpy as np
    from sklearn.base import clone
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC

    return {
        "np": np,
        "clone": clone,
        "TfidfVectorizer": TfidfVectorizer,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "StratifiedKFold": StratifiedKFold,
        "cross_val_predict": cross_val_predict,
        "FeatureUnion": FeatureUnion,
        "Pipeline": Pipeline,
        "LinearSVC": LinearSVC,
    }


def build_model(ml):
    TfidfVectorizer = ml["TfidfVectorizer"]
    FeatureUnion = ml["FeatureUnion"]
    Pipeline = ml["Pipeline"]
    LinearSVC = ml["LinearSVC"]

    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=120_000,
                    sublinear_tf=True,
                    norm="l2",
                ),
            ),
            (
                "character_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=120_000,
                    sublinear_tf=True,
                    norm="l2",
                ),
            ),
        ]
    )

    classifier = LinearSVC(
        C=1.5,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        [
            ("features", features),
            ("classifier", classifier),
        ]
    )


def run_cross_validation(
    model,
    documents: list[str],
    labels: list[str],
    folds: int,
    ml,
) -> None:
    clone = ml["clone"]
    classification_report = ml["classification_report"]
    confusion_matrix = ml["confusion_matrix"]
    f1_score = ml["f1_score"]
    StratifiedKFold = ml["StratifiedKFold"]
    cross_val_predict = ml["cross_val_predict"]

    counts = Counter(labels)
    smallest_class = min(counts.values())
    effective_folds = min(folds, smallest_class)

    print("\nClass distribution:")
    for label in ("phrase", "passage", "multi"):
        print(f"  {label}: {counts.get(label, 0)}")

    if effective_folds < 2:
        print(
            "\nCross-validation skipped: at least one class has fewer than "
            "2 examples."
        )
        return

    print(
        f"\nRunning {effective_folds}-fold stratified cross-validation..."
    )

    cv = StratifiedKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=42,
    )

    predictions = cross_val_predict(
        clone(model),
        documents,
        labels,
        cv=cv,
        n_jobs=-1,
    )

    macro_f1 = f1_score(labels, predictions, average="macro")
    weighted_f1 = f1_score(labels, predictions, average="weighted")

    print(f"\nCross-validation macro F1:    {macro_f1:.5f}")
    print(f"Cross-validation weighted F1: {weighted_f1:.5f}")

    print("\nClassification report:")
    print(
        classification_report(
            labels,
            predictions,
            labels=["phrase", "passage", "multi"],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix")
    print("Rows=true; columns=predicted; order=phrase, passage, multi")
    print(
        confusion_matrix(
            labels,
            predictions,
            labels=["phrase", "passage", "multi"],
        )
    )


def write_submission(
    records: Iterable[dict[str, Any]],
    predictions: Iterable[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_list = list(predictions)

    if len(predictions_list) != len(records):
        raise ValueError(
            "Prediction count does not match the number of test records."
        )

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "spoilerType"])

        for record, prediction in zip(records, predictions_list):
            writer.writerow([get_id(record), prediction])

    expected_lines = len(records) + 1

    with output_path.open("r", encoding="utf-8") as file:
        actual_lines = sum(1 for _ in file)

    if actual_lines != expected_lines:
        raise RuntimeError(
            f"Output validation failed: expected {expected_lines} lines, "
            f"found {actual_lines}."
        )

    print("\nKaggle submission created successfully:")
    print(f"  {output_path}")
    print(f"  Rows excluding header: {len(records)}")
    print("  Columns: id, spoilerType")


def main() -> None:
    if not running_inside_project_venv():
        bootstrap_environment()

    verify_dependencies()
    args = parse_args()
    ml = import_ml_dependencies()

    print("=" * 72)
    print("MSE 641 TASK 1 — COMPLETE RUNNER")
    print("=" * 72)
    print(f"Script folder:\n  {BASE_DIR}")
    print(f"Python environment:\n  {sys.executable}")

    train_path = resolve_data_file(args.train, "train")
    test_path = resolve_data_file(args.test, "test")
    output_path = Path(args.output).expanduser()

    if not output_path.is_absolute():
        output_path = (BASE_DIR / output_path).resolve()

    print(f"\nTraining data:\n  {train_path}")
    print(f"Test data:\n  {test_path}")
    print(f"Output CSV:\n  {output_path}")

    train_records = read_jsonl(train_path)
    test_records = read_jsonl(test_path)

    print(
        f"\nLoaded {len(train_records)} training records and "
        f"{len(test_records)} test records."
    )

    labels = [extract_label(record) for record in train_records]

    train_documents = [
        build_document(
            record,
            max_paragraphs=args.max_paragraphs,
            max_article_chars=args.max_article_chars,
        )
        for record in train_records
    ]

    test_documents = [
        build_document(
            record,
            max_paragraphs=args.max_paragraphs,
            max_article_chars=args.max_article_chars,
        )
        for record in test_records
    ]

    if not any(train_documents):
        raise ValueError("All constructed training documents are empty.")

    model = build_model(ml)

    if not args.no_cv:
        run_cross_validation(
            model=model,
            documents=train_documents,
            labels=labels,
            folds=args.folds,
            ml=ml,
        )

    print("\nTraining the final model on all training records...")
    model.fit(train_documents, labels)

    print("Predicting test records...")
    test_predictions = model.predict(test_documents)

    prediction_counts = Counter(str(label) for label in test_predictions)

    print("\nTest prediction distribution:")
    for label in ("phrase", "passage", "multi"):
        print(f"  {label}: {prediction_counts.get(label, 0)}")

    write_submission(
        records=test_records,
        predictions=test_predictions,
        output_path=output_path,
    )

    print("\nNext step:")
    print(
        "Upload prediction_task1_improved.csv to the Task 1 Kaggle "
        "competition."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(130)
    except Exception as exc:
        print("\n" + "=" * 72)
        print("ERROR")
        print("=" * 72)
        print(exc)
        print(
            "\nThe script, data files, and output should all be inside:\n"
            f"  {BASE_DIR}"
        )
        sys.exit(1)
