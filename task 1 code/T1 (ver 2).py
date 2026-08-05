#!/usr/bin/env python3
"""
Task 1 v2 (Colab) - distilroberta-base, the first move off TF-IDF.
Distilled because the Mac kept running out of memory on roberta-base.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
BASE_DIR = SCRIPT.parent
VENV_DIR = BASE_DIR / ".venv-task1-v2-macsafe"
CACHE_DIR = BASE_DIR / "huggingface_cache"
OUTPUT_DEFAULT = BASE_DIR / "prediction_task1_improved.csv"

MODEL_NAME = "distilroberta-base"
LABELS = ("phrase", "passage", "multi")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

PACKAGES = [
    "numpy",
    "scikit-learn",
    "torch",
    "transformers>=4.44,<5",
    "safetensors",
]


# --- environment bootstrap ---
def env_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def inside_correct_env() -> bool:
    target = env_python()
    if not target.exists():
        return False
    try:
        return Path(sys.executable).resolve() == target.resolve()
    except OSError:
        return False


def run_command(command: list[str], description: str) -> None:
    print(f"\n{description}")
    print(" ".join(command))
    result = subprocess.run(command, cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed with exit code {result.returncode}."
        )


def bootstrap() -> None:
    target = env_python()

    if not target.exists():
        run_command(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            "Creating isolated Python environment",
        )

    run_command(
        [
            str(target), "-m", "pip", "install", "--upgrade",
            "pip", "setuptools", "wheel",
        ],
        "Upgrading package tools",
    )

    run_command(
        [
            str(target), "-m", "pip", "install", "--upgrade",
            *PACKAGES,
        ],
        "Installing Transformer dependencies",
    )

    print("\nSetup complete. Restarting with the correct Python...\n")
    os.execv(
        str(target),
        [str(target), str(SCRIPT), *sys.argv[1:]],
    )


# --- arguments ---
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mac-safe DistilRoBERTa classifier for MSE 641 Task 1."
    )
    parser.add_argument("--train", type=str, default=None)
    parser.add_argument("--test", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(OUTPUT_DEFAULT))

    # Memory-safe defaults
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--validation-size", type=float, default=0.18)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-paragraphs", type=int, default=2)
    parser.add_argument("--max-article-chars", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU. The script automatically uses this after MPS OOM.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="One-epoch setup test.",
    )
    parser.add_argument(
        "--no-auto-cpu-fallback",
        action="store_true",
        help="Do not restart on CPU after an MPS out-of-memory error.",
    )
    return parser.parse_args()


# --- file discovery ---
def find_jsonl() -> list[Path]:
    ignored = {
        ".venv",
        ".venv-transformer",
        ".venv-task1-v2",
        ".venv-task1-v2-macsafe",
        ".git",
        "__pycache__",
    }

    files: list[Path] = []
    for path in BASE_DIR.rglob("*.jsonl"):
        if any(part in ignored for part in path.parts):
            continue
        files.append(path.resolve())
    return sorted(files)


def file_score(path: Path, kind: str) -> tuple[int, int]:
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

    return score, -len(path.relative_to(BASE_DIR).parts)


def resolve_file(explicit: str | None, kind: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{kind} file not found: {path}")
        return path

    candidates = find_jsonl()
    if not candidates:
        raise FileNotFoundError(
            f"No JSONL files found inside:\n  {BASE_DIR}"
        )

    ranked = sorted(
        candidates,
        key=lambda p: file_score(p, kind),
        reverse=True,
    )
    best = ranked[0]

    if file_score(best, kind)[0] <= 0:
        listed = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(
            f"Could not identify {kind}.jsonl automatically.\n"
            f"Found:\n{listed}\n"
            f"Run with --{kind} PATH."
        )

    return best


# --- data ---
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path}, line {line_number}."
                )

            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        return " ".join(flatten(x) for x in value if x is not None)
    if isinstance(value, dict):
        return " ".join(
            f"{flatten(k)} {flatten(v)}"
            for k, v in value.items()
        )
    return str(value)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def get_id(record: dict[str, Any]) -> str:
    for key in ("id", "postId", "uuid"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)

    raise KeyError(
        "No ID field found. Available keys: "
        + ", ".join(sorted(record.keys()))
    )


def extract_label(record: dict[str, Any]) -> str:
    for key in ("spoilerType", "spoiler_type", "label", "target", "tags"):
        if key not in record:
            continue

        value = record[key]

        if isinstance(value, (list, tuple, set)):
            matches = {
                str(x).strip().lower()
                for x in value
            } & set(LABELS)

            if len(matches) == 1:
                return next(iter(matches))
        else:
            label = str(value).strip().lower()
            if label in LABEL_TO_ID:
                return label

    raise KeyError(
        "Could not find phrase/passage/multi label. "
        f"Available keys: {list(record.keys())}"
    )


def make_text(
    record: dict[str, Any],
    max_paragraphs: int,
    max_article_chars: int,
) -> str:
    post = flatten(
        record.get("postText")
        or record.get("post_text")
        or record.get("post")
        or record.get("text")
    )

    title = flatten(
        record.get("targetTitle")
        or record.get("target_title")
        or record.get("title")
    )

    description = flatten(
        record.get("targetDescription")
        or record.get("target_description")
        or record.get("description")
    )

    keywords = flatten(
        record.get("targetKeywords")
        or record.get("target_keywords")
        or record.get("keywords")
    )

    paragraphs = (
        record.get("targetParagraphs")
        or record.get("target_paragraphs")
        or record.get("paragraphs")
        or record.get("article")
        or ""
    )

    if isinstance(paragraphs, list):
        article = " ".join(
            flatten(p)
            for p in paragraphs[:max_paragraphs]
        )
    else:
        article = flatten(paragraphs)

    article = article[:max_article_chars]

    return normalize(
        f"Clickbait post: {post} "
        f"Article title: {title} "
        f"Description: {description} "
        f"Keywords: {keywords} "
        f"Article excerpt: {article}"
    )


# --- ml ---
def import_ml():
    import numpy as np
    import torch
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
    )
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    return {
        "np": np,
        "torch": torch,
        "balanced_accuracy_score": balanced_accuracy_score,
        "classification_report": classification_report,
        "train_test_split": train_test_split,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


def select_device(torch, force_cpu: bool):
    if force_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def set_seed(seed: int, torch, np) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset_class(Dataset):
    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length, labels=None):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, index):
            encoded = self.tokenizer(
                self.texts[index],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_attention_mask=True,
            )

            item = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }

            if self.labels is not None:
                item["labels"] = int(self.labels[index])

            return item

    return TextDataset


def make_collator(tokenizer, torch):
    def collate(batch):
        labels = None

        if "labels" in batch[0]:
            labels = torch.tensor(
                [item.pop("labels") for item in batch],
                dtype=torch.long,
            )

        padded = tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt",
        )

        if labels is not None:
            padded["labels"] = labels

        return padded

    return collate


def move_batch(batch, device):
    return {
        key: tensor.to(device)
        for key, tensor in batch.items()
    }


def calculate_class_weights(label_ids, torch):
    counts = Counter(label_ids)
    total = len(label_ids)
    number_of_classes = len(LABELS)

    values = [
        total / (number_of_classes * counts[class_id])
        for class_id in range(number_of_classes)
    ]

    return torch.tensor(values, dtype=torch.float32)


def evaluate(model, loader, device, weights, ml):
    torch = ml["torch"]
    balanced_accuracy_score = ml["balanced_accuracy_score"]

    model.eval()
    loss_function = torch.nn.CrossEntropyLoss(
        weight=weights.to(device)
    )

    all_logits = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = loss_function(logits, labels)

            total_loss += float(loss.item())
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    predictions = logits.argmax(dim=1)

    score = balanced_accuracy_score(
        labels.numpy(),
        predictions.numpy(),
    )

    return (
        total_loss / max(1, len(loader)),
        score,
        predictions.numpy(),
        labels.numpy(),
    )


def predict(model, loader, device, ml):
    torch = ml["torch"]
    model.eval()
    probabilities = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(**batch).logits
            probabilities.append(
                torch.softmax(logits, dim=-1).cpu()
            )

    return torch.cat(probabilities).numpy()


def train_model(
    train_texts,
    label_ids,
    test_texts,
    tokenizer,
    args,
    device,
    ml,
):
    np = ml["np"]
    torch = ml["torch"]
    train_test_split = ml["train_test_split"]
    DataLoader = ml["DataLoader"]
    Dataset = ml["Dataset"]
    AutoModelForSequenceClassification = ml[
        "AutoModelForSequenceClassification"
    ]
    get_linear_schedule_with_warmup = ml[
        "get_linear_schedule_with_warmup"
    ]
    classification_report = ml["classification_report"]

    set_seed(args.seed, torch, np)

    indices = np.arange(len(train_texts))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=label_ids,
    )

    TextDataset = make_dataset_class(Dataset)
    collate = make_collator(tokenizer, torch)

    train_dataset = TextDataset(
        [train_texts[i] for i in train_indices],
        tokenizer,
        args.max_length,
        [label_ids[i] for i in train_indices],
    )

    validation_dataset = TextDataset(
        [train_texts[i] for i in validation_indices],
        tokenizer,
        args.max_length,
        [label_ids[i] for i in validation_indices],
    )

    test_dataset = TextDataset(
        test_texts,
        tokenizer,
        args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        cache_dir=str(CACHE_DIR),
    )
    model.to(device)

    weights = calculate_class_weights(
        [label_ids[i] for i in train_indices],
        torch,
    )

    loss_function = torch.nn.CrossEntropyLoss(
        weight=weights.to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    update_steps = max(
        1,
        (
            len(train_loader)
            + args.gradient_accumulation
            - 1
        )
        // args.gradient_accumulation,
    )

    total_steps = update_steps * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_score = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            labels = batch.pop("labels")
            logits = model(**batch).logits

            loss = loss_function(logits, labels)
            (loss / args.gradient_accumulation).backward()
            running_loss += float(loss.item())

            update_now = (
                step % args.gradient_accumulation == 0
                or step == len(train_loader)
            )

            if update_now:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 200 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Batch {step}/{len(train_loader)} | "
                    f"Loss {running_loss / step:.4f}"
                )

        val_loss, val_score, val_predictions, val_true = evaluate(
            model,
            validation_loader,
            device,
            weights,
            ml,
        )

        print(
            f"\nEpoch {epoch}: validation loss={val_loss:.4f}, "
            f"balanced accuracy={val_score:.5f}"
        )

        if val_score > best_score:
            best_score = val_score
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            print("New best model.")
            print(
                classification_report(
                    val_true,
                    val_predictions,
                    labels=[0, 1, 2],
                    target_names=list(LABELS),
                    digits=4,
                    zero_division=0,
                )
            )
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("No valid trained model state was created.")

    model.load_state_dict(best_state)
    model.to(device)

    probabilities = predict(
        model,
        test_loader,
        device,
        ml,
    )

    return probabilities, best_score


# --- output ---
def write_submission(
    test_records,
    predictions,
    output_path: Path,
) -> None:
    if len(test_records) != len(predictions):
        raise ValueError("Prediction count does not match test count.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["id", "spoilerType"])

        for record, prediction in zip(test_records, predictions):
            writer.writerow([get_id(record), prediction])

    print("\nKaggle submission created successfully:")
    print(f"  {output_path}")
    print(f"  Rows: {len(predictions)}")
    print("  Columns: id, spoilerType")


# --- cpu fallback ---
def is_mps_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "mps backend out of memory" in message
        or ("mps" in message and "out of memory" in message)
    )


def restart_on_cpu() -> None:
    print(
        "\nMPS ran out of memory. Restarting automatically on CPU.\n"
        "This is slower but avoids the Mac GPU memory limit.\n"
    )

    args = [arg for arg in sys.argv[1:] if arg != "--cpu"]
    os.execv(
        str(env_python()),
        [
            str(env_python()),
            str(SCRIPT),
            *args,
            "--cpu",
            "--no-auto-cpu-fallback",
        ],
    )


# --- main ---
def main() -> None:
    # Colab already provides a managed Python environment; do not create a venv.
    args = parse_args()

    if args.quick:
        args.epochs = 1
        args.max_length = 128
        args.max_paragraphs = 1
        args.max_article_chars = 1500

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ml = import_ml()
    np = ml["np"]
    torch = ml["torch"]

    train_path = resolve_file(args.train, "train")
    test_path = resolve_file(args.test, "test")

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (BASE_DIR / output_path).resolve()

    device = select_device(torch, args.cpu)

    print("=" * 72)
    print("MSE 641 TASK 1 — VERSION 2 COLAB TRANSFORMER")
    print("=" * 72)
    print(f"Project folder: {BASE_DIR}")
    print(f"Python: {sys.executable}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Train file: {train_path}")
    print(f"Test file: {test_path}")
    print(f"Output file: {output_path}")
    print(
        f"Memory settings: max_length={args.max_length}, "
        f"train_batch={args.train_batch_size}, "
        f"eval_batch={args.eval_batch_size}, "
        f"gradient_accumulation={args.gradient_accumulation}"
    )

    train_records = read_jsonl(train_path)
    test_records = read_jsonl(test_path)

    print(
        f"\nLoaded {len(train_records)} training records "
        f"and {len(test_records)} test records."
    )

    labels = [extract_label(record) for record in train_records]
    label_ids = [LABEL_TO_ID[label] for label in labels]

    print("\nClass distribution:")
    counts = Counter(labels)
    for label in LABELS:
        print(f"  {label}: {counts[label]}")

    train_texts = [
        make_text(
            record,
            args.max_paragraphs,
            args.max_article_chars,
        )
        for record in train_records
    ]

    test_texts = [
        make_text(
            record,
            args.max_paragraphs,
            args.max_article_chars,
        )
        for record in test_records
    ]

    tokenizer = ml["AutoTokenizer"].from_pretrained(
        MODEL_NAME,
        cache_dir=str(CACHE_DIR),
        use_fast=True,
    )

    try:
        probabilities, validation_score = train_model(
            train_texts,
            label_ids,
            test_texts,
            tokenizer,
            args,
            device,
            ml,
        )
    except RuntimeError as exc:
        if (
            device.type == "mps"
            and is_mps_oom(exc)
            and not args.no_auto_cpu_fallback
        ):
            restart_on_cpu()
        raise

    prediction_ids = probabilities.argmax(axis=1)
    predictions = [
        ID_TO_LABEL[int(index)]
        for index in prediction_ids
    ]

    print(f"\nBest validation balanced accuracy: {validation_score:.5f}")

    print("\nPrediction distribution:")
    prediction_counts = Counter(predictions)
    for label in LABELS:
        print(f"  {label}: {prediction_counts[label]}")

    write_submission(
        test_records,
        predictions,
        output_path,
    )

    print("\nUpload this file to Kaggle:")
    print(f"  {output_path.name}")
    print(
        "Suggested description: "
        "DistilRoBERTa weighted-loss Mac-safe Transformer"
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
            "\nPlace this script and Task 1 JSONL files inside:\n"
            f"  {BASE_DIR}"
        )
        sys.exit(1)
