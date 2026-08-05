#!/usr/bin/env python3
"""
Task 1 v8 - v6 with roberta-large instead of roberta-base, nothing else changed.
"""

from __future__ import annotations

import os

# Helps reduce CUDA allocator fragmentation on Colab.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import csv
import gc
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_NAME = "FacebookAI/roberta-large"
LABELS = ["phrase", "passage", "multi"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

SEEDS = [42, 123, 777]
MAX_LENGTH = 256
EPOCHS = 3

# RoBERTa-large Colab-safe settings.
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
EVAL_BATCH_SIZE = 8

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0

MAIN_OUTPUT = (
    BASE_DIR
    / "prediction_task1_v8_roberta_large_ensemble.csv"
)
REPORT_OUTPUT = BASE_DIR / "task1_v8_validation_report.txt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_file(kind: str) -> Path:
    files = sorted(BASE_DIR.glob("*.jsonl"))
    lower_to_path = {path.name.lower(): path for path in files}

    exact_names = {
        "train": ["train.jsonl"],
        "val": ["val.jsonl", "validation.jsonl", "dev.jsonl"],
        "test": ["test.jsonl"],
    }[kind]

    for name in exact_names:
        if name in lower_to_path:
            return lower_to_path[name]

    keywords = {
        "train": ["train"],
        "val": ["validation", "val", "dev"],
        "test": ["test"],
    }[kind]

    matches = [
        path
        for path in files
        if any(keyword in path.name.lower() for keyword in keywords)
    ]

    if kind == "val":
        matches = [
            path
            for path in matches
            if "train" not in path.name.lower()
            and "test" not in path.name.lower()
        ]

    if len(matches) == 1:
        return matches[0]

    listed = "\n".join(f"  - {path.name}" for path in files) or "  (none)"
    raise FileNotFoundError(
        f"Could not identify the {kind} JSONL file.\n"
        f"JSONL files found:\n{listed}"
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name}, line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path.name}, line {line_number}."
                )

            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path.name}")

    return records


def post_text(record: dict[str, Any]) -> str:
    value = record.get("postText", "")

    if isinstance(value, list):
        return " ".join(str(item) for item in value)

    return str(value or "")


def title_text(record: dict[str, Any]) -> str:
    return str(record.get("targetTitle", "") or "")


def extract_label(record: dict[str, Any]) -> int:
    value = record.get("tags")

    if isinstance(value, list) and value:
        label = str(value[0]).strip().lower()
    elif isinstance(value, str):
        label = value.strip().lower()
    else:
        label = str(
            record.get("spoilerType")
            or record.get("label")
            or ""
        ).strip().lower()

    if label not in LABEL_TO_ID:
        raise ValueError(
            f"Unknown or missing label {label!r}; expected one of {LABELS}."
        )

    return LABEL_TO_ID[label]


def record_id(record: dict[str, Any]) -> str:
    for key in ("id", "postId", "uuid"):
        value = record.get(key)

        if value not in (None, ""):
            return str(value)

    raise KeyError("No id, postId, or uuid field found in a test record.")


class ClickbaitDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer,
        labelled: bool,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.labelled = labelled

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        encoded = self.tokenizer(
            post_text(record),
            title_text(record),
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        if self.labelled:
            encoded["labels"] = extract_label(record)

        return encoded


def make_train_loader(
    dataset: Dataset,
    collator,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
    )


def make_eval_loader(
    dataset: Dataset,
    collator,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
    )


def create_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


@torch.no_grad()
def predict_probabilities(
    model,
    loader: DataLoader,
    device: torch.device,
    labelled: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()

    probability_batches: list[torch.Tensor] = []
    gold_batches: list[torch.Tensor] = []

    for batch in loader:
        labels = batch.pop("labels", None)

        batch = {
            key: tensor.to(device, non_blocking=True)
            for key, tensor in batch.items()
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            logits = model(**batch).logits

        probability_batches.append(
            torch.softmax(logits.float(), dim=-1).cpu()
        )

        if labelled:
            if labels is None:
                raise RuntimeError(
                    "Validation batch did not contain labels."
                )
            gold_batches.append(labels.cpu())

    probabilities = torch.cat(probability_batches).numpy()

    if labelled:
        return probabilities, torch.cat(gold_batches).numpy()

    return probabilities, None


def weighted_f1(
    probabilities: np.ndarray,
    gold: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    predictions = probabilities.argmax(axis=1)
    accuracy = float(np.mean(predictions == gold))

    weighted_sum = 0.0

    for class_id in range(len(LABELS)):
        true_positive = int(
            np.sum((predictions == class_id) & (gold == class_id))
        )
        false_positive = int(
            np.sum((predictions == class_id) & (gold != class_id))
        )
        false_negative = int(
            np.sum((predictions != class_id) & (gold == class_id))
        )
        support = int(np.sum(gold == class_id))

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        weighted_sum += f1 * support

    return weighted_sum / len(gold), accuracy, predictions


def write_submission(
    path: Path,
    test_records: list[dict[str, Any]],
    probabilities: np.ndarray,
) -> None:
    prediction_ids = probabilities.argmax(axis=1)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "spoilerType"])

        for record, prediction_id in zip(test_records, prediction_ids):
            writer.writerow([
                record_id(record),
                ID_TO_LABEL[int(prediction_id)],
            ])


def train_one_seed(
    seed: int,
    train_dataset: Dataset,
    val_loader: DataLoader,
    test_loader: DataLoader,
    collator,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    set_seed(seed)
    torch.cuda.empty_cache()
    gc.collect()

    train_loader = make_train_loader(
        train_dataset,
        collator,
        seed,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    # Memory-saving implementation details for RoBERTa-large.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    update_steps_per_epoch = (
        len(train_loader)
        + GRADIENT_ACCUMULATION_STEPS
        - 1
    ) // GRADIENT_ACCUMULATION_STEPS

    total_update_steps = update_steps_per_epoch * EPOCHS
    warmup_steps = int(total_update_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    scaler = create_grad_scaler()

    print("\n" + "=" * 72)
    print(f"TRAINING ROBERTA-LARGE — SEED {seed}")
    print("=" * 72)

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels")
            batch = {
                key: tensor.to(device, non_blocking=True)
                for key, tensor in batch.items()
            }
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):
                output = model(
                    **batch,
                    labels=labels,
                )
                loss = output.loss
                scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.item())

            should_update = (
                step % GRADIENT_ACCUMULATION_STEPS == 0
                or step == len(train_loader)
            )

            if should_update:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=MAX_GRAD_NORM,
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 200 == 0 or step == len(train_loader):
                print(
                    f"Seed {seed} | Epoch {epoch}/{EPOCHS} | "
                    f"Batch {step}/{len(train_loader)} | "
                    f"Average loss {running_loss / step:.4f}"
                )

    val_probabilities, gold = predict_probabilities(
        model=model,
        loader=val_loader,
        device=device,
        labelled=True,
    )

    test_probabilities, _ = predict_probabilities(
        model=model,
        loader=test_loader,
        device=device,
        labelled=False,
    )

    if gold is None:
        raise RuntimeError("Validation labels were not returned.")

    del model
    del optimizer
    del scheduler
    del scaler
    del train_loader

    gc.collect()
    torch.cuda.empty_cache()

    return val_probabilities, test_probabilities, gold


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No Colab GPU detected. Select "
            "Runtime → Change runtime type → T4 GPU."
        )

    device = torch.device("cuda")

    train_path = find_file("train")
    val_path = find_file("val")
    test_path = find_file("test")

    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)
    test_records = load_jsonl(test_path)

    print("=" * 72)
    print("TASK 1 VERSION 8 — ROBERTA-LARGE THREE-SEED ENSEMBLE")
    print("=" * 72)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Seeds: {SEEDS}")
    print(f"Train: {train_path.name} ({len(train_records)})")
    print(f"Validation: {val_path.name} ({len(val_records)})")
    print(f"Test: {test_path.name} ({len(test_records)})")
    print(
        f"Settings: max_length={MAX_LENGTH}, epochs={EPOCHS}, "
        f"micro_batch={MICRO_BATCH_SIZE}, "
        f"gradient_accumulation={GRADIENT_ACCUMULATION_STEPS}, "
        f"effective_batch="
        f"{MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}, "
        f"lr={LEARNING_RATE}"
    )

    train_distribution = Counter(
        LABELS[extract_label(record)]
        for record in train_records
    )
    print(f"Train label distribution: {dict(train_distribution)}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    train_dataset = ClickbaitDataset(
        train_records,
        tokenizer,
        labelled=True,
    )
    val_dataset = ClickbaitDataset(
        val_records,
        tokenizer,
        labelled=True,
    )
    test_dataset = ClickbaitDataset(
        test_records,
        tokenizer,
        labelled=False,
    )

    val_loader = make_eval_loader(val_dataset, collator)
    test_loader = make_eval_loader(test_dataset, collator)

    val_probability_sum = np.zeros(
        (len(val_records), len(LABELS)),
        dtype=np.float64,
    )
    test_probability_sum = np.zeros(
        (len(test_records), len(LABELS)),
        dtype=np.float64,
    )

    gold_reference: np.ndarray | None = None

    report_lines = [
        "Task 1 Version 8 — RoBERTa-large three-seed ensemble",
        "",
        "Single modeling change from Version 6:",
        "  roberta-base -> roberta-large",
        "",
        "Seeds: 42, 123, 777",
        "Uniform probability averaging; no seed-subset selection.",
        "",
    ]

    for seed in SEEDS:
        val_probabilities, test_probabilities, gold = train_one_seed(
            seed=seed,
            train_dataset=train_dataset,
            val_loader=val_loader,
            test_loader=test_loader,
            collator=collator,
            device=device,
        )

        if gold_reference is None:
            gold_reference = gold
        elif not np.array_equal(gold_reference, gold):
            raise RuntimeError(
                "Validation labels changed between seed runs."
            )

        val_probability_sum += val_probabilities
        test_probability_sum += test_probabilities

        seed_f1, seed_accuracy, _ = weighted_f1(
            val_probabilities,
            gold,
        )

        seed_output = (
            BASE_DIR
            / f"prediction_task1_v8_roberta_large_seed{seed}.csv"
        )
        write_submission(
            seed_output,
            test_records,
            test_probabilities,
        )

        print(
            f"\nSeed {seed} validation weighted F1: "
            f"{seed_f1:.5f}"
        )
        print(
            f"Seed {seed} validation accuracy: "
            f"{seed_accuracy:.5f}"
        )

        report_lines.extend([
            f"Seed {seed}",
            f"  Validation weighted F1: {seed_f1:.6f}",
            f"  Validation accuracy: {seed_accuracy:.6f}",
            f"  Output: {seed_output.name}",
            "",
        ])

    if gold_reference is None:
        raise RuntimeError("No models were trained.")

    ensemble_val_probabilities = val_probability_sum / len(SEEDS)
    ensemble_test_probabilities = test_probability_sum / len(SEEDS)

    ensemble_f1, ensemble_accuracy, _ = weighted_f1(
        ensemble_val_probabilities,
        gold_reference,
    )

    write_submission(
        MAIN_OUTPUT,
        test_records,
        ensemble_test_probabilities,
    )

    ensemble_prediction_ids = (
        ensemble_test_probabilities.argmax(axis=1)
    )
    prediction_distribution = Counter(
        ID_TO_LABEL[int(index)]
        for index in ensemble_prediction_ids
    )

    print("\n" + "=" * 72)
    print("VERSION 8 ENSEMBLE RESULTS")
    print("=" * 72)
    print(
        f"Validation weighted F1: {ensemble_f1:.5f}"
    )
    print(
        f"Validation accuracy: {ensemble_accuracy:.5f}"
    )
    print(
        f"Test prediction distribution: "
        f"{dict(prediction_distribution)}"
    )
    print(f"Primary Kaggle file: {MAIN_OUTPUT.name}")

    report_lines.extend([
        "Uniform three-seed ensemble",
        f"  Validation weighted F1: {ensemble_f1:.6f}",
        f"  Validation accuracy: {ensemble_accuracy:.6f}",
        f"  Test prediction distribution: "
        f"{dict(prediction_distribution)}",
        f"  Primary output: {MAIN_OUTPUT.name}",
    ])

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(f"Validation report: {REPORT_OUTPUT.name}")


if __name__ == "__main__":
    main()
