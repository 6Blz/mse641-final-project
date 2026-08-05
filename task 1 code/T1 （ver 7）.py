#!/usr/bin/env python3
"""
Task 1 v7 - five seeds, keep whichever subset scores best on val.
"""

from __future__ import annotations

import csv
import gc
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_NAME = "roberta-base"
LABELS = ["phrase", "passage", "multi"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

# Version 6 used the first three. Version 7 adds two diverse seeds.
SEEDS = [42, 123, 777, 2024, 31415]

MAX_LENGTH = 256
EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

SELECTED_OUTPUT = BASE_DIR / "prediction_task1_v7_selected_ensemble.csv"
ALL5_OUTPUT = BASE_DIR / "prediction_task1_v7_all5_uniform.csv"
REPORT_OUTPUT = BASE_DIR / "task1_v7_validation_report.txt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def encode_records(
    records: list[dict[str, Any]],
    tokenizer,
    labelled: bool,
) -> TensorDataset:
    encoded = tokenizer(
        [post_text(record) for record in records],
        [title_text(record) for record in records],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )

    tensors: list[torch.Tensor] = [
        encoded["input_ids"],
        encoded["attention_mask"],
    ]

    if labelled:
        tensors.append(
            torch.tensor(
                [extract_label(record) for record in records],
                dtype=torch.long,
            )
        )

    return TensorDataset(*tensors)


def make_train_loader(dataset: TensorDataset, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        num_workers=2,
    )


def make_eval_loader(dataset: TensorDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=2,
    )


def create_grad_scaler(use_fp16: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_fp16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_fp16)


@torch.no_grad()
def predict_probabilities(
    model,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
    labelled: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()

    probabilities: list[torch.Tensor] = []
    gold_labels: list[torch.Tensor] = []

    for batch in loader:
        input_ids = batch[0].to(device, non_blocking=True)
        attention_mask = batch[1].to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_fp16,
        ):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

        probabilities.append(
            torch.softmax(logits.float(), dim=-1).cpu()
        )

        if labelled:
            gold_labels.append(batch[2].cpu())

    probability_array = torch.cat(probabilities).numpy()

    if labelled:
        return probability_array, torch.cat(gold_labels).numpy()

    return probability_array, None


def classification_metrics(
    probabilities: np.ndarray,
    gold: np.ndarray,
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    accuracy = float(np.mean(predictions == gold))

    per_class: dict[str, dict[str, float | int]] = {}
    weighted_f1_numerator = 0.0

    for class_id, label in enumerate(LABELS):
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

        weighted_f1_numerator += f1 * support

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    weighted_f1 = weighted_f1_numerator / len(gold)

    return {
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "predictions": predictions,
    }


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
    train_dataset: TensorDataset,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    set_seed(seed)

    train_loader = make_train_loader(train_dataset, seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = create_grad_scaler(use_fp16)

    print("\n" + "=" * 72)
    print(f"TRAINING SEED {seed}")
    print("=" * 72)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch[0].to(device, non_blocking=True)
            attention_mask = batch[1].to(device, non_blocking=True)
            labels = batch[2].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_fp16,
            ):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = output.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())

            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"Seed {seed} | Epoch {epoch}/{EPOCHS} | "
                    f"Batch {step}/{len(train_loader)} | "
                    f"Average loss {running_loss / step:.4f}"
                )

    val_probabilities, gold = predict_probabilities(
        model=model,
        loader=val_loader,
        device=device,
        use_fp16=use_fp16,
        labelled=True,
    )

    test_probabilities, _ = predict_probabilities(
        model=model,
        loader=test_loader,
        device=device,
        use_fp16=use_fp16,
        labelled=False,
    )

    if gold is None:
        raise RuntimeError("Validation labels were not returned.")

    del model
    del optimizer
    del scheduler
    del scaler
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_probabilities, test_probabilities, gold


def choose_best_subset(
    val_probabilities_by_seed: dict[int, np.ndarray],
    gold: np.ndarray,
) -> tuple[tuple[int, ...], np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    best_subset: tuple[int, ...] | None = None
    best_probabilities: np.ndarray | None = None
    best_metrics: dict[str, Any] | None = None

    for subset_size in (3, 4, 5):
        for subset in itertools.combinations(SEEDS, subset_size):
            probabilities = np.mean(
                [val_probabilities_by_seed[seed] for seed in subset],
                axis=0,
            )
            metrics = classification_metrics(probabilities, gold)

            results.append({
                "subset": subset,
                "weighted_f1": metrics["weighted_f1"],
                "accuracy": metrics["accuracy"],
            })

            candidate_key = (
                metrics["weighted_f1"],
                metrics["accuracy"],
                subset_size,
            )

            if best_metrics is None:
                is_better = True
            else:
                best_key = (
                    best_metrics["weighted_f1"],
                    best_metrics["accuracy"],
                    len(best_subset),
                )
                is_better = candidate_key > best_key

            if is_better:
                best_subset = subset
                best_probabilities = probabilities
                best_metrics = metrics

    if (
        best_subset is None
        or best_probabilities is None
        or best_metrics is None
    ):
        raise RuntimeError("No valid ensemble subset was evaluated.")

    results.sort(
        key=lambda item: (
            item["weighted_f1"],
            item["accuracy"],
            len(item["subset"]),
        ),
        reverse=True,
    )

    return best_subset, best_probabilities, best_metrics, results


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No Colab GPU detected. Select Runtime → Change runtime type → T4 GPU."
        )

    device = torch.device("cuda")
    use_fp16 = True

    train_path = find_file("train")
    val_path = find_file("val")
    test_path = find_file("test")

    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)
    test_records = load_jsonl(test_path)

    print("=" * 72)
    print("TASK 1 VERSION 7 — VALIDATION-SELECTED ROBERTA ENSEMBLE")
    print("=" * 72)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Seeds: {SEEDS}")
    print(f"Train: {train_path.name} ({len(train_records)})")
    print(f"Validation: {val_path.name} ({len(val_records)})")
    print(f"Test: {test_path.name} ({len(test_records)})")
    print(
        f"Settings: max_length={MAX_LENGTH}, epochs={EPOCHS}, "
        f"batch_size={TRAIN_BATCH_SIZE}, lr={LEARNING_RATE}"
    )

    train_distribution = Counter(
        LABELS[extract_label(record)]
        for record in train_records
    )
    print(f"Train label distribution: {dict(train_distribution)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = encode_records(
        train_records,
        tokenizer,
        labelled=True,
    )
    val_dataset = encode_records(
        val_records,
        tokenizer,
        labelled=True,
    )
    test_dataset = encode_records(
        test_records,
        tokenizer,
        labelled=False,
    )

    val_loader = make_eval_loader(val_dataset)
    test_loader = make_eval_loader(test_dataset)

    val_probabilities_by_seed: dict[int, np.ndarray] = {}
    test_probabilities_by_seed: dict[int, np.ndarray] = {}
    gold_reference: np.ndarray | None = None

    report_lines = [
        "Task 1 Version 7 — Validation-selected RoBERTa seed ensemble",
        "",
        "All Version 3 settings are preserved.",
        "The only new direction is increasing seed diversity and selecting "
        "the best robust 3–5 seed subset using validation weighted F1.",
        "",
    ]

    for seed in SEEDS:
        val_probabilities, test_probabilities, gold = train_one_seed(
            seed=seed,
            train_dataset=train_dataset,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            use_fp16=use_fp16,
        )

        if gold_reference is None:
            gold_reference = gold
        elif not np.array_equal(gold_reference, gold):
            raise RuntimeError("Validation labels changed between seed runs.")

        val_probabilities_by_seed[seed] = val_probabilities
        test_probabilities_by_seed[seed] = test_probabilities

        metrics = classification_metrics(val_probabilities, gold)
        individual_output = BASE_DIR / f"prediction_task1_v7_seed{seed}.csv"

        write_submission(
            individual_output,
            test_records,
            test_probabilities,
        )

        print(
            f"\nSeed {seed} validation accuracy: "
            f"{metrics['accuracy']:.5f}"
        )
        print(
            f"Seed {seed} validation weighted F1: "
            f"{metrics['weighted_f1']:.5f}"
        )

        report_lines.extend([
            f"Seed {seed}",
            f"  Validation accuracy: {metrics['accuracy']:.6f}",
            f"  Validation weighted F1: {metrics['weighted_f1']:.6f}",
            f"  Output: {individual_output.name}",
            "",
        ])

    if gold_reference is None:
        raise RuntimeError("No models were trained.")

    (
        best_subset,
        best_val_probabilities,
        best_metrics,
        subset_results,
    ) = choose_best_subset(
        val_probabilities_by_seed,
        gold_reference,
    )

    selected_test_probabilities = np.mean(
        [test_probabilities_by_seed[seed] for seed in best_subset],
        axis=0,
    )

    all5_val_probabilities = np.mean(
        [val_probabilities_by_seed[seed] for seed in SEEDS],
        axis=0,
    )
    all5_test_probabilities = np.mean(
        [test_probabilities_by_seed[seed] for seed in SEEDS],
        axis=0,
    )
    all5_metrics = classification_metrics(
        all5_val_probabilities,
        gold_reference,
    )

    write_submission(
        SELECTED_OUTPUT,
        test_records,
        selected_test_probabilities,
    )
    write_submission(
        ALL5_OUTPUT,
        test_records,
        all5_test_probabilities,
    )

    selected_ids = selected_test_probabilities.argmax(axis=1)
    selected_distribution = Counter(
        ID_TO_LABEL[int(index)]
        for index in selected_ids
    )

    print("\n" + "=" * 72)
    print("VERSION 7 ENSEMBLE SELECTION")
    print("=" * 72)
    print(f"Selected seeds: {list(best_subset)}")
    print(
        f"Selected validation accuracy: "
        f"{best_metrics['accuracy']:.5f}"
    )
    print(
        f"Selected validation weighted F1: "
        f"{best_metrics['weighted_f1']:.5f}"
    )
    print(
        f"All-five validation weighted F1: "
        f"{all5_metrics['weighted_f1']:.5f}"
    )
    print(f"Prediction distribution: {dict(selected_distribution)}")
    print(f"Primary Kaggle file: {SELECTED_OUTPUT.name}")
    print(f"Fallback all-five file: {ALL5_OUTPUT.name}")

    report_lines.extend([
        "Top validation ensemble subsets:",
    ])

    for rank, item in enumerate(subset_results[:10], start=1):
        report_lines.append(
            f"  {rank}. seeds={list(item['subset'])}, "
            f"weighted_f1={item['weighted_f1']:.6f}, "
            f"accuracy={item['accuracy']:.6f}"
        )

    report_lines.extend([
        "",
        f"Selected subset: {list(best_subset)}",
        f"Selected validation weighted F1: "
        f"{best_metrics['weighted_f1']:.6f}",
        f"Selected validation accuracy: {best_metrics['accuracy']:.6f}",
        f"All-five validation weighted F1: "
        f"{all5_metrics['weighted_f1']:.6f}",
        f"Primary output: {SELECTED_OUTPUT.name}",
        f"Fallback output: {ALL5_OUTPUT.name}",
        "",
        "Submit the selected ensemble first. If its leaderboard score is "
        "unexpectedly lower, submit the all-five uniform ensemble as the "
        "controlled fallback.",
    ])

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(f"Validation report: {REPORT_OUTPUT.name}")


if __name__ == "__main__":
    main()
