#!/usr/bin/env python3
"""
Task 1 v5 - roberta-base, trained on train + val instead of train only.
"""

from __future__ import annotations

import csv
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

MAX_LENGTH = 256
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
SEED = 42

LABELS = ["phrase", "passage", "multi"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

OUTPUT_PATH = BASE_DIR / "prediction_task1_v5.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_file(kind: str) -> Path:
    candidates = list(BASE_DIR.glob("*.jsonl"))

    if kind == "train":
        preferred = ["train.jsonl"]
    elif kind == "val":
        preferred = [
            "val.jsonl",
            "validation.jsonl",
            "dev.jsonl",
        ]
    elif kind == "test":
        preferred = ["test.jsonl"]
    else:
        raise ValueError(f"Unknown file kind: {kind}")

    by_lower_name = {path.name.lower(): path for path in candidates}

    for name in preferred:
        if name in by_lower_name:
            return by_lower_name[name]

    # Fallback for filenames such as task1_train.jsonl.
    keywords = {
        "train": ("train",),
        "val": ("val", "validation", "dev"),
        "test": ("test",),
    }[kind]

    matches = [
        path
        for path in candidates
        if any(keyword in path.name.lower() for keyword in keywords)
    ]

    if kind == "val":
        matches = [
            path for path in matches
            if "train" not in path.name.lower()
            and "test" not in path.name.lower()
        ]

    if len(matches) == 1:
        return matches[0]

    listed = "\n".join(f"  - {path.name}" for path in candidates)
    raise FileNotFoundError(
        f"Could not identify the {kind} JSONL file.\n"
        f"Files found:\n{listed}"
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
                    f"Expected a JSON object in {path.name}, "
                    f"line {line_number}."
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
    tags = record.get("tags")

    if isinstance(tags, list) and tags:
        label = str(tags[0]).strip().lower()
    elif isinstance(tags, str):
        label = tags.strip().lower()
    else:
        label = str(
            record.get("spoilerType")
            or record.get("label")
            or ""
        ).strip().lower()

    if label not in LABEL_TO_ID:
        raise ValueError(
            f"Unknown or missing label {label!r}. "
            f"Expected one of {LABELS}."
        )

    return LABEL_TO_ID[label]


def record_id(record: dict[str, Any]) -> str:
    for key in ("id", "postId", "uuid"):
        value = record.get(key)

        if value not in (None, ""):
            return str(value)

    raise KeyError(
        "No id, postId, or uuid field found in a test record."
    )


def make_dataset(records, tokenizer, labelled: bool) -> TensorDataset:
    encoded = tokenizer(
        [post_text(record) for record in records],
        [title_text(record) for record in records],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )

    tensors = [
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


@torch.no_grad()
def predict(model, dataset, device, use_fp16: bool) -> list[str]:
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    prediction_ids: list[int] = []

    for input_ids, attention_mask in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_fp16,
        ):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

        prediction_ids.extend(
            logits.float().argmax(dim=-1).cpu().tolist()
        )

    return [ID_TO_LABEL[index] for index in prediction_ids]


def main() -> None:
    set_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_fp16 = device.type == "cuda"

    train_path = find_file("train")
    val_path = find_file("val")
    test_path = find_file("test")

    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)
    test_records = load_jsonl(test_path)

    # The only score-oriented change from Version 3:
    all_labelled_records = train_records + val_records

    label_counts = Counter(
        LABELS[extract_label(record)]
        for record in all_labelled_records
    )

    print("=" * 72)
    print("TASK 1 VERSION 5 — ROBERTA-BASE + ALL LABELED DATA")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Train records: {len(train_records)}")
    print(f"Validation records added to training: {len(val_records)}")
    print(f"Total labeled training records: {len(all_labelled_records)}")
    print(f"Test records: {len(test_records)}")
    print(f"Label counts: {dict(label_counts)}")
    print(
        f"Settings: max_length={MAX_LENGTH}, epochs={EPOCHS}, "
        f"batch_size={BATCH_SIZE}, lr={LEARNING_RATE}"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    training_dataset = make_dataset(
        all_labelled_records,
        tokenizer,
        labelled=True,
    )

    test_dataset = make_dataset(
        test_records,
        tokenizer,
        labelled=False,
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

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

    total_steps = len(training_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    model.train()

    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0

        for step, batch in enumerate(training_loader, start=1):
            input_ids, attention_mask, labels = [
                tensor.to(device, non_blocking=True)
                for tensor in batch
            ]

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

            if step % 50 == 0 or step == len(training_loader):
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Batch {step}/{len(training_loader)} | "
                    f"Average loss {running_loss / step:.4f}"
                )

    predictions = predict(
        model=model,
        dataset=test_dataset,
        device=device,
        use_fp16=use_fp16,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["id", "spoilerType"])

        for record, prediction in zip(test_records, predictions):
            writer.writerow([record_id(record), prediction])

    prediction_counts = Counter(predictions)

    print("\n" + "=" * 72)
    print("FINISHED")
    print("=" * 72)
    print(f"Output: {OUTPUT_PATH.name}")
    print(f"Prediction distribution: {dict(prediction_counts)}")
    print(
        "Kaggle description suggestion: "
        "V5 RoBERTa-base trained on train+validation"
    )


if __name__ == "__main__":
    main()
