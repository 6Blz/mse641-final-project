#!/usr/bin/env python3
"""
Task 1 v4 - deberta-v3-base, 3-fold CV over train+val, blended with a TF-IDF SVM.
Too many changes at once to attribute anything.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


# --- configuration ---
BASE_DIR = Path.cwd()
MODEL_NAME = "microsoft/deberta-v3-base"
CACHE_DIR = BASE_DIR / "huggingface_cache"

LABELS = ("phrase", "passage", "multi")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

SEED = 42
N_FOLDS = 3
EPOCHS = 3
PATIENCE = 1
MAX_LENGTH = 192
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
LABEL_SMOOTHING = 0.05

MAIN_OUTPUT = BASE_DIR / "prediction_task1_v4.csv"
DEBERTA_OUTPUT = BASE_DIR / "prediction_task1_v4_deberta_only.csv"
REPORT_OUTPUT = BASE_DIR / "task1_v4_validation_report.txt"


# --- general utilities ---
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        return " ".join(flatten(item) for item in value if item is not None)
    if isinstance(value, dict):
        return " ".join(
            f"{flatten(key)} {flatten(item)}"
            for key, item in value.items()
        )
    return str(value)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def identify_file(kind: str, required: bool = True) -> Path | None:
    candidates = sorted(BASE_DIR.glob("*.jsonl"))

    exact_names = {
        "train": {"train.jsonl", "training.jsonl"},
        "val": {"val.jsonl", "valid.jsonl", "validation.jsonl", "dev.jsonl"},
        "test": {"test.jsonl", "testing.jsonl"},
    }

    for path in candidates:
        if path.name.lower() in exact_names[kind]:
            return path

    tokens = {
        "train": ("train",),
        "val": ("val", "valid", "dev"),
        "test": ("test",),
    }[kind]

    matches = [
        path for path in candidates
        if any(token in path.name.lower() for token in tokens)
    ]

    if matches:
        return matches[0]

    if required:
        names = "\n".join(f"  - {path.name}" for path in candidates) or "  (none)"
        raise FileNotFoundError(
            f"Could not identify the {kind} JSONL file. Files found:\n{names}"
        )
    return None


def extract_label(record: dict[str, Any]) -> str:
    for key in ("spoilerType", "spoiler_type", "label", "target", "tags"):
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, (list, tuple, set)):
            matches = {
                str(item).strip().lower()
                for item in value
            } & set(LABELS)
            if len(matches) == 1:
                return next(iter(matches))
        else:
            label = str(value).strip().lower()
            if label in LABEL_TO_ID:
                return label
    raise KeyError(
        "Could not find phrase/passage/multi in a labelled record. "
        f"Available keys: {list(record.keys())}"
    )


def get_record_id(record: dict[str, Any]) -> str:
    for key in ("id", "postId", "uuid"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    raise KeyError(
        "No id, postId, or uuid field found in a test record. "
        f"Available keys: {list(record.keys())}"
    )


def build_transformer_pair(record: dict[str, Any]) -> tuple[str, str]:
    post = clean(flatten(record.get("postText") or record.get("post_text")))
    title = clean(flatten(record.get("targetTitle") or record.get("target_title")))
    return post, title


def build_linear_text(record: dict[str, Any]) -> str:
    post = clean(flatten(record.get("postText") or record.get("post_text")))
    title = clean(flatten(record.get("targetTitle") or record.get("target_title")))
    description = clean(
        flatten(record.get("targetDescription") or record.get("target_description"))
    )[:700]
    keywords = clean(
        flatten(record.get("targetKeywords") or record.get("target_keywords"))
    )[:300]
    platform = clean(flatten(record.get("postPlatform") or record.get("platform")))

    number_count = len(re.findall(r"\b\d+\b", post))
    question_mark = int("?" in post)
    list_cue = int(bool(re.search(
        r"\b(these|those|ways|reasons|things|signs|facts|steps|tips|habits|people)\b",
        post.lower(),
    )))

    return clean(
        f"POST {post} TITLE {title} DESCRIPTION {description} "
        f"KEYWORDS {keywords} PLATFORM {platform} "
        f"NUMBERCOUNT_{number_count} QUESTION_{question_mark} LISTCUE_{list_cue}"
    )


def softmax_np(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def apply_log_bias(probabilities: np.ndarray, biases: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-9, 1.0)) + biases.reshape(1, -1)
    return softmax_np(logits)


# --- transformer data and training ---
class PairDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        indices: np.ndarray,
        tokenizer,
        labels: np.ndarray | None = None,
    ) -> None:
        self.records = records
        self.indices = np.asarray(indices)
        self.tokenizer = tokenizer
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        source_index = int(self.indices[item])
        post, title = build_transformer_pair(self.records[source_index])
        encoded = self.tokenizer(
            post,
            title,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        if self.labels is not None:
            encoded["labels"] = int(self.labels[source_index])
        return encoded


def move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(LABELS)).astype(np.float64)
    weights = len(y) / (len(LABELS) * np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def predict_probabilities(
    model,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    all_probabilities = []
    all_labels = []

    for batch in loader:
        batch = move_batch(batch, device)
        labels = batch.pop("labels", None)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(**batch).logits

        all_probabilities.append(
            torch.softmax(logits.float(), dim=-1).cpu().numpy()
        )
        if labels is not None:
            all_labels.append(labels.cpu().numpy())

    probabilities = np.concatenate(all_probabilities, axis=0)
    labels_array = np.concatenate(all_labels) if all_labels else None
    return probabilities, labels_array


def train_transformer_fold(
    fold_number: int,
    records: list[dict[str, Any]],
    y: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    test_records: list[dict[str, Any]],
    tokenizer,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    use_amp = device.type == "cuda"
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if use_amp else None,
        return_tensors="pt",
    )

    train_dataset = PairDataset(records, train_indices, tokenizer, y)
    valid_dataset = PairDataset(records, valid_indices, tokenizer, y)
    test_dataset = PairDataset(
        test_records,
        np.arange(len(test_records)),
        tokenizer,
        labels=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=use_amp,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=use_amp,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=use_amp,
    )

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        cache_dir=str(CACHE_DIR),
    )
    config.hidden_dropout_prob = 0.15

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        cache_dir=str(CACHE_DIR),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / GRADIENT_ACCUMULATION
    )
    total_updates = updates_per_epoch * EPOCHS
    warmup_steps = int(total_updates * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    weights = class_weights(y[train_indices], device)
    loss_function = torch.nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=LABEL_SMOOTHING,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint = BASE_DIR / f"task1_v4_fold_{fold_number}_best.pt"
    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    print("\n" + "=" * 72)
    print(f"Transformer fold {fold_number}/{N_FOLDS}")
    print("=" * 72)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            labels_batch = batch.pop("labels")

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(**batch).logits
                loss = loss_function(logits, labels_batch)
                scaled_loss = loss / GRADIENT_ACCUMULATION

            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.item())

            should_update = (
                step % GRADIENT_ACCUMULATION == 0
                or step == len(train_loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 100 == 0 or step == len(train_loader):
                print(
                    f"Fold {fold_number} | Epoch {epoch}/{EPOCHS} | "
                    f"Batch {step}/{len(train_loader)} | "
                    f"Loss {running_loss / step:.4f}"
                )

        valid_probabilities, valid_labels = predict_probabilities(
            model, valid_loader, device, use_amp
        )
        valid_predictions = valid_probabilities.argmax(axis=1)
        macro_f1 = f1_score(valid_labels, valid_predictions, average="macro")

        print(
            f"Fold {fold_number} | Epoch {epoch} | "
            f"Validation macro F1: {macro_f1:.5f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint)
            print("Saved new best fold checkpoint.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print("Early stopping this fold.")
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    valid_probabilities, _ = predict_probabilities(
        model, valid_loader, device, use_amp
    )
    test_probabilities, _ = predict_probabilities(
        model, test_loader, device, use_amp
    )

    if checkpoint.exists():
        checkpoint.unlink()

    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return valid_probabilities, test_probabilities, best_macro_f1


# --- linear model and oof tuning ---
def build_tfidf() -> FeatureUnion:
    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.995,
                max_features=70000,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=90000,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
    ])


def linear_oof_and_test(
    texts: list[str],
    y: np.ndarray,
    test_texts: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    oof_scores = np.zeros((len(texts), len(LABELS)), dtype=np.float64)
    test_scores = np.zeros((len(test_texts), len(LABELS)), dtype=np.float64)

    for fold_number, (train_indices, valid_indices) in enumerate(splits, start=1):
        print(f"\nTraining TF-IDF/SVM fold {fold_number}/{N_FOLDS}...")
        vectorizer = build_tfidf()
        train_matrix = vectorizer.fit_transform(
            [texts[index] for index in train_indices]
        )
        valid_matrix = vectorizer.transform(
            [texts[index] for index in valid_indices]
        )
        test_matrix = vectorizer.transform(test_texts)

        classifier = LinearSVC(
            C=2.0,
            class_weight="balanced",
            random_state=SEED + fold_number,
        )
        classifier.fit(train_matrix, y[train_indices])

        oof_scores[valid_indices] = classifier.decision_function(valid_matrix)
        test_scores += classifier.decision_function(test_matrix) / N_FOLDS

        del vectorizer, classifier, train_matrix, valid_matrix, test_matrix
        gc.collect()

    return oof_scores, test_scores


def tune_blend_and_bias(
    transformer_oof: np.ndarray,
    svm_oof_scores: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, np.ndarray, float]:
    best_macro_f1 = -1.0
    best_weight = 1.0
    best_temperature = 1.0
    best_probabilities = transformer_oof

    for temperature in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        svm_probabilities = softmax_np(svm_oof_scores / temperature)
        for transformer_weight in np.arange(0.50, 1.001, 0.05):
            blended = (
                transformer_weight * transformer_oof
                + (1.0 - transformer_weight) * svm_probabilities
            )
            score = f1_score(y, blended.argmax(axis=1), average="macro")
            if score > best_macro_f1:
                best_macro_f1 = score
                best_weight = float(transformer_weight)
                best_temperature = float(temperature)
                best_probabilities = blended

    best_biases = np.zeros(len(LABELS), dtype=np.float64)
    for phrase_bias in np.arange(-0.60, 0.601, 0.05):
        for multi_bias in np.arange(-0.60, 0.601, 0.05):
            biases = np.array([phrase_bias, 0.0, multi_bias])
            adjusted = apply_log_bias(best_probabilities, biases)
            score = f1_score(y, adjusted.argmax(axis=1), average="macro")
            if score > best_macro_f1:
                best_macro_f1 = score
                best_biases = biases

    return best_weight, best_temperature, best_biases, best_macro_f1


# --- output ---
def write_submission(
    path: Path,
    test_records: list[dict[str, Any]],
    predictions: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "spoilerType"])
        for record, prediction_id in zip(test_records, predictions):
            writer.writerow([
                get_record_id(record),
                ID_TO_LABEL[int(prediction_id)],
            ])


def main() -> None:
    set_seed(SEED)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. In Colab select Runtime > Change runtime type > "
            "T4 GPU, then run the notebook again."
        )

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    train_path = identify_file("train", required=True)
    val_path = identify_file("val", required=False)
    test_path = identify_file("test", required=True)

    train_records = read_jsonl(train_path)
    val_records = read_jsonl(val_path) if val_path else []
    test_records = read_jsonl(test_path)
    labelled_records = train_records + val_records

    y = np.array([
        LABEL_TO_ID[extract_label(record)]
        for record in labelled_records
    ], dtype=np.int64)

    print("\nFiles:")
    print("  train:", train_path.name, len(train_records))
    print("  val:", val_path.name if val_path else "not supplied", len(val_records))
    print("  test:", test_path.name, len(test_records))
    print("  total labelled:", len(labelled_records))
    print("  label distribution:", Counter(ID_TO_LABEL[int(v)] for v in y))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=str(CACHE_DIR),
        use_fast=False,
    )

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )
    splits = list(splitter.split(np.zeros(len(y)), y))

    transformer_oof = np.zeros((len(labelled_records), len(LABELS)))
    transformer_test = np.zeros((len(test_records), len(LABELS)))
    fold_scores = []

    for fold_number, (train_indices, valid_indices) in enumerate(splits, start=1):
        fold_valid_probabilities, fold_test_probabilities, fold_score = (
            train_transformer_fold(
                fold_number=fold_number,
                records=labelled_records,
                y=y,
                train_indices=train_indices,
                valid_indices=valid_indices,
                test_records=test_records,
                tokenizer=tokenizer,
                device=device,
            )
        )
        transformer_oof[valid_indices] = fold_valid_probabilities
        transformer_test += fold_test_probabilities / N_FOLDS
        fold_scores.append(fold_score)

    transformer_oof_f1 = f1_score(
        y, transformer_oof.argmax(axis=1), average="macro"
    )

    linear_texts = [build_linear_text(record) for record in labelled_records]
    linear_test_texts = [build_linear_text(record) for record in test_records]
    svm_oof_scores, svm_test_scores = linear_oof_and_test(
        linear_texts, y, linear_test_texts, splits
    )

    svm_oof_probabilities = softmax_np(svm_oof_scores)
    svm_oof_f1 = f1_score(
        y, svm_oof_probabilities.argmax(axis=1), average="macro"
    )

    blend_weight, svm_temperature, biases, final_oof_f1 = tune_blend_and_bias(
        transformer_oof, svm_oof_scores, y
    )

    svm_test_probabilities = softmax_np(svm_test_scores / svm_temperature)
    blended_test = (
        blend_weight * transformer_test
        + (1.0 - blend_weight) * svm_test_probabilities
    )
    final_test_probabilities = apply_log_bias(blended_test, biases)

    final_oof_blended = (
        blend_weight * transformer_oof
        + (1.0 - blend_weight)
        * softmax_np(svm_oof_scores / svm_temperature)
    )
    final_oof_probabilities = apply_log_bias(final_oof_blended, biases)
    final_oof_predictions = final_oof_probabilities.argmax(axis=1)

    deberta_predictions = transformer_test.argmax(axis=1)
    final_predictions = final_test_probabilities.argmax(axis=1)

    write_submission(DEBERTA_OUTPUT, test_records, deberta_predictions)
    write_submission(MAIN_OUTPUT, test_records, final_predictions)

    report = []
    report.append("MSE 641 Task 1 — Version 4 validation summary")
    report.append("=" * 60)
    report.append(f"Model: {MODEL_NAME}")
    report.append(f"Labelled records used: {len(labelled_records)}")
    report.append(f"Folds: {N_FOLDS}")
    report.append(f"Fold best macro F1: {fold_scores}")
    report.append(f"DeBERTa OOF macro F1: {transformer_oof_f1:.5f}")
    report.append(f"TF-IDF/SVM OOF macro F1: {svm_oof_f1:.5f}")
    report.append(f"Best transformer blend weight: {blend_weight:.2f}")
    report.append(f"Best SVM temperature: {svm_temperature:.2f}")
    report.append(f"OOF class biases [phrase, passage, multi]: {biases.tolist()}")
    report.append(f"Final ensemble OOF macro F1: {final_oof_f1:.5f}")
    report.append("")
    report.append("Classification report:")
    report.append(classification_report(
        y,
        final_oof_predictions,
        labels=[0, 1, 2],
        target_names=list(LABELS),
        digits=4,
        zero_division=0,
    ))
    report.append("Confusion matrix (rows=gold, columns=predicted):")
    report.append(str(confusion_matrix(y, final_oof_predictions, labels=[0, 1, 2])))
    report.append("")
    report.append("Test prediction distribution:")
    report.append(str(Counter(ID_TO_LABEL[int(v)] for v in final_predictions)))

    REPORT_OUTPUT.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 72)
    print("VERSION 4 FINISHED")
    print("=" * 72)
    print(f"DeBERTa OOF macro F1: {transformer_oof_f1:.5f}")
    print(f"TF-IDF/SVM OOF macro F1: {svm_oof_f1:.5f}")
    print(f"Final ensemble OOF macro F1: {final_oof_f1:.5f}")
    print(f"Transformer blend weight: {blend_weight:.2f}")
    print(f"SVM temperature: {svm_temperature:.2f}")
    print(f"Class biases: {biases.tolist()}")
    print("\nCreated:")
    print(" ", MAIN_OUTPUT.name, "<- submit this one first")
    print(" ", DEBERTA_OUTPUT.name, "<- useful ablation/fallback")
    print(" ", REPORT_OUTPUT.name)


if __name__ == "__main__":
    main()
