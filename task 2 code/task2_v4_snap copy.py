"""
Task 2 v4 - the best one. Kaggle public 0.46653.

Model: deepset/deberta-v3-large-squad2 for the spans, plus the same roberta-base
type classifier as v2. Set USE_DEBERTA = False to fall back to three
roberta-base-squad2 models averaged, which is faster and lighter on VRAM.
"""

import collections
import contextlib
import csv
import gc
import glob
import json
import os
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import (AutoModelForQuestionAnswering,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

# ---------------------------------------------------------------- config
USE_DEBERTA = True

if USE_DEBERTA:
    QA_MODEL = "deepset/deberta-v3-large-squad2"
    BATCH_SIZE, GRAD_ACCUM, LR, N_SEEDS = 4, 4, 1e-5, 1
else:
    QA_MODEL = "deepset/roberta-base-squad2"
    BATCH_SIZE, GRAD_ACCUM, LR, N_SEEDS = 16, 1, 3e-5, 3

CLS_MODEL = "roberta-base"

MAX_LEN, STRIDE = 384, 128
EPOCHS = 2
CLS_MAX_LEN, CLS_EPOCHS, CLS_LR = 256, 3, 2e-5
SEED = 42

N_BEST, CAND_MAX_TOK = 20, 90
TYPES = ["phrase", "passage", "multi"]
N_FOLDS = 5

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")
USE_FP16 = (DEVICE == "cuda")


def amp():
    if USE_FP16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def find_data():
    for pattern in ("train.jsonl", "**/train.jsonl"):
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return os.path.dirname(os.path.abspath(hits[0]))
    raise FileNotFoundError("train.jsonl not found")


# ---------------------------------------------------------------- data

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_context(record):
    title = record.get("targetTitle", "") or ""
    paras = record.get("targetParagraphs", []) or []
    pieces, starts = [title], {-1: 0}
    cursor = len(title) + 1
    for i, para in enumerate(paras):
        starts[i] = cursor
        pieces.append(para)
        cursor += len(para) + 1
    return "\n".join(pieces), starts


def sentence_spans(record):
    parts = ([record.get("targetTitle", "") or ""]
             + list(record.get("targetParagraphs", []) or []))
    spans, offset = [], 0
    for part in parts:
        for m in re.finditer(r"[^.!?]+[.!?]*", part):
            a, b = m.span()
            if part[a:b].strip():
                spans.append((offset + a, offset + b))
        offset += len(part) + 1
    return spans


def get_question(record):
    post = record.get("postText", "")
    return " ".join(post) if isinstance(post, list) else str(post)


def get_answers(record):
    spoiler = record.get("spoiler")
    if not spoiler:
        return []
    segments = spoiler if isinstance(spoiler, list) else [spoiler]
    context, starts = build_context(record)
    positions = record.get("spoilerPositions") or []
    found = []
    for i, segment in enumerate(segments):
        offset = None
        if i < len(positions) and len(positions[i]) == 2:
            (p0, c0), (p1, c1) = positions[i]
            if p0 == p1 and p0 in starts:
                begin = starts[p0] + c0
                if context[begin:begin + (c1 - c0)].strip() == segment.strip():
                    offset = begin
        if offset is None:
            hit = context.find(segment)
            offset = hit if hit != -1 else None
        if offset is not None:
            found.append((segment, offset))
    return found


def build_pairs(train_data):
    pairs, weights = [], []
    for record in train_data:
        answers = get_answers(record)
        for answer, offset in answers:
            pairs.append((record, answer, offset))
            weights.append(1.0 / len(answers))
    return pairs, weights


# ---------------------------------------------------------------- features

tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)


def encode(questions, contexts):
    return tokenizer(questions, contexts,
                     truncation="only_second", max_length=MAX_LEN, stride=STRIDE,
                     return_overflowing_tokens=True, return_offsets_mapping=True,
                     padding="max_length")


def make_train_features(pairs, weights):
    enc = encode([get_question(r) for r, _, _ in pairs],
                 [build_context(r)[0] for r, _, _ in pairs])
    starts, ends, window_weights = [], [], []
    for i, offsets in enumerate(enc["offset_mapping"]):
        sample = enc["overflow_to_sample_mapping"][i]
        _, answer, char_start = pairs[sample]
        char_end = char_start + len(answer)
        window_weights.append(weights[sample])

        seq_ids = enc.sequence_ids(i)
        ctx_lo = seq_ids.index(1)
        ctx_hi = len(seq_ids) - 1 - seq_ids[::-1].index(1)
        if not (offsets[ctx_lo][0] <= char_start and offsets[ctx_hi][1] >= char_end):
            starts.append(0)
            ends.append(0)
            continue
        lo = ctx_lo
        while lo <= ctx_hi and offsets[lo][0] <= char_start:
            lo += 1
        hi = ctx_hi
        while hi >= ctx_lo and offsets[hi][1] >= char_end:
            hi -= 1
        starts.append(lo - 1)
        ends.append(hi + 1)

    dataset = TensorDataset(torch.tensor(enc["input_ids"]),
                            torch.tensor(enc["attention_mask"]),
                            torch.tensor(starts), torch.tensor(ends))
    return dataset, np.array(window_weights, dtype=np.float64)


def make_cls_features(records, cls_tokenizer, labelled=True):
    enc = cls_tokenizer([get_question(r) for r in records],
                        [r.get("targetTitle", "") or "" for r in records],
                        truncation=True, max_length=CLS_MAX_LEN, padding="max_length")
    tensors = [torch.tensor(enc["input_ids"]), torch.tensor(enc["attention_mask"])]
    if labelled:
        tensors.append(torch.tensor([TYPES.index(r["tags"][0]) for r in records]))
    return TensorDataset(*tensors)


# ---------------------------------------------------------------- training

def train_loop(model, dataset, epochs, lr, batch_size, tag,
               weights=None, seed=SEED, accum=1):
    torch.manual_seed(seed)
    if weights is not None:
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(dataset), replacement=True)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = (len(loader) // accum) * epochs      # optimiser steps, not batches
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)

    model.train()
    step, running = 0, 0.0
    optimizer.zero_grad(set_to_none=True)
    for _ in range(epochs):
        for i, batch in enumerate(loader):
            batch = [t.to(DEVICE) for t in batch]
            with amp():
                if len(batch) == 4:
                    out = model(input_ids=batch[0], attention_mask=batch[1],
                                start_positions=batch[2], end_positions=batch[3])
                else:
                    out = model(input_ids=batch[0], attention_mask=batch[1],
                                labels=batch[2])
            # divide by accum
            scaler.scale(out.loss / accum).backward()
            running += out.loss.item()

            if (i + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 100 == 0:
                    print(f"[{tag}] {step}/{total}  loss {running / (100 * accum):.4f}")
                    running = 0.0
    return model


# ---------------------------------------------------------------- inference

def build_eval_features(records):
    contexts = [build_context(r)[0] for r in records]
    enc = encode([get_question(r) for r in records], contexts)
    offset_map = []
    for i, offsets in enumerate(enc["offset_mapping"]):
        seq_ids = enc.sequence_ids(i)
        offset_map.append([o if seq_ids[k] == 1 else None
                           for k, o in enumerate(offsets)])
    windows = collections.defaultdict(list)
    for i, sample in enumerate(enc["overflow_to_sample_mapping"]):
        windows[sample].append(i)
    return {"contexts": contexts, "offsets": offset_map, "windows": windows,
            "input_ids": torch.tensor(enc["input_ids"]),
            "mask": torch.tensor(enc["attention_mask"])}


@torch.no_grad()
def model_logits(model, feats, batch_size=32):
    model.eval()
    starts, ends = [], []
    ids, mask = feats["input_ids"], feats["mask"]
    for i in range(0, len(ids), batch_size):
        with amp():
            out = model(input_ids=ids[i:i + batch_size].to(DEVICE),
                        attention_mask=mask[i:i + batch_size].to(DEVICE))
        starts.append(out.start_logits.float().cpu().numpy())
        ends.append(out.end_logits.float().cpu().numpy())
    return np.concatenate(starts), np.concatenate(ends)


def build_candidates(feats, logits):
    start_logits, end_logits = logits
    per_example = []
    for idx in range(len(feats["contexts"])):
        found = []
        for w in feats["windows"][idx]:
            offsets = feats["offsets"][w]
            s_log, e_log = start_logits[w], end_logits[w]
            for s in np.argsort(s_log)[-N_BEST:][::-1]:
                for e in np.argsort(e_log)[-N_BEST:][::-1]:
                    if s >= len(offsets) or e >= len(offsets):
                        continue
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > CAND_MAX_TOK:
                        continue
                    found.append((float(s_log[s] + e_log[e]),
                                  offsets[s][0], offsets[e][1], int(e - s + 1)))
        found.sort(key=lambda c: -c[0])
        per_example.append(found[:200])
    return per_example


# ---------------------------------------------------------------- decoding

def widen(spans, a, b):
    touching = [(s, e) for s, e in spans if s < b and a < e]
    if not touching:
        return a, b
    return min(s for s, _ in touching), max(e for _, e in touching)


def decode(candidates, context, spans, max_tok, k, snap):
    kept = []
    for score, char_start, char_end, n_tok in candidates:
        if n_tok > max_tok:
            continue
        a, b = widen(spans, char_start, char_end) if snap else (char_start, char_end)
        if any(a < e and s < b for s, e in kept):
            continue
        kept.append((a, b))
        if len(kept) >= k:
            break

    kept.sort()

    merged = []
    for a, b in kept:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return " ".join(" ".join(context[a:b] for a, b in merged).split())


def decode_all(cands, contexts, spans, types, config):
    return [decode(c, ctx, sp, *config[t])
            for c, ctx, sp, t in zip(cands, contexts, spans, types)]


def types_from(logits, multi_bias=0.0):
    adjusted = logits.copy()
    adjusted[:, TYPES.index("multi")] += multi_bias
    return [TYPES[i] for i in adjusted.argmax(1)]


# ---------------------------------------------------------------- eval

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tok(s):
    return TOKEN_RE.findall(s.lower())


def meteor(reference, hypothesis):
    from nltk.translate.meteor_score import meteor_score
    r, h = tok(reference), tok(hypothesis)
    return meteor_score([r], h) if r and h else 0.0


def gold_of(record):
    s = record["spoiler"]
    return " ".join(s) if isinstance(s, list) else s


# (max_tok, k, snap). 999 = no cap
GRID = {
    "phrase":  [(m, 1, 0) for m in (4, 6, 8, 12, 20, 40, 999)],
    "passage": [(m, 1, s) for m in (30, 45, 60, 80, 999) for s in (0, 1)]
               + [(60, 2, 0), (60, 2, 1)],
    "multi":   [(m, k, s) for m in (15, 25, 40, 999)
                for k in (1, 2, 3, 4, 5) for s in (0, 1)],
}
BIASES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def main():
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    data_dir = find_data()
    print("data:", data_dir)
    print("device:", DEVICE, "| model:", QA_MODEL)
    print("seeds:", N_SEEDS, "| batch:", BATCH_SIZE, "x accum", GRAD_ACCUM,
          "=", BATCH_SIZE * GRAD_ACCUM)

    train_data = load_jsonl(f"{data_dir}/train.jsonl")
    val = load_jsonl(f"{data_dir}/val.jsonl")
    test = load_jsonl(f"{data_dir}/test.jsonl")

    pairs, pair_weights = build_pairs(train_data)
    raw = collections.Counter(r["tags"][0] for r, _, _ in pairs)
    weighted = collections.Counter()
    for (r, _, _), w in zip(pairs, pair_weights):
        weighted[r["tags"][0]] += w
    total_w = sum(weighted.values())
    print(f"\n{'type':<9}{'pairs':>8}{'raw %':>9}{'weighted %':>13}")
    for t in TYPES:
        print(f"{t:<9}{raw[t]:>8}{raw[t] / len(pairs):>8.1%}"
              f"{weighted[t] / total_w:>12.1%}")

    qa_ds, qa_w = make_train_features(pairs, pair_weights)
    print(len(qa_ds), "training windows")

    # --- span models
    val_f, test_f = build_eval_features(val), build_eval_features(test)
    acc = {"val": [0.0, 0.0], "test": [0.0, 0.0]}
    for s in range(N_SEEDS):
        print(f"\n=== span model seed {s + 1}/{N_SEEDS} ===")
        model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL).to(DEVICE)
        train_loop(model, qa_ds, EPOCHS, LR, BATCH_SIZE, f"qa{s}",
                   weights=qa_w, seed=SEED + s, accum=GRAD_ACCUM)
        for name, feats in (("val", val_f), ("test", test_f)):
            st, en = model_logits(model, feats)
            acc[name][0] = acc[name][0] + st
            acc[name][1] = acc[name][1] + en
        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    val_cands = build_candidates(val_f, (acc["val"][0] / N_SEEDS, acc["val"][1] / N_SEEDS))
    test_cands = build_candidates(test_f, (acc["test"][0] / N_SEEDS, acc["test"][1] / N_SEEDS))
    val_ctx, test_ctx = val_f["contexts"], test_f["contexts"]
    val_spans = [sentence_spans(r) for r in val]
    test_spans = [sentence_spans(r) for r in test]
    print("mean sentences per article:",
          round(float(np.mean([len(s) for s in val_spans])), 1))

    # --- type classifier
    cls_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL)
    cls_model = AutoModelForSequenceClassification.from_pretrained(
        CLS_MODEL, num_labels=len(TYPES)).to(DEVICE)
    train_loop(cls_model, make_cls_features(train_data, cls_tokenizer),
               CLS_EPOCHS, CLS_LR, 16, "cls")

    @torch.no_grad()
    def type_logits(records):
        cls_model.eval()
        out = []
        loader = DataLoader(make_cls_features(records, cls_tokenizer, labelled=False),
                            batch_size=64)
        for ids, att in loader:
            with amp():
                out.append(cls_model(input_ids=ids.to(DEVICE),
                                     attention_mask=att.to(DEVICE)
                                     ).logits.float().cpu().numpy())
        return np.concatenate(out)

    val_tlogits, test_tlogits = type_logits(val), type_logits(test)
    gold_types = [r["tags"][0] for r in val]
    print("type accuracy (bias 0):",
          round(float(np.mean([p == g for p, g
                               in zip(types_from(val_tlogits), gold_types)])), 4))

    golds = [gold_of(r) for r in val]

    def fit_params(idx):
        config = {}
        for t in TYPES:
            sub = [i for i in idx if gold_types[i] == t]
            if not sub:
                config[t] = (999, 1, 0)
                continue
            config[t] = max(GRID[t], key=lambda c: np.mean(
                [meteor(golds[i], decode(val_cands[i], val_ctx[i], val_spans[i], *c))
                 for i in sub]))
        best_bias, best_score = 0.0, -1.0
        for bias in BIASES:
            predicted = types_from(val_tlogits, bias)
            score = np.mean([meteor(golds[i],
                                    decode(val_cands[i], val_ctx[i], val_spans[i],
                                           *config[predicted[i]]))
                             for i in idx])
            if score > best_score:
                best_bias, best_score = bias, score
        return config, best_bias

    def score_with(idx, config, bias):
        predicted = types_from(val_tlogits, bias)
        return float(np.mean([meteor(golds[i],
                                     decode(val_cands[i], val_ctx[i], val_spans[i],
                                            *config[predicted[i]]))
                              for i in idx]))

    rng = np.random.RandomState(SEED)
    folds = np.array_split(rng.permutation(len(val)), N_FOLDS)
    cv = []
    for f, held in enumerate(folds):
        fit_idx = [i for j, fold in enumerate(folds) if j != f for i in fold]
        cfg, bias = fit_params(fit_idx)
        s = score_with(list(held), cfg, bias)
        cv.append(s)
        print(f"fold {f + 1}: bias {bias}  held-out METEOR {s:.4f}  cfg {cfg}")
    print(f"\ncross-validated METEOR: {np.mean(cv):.4f} "
          f"(sd across folds {np.std(cv):.4f})")

    best_config, best_bias = fit_params(list(range(len(val))))
    print("final config:", best_config, "| multi bias:", best_bias)

    # --- results.
    def report(predictions, label):
        per_type, everything = collections.defaultdict(list), []
        for record, prediction in zip(val, predictions):
            s = meteor(gold_of(record), prediction)
            per_type[record["tags"][0]].append(s)
            everything.append(s)

        def mean(xs):
            return float(np.mean(xs)) if len(xs) else 0.0

        print(f"{label:<40}{mean(everything):.4f}   " +
              "   ".join(f"{t} {mean(per_type[t]):.4f}" for t in TYPES))

    flat = {t: (60, 1, 0) for t in TYPES}
    no_snap = {t: (m, k, 0) for t, (m, k, _) in best_config.items()}
    tuned_types = types_from(val_tlogits, best_bias)

    print(f"\n{'system':<40}{'METEOR':<9}   per type")
    report([r.get("targetTitle", "") for r in val], "naive baseline (title)")
    report(decode_all(val_cands, val_ctx, val_spans, gold_types, flat),
           "spans only, flat decode")
    report(decode_all(val_cands, val_ctx, val_spans, tuned_types, no_snap),
           "tuned decode, snapping OFF")
    report(decode_all(val_cands, val_ctx, val_spans, tuned_types, best_config),
           "tuned decode + snapping")
    report(decode_all(val_cands, val_ctx, val_spans, gold_types, best_config),
           "with GOLD types (ceiling)")
    print(f"\n{'cross-validated (honest)':<40}{np.mean(cv):.4f}")
    print("\nfor reference: v1 0.4565 val / 0.45647 LB, v2 0.5192 val / 0.45287 LB")

    # --- submissions
    test_types = types_from(test_tlogits, best_bias)
    test_pred = decode_all(test_cands, test_ctx, test_spans, test_types, best_config)

    with open("prediction_task2_v4.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "spoiler"])
        for record, prediction in zip(test, test_pred):
            w.writerow([record["id"], prediction or record.get("targetTitle", "")])

    with open("prediction_task1_v4.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "spoilerType"])
        for record, spoiler_type in zip(test, test_types):
            w.writerow([record["id"], spoiler_type])

    print("test type distribution:", collections.Counter(test_types))
    print("val gold distribution: ", collections.Counter(gold_types))
    print("empty, backfilled with title:", sum(1 for p in test_pred if not p))


if __name__ == "__main__":
    main()
