"""
Task 2 v3

Same two models as v2 - roberta-base-squad2 for the spans, roberta-base for the
type - with two changes to how the span model is trained.

First, it's trained three times with different seeds and the three sets of
start/end scores are averaged. 
Second, the training examples are sampled with weight 1/n_segments.

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

QA_MODEL = "deepset/roberta-base-squad2"
CLS_MODEL = "roberta-base"

MAX_LEN, STRIDE = 384, 128
EPOCHS, BATCH_SIZE, LR = 2, 16, 3e-5
CLS_MAX_LEN, CLS_EPOCHS, CLS_LR = 256, 3, 2e-5
N_SEEDS = 3           # span models to average; set to 1 to skip the ensemble
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

def train_loop(model, dataset, epochs, lr, batch_size, tag, weights=None, seed=SEED):
    torch.manual_seed(seed)
    if weights is not None:
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(dataset), replacement=True)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)

    model.train()
    step, running = 0, 0.0
    for _ in range(epochs):
        for batch in loader:
            batch = [t.to(DEVICE) for t in batch]
            optimizer.zero_grad(set_to_none=True)
            with amp():
                if len(batch) == 4:
                    out = model(input_ids=batch[0], attention_mask=batch[1],
                                start_positions=batch[2], end_positions=batch[3])
                else:
                    out = model(input_ids=batch[0], attention_mask=batch[1],
                                labels=batch[2])
            scaler.scale(out.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += out.loss.item()
            step += 1
            if step % 200 == 0:
                print(f"[{tag}] {step}/{total}  loss {running / 200:.4f}")
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


def decode(candidates, context, max_tok, k):
    kept = []
    for score, char_start, char_end, n_tok in candidates:
        if n_tok > max_tok:
            continue
        if any(char_start < e and s < char_end for s, e in kept):
            continue
        kept.append((char_start, char_end))
        if len(kept) >= k:
            break
    kept.sort()
    return " ".join(" ".join(context[s:e] for s, e in kept).split())


def decode_all(cands, contexts, types, config):
    return [decode(c, ctx, *config[t]) for c, ctx, t in zip(cands, contexts, types)]


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


# 999 = "don't cap it"
GRID = {
    "phrase":  [(m, 1) for m in (4, 6, 8, 12, 20, 40, 999)],
    "passage": [(m, 1) for m in (30, 45, 60, 80, 999)] + [(40, 2), (60, 2)],
    "multi":   [(m, k) for m in (15, 25, 40, 999) for k in (1, 2, 3, 4, 5)],
}
BIASES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def main():
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    data_dir = find_data()
    print("data:", data_dir, "| device:", DEVICE, "| seeds:", N_SEEDS)
    train_data = load_jsonl(f"{data_dir}/train.jsonl")
    val = load_jsonl(f"{data_dir}/val.jsonl")
    test = load_jsonl(f"{data_dir}/test.jsonl")

    pairs, pair_weights = build_pairs(train_data)
    raw = collections.Counter(r["tags"][0] for r, _, _ in pairs)
    weighted = collections.Counter()
    for (r, _, _), w in zip(pairs, pair_weights):
        weighted[r["tags"][0]] += w
    total_w = sum(weighted.values())
    print(f"{'type':<9}{'pairs':>8}{'raw %':>9}{'weighted %':>13}")
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
                   weights=qa_w, seed=SEED + s)
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
                config[t] = (999, 1)
                continue
            config[t] = max(GRID[t], key=lambda c: np.mean(
                [meteor(golds[i], decode(val_cands[i], val_ctx[i], *c)) for i in sub]))
        best_bias, best_score = 0.0, -1.0
        for bias in BIASES:
            predicted = types_from(val_tlogits, bias)
            score = np.mean([meteor(golds[i], decode(val_cands[i], val_ctx[i],
                                                     *config[predicted[i]]))
                             for i in idx])
            if score > best_score:
                best_bias, best_score = bias, score
        return config, best_bias

    def score_with(idx, config, bias):
        predicted = types_from(val_tlogits, bias)
        return float(np.mean([meteor(golds[i], decode(val_cands[i], val_ctx[i],
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
        print(f"fold {f + 1}: bias {bias}  held-out METEOR {s:.4f}")
    print(f"\ncross-validated METEOR: {np.mean(cv):.4f} "
          f"(sd across folds {np.std(cv):.4f})")

    best_config, best_bias = fit_params(list(range(len(val))))
    print("final config:", best_config, "| multi bias:", best_bias)

    # --- results
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

    flat = {t: (60, 1) for t in TYPES}
    argmax_types = types_from(val_tlogits, 0.0)
    tuned_types = types_from(val_tlogits, best_bias)
    print(f"\n{'system':<40}{'METEOR':<9}   per type")
    report([r.get("targetTitle", "") for r in val], "naive baseline (title)")
    report(decode_all(val_cands, val_ctx, gold_types, flat), "v3 spans, flat decode")
    report(decode_all(val_cands, val_ctx, argmax_types, best_config), "tuned, argmax types")
    report(decode_all(val_cands, val_ctx, tuned_types, best_config), "tuned + multi bias")
    report(decode_all(val_cands, val_ctx, gold_types, best_config), "GOLD types (ceiling)")
    print(f"\n{'cross-validated (honest)':<40}{np.mean(cv):.4f}")


    # --- submissions
    test_types = types_from(test_tlogits, best_bias)
    test_pred = decode_all(test_cands, test_ctx, test_types, best_config)
    with open("prediction_task2_v3.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "spoiler"])
        for record, prediction in zip(test, test_pred):
            w.writerow([record["id"], prediction or record.get("targetTitle", "")])
    with open("prediction_task1_v3.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "spoilerType"])
        for record, spoiler_type in zip(test, test_types):
            w.writerow([record["id"], spoiler_type])

    print("test type distribution:", collections.Counter(test_types))
    print("val gold distribution: ", collections.Counter(gold_types))


if __name__ == "__main__":
    main()
