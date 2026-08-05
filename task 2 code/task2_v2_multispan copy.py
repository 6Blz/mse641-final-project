"""
Task 2 v2

Two models now.

1. The same deepset/roberta-base-squad2 span model as v1, except it's trained
   on every gold segment instead of only the first. Multi spoilers have four
   segments at the median, so this gives them a lot more to learn from.
2. A new roberta-base classifier that reads the post and the title and says
   which kind of spoiler this is: phrase, passage, or multi.
"""

import collections
import contextlib
import csv
import json
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForQuestionAnswering,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

# ---------------------------------------------------------------- config

QA_MODEL = "deepset/roberta-base-squad2"
CLS_MODEL = "roberta-base"        # no SQuAD checkpoint needed for classification
DATA_DIR = "."

MAX_LEN, STRIDE = 384, 128
EPOCHS, BATCH_SIZE, LR = 2, 16, 3e-5
CLS_MAX_LEN, CLS_EPOCHS, CLS_LR = 256, 3, 2e-5
SEED = 42

N_BEST = 20         # start/end candidates per window
CAND_MAX_TOK = 80   # widest span we even keep in the pool
TYPES = ["phrase", "passage", "multi"]

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")
USE_FP16 = (DEVICE == "cuda")


def amp():
    # fp16 only makes sense on CUDA; on mps/cpu just do nothing
    if USE_FP16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


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
        if offset is None:                      # offsets are off by a char or two
            hit = context.find(segment)         # sometimes, so just search
            offset = hit if hit != -1 else None
        if offset is not None:
            found.append((segment, offset))
    return found


# ---------------------------------------------------------------- features

tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)


def encode(questions, contexts):
    return tokenizer(questions, contexts,
                     truncation="only_second", max_length=MAX_LEN, stride=STRIDE,
                     return_overflowing_tokens=True, return_offsets_mapping=True,
                     padding="max_length")


def make_train_features(pairs):
    enc = encode([get_question(r) for r, _, _ in pairs],
                 [build_context(r)[0] for r, _, _ in pairs])
    starts, ends = [], []
    for i, offsets in enumerate(enc["offset_mapping"]):
        _, answer, char_start = pairs[enc["overflow_to_sample_mapping"][i]]
        char_end = char_start + len(answer)
        seq_ids = enc.sequence_ids(i)
        ctx_lo = seq_ids.index(1)
        ctx_hi = len(seq_ids) - 1 - seq_ids[::-1].index(1)

        if not (offsets[ctx_lo][0] <= char_start and offsets[ctx_hi][1] >= char_end):
            starts.append(0)
            ends.append(0)          # answer not in this window
            continue

        lo = ctx_lo
        while lo <= ctx_hi and offsets[lo][0] <= char_start:
            lo += 1
        hi = ctx_hi
        while hi >= ctx_lo and offsets[hi][1] >= char_end:
            hi -= 1
        starts.append(lo - 1)
        ends.append(hi + 1)

    return TensorDataset(torch.tensor(enc["input_ids"]),
                         torch.tensor(enc["attention_mask"]),
                         torch.tensor(starts), torch.tensor(ends))


def make_cls_features(records, cls_tokenizer, labelled=True):
    enc = cls_tokenizer([get_question(r) for r in records],
                        [r.get("targetTitle", "") or "" for r in records],
                        truncation=True, max_length=CLS_MAX_LEN, padding="max_length")
    tensors = [torch.tensor(enc["input_ids"]), torch.tensor(enc["attention_mask"])]
    if labelled:
        tensors.append(torch.tensor([TYPES.index(r["tags"][0]) for r in records]))
    return TensorDataset(*tensors)


# ---------------------------------------------------------------- training

def train_loop(model, dataset, epochs, lr, batch_size, tag):
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
            if step % 100 == 0:
                print(f"[{tag}] step {step}/{total}  loss {running / 100:.4f}")
                running = 0.0
    return model


# ---------------------------------------------------------------- inference

@torch.no_grad()
def collect_candidates(qa_model, records, batch_size=32):
    contexts = [build_context(r)[0] for r in records]
    enc = encode([get_question(r) for r in records], contexts)

    offset_map = []
    for i, offsets in enumerate(enc["offset_mapping"]):
        seq_ids = enc.sequence_ids(i)
        offset_map.append([o if seq_ids[k] == 1 else None
                           for k, o in enumerate(offsets)])

    input_ids = torch.tensor(enc["input_ids"])
    masks = torch.tensor(enc["attention_mask"])
    qa_model.eval()
    starts, ends = [], []
    for i in range(0, len(input_ids), batch_size):
        with amp():
            out = qa_model(input_ids=input_ids[i:i + batch_size].to(DEVICE),
                           attention_mask=masks[i:i + batch_size].to(DEVICE))
        starts.append(out.start_logits.float().cpu().numpy())
        ends.append(out.end_logits.float().cpu().numpy())
    start_logits, end_logits = np.concatenate(starts), np.concatenate(ends)

    windows = collections.defaultdict(list)
    for i, sample in enumerate(enc["overflow_to_sample_mapping"]):
        windows[sample].append(i)

    per_example = []
    for idx in range(len(records)):
        found = []
        for w in windows[idx]:
            offsets = offset_map[w]
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
    return per_example, contexts


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

GRID = {
    "phrase":  [(m, 1) for m in (4, 6, 8, 12, 20)],
    "passage": [(m, 1) for m in (30, 45, 60, 80)] + [(40, 2)],
    "multi":   [(m, k) for m in (15, 25, 40) for k in (2, 3, 4, 5)],
}


def main():
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    train_data = load_jsonl(f"{DATA_DIR}/train.jsonl")
    val = load_jsonl(f"{DATA_DIR}/val.jsonl")
    test = load_jsonl(f"{DATA_DIR}/test.jsonl")
    print(f"train {len(train_data)} | val {len(val)} | test {len(test)}")

    pairs = [(r, a, s) for r in train_data for a, s in get_answers(r)]
    print(f"{len(pairs)} (article, segment) pairs from {len(train_data)} articles")
    print("by type:", collections.Counter(r["tags"][0] for r, _, _ in pairs))

    # --- span model
    qa_ds = make_train_features(pairs)
    print(len(qa_ds), "training windows")
    qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL).to(DEVICE)
    train_loop(qa_model, qa_ds, EPOCHS, LR, BATCH_SIZE, "qa")

    # --- type classifier
    cls_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL)
    cls_model = AutoModelForSequenceClassification.from_pretrained(
        CLS_MODEL, num_labels=len(TYPES)).to(DEVICE)
    train_loop(cls_model, make_cls_features(train_data, cls_tokenizer),
               CLS_EPOCHS, CLS_LR, 16, "cls")

    @torch.no_grad()
    def predict_types(records):
        ds = make_cls_features(records, cls_tokenizer, labelled=False)
        cls_model.eval()
        out = []
        for ids, att in DataLoader(ds, batch_size=64):
            with amp():
                logits = cls_model(input_ids=ids.to(DEVICE),
                                   attention_mask=att.to(DEVICE)).logits
            out += logits.float().argmax(-1).cpu().tolist()
        return [TYPES[i] for i in out]

    val_types = predict_types(val)
    gold_types = [r["tags"][0] for r in val]
    print("type accuracy on val:",
          round(float(np.mean([p == g for p, g in zip(val_types, gold_types)])), 4))

    # --- tune the decode on val
    val_cands, val_ctx = collect_candidates(qa_model, val)
    golds = [gold_of(r) for r in val]

    best_config = {}
    for t in TYPES:
        idx = [i for i, r in enumerate(val) if r["tags"][0] == t]
        best, best_score = None, -1
        for max_tok, k in GRID[t]:
            score = np.mean([meteor(golds[i], decode(val_cands[i], val_ctx[i], max_tok, k))
                             for i in idx])
            if score > best_score:
                best, best_score = (max_tok, k), score
        best_config[t] = best
        print(f"{t:<9} best (max_tok, k) = {best}   METEOR {best_score:.4f}  n={len(idx)}")

    # --- results table
    def report(predictions, label):
        per_type, everything = collections.defaultdict(list), []
        for record, prediction in zip(val, predictions):
            s = meteor(gold_of(record), prediction)
            per_type[record["tags"][0]].append(s)
            everything.append(s)

        def mean(xs):
            return float(np.mean(xs)) if len(xs) else 0.0

        print(f"{label:<34}{mean(everything):.4f}   " +
              "   ".join(f"{t} {mean(per_type[t]):.4f}" for t in TYPES))

    flat = {t: (60, 1) for t in TYPES}   # what v1 effectively did
    print(f"\n{'system':<34}{'METEOR':<9}   per type")
    report([r.get("targetTitle", "") for r in val], "naive baseline (title)")
    report(decode_all(val_cands, val_ctx, gold_types, flat), "single span, flat cap")
    report(decode_all(val_cands, val_ctx, val_types, best_config), "predicted type + tuned")
    report(decode_all(val_cands, val_ctx, gold_types, best_config), "GOLD type (upper bound)")

    # --- submissions
    test_cands, test_ctx = collect_candidates(qa_model, test)
    test_types = predict_types(test)
    test_pred = decode_all(test_cands, test_ctx, test_types, best_config)

    with open("prediction_task2.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "spoiler"])
        for record, prediction in zip(test, test_pred):
            writer.writerow([record["id"], prediction or record.get("targetTitle", "")])

    # the classifier is a Task 1 submission for free
    with open("prediction_task1.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "spoilerType"])
        for record, spoiler_type in zip(test, test_types):
            writer.writerow([record["id"], spoiler_type])

    print("predicted type distribution:", collections.Counter(test_types))
    print("empty spoilers backfilled with title:", sum(1 for p in test_pred if not p))


if __name__ == "__main__":
    main()
