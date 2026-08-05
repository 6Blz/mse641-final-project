"""
Task 2 v1

Model: deepset/roberta-base-squad2. That's RoBERTa with a question-answering
head, already fine-tuned on SQuAD2, which we then fine-tune again on the
clickbait data.

It takes the post as the question and the article as the context, and predicts
two positions: where the spoiler starts and where it ends. The text in between
is the answer. It never writes anything new, it only points.

val METEOR 0.4565 / Kaggle public 0.45647
"""

import collections
import csv
import json
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                          get_linear_schedule_with_warmup)

# ---------------------------------------------------------------- config

MODEL_NAME = "deepset/roberta-base-squad2"   # already SQuAD2-tuned, saves us a lot
DATA_DIR = "."

MAX_LEN = 384      # tokens per window
STRIDE = 128       # overlap between windows
EPOCHS = 2
BATCH_SIZE = 16
LR = 3e-5
SEED = 42

N_BEST = 20        # how many start/end candidates to keep per window
MAX_ANS_LEN = 60   # longest answer we'll accept, in tokens

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- data

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_context(record):
    title = record.get("targetTitle", "") or ""
    paras = record.get("targetParagraphs", []) or []
    pieces, starts = [title], {-1: 0}
    cursor = len(title) + 1                     # +1 for the newline we join with
    for i, para in enumerate(paras):
        starts[i] = cursor
        pieces.append(para)
        cursor += len(para) + 1
    return "\n".join(pieces), starts


def get_question(record):
    post = record.get("postText", "")
    return " ".join(post) if isinstance(post, list) else str(post)


def get_answer(record):
    spoiler = record.get("spoiler")
    if not spoiler:
        return None, None
    target = spoiler[0] if isinstance(spoiler, list) else spoiler
    context, starts = build_context(record)

    positions = record.get("spoilerPositions") or []
    if positions and len(positions[0]) == 2:
        (p_start, c_start), (p_end, c_end) = positions[0]
        if p_start == p_end and p_start in starts:
            begin = starts[p_start] + c_start
            if context[begin:begin + (c_end - c_start)].strip() == target.strip():
                return target, begin

    found = context.find(target)
    return (target, found) if found != -1 else (None, None)


# ---------------------------------------------------------------- features

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def encode(questions, contexts):
    # Articles are ~360 words at the median and 1000+ at p90, way over 512
    # tokens, so one article turns into several overlapping windows.
    return tokenizer(
        questions, contexts,
        truncation="only_second",        # never cut the question
        max_length=MAX_LEN,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )


def make_train_features(records):
    keep = [(r, *get_answer(r)) for r in records]
    keep = [(r, a, s) for r, a, s in keep if a is not None]

    enc = encode([get_question(r) for r, _, _ in keep],
                 [build_context(r)[0] for r, _, _ in keep])

    starts, ends = [], []
    for i, offsets in enumerate(enc["offset_mapping"]):
        sample_idx = enc["overflow_to_sample_mapping"][i]
        _, answer, char_start = keep[sample_idx]
        char_end = char_start + len(answer)
        seq_ids = enc.sequence_ids(i)

        # where the context part of this window starts and ends
        ctx_lo = seq_ids.index(1)
        ctx_hi = len(seq_ids) - 1 - seq_ids[::-1].index(1)

        # answer isn't in this window -> point at <s>, i.e. "nothing here"
        if not (offsets[ctx_lo][0] <= char_start and offsets[ctx_hi][1] >= char_end):
            starts.append(0)
            ends.append(0)
            continue

        # walk in from both ends until we're inside the answer
        lo = ctx_lo
        while lo <= ctx_hi and offsets[lo][0] <= char_start:
            lo += 1
        hi = ctx_hi
        while hi >= ctx_lo and offsets[hi][1] >= char_end:
            hi -= 1
        starts.append(lo - 1)
        ends.append(hi + 1)

    return TensorDataset(
        torch.tensor(enc["input_ids"]),
        torch.tensor(enc["attention_mask"]),
        torch.tensor(starts),
        torch.tensor(ends),
    )


# ---------------------------------------------------------------- training

def train(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    model.train()
    step, running = 0, 0.0
    for epoch in range(EPOCHS):
        for batch in loader:
            input_ids, attention_mask, start_pos, end_pos = [t.to(DEVICE) for t in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, dtype=torch.float16,
                                enabled=(DEVICE == "cuda")):
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            start_positions=start_pos, end_positions=end_pos)
            scaler.scale(out.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += out.loss.item()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch + 1}  step {step}/{total_steps}  "
                      f"loss {running / 50:.4f}")
                running = 0.0
    return model


# ---------------------------------------------------------------- inference

@torch.no_grad()
def predict(model, records, batch_size=32):
    contexts = [build_context(r)[0] for r in records]
    questions = [get_question(r) for r in records]
    enc = encode(questions, contexts)

    # null out offsets for question/padding tokens so they can never be picked
    offset_map = []
    for i, offsets in enumerate(enc["offset_mapping"]):
        seq_ids = enc.sequence_ids(i)
        offset_map.append([o if seq_ids[k] == 1 else None
                           for k, o in enumerate(offsets)])

    input_ids = torch.tensor(enc["input_ids"])
    masks = torch.tensor(enc["attention_mask"])

    model.eval()
    all_start, all_end = [], []
    for i in range(0, len(input_ids), batch_size):
        ids = input_ids[i:i + batch_size].to(DEVICE)
        att = masks[i:i + batch_size].to(DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.float16,
                            enabled=(DEVICE == "cuda")):
            out = model(input_ids=ids, attention_mask=att)
        all_start.append(out.start_logits.float().cpu().numpy())
        all_end.append(out.end_logits.float().cpu().numpy())
    start_logits = np.concatenate(all_start)
    end_logits = np.concatenate(all_end)

    windows = collections.defaultdict(list)
    for i, sample_idx in enumerate(enc["overflow_to_sample_mapping"]):
        windows[sample_idx].append(i)

    predictions = []
    for idx in range(len(records)):
        context = contexts[idx]
        best_score, best_text = -1e9, ""
        for w in windows[idx]:
            offsets = offset_map[w]
            s_log, e_log = start_logits[w], end_logits[w]
            s_cand = np.argsort(s_log)[-N_BEST:][::-1]
            e_cand = np.argsort(e_log)[-N_BEST:][::-1]
            for s in s_cand:
                for e in e_cand:
                    if s >= len(offsets) or e >= len(offsets):
                        continue
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > MAX_ANS_LEN:
                        continue
                    score = s_log[s] + e_log[e]
                    if score > best_score:
                        best_score = score
                        best_text = context[offsets[s][0]:offsets[e][1]]
        predictions.append(" ".join(best_text.split()))   # squash whitespace
    return predictions


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


def report(records, predictions, label):
    scores, by_tag = [], collections.defaultdict(list)
    for record, prediction in zip(records, predictions):
        s = meteor(gold_of(record), prediction)
        scores.append(s)
        by_tag[record["tags"][0]].append(s)

    def mean(xs):
        return float(np.mean(xs)) if len(xs) else 0.0

    print(f"{label:<28}{mean(scores):.4f}   " + "  ".join(
        f"{t} {mean(by_tag[t]):.4f}" for t in ("phrase", "passage", "multi")))
    return mean(scores)


# ---------------------------------------------------------------- main

def main():
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    train_data = load_jsonl(f"{DATA_DIR}/train.jsonl")
    val = load_jsonl(f"{DATA_DIR}/val.jsonl")
    test = load_jsonl(f"{DATA_DIR}/test.jsonl")
    print(f"train {len(train_data)} | val {len(val)} | test {len(test)}")

    resolved = sum(get_answer(r)[0] is not None for r in train_data)
    print(f"found the answer span in {resolved}/{len(train_data)} "
          f"({resolved / len(train_data):.1%})")

    dataset = make_train_features(train_data)
    print(len(dataset), "training windows")

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
    train(model, dataset)

    val_pred = predict(model, val)
    for r, p in list(zip(val, val_pred))[:5]:
        print("POST :", get_question(r)[:90])
        print("GOLD :", gold_of(r)[:90])
        print("PRED :", p[:90])
        print("-" * 70)

    print(f"\n{'system':<28}{'METEOR':<9}  per type")
    report(val, [r.get("targetTitle", "") for r in val], "naive baseline (title)")
    report(val, val_pred, "roberta-base-squad2")

    test_pred = predict(model, test)
    with open("prediction_task2.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "spoiler"])
        for record, prediction in zip(test, test_pred):
            # if the model somehow returned nothing, fall back to the title
            writer.writerow([record["id"], prediction or record.get("targetTitle", "")])
    print(f"wrote {len(test_pred)} predictions, "
          f"{sum(1 for p in test_pred if not p)} were empty")


if __name__ == "__main__":
    main()
