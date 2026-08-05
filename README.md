# MSE 641 Final Project — Clickbait Spoiling

Code for our MSE 641 (Text Analytics) final project.

- **Task 1** — classify which kind of spoiler a clickbait post needs: `phrase`, `passage`, or `multi`.
- **Task 2** — produce the spoiler itself, framed as extractive question answering over the linked article.

Authors: Archi Zhao, Leo Zhou.

## Data

The competition data is **not** in this repository. Download `train.jsonl`,
`val.jsonl` and `test.jsonl` from the Kaggle competition page and put them next
to whichever script you want to run — every script searches the current
directory and its subdirectories for `train.jsonl`.

## Running

```bash
pip install torch transformers scikit-learn numpy nltk
python "task 2 code/task2_v4_snap.py"
```

Task 1 runs on CPU but is far quicker on a GPU. Task 2 needs one: the final
version takes roughly 50 minutes on a Colab T4. Each script writes its own
`prediction_*.csv` in Kaggle submission format.

## Task 1 — spoiler type classification

Scored by weighted F1 on the public leaderboard.

| Version | Main change | Model | Kaggle | File |
|---|---|---|---|---|
All files live in `task 1 code/`.

| Version | Main change | Model | Kaggle | File |
|---|---|---|---|---|
| V1 | word + character TF–IDF, class-balanced linear SVM, 5-fold CV | LinearSVC | 0.53313 | `T1 (ver 1).py` |
| V2 | first move off sparse features | DistilRoBERTa | 0.71359 | `T1 (ver 2).py` |
| V3 | post and title as a paired input, single model | RoBERTa-base | 0.75389 | `T1 (ver 3).py` |
| V4 | three-fold CV, blended with the TF–IDF SVM | DeBERTa-v3-base + SVM | 0.71581 | `T1 （ver 4）.py` |
| V5 | validation records added to training | RoBERTa-base | 0.72383 | `T1 （ver 5）.py` |
| V6 | three seeds (42/123/777), probabilities averaged | RoBERTa-base | 0.77160 | `T1 （ver 6）.py` |
| V7 | best subset of five seeds, chosen on validation | RoBERTa-base | 0.75483 | `T1 （ver 7）.py` |
| V8 | larger encoder, nothing else changed | RoBERTa-large | 0.73291 | `T1 （ver 8）.py` |
| **V9** | **common epoch for the whole three-seed ensemble** | **RoBERTa-base** | **0.77354** | `T1 （ver 9）.py` |

V9 is the submitted system.

**V3 is a byproduct of Task 2.** `task 2 code/task2_v3_rebalanced.py` trains a
RoBERTa-base spoiler-type classifier to route its span decoder, and that
classifier's test predictions are themselves a complete Task 1 submission
(`prediction_task1_v3.csv`). `T1 (ver 3).py` is the same script. Note that its
predicted types are taken after a validation-tuned bias is added to the `multi`
logit, not by a plain argmax.

## Task 2 — spoiler generation

Scored by METEOR. Validation METEOR is what we tuned on; the leaderboard column
is the public Kaggle score.

| Version | Main change | Model | Val | Kaggle | File |
|---|---|---|---|---|---|
| V1 | single span, one length cap for every type | roberta-base-squad2 | 0.4565 | 0.45647 | `task 2 code/task2_v1_qa.py` |
| V2 | trained on every gold segment, type classifier routes the decoder | roberta-base-squad2 + roberta-base | 0.5192 | 0.45287 | `task 2 code/task2_v2_multispan.py` |
| V3 | `1/n_segments` sampling, three-seed span ensemble, five-fold decoder tuning, multi-logit bias | roberta-base-squad2 + roberta-base | — | 0.45133 | `task 2 code/task2_v3_rebalanced.py` |
| **V4** | **sentence-boundary snapping for passage and multi, stronger backbone** | **deberta-v3-large-squad2 + roberta-base** | — | **0.46653** | `task 2 code/task2_v4_snap.py` |

V4 is the submitted system.

The framing is extractive rather than generative because the training data ships
`spoilerPositions`, the character offsets of the spoiler inside the article, so
the answer is almost always a literal substring. Articles are longer than the
512-token limit at the 90th percentile, so the context is split into 384-token
windows with a stride of 128.

Sentence snapping is the change that moved the leaderboard. Gold `passage` and
`multi` answers cover about 84% of the sentence they sit in, so a span that
stops short loses recall — and METEOR weights recall nine times more heavily
than precision. `phrase` answers cover only 13% of their sentence and are never
snapped.

`USE_DEBERTA = False` in V4 falls back to three averaged roberta-base-squad2
models, which is faster and needs less VRAM.
