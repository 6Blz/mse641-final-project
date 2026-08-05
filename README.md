# MSE 641 Final Project - Clickbait Spoiling

Task 1 classifies which kind of spoiler a clickbait post needs (`phrase`,
`passage`, `multi`). Task 2 produces the spoiler itself, as an extractive QA
problem. Archi Zhao, Leo Zhou.

## Running

Download `train.jsonl`, `val.jsonl` and `test.jsonl` from the Kaggle competition
page and put them next to the script you want to run.

```bash
pip install torch transformers scikit-learn numpy nltk
python "task 2 code/task2_v4_snap.py"
```

Task 2 needs a GPU, roughly 50 minutes on a Colab T4. Each script writes its own
`prediction_*.csv` in submission format.

## Task 1 (`task 1 code/`)

| Version | What changed | Model | Kaggle |
|---|---|---|---|
| `T1 (ver 1).py` | word and character TF-IDF into a balanced linear SVM | LinearSVC | 0.53313 |
| `T1 (ver 2).py` | swapped TF-IDF for a transformer | DistilRoBERTa | 0.71359 |
| `T1 (ver 3).py` | feed the post and the title as a pair | RoBERTa-base | 0.75389 |
| `T1 （ver 4）.py` | 3-fold CV, blended with the old SVM | DeBERTa-v3-base + SVM | 0.71581 |
| `T1 （ver 5）.py` | train on train+val instead of train only | RoBERTa-base | 0.72383 |
| `T1 （ver 6）.py` | three seeds, probabilities averaged | RoBERTa-base | 0.77160 |
| `T1 （ver 7）.py` | five seeds, keep whichever subset scores best | RoBERTa-base | 0.75483 |
| `T1 （ver 8）.py` | tried roberta-large | RoBERTa-large | 0.73291 |
| `T1 （ver 9）.py` | one shared epoch for the whole ensemble | RoBERTa-base | 0.77354 |

V9 is the submission. V3 is the spoiler-type classifier out of the Task 2 v3
pipeline, so that file is the same script as `task 2 code/task2_v3_rebalanced.py`.

## Task 2 (`task 2 code/`)

| Version | What changed | Model | Kaggle |
|---|---|---|---|
| `task2_v1_qa.py` | one span, same length cap for every type | roberta-base-squad2 | 0.45647 |
| `task2_v2_multispan.py` | train on every gold segment, add a type classifier to route the decoder | roberta-base-squad2 + roberta-base | 0.45287 |
| `task2_v3_rebalanced.py` | weight segments by `1/n`, average three seeds, tune the decoder with 5-fold CV | roberta-base-squad2 + roberta-base | 0.45133 |
| `task2_v4_snap.py` | grow passage and multi spans out to their sentence, bigger backbone | deberta-v3-large-squad2 + roberta-base | 0.46653 |

V4 is the submission.
