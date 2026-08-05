# MSE 641 Final Project — Clickbait Spoiling

Task 1 classifies which kind of spoiler a clickbait post needs (`phrase`,
`passage`, `multi`). Task 2 produces the spoiler itself, framed as extractive
question answering. Archi Zhao, Leo Zhou.

## Running

Download `train.jsonl`, `val.jsonl` and `test.jsonl` from the Kaggle competition
page and put them next to the script you want to run.

```bash
pip install torch transformers scikit-learn numpy nltk
python "task 2 code/task2_v4_snap.py"
```

Task 2 needs a GPU — about 50 minutes on a Colab T4. Each script writes its own
`prediction_*.csv` in submission format.

## Task 1 — `task 1 code/`

| Version | Main change | Model | Kaggle |
|---|---|---|---|
| `T1 (ver 1).py` | word + character TF–IDF, balanced linear SVM | LinearSVC | 0.53313 |
| `T1 (ver 2).py` | first move off sparse features | DistilRoBERTa | 0.71359 |
| `T1 (ver 3).py` | post and title as a paired input, single model | RoBERTa-base | 0.75389 |
| `T1 （ver 4）.py` | three-fold CV, blended with the TF–IDF SVM | DeBERTa-v3-base + SVM | 0.71581 |
| `T1 （ver 5）.py` | validation records added to training | RoBERTa-base | 0.72383 |
| `T1 （ver 6）.py` | three seeds averaged | RoBERTa-base | 0.77160 |
| `T1 （ver 7）.py` | best subset of five seeds | RoBERTa-base | 0.75483 |
| `T1 （ver 8）.py` | larger encoder | RoBERTa-large | 0.73291 |
| **`T1 （ver 9）.py`** | **common epoch for the three-seed ensemble** | **RoBERTa-base** | **0.77354** |

V3 is the spoiler-type classifier from the Task 2 v3 pipeline; the file is the
same script as `task 2 code/task2_v3_rebalanced.py`.

## Task 2 — `task 2 code/`

| Version | Main change | Model | Kaggle |
|---|---|---|---|
| `task2_v1_qa.py` | single span, one length cap for every type | roberta-base-squad2 | 0.45647 |
| `task2_v2_multispan.py` | every gold segment, type classifier routes the decoder | roberta-base-squad2 + roberta-base | 0.45287 |
| `task2_v3_rebalanced.py` | `1/n_segments` sampling, seed ensemble, five-fold decoder tuning | roberta-base-squad2 + roberta-base | 0.45133 |
| **`task2_v4_snap.py`** | **sentence-boundary snapping, stronger backbone** | **deberta-v3-large-squad2 + roberta-base** | **0.46653** |
