# DSAN6000-Project

Scalable fraud detection in banking chatbot conversations.

Banking77 supplies the labeled customer sentences. This repo turns that static file into a local OLTP-style message stream: most new messages have no intent label, a minority are manually labeled, and a small share follow simulated fraud patterns. Later work can point the same paths at S3 and Spark. The bucket is still a placeholder.

## Data quick start

From this directory, with Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/data/prepare_banking77.py
python src/data/simulate_stream.py --profile dev
```

`dev` builds 10,000 users and 500,000 messages. `python src/data/simulate_stream.py --profile full` builds the 1,000,000-user / 50,000,000-message stream in chunks. Generated files under `data/` are gitignored. Details, column schemas, and the fraud rules are in [docs/data_setup.md](docs/data_setup.md).

## ML

The course model is a Spark MLlib `Pipeline`: `RegexTokenizer`, word and bigram `CountVectorizer`, `IDF`, `VectorAssembler`, then `LogisticRegression`. Run it from the repo root after the data quick start. It needs a JDK and `pyspark`.

```bash
python spark/ml_run.py
```

Intent uses the official `split`. Fraud predicts simulated `is_fraud` and splits users with seed 7 (6,400 fit, 1,600 validation, 2,000 test on the dev stream). The text vectorizer, the behavior `StandardScaler`, and the classifier are fit on the fit users only. `fraud_pattern`, `persona`, and `n_fraud_messages` are not features. The script writes `ml/output/spark_metrics.json`, which is gitignored. Those numbers are not filled in here until that command has been run.

`ON_CLUSTER=1` points the same script at `spark://<private-ip>:7077` and `s3a://`. It reads `S3_BUCKET` and `S3_PREFIX` and does not read AWS keys. Unlabeled-stream intent scoring stays in `spark/intent_classifier.ipynb` on `main`.

| File | What it does |
|---|---|
| `spark/ml_run.py` | Course intent model and the text-only versus text-plus-behavior fraud comparison. |
| `src/ml/behavior.py` | Feature definitions shared with the Spark script: gap, first-message flag, messages so far, session minutes so far, hour, late night (00:00–05:00), distance from home. |
| `src/ml/run.py` | Local scikit-learn prototype of the same two tasks. Not the course result. |

The fraud label is `messages.is_fraud`, not the intent id in `label` and not `dim_intent.risk_level`. About 70% of stream rows have a null intent label. The fraud models do not fill those in by matching the sentence back to Banking77. TF-IDF is fit on the training users only.

Users are split with seed 7: 6,400 fit, 1,600 validation, 2,000 test. The same users are used for both fraud models. The threshold maximizes F1 on the validation users. Test users are not used to choose it. `fraud_pattern`, `persona`, and `n_fraud_messages` are not features. `class_weight="balanced"` is on, because about 1.3% of messages are fraud.

`is_fraud` comes from the simulator rules, not from confirmed bank cases.

### Local scikit-learn prototype, already run

These numbers are from `python -m src.ml.run`. They are not the Spark MLlib result.

Intent, on the Banking77 test split: accuracy 0.891, macro F1 0.891, weighted F1 0.891.

Fraud, on the 2,000 test users (99,515 messages, 1.27% fraud). PR-AUC is the comparison that does not depend on the threshold.

| Model | Precision | Recall | F1 | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|
| Text only | 0.144 | 0.405 | 0.212 | 0.124 | 0.894 |
| Text + behavior | 0.713 | 0.472 | 0.568 | 0.606 | 0.986 |

Those two thresholds are different, so the per-pattern recalls below are at 0.5 for both models. Behavior helps money-mule, location-anomaly, and night-activity sessions. Card testing is already high from the text. Account takeover is a short daytime session at home, and behavior does not improve it.

| Pattern | Text only | Text + behavior |
|---|---:|---:|
| Account takeover | 0.763 | 0.693 |
| Money mule | 0.866 | 0.984 |
| Card testing | 0.954 | 0.934 |
| Location anomaly | 0.715 | 0.849 |
| Night activity | 0.763 | 0.905 |

At 0.5, precision is 0.081 for text only and 0.168 for text plus behavior. That threshold is not the operating point. It is there so the pattern rates can be compared.

That fraud comparison also scores cutoffs chosen only on the 1,600 validation users. On the 2,000 test users, text + behavior has 1,264 fraud messages out of 99,515. At the F1 cutoff the counts are 596 true positives, 240 false positives, and 668 false negatives (836 messages flagged). A validation rule that asks for at least 70% recall lands at 0.866 and, on the test users, precision 0.316, recall 0.692, 2,770 flagged. The 80% validation rule lands at 0.743 and, on the test users, precision 0.235, recall 0.780, 4,203 flagged. At 0.5, recall is 0.882 and 6,640 messages are flagged. None of these test numbers were used to pick the cutoff.
