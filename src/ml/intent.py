"""Banking77 intent classifier.

Trains on the published train split and scores the published test split.
The simulated stream is not used here: its sentences are drawn from the
train split only, so scoring it would not be an unseen test.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

EXPECTED_ROWS = {"train": 10003, "test": 3080}


def fit_intent(clean: pd.DataFrame) -> dict:
    """TF-IDF (1–2 grams) plus logistic regression, same setup as the Part B EDA."""

    missing = {"text", "label", "split"} - set(clean.columns)
    if missing:
        raise SystemExit(f"banking77_clean is missing columns: {sorted(missing)}")

    train = clean.loc[clean["split"] == "train"].copy()
    test = clean.loc[clean["split"] == "test"].copy()
    counts = {"train": len(train), "test": len(test)}
    for split, expected in EXPECTED_ROWS.items():
        if counts[split] != expected:
            raise SystemExit(f"{split} split has {counts[split]} rows, expected {expected}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )
    x_train = vectorizer.fit_transform(train["text"])
    x_test = vectorizer.transform(test["text"])
    model = LogisticRegression(max_iter=2000, C=10.0, random_state=42)
    model.fit(x_train, train["label"].astype("int64"))
    predicted = model.predict(x_test)
    labels = test["label"].astype("int64")
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_classes": int(labels.nunique()),
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "weighted_f1": float(f1_score(labels, predicted, average="weighted")),
    }
