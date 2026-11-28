"""Text-only fraud model versus the same model with behavior features.

Both models share one user split. The vectorizer and the scaler are fit on
the fit users only. The decision threshold maximizes F1 on the validation
users and is never chosen from the test users.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from src.ml.behavior import BEHAVIOR_COLUMNS

PATTERNS = (
    "account_takeover",
    "money_mule",
    "card_testing",
    "location_anomaly",
    "night_activity",
)


def split_user_ids(
    user_ids: np.ndarray,
    seed: int = 7,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> dict[str, np.ndarray]:
    """80/20 user holdout, then 80/20 fit/validation inside the training users."""

    users = np.array(sorted(set(user_ids.tolist())))
    outer = np.random.default_rng(seed)
    outer.shuffle(users)
    n_test = int(round(len(users) * test_size))
    if n_test == 0 or n_test >= len(users):
        raise SystemExit("user split left an empty train or test set")
    test_users = users[:n_test]
    train_users = users[n_test:]

    inner = np.random.default_rng(seed)
    inner.shuffle(train_users)
    n_val = int(round(len(train_users) * val_size))
    if n_val == 0 or n_val >= len(train_users):
        raise SystemExit("validation split left an empty fit or validation set")
    return {
        "fit": np.sort(train_users[n_val:]),
        "val": np.sort(train_users[:n_val]),
        "test": np.sort(test_users),
    }


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_features=30000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )


def _classifier() -> LogisticRegression:
    # liblinear keeps the sparse text matrix sparse. The default solver
    # densifies it, which does not fit a few hundred thousand messages.
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=7,
        solver="liblinear",
    )


def _best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1 = (2 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1))])


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    y_bool = y_true.astype(bool)
    tp = int(np.logical_and(y_bool, predicted).sum())
    fp = int(np.logical_and(~y_bool, predicted).sum())
    fn = int(np.logical_and(y_bool, ~predicted).sum())
    tn = int(np.logical_and(~y_bool, ~predicted).sum())
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_flagged": tp + fp,
        "n_fraud": tp + fn,
    }


def _recall_targets(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    """Validation-only cutoffs. Ties keep the higher precision, then the higher recall."""

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return {"recall_at_least_0_70": None, "recall_at_least_0_80": None}
    precision = precision[:-1]
    recall = recall[:-1]
    chosen: dict[str, float | None] = {}
    for level, key in ((0.70, "recall_at_least_0_70"), (0.80, "recall_at_least_0_80")):
        eligible = np.flatnonzero(recall >= level)
        if len(eligible) == 0:
            chosen[key] = None
            continue
        order = np.lexsort((thresholds[eligible], recall[eligible], precision[eligible]))
        chosen[key] = float(thresholds[eligible[order[-1]]])
    return chosen


def _operating_points(
    y_val: np.ndarray,
    val_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
) -> list[dict]:
    """Cutoffs are fixed or chosen on validation users, then scored on test users."""

    rules: list[tuple[str, float | None]] = [
        ("max_f1", _best_f1_threshold(y_val, val_scores)),
        *_recall_targets(y_val, val_scores).items(),
        ("fixed_0_5", 0.5),
    ]
    points = []
    for rule, threshold in rules:
        if threshold is None:
            points.append({"rule": rule, "chosen_on": "validation", "threshold": None, "reachable_on_validation": False})
            continue
        points.append({
            "rule": rule,
            "chosen_on": "validation" if rule != "fixed_0_5" else "predeclared",
            "threshold": float(threshold),
            "reachable_on_validation": True,
            "validation": _metrics(y_val, val_scores, threshold),
            "test": _metrics(y_test, test_scores, threshold),
        })
    return points


def _recall_by_pattern(
    pattern: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    predicted = scores >= threshold
    fraud = y_true == 1
    out = {}
    for name in PATTERNS:
        mask = fraud & (pattern == name)
        total = int(mask.sum())
        hits = int(np.logical_and(mask, predicted).sum())
        out[name] = {
            "n_fraud": total,
            "n_detected": hits,
            "recall": float(hits / total) if total else None,
        }
    return out


def _design_matrix(texts, behavior: pd.DataFrame, vectorizer, scaler, fit: bool, with_behavior: bool):
    if fit:
        text_matrix = vectorizer.fit_transform(texts)
    else:
        text_matrix = vectorizer.transform(texts)
    if not with_behavior:
        return text_matrix
    values = behavior.loc[:, BEHAVIOR_COLUMNS].to_numpy(dtype=np.float64)
    scaled = scaler.fit_transform(values) if fit else scaler.transform(values)
    return sparse.hstack([text_matrix, sparse.csr_matrix(scaled)], format="csr")


def _slice(frame: pd.DataFrame, users: np.ndarray) -> pd.DataFrame:
    chosen = frame.loc[frame["user_id"].isin(users)]
    if chosen.empty:
        raise SystemExit("a user split selected no messages")
    return chosen


def compare_fraud_models(frame: pd.DataFrame, seed: int = 7) -> dict:
    """Fit both models on the same users and score the same test users."""

    users = split_user_ids(frame["user_id"].to_numpy(), seed=seed)
    parts = {name: _slice(frame, ids) for name, ids in users.items()}
    y = {name: part["is_fraud"].to_numpy(dtype=np.int8) for name, part in parts.items()}
    if y["fit"].sum() == 0 or y["val"].sum() == 0 or y["test"].sum() == 0:
        raise SystemExit("a split has no fraud messages; the threshold cannot be chosen")

    results = {
        "seed": seed,
        "n_users": {name: int(len(ids)) for name, ids in users.items()},
        "n_messages": {name: int(len(part)) for name, part in parts.items()},
        "fraud_rate": {name: float(y[name].mean()) for name in parts},
        "models": {},
    }

    for name, with_behavior in (("text_only", False), ("text_behavior", True)):
        vectorizer = _vectorizer()
        scaler = StandardScaler()
        model = _classifier()
        x_fit = _design_matrix(
            parts["fit"]["text"], parts["fit"], vectorizer, scaler, fit=True, with_behavior=with_behavior
        )
        model.fit(x_fit, y["fit"])
        val_scores = model.predict_proba(
            _design_matrix(
                parts["val"]["text"], parts["val"], vectorizer, scaler, fit=False, with_behavior=with_behavior
            )
        )[:, 1]
        threshold = _best_f1_threshold(y["val"], val_scores)
        test_scores = model.predict_proba(
            _design_matrix(
                parts["test"]["text"], parts["test"], vectorizer, scaler, fit=False, with_behavior=with_behavior
            )
        )[:, 1]
        at_threshold = _metrics(y["test"], test_scores, threshold)
        at_half = _metrics(y["test"], test_scores, 0.5)
        pattern = parts["test"]["fraud_pattern"].to_numpy()
        results["models"][name] = {
            "features": "tfidf" if not with_behavior else "tfidf+behavior",
            **at_threshold,
            "at_threshold_0_5": {
                "precision": at_half["precision"],
                "recall": at_half["recall"],
                "f1": at_half["f1"],
            },
            "operating_points": _operating_points(y["val"], val_scores, y["test"], test_scores),
            "recall_by_pattern": _recall_by_pattern(pattern, y["test"], test_scores, threshold),
            "recall_by_pattern_at_0_5": _recall_by_pattern(pattern, y["test"], test_scores, 0.5),
        }
    return results
