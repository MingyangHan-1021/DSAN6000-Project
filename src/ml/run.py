"""Local scikit-learn prototype. The course Pipeline is spark/ml_run.py.

Run from the repo root, after the dev stream exists:

    python src/data/prepare_banking77.py
    python src/data/simulate_stream.py --profile dev
    python -m src.ml.run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.behavior import add_behavior, load_home_locations, load_messages
from src.ml.fraud import compare_fraud_models
from src.ml.intent import fit_intent

CLEAN_PATH = REPO_ROOT / "data/processed/banking77_clean.parquet"
MESSAGES_PATH = REPO_ROOT / "data/simulated/messages"
USERS_PATH = REPO_ROOT / "data/simulated/users.parquet"
OUTPUT_PATH = REPO_ROOT / "ml/output/metrics.json"


def _require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. From the repo root, run "
            "python src/data/prepare_banking77.py and "
            "python src/data/simulate_stream.py --profile dev"
        )
    return path


def _print_operating_points(name: str, points: list[dict]) -> None:
    print(f"{name} operating points (cutoffs chosen on validation, scored on test)")
    for point in points:
        test = point.get("test")
        if test is None:
            print(f"  {point['rule']}: not reachable on validation")
            continue
        print(
            f"  {point['rule']} threshold={test['threshold']:.3f} "
            f"precision={test['precision']:.3f} recall={test['recall']:.3f} "
            f"f1={test['f1']:.3f} flagged={test['n_flagged']} "
            f"tp={test['tp']} fp={test['fp']} fn={test['fn']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local scikit-learn prototype.")
    parser.add_argument("--fraud-only", action="store_true")
    args = parser.parse_args()

    payload = {
        "note": (
            "is_fraud comes from the simulator rules, not from confirmed bank cases. "
            "The fraud threshold maximizes F1 on validation users. "
            "Recall targets are also chosen on validation users. "
            "Test users are not used to choose a threshold."
        )
    }
    if not args.fraud_only:
        clean = pd.read_parquet(_require(CLEAN_PATH))
        print("fitting intent classifier on the Banking77 split")
        intent_metrics = fit_intent(clean)
        print(
            "intent "
            f"accuracy={intent_metrics['accuracy']:.3f} "
            f"macro_f1={intent_metrics['macro_f1']:.3f} "
            f"weighted_f1={intent_metrics['weighted_f1']:.3f}"
        )
        payload["intent"] = intent_metrics

    print("building behavior features on the full dev stream")
    messages = load_messages(_require(MESSAGES_PATH))
    users = load_home_locations(_require(USERS_PATH))
    featured = add_behavior(messages, users)
    print(f"messages={len(featured)} fraud_rate={featured['is_fraud'].mean():.4f}")

    print("comparing text-only and text-plus-behavior fraud models")
    fraud_metrics = compare_fraud_models(featured)
    for name, metrics in fraud_metrics["models"].items():
        print(
            f"{name} precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
            f"pr_auc={metrics['pr_auc']:.3f} threshold={metrics['threshold']:.3f}"
        )
        _print_operating_points(name, metrics["operating_points"])

    payload["fraud"] = fraud_metrics
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
