"""Course ML: Spark MLlib pipelines for Banking77.

Week 8 session (local[*] or spark://<private-ip>:7077) and a week 10 Pipeline.
Logistic regression only. A decision tree on TF-IDF crosses the 4096-feature
switch taught in week 10.

From the repo root, after the dev stream exists:

    python spark/ml_run.py
    python spark/ml_run.py --intent-only
    python spark/ml_run.py --fraud-only

Intent uses the official Banking77 split (train 10,003 / test 3,080).
Fraud predicts simulated is_fraud. Users are split with seed 7 before any
tokenizer, IDF, or scaler is fit. The two fraud models share that split.

This script does not score unlabeled stream rows. That step is
spark/intent_classifier.ipynb on main. is_fraud is generator ground truth,
not a bank investigation, and not dim_intent.risk_level.

Cluster path, only after the bucket exists:

    ON_CLUSTER=1 S3_BUCKET=your-bucket S3_PREFIX=banking77 python spark/ml_run.py

No AWS keys are read here. SPARK_DRIVER_MEMORY defaults to 6g.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.behavior import BEHAVIOR_COLUMNS, FIRST_GAP_SECONDS
from src.ml.fraud import (
    _best_f1_threshold,
    _metrics,
    _recall_by_pattern,
    split_user_ids,
)

EXPECTED_INTENT = {"train": 10003, "test": 3080}
EXPECTED_USERS = {"fit": 6400, "val": 1600, "test": 2000}
OUTPUT_PATH = ROOT / "ml" / "output" / "spark_metrics.json"


def _paths() -> tuple[str, str, str, str]:
    """Return master, clean path, messages path, users path."""

    on_cluster = os.environ.get("ON_CLUSTER") == "1"
    if not on_cluster:
        clean = ROOT / "data" / "processed" / "banking77_clean.parquet"
        messages = ROOT / "data" / "simulated" / "messages"
        users = ROOT / "data" / "simulated" / "users.parquet"
        missing = [path for path in (clean, messages, users) if not path.exists()]
        if missing:
            joined = ", ".join(str(path) for path in missing)
            raise SystemExit(
                f"Missing {joined}. From the repo root, run "
                "python src/data/prepare_banking77.py and "
                "python src/data/simulate_stream.py --profile dev"
            )
        return "local[*]", clean.as_posix(), messages.as_posix(), users.as_posix()

    bucket = os.environ.get("S3_BUCKET", "").strip()
    prefix = os.environ.get("S3_PREFIX", "").strip().strip("/")
    if bucket in ("", "TODO") or prefix in ("", "TODO"):
        raise SystemExit("ON_CLUSTER=1 requires S3_BUCKET and S3_PREFIX. No keys are read from the environment.")
    ip = subprocess.check_output(["hostname", "-I"], text=True).split()
    if not ip:
        raise SystemExit("hostname -I returned no address. Start the master before this script.")
    base = f"s3a://{bucket}/{prefix}"
    return (
        f"spark://{ip[0]}:7077",
        f"{base}/processed/banking77_clean.parquet",
        f"{base}/simulated/messages",
        f"{base}/simulated/users.parquet",
    )


def _session(master: str):
    from pyspark.sql import SparkSession

    memory = os.environ.get("SPARK_DRIVER_MEMORY", "6g")
    partitions = "48" if master != "local[*]" else "16"
    builder = (
        SparkSession.builder.appName("banking77-ml")
        .master(master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", partitions)
        .config("spark.driver.memory", memory)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )
    if master != "local[*]":
        builder = builder.config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.5.0")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print(f"master: {master} | cores: {spark.sparkContext.defaultParallelism} | driver: {memory}")
    return spark


def _logistic_regression(family: str, **kwargs):
    from pyspark.ml.classification import LogisticRegression

    model = LogisticRegression(**kwargs)
    if hasattr(model, "setFamily") and "family" in model.explainParams():
        model.setFamily(family)
    return model


def _text_stages():
    """Same text stack as spark/intent_classifier.ipynb on main."""

    from pyspark.ml.feature import CountVectorizer, IDF, NGram, RegexTokenizer

    return [
        RegexTokenizer(inputCol="text", outputCol="tokens", pattern="[^a-z0-9']+", toLowercase=True),
        NGram(n=2, inputCol="tokens", outputCol="bigrams"),
        CountVectorizer(inputCol="tokens", outputCol="uni_tf", minDF=2),
        CountVectorizer(inputCol="bigrams", outputCol="bi_tf", minDF=2),
        IDF(inputCol="uni_tf", outputCol="uni_tfidf"),
        IDF(inputCol="bi_tf", outputCol="bi_tfidf"),
    ]


def _timed(name: str, fn):
    start = time.time()
    out = fn()
    print(f"{name}: {time.time() - start:.1f} s")
    return out


def run_intent(spark, clean_path: str) -> dict:
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import functions as F

    clean = spark.read.parquet(clean_path)
    train = clean.where(F.col("split") == "train").select("text", F.col("label").cast("double").alias("label"))
    test = clean.where(F.col("split") == "test").select("text", F.col("label").cast("double").alias("label"))
    counts = {"train": train.count(), "test": test.count()}
    print(f"intent rows: train={counts['train']} test={counts['test']}")
    for split, expected in EXPECTED_INTENT.items():
        if counts[split] != expected:
            raise SystemExit(f"intent {split} split has {counts[split]} rows, expected {expected}")

    pipeline = Pipeline(stages=[
        *_text_stages(),
        VectorAssembler(inputCols=["uni_tfidf", "bi_tfidf"], outputCol="features"),
        _logistic_regression(
            "multinomial",
            featuresCol="features",
            labelCol="label",
            maxIter=100,
            regParam=0.01,
            elasticNetParam=0.0,
        ),
    ])
    model = _timed("intent pipeline", lambda: pipeline.fit(train))
    predicted = model.transform(test)
    accuracy = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    ).evaluate(predicted)
    weighted_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    ).evaluate(predicted)
    print(f"intent accuracy={accuracy:.4f} weighted_f1={weighted_f1:.4f}")
    return {
        "stack": "spark_mllib",
        "split": "banking77 official split",
        "n_train": counts["train"],
        "n_test": counts["test"],
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "f1_metric": "Spark MulticlassClassificationEvaluator metricName=f1, which is weighted F1",
    }


def _haversine_km(lat1, lon1, lat2, lon2):
    from pyspark.sql import functions as F

    phi1 = F.radians(lat1)
    phi2 = F.radians(lat2)
    dphi = F.radians(lat2 - lat1)
    dlambda = F.radians(lon2 - lon1)
    a = F.pow(F.sin(dphi / 2.0), 2.0) + F.cos(phi1) * F.cos(phi2) * F.pow(F.sin(dlambda / 2.0), 2.0)
    clipped = F.least(F.greatest(a, F.lit(0.0)), F.lit(1.0))
    return F.lit(2.0 * 6371.0) * F.asin(F.sqrt(clipped))


def _with_behavior(spark, messages_path: str, users_path: str):
    """Causal behavior on the full stream. The intent id column is not read."""

    from pyspark.sql import Window
    from pyspark.sql import functions as F

    messages = spark.read.parquet(messages_path).select(
        "message_id", "received_ts", "text", "user_id", "session_id", "lat", "lon", "is_fraud", "fraud_pattern"
    )
    users = spark.read.parquet(users_path).select("user_id", "home_lat", "home_lon")
    if messages.groupBy("message_id").count().where(F.col("count") > 1).limit(1).count():
        raise SystemExit("message_id is not unique")

    frame = messages.join(users, on="user_id", how="left")
    if frame.where(F.col("home_lat").isNull() | F.col("text").isNull()).limit(1).count():
        raise SystemExit("a message is missing text or a home location")

    received = F.col("received_ts").cast("timestamp")
    by_user = Window.partitionBy("user_id").orderBy(received, "message_id")
    by_session = Window.partitionBy("user_id", "session_id").orderBy(received, "message_id")
    session_all = Window.partitionBy("user_id", "session_id")
    previous = F.lag(received).over(by_user)
    gap = F.unix_timestamp(received) - F.unix_timestamp(previous)

    featured = frame.select(
        "message_id",
        "user_id",
        "text",
        "fraud_pattern",
        F.col("is_fraud").cast("double").alias("fraud_label"),
        F.when(previous.isNull(), F.lit(float(FIRST_GAP_SECONDS))).otherwise(gap).cast("double").alias("gap_seconds"),
        F.when(previous.isNull(), F.lit(1.0)).otherwise(F.lit(0.0)).alias("is_first_message"),
        F.row_number().over(by_session).cast("double").alias("messages_so_far"),
        ((F.unix_timestamp(received) - F.unix_timestamp(F.min(received).over(session_all))) / F.lit(60.0))
        .cast("double")
        .alias("session_minutes_so_far"),
        F.hour(received).cast("double").alias("hour"),
        F.when((F.hour(received) >= 0) & (F.hour(received) < 5), F.lit(1.0)).otherwise(F.lit(0.0)).alias("is_late_night"),
        _haversine_km(F.col("lat"), F.col("lon"), F.col("home_lat"), F.col("home_lon")).cast("double").alias("distance_km"),
    )
    missing = [name for name in BEHAVIOR_COLUMNS if name not in featured.columns]
    if missing:
        raise SystemExit(f"behavior columns were not created: {missing}")
    return featured


def _attach_split(spark, featured):
    from pyspark.sql import functions as F

    user_ids = [row.user_id for row in featured.select("user_id").distinct().collect()]
    users = split_user_ids(np.asarray(user_ids), seed=7)
    user_counts = {name: int(len(ids)) for name, ids in users.items()}
    if len(user_ids) == 10000 and user_counts != EXPECTED_USERS:
        raise SystemExit(f"user split {user_counts} != {EXPECTED_USERS}")
    print("fraud users: " + " ".join(f"{name}={user_counts[name]}" for name in ("fit", "val", "test")))
    rows = [(int(user_id), name) for name, ids in users.items() for user_id in ids]
    parts = spark.createDataFrame(rows, "user_id long, part string")
    before = featured.count()
    assigned = featured.join(F.broadcast(parts), on="user_id", how="inner")
    if assigned.count() != before:
        raise SystemExit("user split dropped messages")
    return assigned, user_counts


def _balanced_weight(featured):
    """sklearn class_weight='balanced', from the fit users only."""

    from pyspark.sql import functions as F

    counts = {
        int(row.fraud_label): int(row["count"])
        for row in featured.where(F.col("part") == "fit").groupBy("fraud_label").count().collect()
    }
    if sorted(counts) != [0, 1] or counts[1] == 0:
        raise SystemExit(f"fit users do not contain both classes: {counts}")
    total = counts[0] + counts[1]
    weight_neg = total / (2.0 * counts[0])
    weight_pos = total / (2.0 * counts[1])
    print(f"fit class counts: non_fraud={counts[0]} fraud={counts[1]} weights={weight_neg:.4f}/{weight_pos:.4f}")
    return featured.withColumn(
        "weight",
        F.when(F.col("fraud_label") == 1.0, F.lit(weight_pos)).otherwise(F.lit(weight_neg)),
    )


def _score_frame(model, frame):
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F

    scored = model.transform(frame).select(
        F.col("fraud_label").alias("y"),
        vector_to_array(F.col("probability")).getItem(1).alias("score"),
        "fraud_pattern",
    ).toPandas()
    return scored["y"].to_numpy(), scored["score"].to_numpy(), scored["fraud_pattern"].to_numpy()


def _fraud_metrics(model, val_frame, test_frame) -> dict:
    y_val, score_val, _ = _score_frame(model, val_frame)
    if y_val.sum() == 0:
        raise SystemExit("validation users have no fraud messages")
    threshold = _best_f1_threshold(y_val, score_val)
    y_test, score_test, pattern = _score_frame(model, test_frame)
    if y_test.sum() == 0:
        raise SystemExit("test users have no fraud messages")
    at_threshold = _metrics(y_test, score_test, threshold)
    at_half = _metrics(y_test, score_test, 0.5)
    return {
        **at_threshold,
        "at_threshold_0_5": {
            "precision": at_half["precision"],
            "recall": at_half["recall"],
            "f1": at_half["f1"],
        },
        "recall_by_pattern": _recall_by_pattern(pattern, y_test, score_test, threshold),
        "recall_by_pattern_at_0_5": _recall_by_pattern(pattern, y_test, score_test, 0.5),
    }


def run_fraud(spark, messages_path: str, users_path: str) -> dict:
    from pyspark import StorageLevel
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StandardScaler, VectorAssembler
    from pyspark.sql import functions as F

    featured, user_counts = _attach_split(spark, _with_behavior(spark, messages_path, users_path))
    totals = {
        row.part: (int(row.n), int(row.n_fraud))
        for row in featured.groupBy("part").agg(
            F.count("*").alias("n"),
            F.sum("fraud_label").alias("n_fraud"),
        ).collect()
    }
    print(
        "fraud messages: "
        + " ".join(f"{name}={totals[name][0]} fraud={totals[name][1]}" for name in ("fit", "val", "test"))
    )
    for name in ("fit", "val", "test"):
        n_rows, n_fraud = totals[name]
        if n_fraud == 0 or n_fraud == n_rows:
            raise SystemExit(f"{name} split has no class contrast ({n_fraud} fraud of {n_rows})")

    weighted = _balanced_weight(featured)
    fit = weighted.where(F.col("part") == "fit")
    text_model = _timed("fraud text pipeline", lambda: Pipeline(stages=_text_stages()).fit(fit))

    keep = ["fraud_label", "weight", "fraud_pattern", "uni_tfidf", "bi_tfidf", *BEHAVIOR_COLUMNS]
    parts = {}
    for name in ("fit", "val", "test"):
        transformed = text_model.transform(weighted.where(F.col("part") == name)).select(*keep)
        parts[name] = transformed.persist(StorageLevel.MEMORY_AND_DISK)
        parts[name].count()

    classifier = _logistic_regression(
        "binomial",
        featuresCol="features",
        labelCol="fraud_label",
        weightCol="weight",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0,
    )
    text_only = Pipeline(stages=[
        VectorAssembler(inputCols=["uni_tfidf", "bi_tfidf"], outputCol="features"),
        classifier.copy(),
    ])
    text_behavior = Pipeline(stages=[
        VectorAssembler(inputCols=list(BEHAVIOR_COLUMNS), outputCol="behavior_raw"),
        # Mean-center these 7 columns only. withMean on the TF-IDF vector would densify it.
        StandardScaler(inputCol="behavior_raw", outputCol="behavior_scaled", withMean=True, withStd=True),
        VectorAssembler(inputCols=["uni_tfidf", "bi_tfidf", "behavior_scaled"], outputCol="features"),
        classifier.copy(),
    ])

    models = {}
    for name, pipeline, features in (
        ("text_only", text_only, "tfidf"),
        ("text_behavior", text_behavior, "tfidf+behavior"),
    ):
        fitted = _timed(name, lambda pipeline=pipeline: pipeline.fit(parts["fit"]))
        metrics = _fraud_metrics(fitted, parts["val"], parts["test"])
        models[name] = {"features": features, **metrics}
        print(
            f"{name} precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f} pr_auc={metrics['pr_auc']:.3f} threshold={metrics['threshold']:.3f}"
        )

    for frame in parts.values():
        frame.unpersist()
    return {
        "stack": "spark_mllib",
        "target": "is_fraud",
        "seed": 7,
        "n_users": user_counts,
        "n_messages": {name: totals[name][0] for name in ("fit", "val", "test")},
        "fraud_count": {name: totals[name][1] for name in ("fit", "val", "test")},
        "fraud_rate": {name: totals[name][1] / totals[name][0] for name in ("fit", "val", "test")},
        "models": models,
        "threshold_rule": (
            "Each model picks the threshold that maximizes F1 on validation users. "
            "recall_by_pattern uses that threshold. recall_by_pattern_at_0_5 uses 0.5 for both models. "
            "pr_auc does not use a threshold."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Spark MLlib intent model and fraud comparison.")
    parser.add_argument("--intent-only", action="store_true")
    parser.add_argument("--fraud-only", action="store_true")
    args = parser.parse_args()
    if args.intent_only and args.fraud_only:
        raise SystemExit("Choose one of --intent-only or --fraud-only.")

    master, clean_path, messages_path, users_path = _paths()
    spark = _session(master)
    payload = {
        "note": (
            "Intent metrics are Banking77 label accuracy on the official test split. "
            "Fraud metrics recover simulated is_fraud on held-out users. "
            "They are not confirmed bank fraud."
        )
    }
    try:
        if not args.fraud_only:
            payload["intent"] = run_intent(spark, clean_path)
        if not args.intent_only:
            payload["fraud"] = run_fraud(spark, messages_path, users_path)
    finally:
        spark.stop()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
