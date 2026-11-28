"""Behavior features that are known when a message arrives.

History is computed on the full stream, then the caller splits by user.
Within a user, a row may see only earlier rows. Session totals, persona,
fraud_pattern, and n_fraud_messages are not features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Gap written on a user's first message. The flag below tells the model
# this is "no previous message", not a real one-day wait.
FIRST_GAP_SECONDS = 24 * 60 * 60

BEHAVIOR_COLUMNS = [
    "gap_seconds",
    "is_first_message",
    "messages_so_far",
    "session_minutes_so_far",
    "hour",
    "is_late_night",
    "distance_km",
]

MESSAGE_COLUMNS = [
    "message_id",
    "received_ts",
    "text",
    "user_id",
    "session_id",
    "lat",
    "lon",
    "is_fraud",
    "fraud_pattern",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres. Inputs are degrees."""

    radius = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_messages(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=MESSAGE_COLUMNS)
    if frame["message_id"].duplicated().any():
        raise SystemExit("message_id is not unique")
    if frame["text"].isna().any():
        raise SystemExit("messages.text has nulls")
    frame["received_ts"] = pd.to_datetime(frame["received_ts"])
    frame["is_fraud"] = frame["is_fraud"].astype(bool)
    return frame


def load_home_locations(path: Path) -> pd.DataFrame:
    """Home coordinates only. persona and n_fraud_messages would leak the label."""

    users = pd.read_parquet(path, columns=["user_id", "home_lat", "home_lon"])
    if users["user_id"].duplicated().any():
        raise SystemExit("user_id is duplicated in users.parquet")
    return users


def add_behavior(messages: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Sort by user, time, message_id, then add causal behavior columns."""

    frame = messages.merge(users, on="user_id", how="left", validate="many_to_one")
    if frame["home_lat"].isna().any():
        raise SystemExit("a message has no home location")

    frame = frame.sort_values(
        ["user_id", "received_ts", "message_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    gap = frame.groupby("user_id", sort=False)["received_ts"].diff().dt.total_seconds()
    frame["is_first_message"] = gap.isna().astype(np.int8)
    frame["gap_seconds"] = gap.fillna(FIRST_GAP_SECONDS).astype(np.float64)

    session = frame.groupby(["user_id", "session_id"], sort=False)
    frame["messages_so_far"] = session.cumcount() + 1
    session_start = session["received_ts"].transform("first")
    frame["session_minutes_so_far"] = (
        (frame["received_ts"] - session_start).dt.total_seconds() / 60.0
    ).astype(np.float64)

    frame["hour"] = frame["received_ts"].dt.hour.astype(np.int8)
    frame["is_late_night"] = ((frame["hour"] >= 0) & (frame["hour"] < 5)).astype(np.int8)
    frame["distance_km"] = haversine_km(
        frame["lat"].to_numpy(),
        frame["lon"].to_numpy(),
        frame["home_lat"].to_numpy(),
        frame["home_lon"].to_numpy(),
    ).astype(np.float64)

    keep = [
        "message_id",
        "user_id",
        "text",
        "is_fraud",
        "fraud_pattern",
        *BEHAVIOR_COLUMNS,
    ]
    return frame.loc[:, keep]
