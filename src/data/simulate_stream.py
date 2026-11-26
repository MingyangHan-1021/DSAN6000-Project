"""Simulate an OLTP-style chatbot stream from the cleaned Banking77 sentences.

Run from the repo root:

    python src/data/simulate_stream.py --profile dev

`full` is about 50 million messages. It is implemented, but it runs only when
you pass --profile full explicitly.

All times in `received` and `received_ts` are the user's local clock time.
Stream text is sampled only from the `source_split` split (train by default),
so the Banking77 test set stays unseen.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.storage import load_config, locate

MESSAGE_COLUMNS = [
    "message_id",
    "received",
    "received_ts",
    "text",
    "label",
    "label_source",
    "intent_set_version",
    "user_id",
    "session_id",
    "city",
    "lat",
    "lon",
    "is_fraud",
    "fraud_pattern",
]
USER_COLUMNS = [
    "user_id",
    "home_city",
    "home_lat",
    "home_lon",
    "persona",
    "n_messages",
    "n_fraud_messages",
]
ROW_FIELDS = [
    "received_ts",
    "received",
    "text",
    "intent",
    "user_id",
    "session_id",
    "city",
    "lat",
    "lon",
    "is_fraud",
    "fraud_pattern",
]


class Counter:
    def __init__(self) -> None:
        self.value = 1

    def take(self, count: int = 1) -> int:
        start = self.value
        self.value += count
        return start


@dataclass
class UserPlan:
    user_id: int
    persona: str
    city: str
    lat: float
    lon: float
    n_messages: int
    n_fraud_messages: int
    pattern_plan: dict = field(default_factory=dict)


@dataclass
class World:
    seed: int
    day0: np.datetime64
    n_days: int
    hour_start: int
    hour_end: int
    margin: timedelta
    place_attempts: int
    size_min: int
    size_max: int
    span_min: int
    span_max: int
    night_share: float
    evening_share: float
    night_hour_start: int
    night_hour_end: int
    evening_hour_start: int
    evening_hour_end: int
    away_session_share: float
    sensitive_intent_share: float
    sensitive_intents: list[str]
    pools: dict[str, np.ndarray]
    stages: list[dict]
    stage_cum: np.ndarray
    cities: list[dict]
    min_away_km: float
    min_history_days: int
    name_to_id: dict[str, int]
    label_keep_rate: float
    label_source: str
    intent_set_version: int
    patterns: dict
    sessions: Counter


def user_generator(seed: int, user_id: int, stream: int) -> np.random.Generator:
    """Independent stream. stream 1 draws sizes, stream 2 draws messages."""

    return np.random.default_rng(np.random.SeedSequence([seed, user_id, stream]))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def new_cols() -> dict[str, list]:
    return {name: [] for name in ROW_FIELDS}


def at(day0: np.datetime64, day_offset: int, hour: int, minute: int, second: int = 0) -> np.datetime64:
    return (
        np.datetime64(day0, "s")
        + np.timedelta64(int(day_offset), "D")
        + np.timedelta64(int(hour), "h")
        + np.timedelta64(int(minute), "m")
        + np.timedelta64(int(second), "s")
    )


def burst_times(rng: np.random.Generator, count: int, start: np.datetime64, window_minutes: int) -> list[datetime]:
    span_seconds = max(int(window_minutes) * 60, count)
    offsets = np.sort(rng.choice(span_seconds, size=count, replace=False))
    stamps = np.datetime64(start, "s") + offsets.astype("timedelta64[s]")
    return list(pd.DatetimeIndex(stamps).to_pydatetime())


def daytime_start(
    rng: np.random.Generator,
    day0: np.datetime64,
    day_lo: int,
    day_hi: int,
    hour_start: int,
    hour_end: int,
) -> np.datetime64:
    if day_hi <= day_lo:
        raise SystemExit(f"Empty day range [{day_lo}, {day_hi})")
    if hour_end <= hour_start:
        raise SystemExit("daytime hour_end must be after hour_start")
    return at(
        day0,
        int(rng.integers(day_lo, day_hi)),
        int(rng.integers(hour_start, hour_end)),
        int(rng.integers(0, 60)),
        int(rng.integers(0, 60)),
    )


def overlaps(times: list[datetime], blocked: list[tuple[datetime, datetime]], margin: timedelta) -> bool:
    if not blocked:
        return False
    start = times[0] - margin
    end = times[-1] + margin
    return any(start <= stop and end >= begin for begin, stop in blocked)


def split_session_sizes(rng: np.random.Generator, count: int, size_min: int, size_max: int) -> list[int]:
    sizes = []
    left = count
    while left > 0:
        high = min(size_max, left)
        low = min(size_min, high)
        sizes.append(int(rng.integers(low, high + 1)))
        left -= sizes[-1]
    return sizes


def session_intents(rng: np.random.Generator, names: list[str], size: int) -> list[str]:
    if len(names) == 1:
        return [names[0]] * size
    order = rng.permutation(np.array(names, dtype=object))
    tiled = np.tile(order, int(math.ceil(size / len(order))))
    return tiled[:size].tolist()


def take_texts(rng: np.random.Generator, pools: dict[str, np.ndarray], intents: list[str]) -> list[str]:
    texts = []
    for name in intents:
        pool = pools.get(name)
        if pool is None or len(pool) == 0:
            raise SystemExit(f"No Banking77 sentences for intent {name!r}")
        texts.append(str(pool[int(rng.integers(0, len(pool)))]))
    return texts


def add_session(
    cols: dict[str, list],
    times: list[datetime],
    intents: list[str],
    texts: list[str],
    user: UserPlan,
    session_id: int,
    city: str,
    lat: float,
    lon: float,
    is_fraud: bool,
    pattern: str | None,
) -> None:
    for ts, intent, text in zip(times, intents, texts, strict=True):
        cols["received_ts"].append(ts)
        cols["received"].append(ts.strftime("%Y-%m-%d_%H:%M:%S"))
        cols["text"].append(text)
        cols["intent"].append(intent)
        cols["user_id"].append(user.user_id)
        cols["session_id"].append(session_id)
        cols["city"].append(city)
        cols["lat"].append(float(lat))
        cols["lon"].append(float(lon))
        cols["is_fraud"].append(bool(is_fraud))
        cols["fraud_pattern"].append(pattern)


def choose_names(
    rng: np.random.Generator,
    names: list[str],
    count: int,
    weights: list[float] | None = None,
) -> list[str]:
    options = np.array(names, dtype=object)
    if weights is None:
        return rng.choice(options, size=count, replace=True).tolist()
    probability = np.array(weights, dtype=float)
    if len(probability) != len(names) or (probability < 0).any() or probability.sum() <= 0:
        raise SystemExit(f"Bad weights for {names}")
    probability = probability / probability.sum()
    return rng.choice(options, size=count, p=probability, replace=True).tolist()


def choose_ordinary_clock(rng: np.random.Generator, world: World) -> tuple[int, int]:
    """Daytime is the default. A small share of sessions fall at night or in the evening."""

    roll = float(rng.random())
    if roll < world.night_share:
        return world.night_hour_start, world.night_hour_end
    if roll < world.night_share + world.evening_share:
        return world.evening_hour_start, world.evening_hour_end
    return world.hour_start, world.hour_end


def place_ordinary_session(
    rng: np.random.Generator,
    size: int,
    world: World,
    day_lo: int,
    day_hi: int,
    blocked: list[tuple[datetime, datetime]],
) -> list[datetime]:
    span = int(rng.integers(world.span_min, world.span_max + 1))
    hour_start, hour_end = choose_ordinary_clock(rng, world)
    # Keep the whole short session inside the chosen window, including its span.
    earliest = hour_start * 60
    latest = hour_end * 60 - span
    if latest <= earliest:
        raise SystemExit(
            f"Ordinary span of {span} minutes does not fit between {hour_start}:00 and {hour_end}:00."
        )
    for _ in range(world.place_attempts):
        minute_of_day = int(rng.integers(earliest, latest))
        hour, minute = divmod(minute_of_day, 60)
        start = at(
            world.day0,
            int(rng.integers(day_lo, day_hi)),
            hour,
            minute,
            0,
        )
        candidate = burst_times(rng, size, start, span)
        if not overlaps(candidate, blocked, world.margin):
            return candidate
    raise SystemExit(
        "Could not place an ordinary session inside the date window. "
        "Increase ordinary_sessions.place_attempts."
    )


def build_ordinary(
    rng: np.random.Generator,
    user: UserPlan,
    count: int,
    world: World,
    day_lo: int,
    day_hi: int,
    blocked: list[tuple[datetime, datetime]],
) -> dict[str, list]:
    """Ordinary sessions for normal users and for fraud-user background.

    Most sessions are daytime and at home, inside one journey stage.
    A configured share is at night, in the evening, in another city, or about
    one sensitive intent. Those sessions stay size_min..size_max, so they do
    not copy a fraud burst or an account-takeover chain.
    """

    cols = new_cols()
    occupied = list(blocked)
    for size in split_session_sizes(rng, count, world.size_min, world.size_max):
        times = place_ordinary_session(rng, size, world, day_lo, day_hi, occupied)
        if float(rng.random()) < world.sensitive_intent_share:
            intent = world.sensitive_intents[int(rng.integers(0, len(world.sensitive_intents)))]
            intents = [intent] * size
        else:
            # Pick a journey stage in proportion to how many intents it holds, so every
            # ordinary intent is about equally common (a uniform stage pick would make
            # one-intent stages such as `limits` five times more common than the rest).
            index = int(np.searchsorted(world.stage_cum, float(rng.random()), side="right"))
            stage = world.stages[min(index, len(world.stages) - 1)]
            intents = session_intents(rng, stage["intents"], size)
        if float(rng.random()) < world.away_session_share:
            away = pick_away(rng, user, world)
            city, lat, lon = str(away["name"]), float(away["lat"]), float(away["lon"])
        else:
            city, lat, lon = user.city, user.lat, user.lon
        texts = take_texts(rng, world.pools, intents)
        add_session(
            cols,
            times,
            intents,
            texts,
            user,
            world.sessions.take(),
            city,
            lat,
            lon,
            False,
            None,
        )
        occupied.append((times[0], times[-1]))
    return cols


def build_account_takeover(rng: np.random.Generator, user: UserPlan, world: World) -> tuple[dict[str, list], datetime, datetime]:
    """Account takeover.

    Real scheme: the attacker resets the PIN or passcode, edits the personal
    details on the account, then moves money. Banking77 has no "add
    beneficiary" intent, so the money-movement step uses beneficiary_not_allowed
    and the other transfer intents. The steps stay in that order inside one
    short window.
    """

    spec = world.patterns["account_takeover"]
    intents: list[str] = []
    for step, count in zip(spec["steps"], user.pattern_plan["step_counts"], strict=True):
        intents.extend(choose_names(rng, step["intents"], int(count)))
    window = int(rng.integers(spec["window_minutes_min"], spec["window_minutes_max"] + 1))
    start = daytime_start(rng, world.day0, 0, world.n_days, world.hour_start, world.hour_end)
    times = burst_times(rng, len(intents), start, window)
    cols = new_cols()
    add_session(
        cols,
        times,
        intents,
        take_texts(rng, world.pools, intents),
        user,
        world.sessions.take(),
        user.city,
        user.lat,
        user.lon,
        True,
        "account_takeover",
    )
    return cols, times[0], times[-1]


def build_money_mule(rng: np.random.Generator, user: UserPlan, world: World) -> tuple[dict[str, list], datetime, datetime]:
    """Money mule.

    Real scheme: the account receives several payments, then forwards them
    out within minutes. Incoming text is receiving_money. Outgoing text uses
    transfer intents, because the dataset has no separate send-money label.
    """

    spec = world.patterns["money_mule"]
    receive_n = int(user.pattern_plan["receive_n"])
    send_n = int(user.pattern_plan["send_n"])
    receive_span = int(rng.integers(spec["receive_span_minutes_min"], spec["receive_span_minutes_max"] + 1))
    start = daytime_start(
        rng,
        world.day0,
        0,
        world.n_days,
        world.hour_start,
        int(spec["receive_hour_end"]),
    )
    receive_times = burst_times(rng, receive_n, start, receive_span)
    gap = int(rng.integers(spec["send_gap_minutes_min"], spec["send_gap_minutes_max"] + 1))
    send_span = int(rng.integers(spec["send_span_minutes_min"], spec["send_span_minutes_max"] + 1))
    send_start = np.datetime64(receive_times[-1] + timedelta(minutes=gap), "s")
    send_times = burst_times(rng, send_n, send_start, send_span)
    receive_intents = choose_names(rng, spec["receive_intents"], receive_n)
    send_intents = choose_names(rng, spec["send_intents"], send_n)
    times = receive_times + send_times
    intents = receive_intents + send_intents
    cols = new_cols()
    add_session(
        cols,
        times,
        intents,
        take_texts(rng, world.pools, intents),
        user,
        world.sessions.take(),
        user.city,
        user.lat,
        user.lon,
        True,
        "money_mule",
    )
    return cols, times[0], times[-1]


def build_card_testing(rng: np.random.Generator, user: UserPlan, world: World) -> tuple[dict[str, list], datetime, datetime]:
    """Card testing.

    Real scheme: a stolen card number is tried in a rapid burst of payments
    to see which authorizations go through, mixed with questions about limits.
    """

    spec = world.patterns["card_testing"]
    count = int(user.pattern_plan["n"])
    intents = choose_names(rng, spec["intents"], count, spec["weights"])
    window = int(rng.integers(spec["window_minutes_min"], spec["window_minutes_max"] + 1))
    start = daytime_start(rng, world.day0, 0, world.n_days, world.hour_start, world.hour_end)
    times = burst_times(rng, count, start, window)
    cols = new_cols()
    add_session(
        cols,
        times,
        intents,
        take_texts(rng, world.pools, intents),
        user,
        world.sessions.take(),
        user.city,
        user.lat,
        user.lon,
        True,
        "card_testing",
    )
    return cols, times[0], times[-1]


def build_location_anomaly(rng: np.random.Generator, user: UserPlan, world: World) -> tuple[dict[str, list], datetime, datetime]:
    """Location anomaly.

    Real scheme: after a history of home-city use, the account shows up in
    a distant city (stolen session or impossible travel). Only the away
    session is flagged. Background messages stay in the home city.

    Each user gets their own anomaly day, drawn across the whole window after
    min_history_days. The user's home-city history is placed only before that
    day (see build_user), so anomalies do not pile up at the end of the window.
    """

    spec = world.patterns["location_anomaly"]
    count = int(user.pattern_plan["n"])
    away = pick_away(rng, user, world)
    intents = choose_names(rng, spec["intents"], count)
    window = int(rng.integers(spec["window_minutes_min"], spec["window_minutes_max"] + 1))
    day = int(rng.integers(world.min_history_days, world.n_days))
    user.pattern_plan["anomaly_day"] = day
    start = daytime_start(rng, world.day0, day, day + 1, world.hour_start, world.hour_end)
    times = burst_times(rng, count, start, window)
    cols = new_cols()
    add_session(
        cols,
        times,
        intents,
        take_texts(rng, world.pools, intents),
        user,
        world.sessions.take(),
        away["name"],
        away["lat"],
        away["lon"],
        True,
        "location_anomaly",
    )
    return cols, times[0], times[-1]


def build_night_activity(rng: np.random.Generator, user: UserPlan, world: World) -> tuple[dict[str, list], datetime, datetime]:
    """Night activity.

    Real scheme: the account is driven in one short cluster after midnight,
    when the legitimate customer is usually asleep. Daytime messages from
    the same user are background and are not flagged.
    """

    spec = world.patterns["night_activity"]
    count = int(user.pattern_plan["n"])
    window_hours = int(rng.integers(spec["window_hours_min"], spec["window_hours_max"] + 1))
    latest_hour = int(spec["hour_end"]) - window_hours
    if latest_hour < int(spec["hour_start"]):
        raise SystemExit("night window does not fit between hour_start and hour_end")
    hour0 = int(rng.integers(int(spec["hour_start"]), latest_hour + 1))
    day = int(rng.integers(0, world.n_days))
    start = at(world.day0, day, hour0, 0, 0)
    times = burst_times(rng, count, start, window_hours * 60)
    intents = choose_names(rng, spec["intents"], count)
    cols = new_cols()
    add_session(
        cols,
        times,
        intents,
        take_texts(rng, world.pools, intents),
        user,
        world.sessions.take(),
        user.city,
        user.lat,
        user.lon,
        True,
        "night_activity",
    )
    return cols, times[0], times[-1]


BUILDERS = {
    "account_takeover": build_account_takeover,
    "money_mule": build_money_mule,
    "card_testing": build_card_testing,
    "location_anomaly": build_location_anomaly,
    "night_activity": build_night_activity,
}


def pick_away(rng: np.random.Generator, user: UserPlan, world: World) -> dict:
    far = []
    for city in world.cities:
        if city["name"] == user.city:
            continue
        distance = haversine_km(user.lat, user.lon, float(city["lat"]), float(city["lon"]))
        if distance >= world.min_away_km:
            far.append(city)
    if not far:
        raise SystemExit(f"No city is at least {world.min_away_km} km from {user.city}")
    return far[int(rng.integers(0, len(far)))]


def concat_cols(parts: list[dict[str, list]]) -> dict[str, list]:
    cols = new_cols()
    for part in parts:
        for key in cols:
            cols[key].extend(part[key])
    return cols


def apply_label_mask(cols: dict[str, list], rng: np.random.Generator, world: World) -> None:
    """Keep the intent id on about label_keep_rate of rows. The rest stay null."""

    count = len(cols["text"])
    keep_n = int(round(count * world.label_keep_rate))
    mask = np.zeros(count, dtype=bool)
    if keep_n:
        mask[:keep_n] = True
        rng.shuffle(mask)
    labels: list[int | None] = []
    sources: list[str | None] = []
    for keep, intent in zip(mask, cols["intent"], strict=True):
        if bool(keep):
            labels.append(int(world.name_to_id[intent]))
            sources.append(world.label_source)
        else:
            labels.append(None)
            sources.append(None)
    cols["label"] = labels
    cols["label_source"] = sources
    del cols["intent"]


def sort_cols(cols: dict[str, list]) -> None:
    order = sorted(range(len(cols["received_ts"])), key=cols["received_ts"].__getitem__)
    for key, values in list(cols.items()):
        cols[key] = [values[index] for index in order]


def build_user(user: UserPlan, world: World) -> dict[str, list]:
    rng = user_generator(world.seed, user.user_id, 2)
    parts: list[dict[str, list]] = []
    blocked: list[tuple[datetime, datetime]] = []
    if user.persona != "normal":
        builder = BUILDERS[user.persona]
        pattern_cols, start, end = builder(rng, user, world)
        if len(pattern_cols["text"]) != user.n_fraud_messages:
            raise SystemExit(
                f"user {user.user_id} emitted {len(pattern_cols['text'])} fraud rows, "
                f"planned {user.n_fraud_messages}"
            )
        parts.append(pattern_cols)
        blocked.append((start, end))
    ordinary_n = user.n_messages - user.n_fraud_messages
    day_lo, day_hi = 0, world.n_days
    if user.persona == "location_anomaly":
        # Home-city history happens only before this user's anomaly day.
        day_lo, day_hi = 0, int(user.pattern_plan["anomaly_day"])
    if ordinary_n:
        parts.append(build_ordinary(rng, user, ordinary_n, world, day_lo, day_hi, blocked))
    cols = concat_cols(parts)
    if len(cols["text"]) != user.n_messages:
        raise SystemExit(
            f"user {user.user_id} emitted {len(cols['text'])} rows, planned {user.n_messages}"
        )
    apply_label_mask(cols, rng, world)
    sort_cols(cols)
    return cols


def draw_plan(persona: str, rng: np.random.Generator, spec: dict) -> tuple[dict, int]:
    background_n = int(rng.integers(spec["background_min"], spec["background_max"] + 1))
    if persona == "account_takeover":
        step_counts = [
            int(rng.integers(step["n_min"], step["n_max"] + 1)) for step in spec["steps"]
        ]
        return {"step_counts": step_counts, "background_n": background_n}, sum(step_counts)
    if persona == "money_mule":
        receive_n = int(rng.integers(spec["receive_min"], spec["receive_max"] + 1))
        send_n = int(rng.integers(spec["send_min"], spec["send_max"] + 1))
        return {
            "receive_n": receive_n,
            "send_n": send_n,
            "background_n": background_n,
        }, receive_n + send_n
    if persona in {"card_testing", "location_anomaly", "night_activity"}:
        count = int(rng.integers(spec["n_min"], spec["n_max"] + 1))
        return {"n": count, "background_n": background_n}, count
    raise SystemExit(f"Unknown fraud pattern {persona!r}")


def _require_keys(spec: dict, keys: list[str], label: str) -> None:
    missing = [key for key in keys if key not in spec]
    if missing:
        raise SystemExit(f"{label} is missing {missing}")


def collect_intent_names(patterns: dict) -> list[str]:
    names: list[str] = []
    for persona, spec in patterns.items():
        if persona == "account_takeover":
            for step in spec["steps"]:
                names.extend(step["intents"])
        elif persona == "money_mule":
            names.extend(spec["receive_intents"])
            names.extend(spec["send_intents"])
        else:
            names.extend(spec["intents"])
    return names


def validate_config(cfg: dict) -> str:
    profile = str(cfg["active_profile"])
    if profile not in cfg["profiles"]:
        raise SystemExit(f"Unknown active_profile {profile!r}")
    rate = float(cfg["label_keep_rate"])
    if not 0 <= rate <= 1:
        raise SystemExit("label_keep_rate must be between 0 and 1")
    share = float(cfg["fraud_user_share"])
    if not 0 < share < 1:
        raise SystemExit("fraud_user_share must be between 0 and 1")
    if not isinstance(cfg["label_source"], str) or not cfg["label_source"].isidentifier():
        raise SystemExit("label_source must be a simple token such as manual")
    if str(cfg.get("source_split", "train")) not in {"train", "test"}:
        raise SystemExit("source_split must be train or test")
    for name, size in cfg["profiles"].items():
        window = size.get("window", cfg.get("window"))
        if window is None or "start" not in window or "end" not in window:
            raise SystemExit(f"profile {name!r} needs a window with start and end")
    patterns = cfg["patterns"]
    weights = cfg["pattern_weights"]
    if set(patterns) != set(BUILDERS) or set(weights) != set(BUILDERS):
        raise SystemExit(
            "pattern_weights and patterns must both be "
            f"{sorted(BUILDERS)}"
        )
    ordinary = cfg["ordinary_sessions"]
    _require_keys(
        ordinary,
        [
            "size_min",
            "size_max",
            "span_minutes_min",
            "span_minutes_max",
            "place_attempts",
            "normal_night_share",
            "night_hour_start",
            "night_hour_end",
            "normal_evening_share",
            "evening_hour_start",
            "evening_hour_end",
            "normal_away_session_share",
            "sensitive_intent_share",
            "sensitive_intents",
        ],
        "ordinary_sessions",
    )
    night_share = float(ordinary["normal_night_share"])
    evening_share = float(ordinary["normal_evening_share"])
    away_share = float(ordinary["normal_away_session_share"])
    sensitive_share = float(ordinary["sensitive_intent_share"])
    for name, value in (
        ("normal_night_share", night_share),
        ("normal_evening_share", evening_share),
        ("normal_away_session_share", away_share),
        ("sensitive_intent_share", sensitive_share),
    ):
        if not 0 <= value <= 1:
            raise SystemExit(f"{name} must be between 0 and 1")
    if night_share + evening_share >= 1:
        raise SystemExit("normal_night_share + normal_evening_share must be below 1")
    if int(ordinary["night_hour_end"]) <= int(ordinary["night_hour_start"]):
        raise SystemExit("ordinary night hours are empty")
    if int(ordinary["evening_hour_end"]) <= int(ordinary["evening_hour_start"]):
        raise SystemExit("ordinary evening hours are empty")
    if not ordinary["sensitive_intents"]:
        raise SystemExit("sensitive_intents must not be empty")
    fraud_night = patterns["night_activity"]
    if int(ordinary["night_hour_start"]) != int(fraud_night["hour_start"]) or int(ordinary["night_hour_end"]) != int(fraud_night["hour_end"]):
        raise SystemExit(
            "Ordinary night hours must match night_activity hours, "
            "so late-night messages are not fraud by the clock alone."
        )
    _require_keys(patterns["account_takeover"], ["steps", "window_minutes_min", "window_minutes_max", "background_min", "background_max"], "account_takeover")
    for index, step in enumerate(patterns["account_takeover"]["steps"]):
        _require_keys(step, ["intents", "n_min", "n_max"], f"account_takeover step {index}")
        if not step["intents"]:
            raise SystemExit(f"account_takeover step {index} has no intents")
    _require_keys(
        patterns["money_mule"],
        [
            "receive_intents",
            "receive_min",
            "receive_max",
            "receive_span_minutes_min",
            "receive_span_minutes_max",
            "send_intents",
            "send_min",
            "send_max",
            "send_gap_minutes_min",
            "send_gap_minutes_max",
            "send_span_minutes_min",
            "send_span_minutes_max",
            "receive_hour_end",
            "background_min",
            "background_max",
        ],
        "money_mule",
    )
    _require_keys(
        patterns["card_testing"],
        ["intents", "weights", "n_min", "n_max", "window_minutes_min", "window_minutes_max", "background_min", "background_max"],
        "card_testing",
    )
    if len(patterns["card_testing"]["weights"]) != len(patterns["card_testing"]["intents"]):
        raise SystemExit("card_testing weights must match intents")
    _require_keys(
        patterns["location_anomaly"],
        ["intents", "n_min", "n_max", "window_minutes_min", "window_minutes_max", "min_history_days", "background_min", "background_max"],
        "location_anomaly",
    )
    if int(patterns["location_anomaly"]["background_min"]) < 1:
        raise SystemExit("location_anomaly needs background history")
    if int(patterns["location_anomaly"]["min_history_days"]) < 1:
        raise SystemExit("location_anomaly min_history_days must be at least 1")
    _require_keys(
        patterns["night_activity"],
        ["intents", "n_min", "n_max", "hour_start", "hour_end", "window_hours_min", "window_hours_max", "background_min", "background_max"],
        "night_activity",
    )
    cities = cfg["cities"]
    names = [city["name"] for city in cities]
    if len(names) < 2 or len(set(names)) != len(names):
        raise SystemExit("cities need at least two unique names")
    for city in cities:
        lat = float(city["lat"])
        lon = float(city["lon"])
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise SystemExit(f"Bad coordinates for {city['name']}")
    return profile


def allocate_users(cfg: dict, profile: str, master: np.random.Generator) -> list[UserPlan]:
    """Draw who is fraudulent before any message text is sampled.

    Master-stream order is part of the seed: cities, fraud membership,
    pattern labels, then normal-user message counts. Per-user sizes use
    SeedSequence([seed, user_id, 1]) and do not consume the master stream.
    """

    size = cfg["profiles"][profile]
    n_users = int(size["n_users"])
    n_messages = int(size["n_messages"])
    if n_messages < n_users:
        raise SystemExit("n_messages must be at least n_users")
    patterns = cfg["patterns"]
    names = list(BUILDERS)
    n_fraud = int(round(n_users * float(cfg["fraud_user_share"])))
    if n_fraud < len(names):
        raise SystemExit("fraud_user_share is too small to cover every pattern")
    cities = cfg["cities"]
    city_index = master.integers(0, len(cities), size=n_users)
    fraud_index = np.sort(master.choice(n_users, size=n_fraud, replace=False))
    weights = np.array([float(cfg["pattern_weights"][name]) for name in names], dtype=float)
    if (weights <= 0).any():
        raise SystemExit("pattern_weights must be positive")
    drawn = master.multinomial(n_fraud, weights / weights.sum())
    if (drawn == 0).any():
        raise SystemExit(f"A pattern was assigned to nobody: {dict(zip(names, drawn.tolist()))}")
    labels = np.repeat(np.array(names, dtype=object), drawn)
    master.shuffle(labels)
    personas = np.array(["normal"] * n_users, dtype=object)
    personas[fraud_index] = labels

    plans: list[UserPlan | None] = [None] * n_users
    fraud_messages = 0
    for index in range(n_users):
        city = cities[int(city_index[index])]
        persona = str(personas[index])
        if persona == "normal":
            continue
        plan, n_pattern = draw_plan(
            persona,
            user_generator(int(cfg["seed"]), index + 1, 1),
            patterns[persona],
        )
        n_user = n_pattern + int(plan["background_n"])
        fraud_messages += n_user
        plans[index] = UserPlan(
            user_id=index + 1,
            persona=persona,
            city=str(city["name"]),
            lat=float(city["lat"]),
            lon=float(city["lon"]),
            n_messages=n_user,
            n_fraud_messages=n_pattern,
            pattern_plan=plan,
        )
    normal_index = [index for index, persona in enumerate(personas) if persona == "normal"]
    budget = n_messages - fraud_messages
    if budget < len(normal_index):
        raise SystemExit(
            f"Fraud users already use {fraud_messages} messages. "
            f"Only {budget} remain for {len(normal_index)} normal users."
        )
    shape = master.lognormal(0.0, 0.35, size=len(normal_index))
    counts = master.multinomial(budget, shape / shape.sum())
    for zero in np.flatnonzero(counts == 0):
        donors = np.flatnonzero(counts > 1)
        if len(donors) == 0:
            raise SystemExit("Could not give every normal user at least one message")
        donor = int(donors[int(master.integers(0, len(donors)))])
        counts[donor] -= 1
        counts[zero] += 1
    for offset, index in enumerate(normal_index):
        city = cities[int(city_index[index])]
        plans[index] = UserPlan(
            user_id=index + 1,
            persona="normal",
            city=str(city["name"]),
            lat=float(city["lat"]),
            lon=float(city["lon"]),
            n_messages=int(counts[offset]),
            n_fraud_messages=0,
        )
    filled = [plan for plan in plans if plan is not None]
    if len(filled) != n_users or sum(plan.n_messages for plan in filled) != n_messages:
        raise SystemExit("User allocation does not sum to the configured message count")
    return filled


def load_world(cfg: dict, plans_seed: int, profile: str) -> World:
    clean_path = locate(cfg["inputs"]["clean"]).require_local()
    dim_path = locate(cfg["inputs"]["dim_intent"]).require_local()
    if not clean_path.is_file() or not dim_path.is_file():
        raise SystemExit("Run python src/data/prepare_banking77.py before the simulator.")
    clean = pd.read_parquet(clean_path)
    # Only sample stream text from one split, so the Banking77 test set stays unseen.
    source_split = str(cfg.get("source_split", "train"))
    clean = clean[clean["split"] == source_split].reset_index(drop=True)
    if clean.empty:
        raise SystemExit(f"No sentences found for source_split {source_split!r}")
    dim = pd.read_parquet(dim_path)
    pools = {
        str(name): group["text"].to_numpy()
        for name, group in clean.groupby("intent", sort=False)
    }
    missing = sorted(set(collect_intent_names(cfg["patterns"])) - set(pools))
    if missing:
        raise SystemExit(f"Pattern intents are not in banking77_clean: {missing}")
    sensitive = [str(name) for name in cfg["ordinary_sessions"]["sensitive_intents"]]
    missing_sensitive = sorted(set(sensitive) - set(pools))
    if missing_sensitive:
        raise SystemExit(f"sensitive_intents are not in banking77_clean: {missing_sensitive}")
    ordinary = dim[dim["fraud_type"].isna()]
    stages = []
    for stage, group in ordinary.groupby("journey_stage", sort=True):
        names = [str(name) for name in group["intent"].tolist()]
        if names:
            stages.append({"name": str(stage), "intents": names})
    if not stages:
        raise SystemExit("No ordinary journey stages are available for normal users")
    stage_sizes = np.array([len(stage["intents"]) for stage in stages], dtype=float)
    stage_cum = np.cumsum(stage_sizes / stage_sizes.sum())
    missing_stage = sorted({name for stage in stages for name in stage["intents"]} - set(pools))
    if missing_stage:
        raise SystemExit(f"Ordinary stage intents are not in banking77_clean: {missing_stage}")
    # Each profile has its own window: dev is one month, full is one year.
    window = cfg["profiles"][profile].get("window", cfg.get("window"))
    if window is None:
        raise SystemExit(f"No window set for profile {profile!r}")
    start = np.datetime64(str(window["start"]), "D")
    end = np.datetime64(str(window["end"]), "D")
    n_days = int((end - start) / np.timedelta64(1, "D")) + 1
    if n_days < 2:
        raise SystemExit("window must cover at least two days")
    min_history_days = int(cfg["patterns"]["location_anomaly"]["min_history_days"])
    if not 1 <= min_history_days < n_days:
        raise SystemExit("min_history_days must leave at least one day for the anomaly")
    sessions_cfg = cfg["ordinary_sessions"]
    return World(
        seed=plans_seed,
        day0=np.datetime64(str(window["start"]), "s"),
        n_days=n_days,
        hour_start=int(cfg["daytime"]["hour_start"]),
        hour_end=int(cfg["daytime"]["hour_end"]),
        margin=timedelta(minutes=int(cfg["session_gap_minutes"])),
        place_attempts=int(sessions_cfg["place_attempts"]),
        size_min=int(sessions_cfg["size_min"]),
        size_max=int(sessions_cfg["size_max"]),
        span_min=int(sessions_cfg["span_minutes_min"]),
        span_max=int(sessions_cfg["span_minutes_max"]),
        night_share=float(sessions_cfg["normal_night_share"]),
        evening_share=float(sessions_cfg["normal_evening_share"]),
        night_hour_start=int(sessions_cfg["night_hour_start"]),
        night_hour_end=int(sessions_cfg["night_hour_end"]),
        evening_hour_start=int(sessions_cfg["evening_hour_start"]),
        evening_hour_end=int(sessions_cfg["evening_hour_end"]),
        away_session_share=float(sessions_cfg["normal_away_session_share"]),
        sensitive_intent_share=float(sessions_cfg["sensitive_intent_share"]),
        sensitive_intents=sensitive,
        pools=pools,
        stages=stages,
        stage_cum=stage_cum,
        cities=cfg["cities"],
        min_away_km=float(cfg["min_away_km"]),
        min_history_days=min_history_days,
        name_to_id={str(intent): int(intent_id) for intent_id, intent in zip(dim["intent_id"], dim["intent"])},
        label_keep_rate=float(cfg["label_keep_rate"]),
        label_source=str(cfg["label_source"]),
        intent_set_version=int(cfg["intent_set_version"]),
        patterns=cfg["patterns"],
        sessions=Counter(),
    )


def buffer_to_table(buffer: dict[str, list]) -> pa.Table:
    return pa.table(
        {
            "message_id": pa.array(buffer["message_id"], type=pa.int64()),
            "received": pa.array(buffer["received"], type=pa.string()),
            "received_ts": pa.array(buffer["received_ts"], type=pa.timestamp("us")),
            "text": pa.array(buffer["text"], type=pa.string()),
            "label": pa.array(buffer["label"], type=pa.int64()),
            "label_source": pa.array(buffer["label_source"], type=pa.string()),
            "intent_set_version": pa.array(buffer["intent_set_version"], type=pa.int64()),
            "user_id": pa.array(buffer["user_id"], type=pa.int64()),
            "session_id": pa.array(buffer["session_id"], type=pa.int64()),
            "city": pa.array(buffer["city"], type=pa.string()),
            "lat": pa.array(buffer["lat"], type=pa.float64()),
            "lon": pa.array(buffer["lon"], type=pa.float64()),
            "is_fraud": pa.array(buffer["is_fraud"], type=pa.bool_()),
            "fraud_pattern": pa.array(buffer["fraud_pattern"], type=pa.string()),
        }
    )


def validate_buffer(buffer: dict[str, list]) -> None:
    count = len(buffer["message_id"])
    if any(len(values) != count for values in buffer.values()):
        raise SystemExit("Message buffer columns have different lengths")
    for received, flag, pattern, label, source in zip(
        buffer["received"],
        buffer["is_fraud"],
        buffer["fraud_pattern"],
        buffer["label"],
        buffer["label_source"],
        strict=True,
    ):
        if len(received) != 19 or received[10] != "_":
            raise SystemExit(f"Bad received timestamp: {received!r}")
        if bool(flag) == (pattern is None):
            raise SystemExit("is_fraud and fraud_pattern disagree")
        if (label is None) != (source is None):
            raise SystemExit("label and label_source must be null together")


def write_chunk(buffer: dict[str, list], root: Path, chunk_id: int) -> None:
    if not buffer["message_id"]:
        return
    validate_buffer(buffer)
    table = buffer_to_table(buffer)
    if table.column_names != MESSAGE_COLUMNS:
        raise SystemExit(f"message columns drifted: {table.column_names}")
    dates = [ts.strftime("%Y-%m-%d") for ts in buffer["received_ts"]]
    groups: dict[str, list[int]] = {}
    for index, day in enumerate(dates):
        groups.setdefault(day, []).append(index)
    for day, indexes in groups.items():
        folder = root / f"date={day}"
        folder.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.take(pa.array(indexes, type=pa.int64())), folder / f"part-{chunk_id:05d}.parquet")
    for values in buffer.values():
        values.clear()


def extend_buffer(buffer: dict[str, list], cols: dict[str, list], message_ids: Counter, version: int) -> None:
    count = len(cols["text"])
    start = message_ids.take(count)
    buffer["message_id"].extend(range(start, start + count))
    for key in (
        "received",
        "received_ts",
        "text",
        "label",
        "label_source",
        "user_id",
        "session_id",
        "city",
        "lat",
        "lon",
        "is_fraud",
        "fraud_pattern",
    ):
        buffer[key].extend(cols[key])
    buffer["intent_set_version"].extend([version] * count)


def write_users(plans: list[UserPlan], path: Path) -> None:
    table = pa.table(
        {
            "user_id": pa.array([plan.user_id for plan in plans], type=pa.int64()),
            "home_city": pa.array([plan.city for plan in plans], type=pa.string()),
            "home_lat": pa.array([plan.lat for plan in plans], type=pa.float64()),
            "home_lon": pa.array([plan.lon for plan in plans], type=pa.float64()),
            "persona": pa.array([plan.persona for plan in plans], type=pa.string()),
            "n_messages": pa.array([plan.n_messages for plan in plans], type=pa.int64()),
            "n_fraud_messages": pa.array([plan.n_fraud_messages for plan in plans], type=pa.int64()),
        }
    )
    if table.column_names != USER_COLUMNS:
        raise SystemExit(f"user columns drifted: {table.column_names}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def parquet_glob(messages_dir: Path) -> str:
    return (messages_dir / "**" / "*.parquet").as_posix()


def write_daily_report(messages_dir: Path, report_path: Path, label_source: str) -> pd.DataFrame:
    query = f"""
        SELECT
            CAST(received_ts AS DATE) AS day,
            COUNT(*) AS messages,
            COUNT(label) AS labeled_messages,
            ROUND(
                COUNT(*) FILTER (WHERE label_source = '{label_source}') * 1.0 / COUNT(*),
                3
            ) AS manual_coverage,
            ROUND(COUNT(*) FILTER (WHERE is_fraud) * 1.0 / COUNT(*), 3) AS fraud_rate
        FROM read_parquet('{parquet_glob(messages_dir)}', hive_partitioning = false)
        GROUP BY 1
        ORDER BY 1
    """
    frame = duckdb.sql(query).df()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(report_path, index=False)
    return frame


def print_summary(messages_dir: Path, users_path: Path, report: pd.DataFrame, label_source: str) -> None:
    glob = parquet_glob(messages_dir)
    overview = duckdb.sql(
        f"""
        SELECT
            COUNT(*) AS messages,
            COUNT(DISTINCT user_id) AS users,
            COUNT(label) AS labeled_messages,
            COUNT(*) FILTER (WHERE label_source = '{label_source}') AS manual_labels,
            ROUND(COUNT(label) * 1.0 / COUNT(*), 4) AS label_coverage,
            ROUND(AVG(CASE WHEN is_fraud THEN 1 ELSE 0 END), 4) AS fraud_rate,
            MIN(message_id) AS min_id,
            MAX(message_id) AS max_id,
            COUNT(DISTINCT message_id) AS distinct_ids
        FROM read_parquet('{glob}', hive_partitioning = false)
        """
    ).df()
    by_pattern = duckdb.sql(
        f"""
        SELECT
            fraud_pattern,
            COUNT(*) AS messages,
            ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (), 4) AS share_of_fraud
        FROM read_parquet('{glob}', hive_partitioning = false)
        WHERE is_fraud
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    print("\n== overview ==")
    print(overview.to_string(index=False))
    print("\n== fraud rate by pattern ==")
    print(by_pattern.to_string(index=False))
    print("\n== daily report (first 3 and last 3 days) ==")
    print(pd.concat([report.head(3), report.tail(3)]).to_string(index=False))
    sample = duckdb.sql(
        f"""
        SELECT message_id, received, label, label_source, user_id, session_id,
               city, is_fraud, fraud_pattern, substr(text, 1, 90) AS text
        FROM read_parquet('{glob}', hive_partitioning = false)
        WHERE NOT is_fraud
        ORDER BY message_id
        LIMIT 3
        """
    ).df()
    takeover = duckdb.sql(
        f"""
        SELECT m.message_id, m.received, m.label, m.city, m.is_fraud, m.fraud_pattern,
               substr(m.text, 1, 90) AS text
        FROM read_parquet('{glob}', hive_partitioning = false) AS m
        WHERE m.user_id = (
            SELECT MIN(user_id) FROM read_parquet('{users_path.as_posix()}')
            WHERE persona = 'account_takeover'
        )
          AND m.is_fraud
        ORDER BY m.received_ts
        """
    ).df()
    print("\n== sample normal rows ==")
    print(sample.to_string(index=False))
    print("\n== one account_takeover sequence ==")
    print(takeover.to_string(index=False))
    labeled = int(overview.loc[0, "labeled_messages"])
    manual = int(overview.loc[0, "manual_labels"])
    if labeled != manual:
        raise SystemExit(f"labeled rows ({labeled}) do not match {label_source} rows ({manual})")
    coverage = labeled / int(overview.loc[0, "messages"])
    if abs(coverage - 0.30) > 0.015:
        raise SystemExit(f"label coverage {coverage:.4f} is far from the configured 0.30")
    if int(overview.loc[0, "min_id"]) != 1 or int(overview.loc[0, "max_id"]) != int(overview.loc[0, "messages"]):
        raise SystemExit("message_id is not 1..n")
    if int(overview.loc[0, "distinct_ids"]) != int(overview.loc[0, "messages"]):
        raise SystemExit("message_id values are not unique")


def generate(cfg: dict, profile: str) -> None:
    size = cfg["profiles"][profile]
    window = size.get("window", cfg.get("window"))
    print(
        f"profile={profile} users={size['n_users']} messages={size['n_messages']} "
        f"window={window['start']}..{window['end']} "
        f"source_split={cfg.get('source_split', 'train')} seed={cfg['seed']}"
    )
    master = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), 0]))
    plans = allocate_users(cfg, profile, master)
    world = load_world(cfg, int(cfg["seed"]), profile)
    fraud_rows = sum(plan.n_fraud_messages for plan in plans)
    print(
        f"planned fraud messages={fraud_rows} "
        f"personas={pd.Series([plan.persona for plan in plans]).value_counts().to_dict()}"
    )
    messages_dir = locate(cfg["outputs"]["messages"]).require_local()
    if messages_dir.exists():
        shutil.rmtree(messages_dir)
    messages_dir.mkdir(parents=True)
    buffer = {name: [] for name in MESSAGE_COLUMNS}
    message_ids = Counter()
    chunk_size = int(size["chunk_size"])
    chunk_id = 0
    written = 0
    for plan in plans:
        cols = build_user(plan, world)
        extend_buffer(buffer, cols, message_ids, world.intent_set_version)
        if len(buffer["message_id"]) >= chunk_size:
            written += len(buffer["message_id"])
            write_chunk(buffer, messages_dir, chunk_id)
            chunk_id += 1
            print(f"  wrote chunk {chunk_id} cumulative_rows={written}", flush=True)
    if buffer["message_id"]:
        written += len(buffer["message_id"])
        write_chunk(buffer, messages_dir, chunk_id)
        print(f"  wrote final chunk cumulative_rows={written}", flush=True)
    if written != int(size["n_messages"]):
        raise SystemExit(f"Wrote {written} messages, expected {size['n_messages']}")
    users_path = locate(cfg["outputs"]["users"]).require_local()
    write_users(plans, users_path)
    report_path = locate(cfg["outputs"]["daily_report"]).require_local()
    report = write_daily_report(messages_dir, report_path, world.label_source)
    print(f"wrote {users_path}")
    print(f"wrote {report_path} days={len(report)}")
    print_summary(messages_dir, users_path, report, world.label_source)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the Banking77 message stream.")
    parser.add_argument("--profile", choices=["dev", "full"], default=None)
    args = parser.parse_args()
    cfg = load_config("simulation.yaml")
    selected = validate_config(cfg)
    if args.profile:
        selected = args.profile
    if selected == "full" and args.profile != "full":
        raise SystemExit(
            "active_profile is full (about 50 million messages). "
            "Pass --profile full to run it."
        )
    generate(cfg, selected)


if __name__ == "__main__":
    main()
