"""Copy Banking77 locally, clean it, and build dim_intent.

Run from the repo root:

    python src/data/prepare_banking77.py
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.storage import REPO_ROOT as ROOT
from src.data.storage import load_config, load_yaml, locate

RISK_LEVELS = ("low", "medium", "high")
CLEAN_COLUMNS = ["text", "label", "intent", "split", "text_length"]
DIM_COLUMNS = [
    "intent_id",
    "intent",
    "fraud_relevant",
    "valid_from",
    "valid_to",
    "intent_set_version",
    "journey_stage",
    "fraud_type",
    "risk_level",
]


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"Missing input file: {path}")
    return path


def copy_inputs(cfg: dict) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    for key, raw_name in cfg["raw_names"].items():
        source = _require_file((ROOT / cfg["inputs"][key]).resolve())
        target = locate(f"raw/{raw_name}").require_local()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[key] = target
        print(f"copied {source} -> {target}")
    return copied


def load_messages(paths: dict[str, Path]) -> pd.DataFrame:
    frames = []
    expected = {"train": 10003, "test": 3080}
    for split in ("train", "test"):
        frame = pd.read_parquet(paths[split])
        missing = {"text", "label"} - set(frame.columns)
        if missing:
            raise SystemExit(f"{paths[split]} is missing columns: {sorted(missing)}")
        if len(frame) != expected[split]:
            raise SystemExit(
                f"{paths[split]} has {len(frame)} rows, expected {expected[split]}"
            )
        piece = frame.loc[:, ["text", "label"]].copy()
        piece["split"] = split
        frames.append(piece)
    combined = pd.concat(frames, ignore_index=True)
    combined["label"] = combined["label"].astype("int64")
    return combined


def clean_text(messages: pd.DataFrame) -> pd.DataFrame:
    original = messages["text"].astype("string")
    cleaned = original.str.replace(r"\s+", " ", regex=True).str.strip()
    empty = cleaned.isna() | (cleaned.str.len() == 0)
    if bool(empty.any()):
        raise SystemExit(f"{int(empty.sum())} messages are empty after whitespace cleaning")
    changed = int((original != cleaned).sum())
    print(f"whitespace cleaned on {changed} messages")
    out = messages.copy()
    out["text"] = cleaned.astype(str)
    out["text_length"] = cleaned.str.len().astype("int64")
    return out


def attach_intent_names(messages: pd.DataFrame, intents: pd.DataFrame) -> pd.DataFrame:
    if intents["label"].duplicated().any() or intents["intent"].duplicated().any():
        raise SystemExit("intents.csv has duplicate label or intent values")
    names = intents.set_index("label")["intent"]
    unknown = sorted(set(messages["label"]) - set(names.index))
    if unknown:
        raise SystemExit(f"labels missing from intents.csv: {unknown}")
    out = messages.copy()
    out["intent"] = out["label"].map(names)
    if out["intent"].isna().any():
        raise SystemExit("intent join left null names")
    return out


def validate_taxonomy(intent_names: set[str], taxonomy: dict) -> None:
    allowed = set(taxonomy.get("risk_levels", RISK_LEVELS))
    if allowed != set(RISK_LEVELS):
        raise SystemExit(f"risk_levels must be {list(RISK_LEVELS)}")
    roster = taxonomy.get("fraud_type_roster")
    per_intent = taxonomy.get("intents")
    if not isinstance(roster, dict) or not isinstance(per_intent, dict):
        raise SystemExit("intent_taxonomy.yaml needs fraud_type_roster and intents")

    tax_names = set(per_intent)
    missing = sorted(intent_names - tax_names)
    extra = sorted(tax_names - intent_names)
    if missing or extra:
        raise SystemExit(
            "intent names do not match intents.csv. "
            f"missing from taxonomy: {missing or 'none'}. "
            f"not in intents.csv: {extra or 'none'}."
        )

    listed: dict[str, str] = {}
    for fraud_type, spec in roster.items():
        if not isinstance(spec, dict):
            raise SystemExit(f"{fraud_type} roster entry must be a mapping")
        risk = spec.get("risk_level")
        if risk not in allowed or risk == "low":
            raise SystemExit(f"{fraud_type} risk_level must be medium or high, got {risk!r}")
        members = spec.get("intents")
        if not isinstance(members, list) or not members:
            raise SystemExit(f"{fraud_type} needs a non-empty intents list")
        for name in members:
            if name not in per_intent:
                raise SystemExit(
                    f"fraud type {fraud_type} lists {name!r}, which is not an intent name"
                )
            if name in listed:
                raise SystemExit(f"{name!r} is listed under both {listed[name]} and {fraud_type}")
            listed[name] = fraud_type

    for name, entry in per_intent.items():
        if not isinstance(entry, dict):
            raise SystemExit(f"{name} must be a mapping")
        for field in ("journey_stage", "fraud_type", "risk_level"):
            if field not in entry:
                raise SystemExit(f"{name} is missing {field}")
        stage = entry["journey_stage"]
        if not isinstance(stage, str) or not stage.strip():
            raise SystemExit(f"{name} has an empty journey_stage")
        risk = entry["risk_level"]
        if risk not in allowed:
            raise SystemExit(f"{name} has unknown risk_level {risk!r}")
        fraud_type = entry["fraud_type"]
        if fraud_type is None:
            if risk != "low":
                raise SystemExit(f"{name} has no fraud_type, so risk_level must be low")
            continue
        if fraud_type not in roster:
            raise SystemExit(f"{name} has unknown fraud_type {fraud_type!r}")
        if listed.get(name) != fraud_type:
            raise SystemExit(
                f"{name} is tagged {fraud_type}, but the roster does not list it there"
            )
        if risk != roster[fraud_type]["risk_level"]:
            raise SystemExit(
                f"{name} risk_level is {risk}, roster says {roster[fraud_type]['risk_level']}"
            )

    untagged = sorted(name for name in listed if per_intent[name]["fraud_type"] != listed[name])
    if untagged:
        raise SystemExit(f"roster and per-intent fraud_type disagree for {untagged}")


def write_dim_intent(intents: pd.DataFrame, taxonomy: dict, cfg: dict, path: Path) -> None:
    dim_cfg = cfg["dim_intent"]
    valid_from = datetime.fromisoformat(str(dim_cfg["valid_from"]))
    valid_to = datetime.fromisoformat(str(dim_cfg["valid_to"]))
    version = int(dim_cfg["intent_set_version"])
    ordered = intents.sort_values("label")
    per_intent = taxonomy["intents"]
    fraud_type = [per_intent[name]["fraud_type"] for name in ordered["intent"]]
    table = pa.table(
        {
            "intent_id": pa.array(ordered["label"].astype("int64"), type=pa.int64()),
            "intent": pa.array(ordered["intent"].astype(str), type=pa.string()),
            "fraud_relevant": pa.array([item is not None for item in fraud_type], type=pa.bool_()),
            "valid_from": pa.array([valid_from] * len(ordered), type=pa.timestamp("us")),
            "valid_to": pa.array([valid_to] * len(ordered), type=pa.timestamp("us")),
            "intent_set_version": pa.array([version] * len(ordered), type=pa.int64()),
            "journey_stage": pa.array(
                [per_intent[name]["journey_stage"] for name in ordered["intent"]],
                type=pa.string(),
            ),
            "fraud_type": pa.array(fraud_type, type=pa.string()),
            "risk_level": pa.array(
                [per_intent[name]["risk_level"] for name in ordered["intent"]],
                type=pa.string(),
            ),
        }
    )
    if table.column_names != DIM_COLUMNS:
        raise SystemExit(f"dim_intent columns drifted: {table.column_names}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    relevant = int(table.column("fraud_relevant").to_pylist().count(True))
    print(f"wrote {path} rows={table.num_rows} fraud_relevant={relevant}")


def main() -> None:
    cfg = load_config("prepare.yaml")
    taxonomy = load_yaml(ROOT / cfg["taxonomy"])
    copied = copy_inputs(cfg)
    intents = pd.read_csv(copied["intents"])
    if list(intents.columns) != ["label", "intent"]:
        raise SystemExit(f"intents.csv columns must be label, intent; got {list(intents.columns)}")
    messages = attach_intent_names(clean_text(load_messages(copied)), intents)
    validate_taxonomy(set(intents["intent"]), taxonomy)
    clean = messages.loc[:, CLEAN_COLUMNS]
    clean_path = locate(cfg["outputs"]["clean"]).require_local()
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(clean_path, index=False)
    print(
        f"wrote {clean_path} rows={len(clean)} "
        f"train={(clean['split'] == 'train').sum()} test={(clean['split'] == 'test').sum()} "
        f"intents={clean['intent'].nunique()}"
    )
    write_dim_intent(
        intents,
        taxonomy,
        cfg,
        locate(cfg["outputs"]["dim_intent"]).require_local(),
    )


if __name__ == "__main__":
    main()
