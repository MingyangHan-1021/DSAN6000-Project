"""Shared paths and plotting style for the EDA notebooks (Wenhao Zhou, part B)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
FIG_DIR = REPO_ROOT / "figures"
OUT_DIR = REPO_ROOT / "eda" / "output"

RISK_ORDER = ["high", "medium", "low"]
RISK_COLOR = {"high": "#c0392b", "medium": "#e67e22", "low": "#7f8c8d"}

# Blue for a harmless mix-up inside one queue, red for a fraud signal that is lost.
DANGER_COLOR = {"harmless": "#4a90c2", "dangerous": "#c0392b"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a share; behaves near 0, unlike the normal approximation."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two read-only inputs from the data pipeline. Never write to them."""
    clean = pd.read_parquet(DATA_DIR / "banking77_clean.parquet")
    dim = pd.read_parquet(DATA_DIR / "dim_intent.parquet")
    return clean, dim


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig, name: str) -> Path:
    """Save under figures/B_*.png so each person's files stay separate."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    print(f"wrote {path}")
    return path
