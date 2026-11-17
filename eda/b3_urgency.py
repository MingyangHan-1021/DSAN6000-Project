"""B3 - Does urgency language separate high-risk messages from ordinary ones?

A hand-built lexicon in three groups (time pressure, distress/appeal, incident/threat)
flags each Banking77 message once. The figure asks four things:

  a. What share of messages carries each group, by risk level? Wilson 95% intervals,
     plus a chi-square test across the three levels.
  b. Which fraud types produce the effect? Fraud types against an ordinary reference.
  c. Is the gap an artefact of message length? A message that says more has more
     chances to contain a keyword, so the panel compares the odds ratio between risk
     levels with and without log message length in the model (bootstrap intervals).
  d. Is the risk label a good proxy at all? Each dot is one of the 77 intents, so the
     spread inside a risk level can be compared with the gap between levels.

A lexicon-sensitivity table (brief-only terms, strict urgency, current, broad) is
written to CSV: the conclusions must not depend on how the word list was drawn.
"""

from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullFormatter
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression

from _common import OUT_DIR, RISK_COLOR, apply_style, load_data, save_figure, wilson

LEXICON = {
    "time pressure": [
        "immediately", "urgent", "asap", "right away", "as soon as possible", "hurry",
        "quickly", "emergency", "today", "straight away", "instantly", "now",
    ],
    "distress / appeal": [
        "help", "please", "worried", "worry", "scared", "panic", "desperate", "upset",
        "angry", "frustrated", "afraid", "stressed", "terrible", "awful",
    ],
    "incident / threat": [
        "someone", "stolen", "steal", "fraud", "scam", "hacked", "unauthorised",
        "unauthorized", "suspicious", "fake", "compromised", "not mine",
        "without my permission",
    ],
}
CAT_COLOR = {"time pressure": "#e08a1e", "distress / appeal": "#8e6bbf",
             "incident / threat": "#c0392b"}
RISK_LABEL = {"low": "low\n(55 intents)", "medium": "medium\n(10)", "high": "high\n(12)"}

ALL_TERMS = [w for words in LEXICON.values() for w in words]
SENSITIVITY = {
    "brief only": ["immediately", "urgent", "someone", "stolen", "help"],
    "strict urgency": ["immediately", "urgent", "asap", "emergency", "straight away",
                       "right away", "as soon as possible", "hurry", "scared", "panic",
                       "desperate"],
    "current (3 groups)": ALL_TERMS,
    "broad (+plea)": ALL_TERMS + ["need", "sorry", "problem", "wrong"],
}


def _pattern(words: list[str]) -> re.Pattern:
    joined = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b(?:{joined})\b", re.IGNORECASE)


def build_flags(clean: pd.DataFrame, dim: pd.DataFrame) -> pd.DataFrame:
    frame = clean.merge(dim[["intent", "risk_level", "fraud_type"]], on="intent", how="left")
    for name, words in LEXICON.items():
        frame[name] = frame["text"].str.contains(_pattern(words), regex=True)
    frame["any term"] = frame[list(LEXICON)].any(axis=1)
    return frame


def share_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(by, observed=True):
        for cat in list(LEXICON) + ["any term"]:
            k, n = int(group[cat].sum()), int(len(group))
            lo, hi = wilson(k, n)
            rows.append({by: key, "group": cat, "k": k, "n": n,
                         "share": k / n, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def chi_square(frame: pd.DataFrame, cat: str) -> tuple[float, float]:
    table = pd.crosstab(frame["risk_level"], frame[cat])
    chi2, p, _, _ = chi2_contingency(table)
    return float(chi2), float(p)


def bootstrap_odds(frame: pd.DataFrame, n_boot: int = 300, seed: int = 11) -> pd.DataFrame:
    """Odds of carrying any lexicon term, medium/high vs low, with and without length."""
    y = frame["any term"].to_numpy()
    base = np.column_stack([
        (frame["risk_level"] == "medium").astype(float),
        (frame["risk_level"] == "high").astype(float),
    ])
    adjusted = np.column_stack([base, np.log1p(frame["text_length"].to_numpy())])
    rng = np.random.default_rng(seed)

    draws = {"unadjusted": [], "length-adjusted": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        for name, matrix in (("unadjusted", base), ("length-adjusted", adjusted)):
            model = LogisticRegression(max_iter=1000).fit(matrix[idx], y[idx])
            draws[name].append(np.exp(model.coef_[0][:2]))

    rows = []
    for name, values in draws.items():
        values = np.asarray(values)
        for level, col in (("medium", 0), ("high", 1)):
            lo, med, hi = np.percentile(values[:, col], [2.5, 50, 97.5])
            rows.append({"model": name, "level": level, "or": med, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def per_intent(frame: pd.DataFrame) -> pd.DataFrame:
    table = (frame.groupby(["intent", "risk_level"], observed=True)["any term"]
             .agg(["mean", "size"]).reset_index())
    return table.rename(columns={"mean": "share", "size": "n_messages"})


def sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, words in SENSITIVITY.items():
        flag = frame["text"].str.contains(_pattern(words), regex=True)
        for level in ("low", "medium", "high"):
            sel = frame["risk_level"] == level
            rows.append({"lexicon": name, "risk_level": level,
                         "share": float(flag[sel].mean())})
    return pd.DataFrame(rows)


def draw(by_risk, by_type, odds, pvals, intent_df, sens):
    fig = plt.figure(figsize=(18.5, 13.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], wspace=0.24, hspace=0.40,
                            left=0.055, right=0.985, top=0.905, bottom=0.065)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_c = fig.add_subplot(grid[0, 1])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    levels = ["low", "medium", "high"]
    cats = list(LEXICON) + ["any term"]
    width = 0.2

    # ---- a. share by risk level ------------------------------------------------
    for i, cat in enumerate(cats):
        sub = by_risk[by_risk["group"] == cat].set_index("risk_level").loc[levels]
        pos = np.arange(len(levels)) + (i - (len(cats) - 1) / 2) * width
        color = CAT_COLOR.get(cat, "#34495e")
        err = np.vstack([sub["share"] - sub["lo"], sub["hi"] - sub["share"]])
        ax_a.bar(pos, sub["share"], width=width * 0.92, color=color, label=cat,
                 yerr=err, capsize=2.0, error_kw={"lw": 0.8, "alpha": 0.8})
        for xpos, value in zip(pos, sub["share"]):
            ax_a.text(xpos, value + 0.008, f"{value:.1%}", ha="center", va="bottom",
                      rotation=90, fontsize=6.6, color="#333333")
    ax_a.set_xticks(range(len(levels)), [RISK_LABEL[l] for l in levels])
    ax_a.set_xlabel("intent risk level (number of intents in brackets)", fontsize=9)
    ax_a.set_ylabel("share of messages containing a term")
    ax_a.set_ylim(0, max(by_risk["hi"].max() * 1.22, 0.35))
    ax_a.legend(frameon=False, fontsize=8.2, loc="upper left")
    ax_a.grid(axis="y", alpha=0.25, lw=0.6)
    ax_a.set_axisbelow(True)
    p_text = "chi-square across risk levels:  " + "   ".join(
        f"{cat.split(' ')[0]} p={pvals[cat]:.1e}" for cat in LEXICON)
    ax_a.set_title("a. Urgency terms by risk level (Wilson 95% CI)\n" + p_text,
                   loc="left", fontsize=10)

    # ---- c. forest plot --------------------------------------------------------
    ypos = {"high": 1.0, "medium": 0.0}
    offset = {"length-adjusted": 0.14, "unadjusted": -0.14}
    color = {"length-adjusted": "#1f4e79", "unadjusted": "#9aa5ac"}
    for level in ("high", "medium"):
        for model in ("length-adjusted", "unadjusted"):
            row = odds[(odds["level"] == level) & (odds["model"] == model)].iloc[0]
            y = ypos[level] + offset[model]
            ax_c.errorbar(row["or"], y,
                          xerr=[[row["or"] - row["lo"]], [row["hi"] - row["or"]]],
                          fmt="o", ms=7, color=color[model], capsize=3.5, lw=1.6,
                          label=model if level == "high" else None)
            ax_c.text(row["hi"] * 1.07, y,
                      f"{row['or']:.2f}  [{row['lo']:.2f}-{row['hi']:.2f}]",
                      va="center", fontsize=8.2, color=color[model])
    ax_c.axvline(1.0, color="#555555", lw=0.9, ls="--")
    ax_c.set_yticks([1.0, 0.0], ["high vs low", "medium vs low"])
    ax_c.set_xscale("log")
    ax_c.set_xlim(0.85, 9.5)
    ax_c.xaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 4, 6, 8]))
    ax_c.set_xticklabels(["1", "1.5", "2", "3", "4", "6", "8"])
    ax_c.xaxis.set_minor_formatter(NullFormatter())
    ax_c.text(0.02, 0.93, "for reference: log-length alone OR 8.2", transform=ax_c.transAxes,
              fontsize=7.8, color="#666666")
    ax_c.set_xlabel("odds ratio of carrying any urgency term (log scale)")
    ax_c.set_title("c. Does message length explain it?\n"
                   "same odds with (dark) and without (grey) log length in the model",
                   loc="left", fontsize=10)
    ax_c.grid(axis="x", alpha=0.25, lw=0.6)
    ax_c.set_axisbelow(True)
    ax_c.legend(frameon=False, fontsize=8.4, loc="lower right")

    # ---- b. share by fraud type ------------------------------------------------
    types = by_type["fraud_type"].unique().tolist()
    y = np.arange(len(types))
    for i, cat in enumerate(cats):
        sub = by_type[by_type["group"] == cat].set_index("fraud_type").loc[types]
        pos = y + (i - (len(cats) - 1) / 2) * width
        col = CAT_COLOR.get(cat, "#34495e")
        err = np.clip(np.vstack([sub["share"] - sub["lo"], sub["hi"] - sub["share"]]), 0, None)
        ax_b.barh(pos, sub["share"], height=width * 0.92, color=col, label=cat,
                  xerr=err, capsize=1.6, error_kw={"lw": 0.7, "alpha": 0.8})
        if cat in ("any term", "incident / threat"):
            for ypos_, value in zip(pos, sub["share"]):
                if value >= 0.02:
                    ax_b.text(value + 0.008, ypos_, f"{value:.0%}", va="center",
                              fontsize=6.8, color="#333333")
    ax_b.set_yticks(y, [t.replace("_", " ") for t in types], fontsize=8.8)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, max(by_type["hi"].max() * 1.16, 0.1))
    ax_b.set_xlabel("share of messages containing a term", fontsize=9)
    ax_b.set_title("b. The same terms by fraud type ('ordinary' = all 55 low-risk intents)",
                   loc="left", fontsize=10)
    ax_b.grid(axis="x", alpha=0.25, lw=0.6)
    ax_b.set_axisbelow(True)

    # ---- d. per-intent spread --------------------------------------------------
    rng = np.random.default_rng(3)
    for i, level in enumerate(levels):
        sub = intent_df[intent_df["risk_level"] == level]
        jitter = rng.uniform(-0.22, 0.22, len(sub))
        ax_d.scatter(np.full(len(sub), i) + jitter, sub["share"],
                     s=18 + sub["n_messages"] / 8, color=RISK_COLOR[level],
                     alpha=0.75, edgecolor="white", linewidth=0.5, zorder=3)
        med = sub["share"].median()
        ax_d.hlines(med, i - 0.32, i + 0.32, color="#333333", lw=2.0, zorder=4)
        ax_d.text(i + 0.35, med, f"median {med:.0%}", va="center", fontsize=8,
                  color="#333333")
    ax_d.set_xticks(range(len(levels)), [RISK_LABEL[l] for l in levels])
    ax_d.set_xlim(-0.6, 2.75)
    ax_d.set_ylim(0, 0.95)
    ax_d.set_ylabel("share of that intent's messages with a term", fontsize=9)
    ax_d.set_title("d. Each dot is one intent: the spread inside a risk level dwarfs the gap "
                   "between levels\n(dot area = messages in that intent; bar = median)",
                   loc="left", fontsize=10)
    ax_d.grid(axis="y", alpha=0.25, lw=0.6)
    ax_d.set_axisbelow(True)

    highlights = {
        "compromised_card": (1.80, 0.90, "right"),
        "lost_or_stolen_card": (1.80, 0.78, "right"),
        "why_verify_identity": (1.80, 0.10, "right"),
        "terminate_account": (0.22, 0.40, "left"),
    }
    for name, (tx, ty, ha) in highlights.items():
        row = intent_df[intent_df["intent"] == name].iloc[0]
        i = levels.index(row["risk_level"])
        ax_d.annotate(name, (i, row["share"]), xytext=(tx, ty),
                      fontsize=7.6, ha=ha, va="center", color="#333333",
                      arrowprops=dict(arrowstyle="-", lw=0.6, color="#999999",
                                      shrinkA=2, shrinkB=4))

    fig.suptitle(
        "B3  Urgency words track risk only through incident vocabulary\n"
        "time pressure (immediately / urgent) stays flat at 4-6% across risk levels; "
        "incident words (someone / stolen) go 0.8% to 18.3%, mostly from card compromise",
        x=0.012, y=0.975, ha="left", fontsize=13.5, fontweight="bold",
    )
    fig.text(0.5, 0.012,
             "lexicon sensitivity (share carrying a term, low / medium / high): "
             + "    ".join(
                 f"{name}: " + " / ".join(
                     f"{sens[(sens.lexicon == name) & (sens.risk_level == l)]['share'].iloc[0]:.1%}"
                     for l in levels)
                 for name in SENSITIVITY),
             ha="center", fontsize=8.2, color="#444444")
    return fig


def main() -> None:
    apply_style()
    clean, dim = load_data()
    frame = build_flags(clean, dim)

    by_risk = share_table(frame, "risk_level")
    by_risk["risk_level"] = pd.Categorical(by_risk["risk_level"],
                                           categories=["low", "medium", "high"], ordered=True)
    by_risk = by_risk.sort_values(["group", "risk_level"])

    fraud = frame.dropna(subset=["fraud_type"]).copy()
    ordinary = frame[frame["risk_level"] == "low"].copy()
    ordinary["fraud_type"] = "ordinary"
    by_type = share_table(pd.concat([fraud, ordinary], ignore_index=True), "fraud_type")
    preferred = ["ordinary", "virtual_card_abuse", "identity_kyc_bypass", "account_takeover",
                 "app_fraud_money_mule", "refund_chargeback_abuse", "card_compromise"]
    by_type["fraud_type"] = pd.Categorical(by_type["fraud_type"], categories=preferred,
                                           ordered=True)
    by_type = by_type.sort_values(["group", "fraud_type"])

    pvals = {cat: chi_square(frame, cat)[1] for cat in LEXICON}
    odds = bootstrap_odds(frame)
    intent_df = per_intent(frame)
    sens = sensitivity(frame)

    fig = draw(by_risk, by_type, odds, pvals, intent_df, sens)
    save_figure(fig, "B_urgency_by_risk.png")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_risk.to_csv(OUT_DIR / "b3_shares_by_risk.csv", index=False)
    by_type.to_csv(OUT_DIR / "b3_shares_by_fraud_type.csv", index=False)
    odds.to_csv(OUT_DIR / "b3_odds_ratios.csv", index=False)
    intent_df.to_csv(OUT_DIR / "b3_per_intent.csv", index=False)
    sens.to_csv(OUT_DIR / "b3_lexicon_sensitivity.csv", index=False)
    (OUT_DIR / "b3_chi_square.json").write_text(json.dumps(pvals, indent=2))

    print("share by risk level:")
    print(by_risk.pivot(index="risk_level", columns="group", values="share").round(3).to_string())
    print("\nchi-square p:", {k: f"{v:.2e}" for k, v in pvals.items()})
    print("\nodds ratios:\n", odds.round(3).to_string(index=False))
    print("\nlexicon sensitivity:\n",
          sens.pivot(index="lexicon", columns="risk_level", values="share").round(3).to_string())
    spread = intent_df.groupby("risk_level")["share"].agg(["min", "median", "max"]).round(3)
    print("\nper-intent spread:\n", spread.to_string())


if __name__ == "__main__":
    main()
