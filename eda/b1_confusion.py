"""B1 - Intent confusion heat map and where the fraud signal is lost.

TF-IDF + multinomial logistic regression, trained on the `train` split only and
scored on the `test` split.

Panels
  1. 77 x 77 view, intents ordered by journey_stage, so blocks on the diagonal are
     confusions inside one stage and off-block cells are cross-stage errors.
  2. Detail view of the intents that take part in the 20 most frequent mix-ups.
  3. Those 20 pairs, split into "dangerous" (fraud queue lost or split) and
     "harmless" (both intents feed the same queue).
  4. Fraud leakage: for every message whose true intent is fraud-relevant, where
     the prediction actually sends it - same fraud team, another fraud team, or
     an ordinary servicing queue. Rates carry Wilson 95% intervals, and the pair
     ranking is resampled (bootstrap) because several counts are only 3-5.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from _common import DANGER_COLOR, OUT_DIR, apply_style, load_data, save_figure, wilson

TOP_PAIRS = 20
CLIP = 8  # colour scale upper end; the diagonal is far above it
LEAK_COLOR = {"same_team": "#95a5a6", "wrong_team": "#e67e22", "ordinary": "#c0392b"}
RISK_RANK = {"high": 0, "medium": 1, "low": 2}


def fit_classifier(clean: pd.DataFrame):
    train = clean[clean["split"] == "train"]
    test = clean[clean["split"] == "test"]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, min_df=2, strip_accents="unicode"
    )
    x_train = vectorizer.fit_transform(train["text"])
    x_test = vectorizer.transform(test["text"])
    model = LogisticRegression(max_iter=2000, C=10.0, random_state=42).fit(
        x_train, train["label"]
    )
    return train, test, model.predict(x_test), model


def label_pairs(cm: np.ndarray, names: list[str], top: int = TOP_PAIRS) -> pd.DataFrame:
    """Rank undirected intent pairs by how often the test set mixed them up."""
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            forward, backward = int(cm[i, j]), int(cm[j, i])
            if forward + backward == 0:
                continue
            rows.append({"i": i, "j": j, "a": names[i], "b": names[j],
                         "count": forward + backward, "a_to_b": forward, "b_to_a": backward})
    pairs = pd.DataFrame(rows).sort_values(["count", "i", "j"], ascending=[False, True, True])
    return pairs.head(top).reset_index(drop=True)


def bootstrap_pairs(test, pred, n_intents: int, n_boot: int = 500,
                    top: int = TOP_PAIRS) -> pd.DataFrame:
    """Resample the test set to see how stable the top-pair ranking is."""
    rng = np.random.default_rng(7)
    flat = test["label"].to_numpy() * n_intents + pred
    tri = np.triu_indices(n_intents, 1)
    n_pairs = len(tri[0])
    vals = np.empty((n_boot, n_pairs), dtype=np.int32)
    for b in range(n_boot):
        sample = flat[rng.integers(0, len(flat), len(flat))]
        cmb = np.bincount(sample, minlength=n_intents ** 2).reshape(n_intents, n_intents)
        vals[b] = cmb[tri] + cmb.T[tri]

    in_top = np.zeros(n_pairs)
    for b in range(n_boot):
        in_top[np.argsort(-vals[b])[:top]] += 1

    return pd.DataFrame({
        "i": tri[0], "j": tri[1],
        "stab": in_top / n_boot,
        "lo": np.percentile(vals, 2.5, axis=0),
        "hi": np.percentile(vals, 97.5, axis=0),
    })


def classify_pair(row, ftype: dict) -> str:
    fa, fb = ftype[row["a"]], ftype[row["b"]]
    if pd.isna(fa) and pd.isna(fb):
        return "harmless"
    if fa == fb:
        return "harmless"
    return "dangerous"


def fraud_leakage(cm: np.ndarray, dim: pd.DataFrame) -> pd.DataFrame:
    """For each true fraud type, where do its test messages end up?"""
    name = dim.sort_values("intent_id")["intent"].tolist()
    ftype = dict(zip(name, dim.sort_values("intent_id")["fraud_type"]))
    rows = []
    for i, true_name in enumerate(name):
        if pd.isna(ftype[true_name]):
            continue
        for j, pred_name in enumerate(name):
            n = int(cm[i, j])
            if n == 0:
                continue
            if ftype[true_name] == ftype[pred_name]:
                bucket = "same_team"
            elif pd.isna(ftype[pred_name]):
                bucket = "ordinary"
            else:
                bucket = "wrong_team"
            rows.append({"fraud_type": ftype[true_name], "true": true_name,
                         "pred": pred_name, "n": n, "bucket": bucket})
    detail = pd.DataFrame(rows)
    table = (
        detail.pivot_table(index="fraud_type", columns="bucket", values="n",
                           aggfunc="sum", fill_value=0)
        .reindex(columns=["same_team", "wrong_team", "ordinary"], fill_value=0)
    )
    table["n"] = table.sum(axis=1)
    table = table.sort_values("n", ascending=False)
    return table, detail


def draw(pairs, detail_cm, overview_cm, names, dim, metrics, leak_table, order, stage_names):
    ftype = dict(zip(names, dim.sort_values("intent_id")["fraud_type"]))
    risk = dict(zip(names, dim.sort_values("intent_id")["risk_level"]))
    off_diag = detail_cm.sum(axis=1) - np.diag(detail_cm)
    # Clip the colour scale well below the diagonal so off-diagonal structure shows.
    vmax = np.log1p(CLIP)

    fig = plt.figure(figsize=(23, 15.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.34,
                             left=0.035, right=0.985, top=0.925, bottom=0.055)
    left = outer[0].subgridspec(2, 1, height_ratios=[0.95, 1.0], hspace=0.20)
    right = outer[1].subgridspec(2, 1, height_ratios=[1.62, 1.0], hspace=0.34)
    ax_o, ax_h, ax_b, ax_l = (fig.add_subplot(s) for s in
                              (left[0], left[1], right[0], right[1]))

    # ---- 1. full 77 x 77 view, ordered by journey stage -------------------------
    plot_o = np.log1p(overview_cm[np.ix_(order, order)])
    im_o = ax_o.imshow(plot_o, cmap="Blues", vmin=0, vmax=vmax, aspect="equal",
                       interpolation="nearest")

    stage_pos = {}
    for pos, k in enumerate(order):
        stage_pos.setdefault(stage_names[k], []).append(pos)
    bounds = [0] + [max(v) + 0.5 for v in stage_pos.values()]
    for b in bounds[1:-1]:
        ax_o.axhline(b, color="white", lw=0.9)
        ax_o.axvline(b, color="white", lw=0.9)
    ax_o.set_yticks([np.mean(v) for v in stage_pos.values()])
    ax_o.set_yticklabels(list(stage_pos), fontsize=5.6)
    ax_o.set_xticks([])
    ax_o.set_xlabel("predicted intent (same stage order)", fontsize=8)
    ax_o.set_ylabel("true intent, grouped by journey stage", fontsize=8)
    ax_o.set_title("All 77 intents, ordered by journey stage\n"
                   "blocks on the diagonal = mix-ups inside one stage",
                   loc="left", fontsize=10)

    # ---- 2. detail view ---------------------------------------------------------
    involved = sorted({int(v) for v in pd.concat([pairs["i"], pairs["j"]])})
    if len(involved) > 26:
        involved = sorted(involved, key=lambda k: -off_diag[k])[:26]
    involved.sort(key=lambda k: (stage_names[k], RISK_RANK[risk[names[k]]], -off_diag[k]))

    sub = detail_cm[np.ix_(involved, involved)]
    short = [names[k] for k in involved]
    plot = np.log1p(sub)
    ax_h.imshow(plot, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    ax_h.set_xticks(range(len(short)), short, rotation=90, fontsize=6.2)
    ax_h.set_yticks(range(len(short)), short, fontsize=6.2)
    for tick, k in zip(ax_h.get_yticklabels(), involved):
        tick.set_color({"high": "#c0392b", "medium": "#e67e22", "low": "#4d4d4d"}[risk[names[k]]])
    for r in range(len(short)):
        for c in range(len(short)):
            if sub[r, c] >= 2:
                ax_h.text(c, r, int(sub[r, c]), ha="center", va="center", fontsize=5.8,
                          color="white" if plot[r, c] >= vmax * 0.82 else "#333333")
    for _, row in pairs.iterrows():
        for a, b in ((row["i"], row["j"]), (row["j"], row["i"])):
            if a in involved and b in involved:
                r, c = involved.index(a), involved.index(b)
                ax_h.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                         edgecolor=DANGER_COLOR["dangerous"], lw=1.0, ls=":"))
    ax_h.set_xlabel("predicted intent")
    ax_h.set_ylabel("true intent")
    ax_h.set_title(f"Intents in the top {len(pairs)} mix-ups, ordered by stage then risk\n"
                   "cell = test messages; 1-message cells unlabelled; dotted = top pairs",
                   loc="left", fontsize=10)
    cb = fig.colorbar(im_o, ax=ax_o, fraction=0.030, pad=0.025)
    cb.set_label(f"log1p(test messages), clipped at {CLIP}", fontsize=7.5)

    # ---- 3. ranked pairs --------------------------------------------------------
    pairs = pairs.assign(kind=[classify_pair(r, ftype) for _, r in pairs.iterrows()])
    ordered = pairs.sort_values("count")
    xerr = np.vstack([ordered["count"] - ordered["lo"], ordered["hi"] - ordered["count"]]).clip(min=0)
    bars_b = ax_b.barh(range(len(ordered)), ordered["count"],
              color=[DANGER_COLOR[k] for k in ordered["kind"]], height=0.62,
              xerr=xerr, ecolor="#4d4d4d", capsize=2.2, error_kw={"lw": 0.8, "alpha": 0.85})
    ax_b.set_yticks(range(len(ordered)),
                    [f"{r.a}\n\u21c4  {r.b}" for r in ordered.itertuples()],
                    fontsize=6.9, linespacing=1.3)
    ax_b.set_xlabel("test messages confused across the pair (both directions)")
    n_d = int((pairs["kind"] == "dangerous").sum())
    ax_b.set_title(f"Top {len(pairs)} confused intent pairs: "
                   f"{n_d} dangerous (red), {len(pairs) - n_d} harmless (blue)\n"
                   "bar = test messages, whisker = 95% bootstrap interval, "
                   "right column = share of 500 resamples keeping the pair in the top 20",
                   loc="left", fontsize=9.5)
    # A separate right-hand column keeps the bootstrap stability readable.
    xmax = ordered["hi"].max()
    ax_b.set_xlim(0, xmax * 1.20)
    ax_b.axvline(xmax * 1.04, color="#b0b0b0", lw=0.7, ls=":")
    for yi, r in enumerate(ordered.itertuples()):
        ax_b.text(xmax * 1.17, yi, f"{r.stab:.0%}", va="center", ha="right",
                  fontsize=6.8, color="#555555")
    ax_b.grid(axis="x", alpha=0.25, lw=0.6)
    ax_b.set_axisbelow(True)
    ax_b.legend(handles=[Patch(color=DANGER_COLOR["dangerous"], label="dangerous"),
                         Patch(color=DANGER_COLOR["harmless"], label="harmless")],
                loc="upper left", bbox_to_anchor=(0.0, -0.055), ncol=2,
                frameon=False, fontsize=8)

    # ---- 4. fraud leakage -------------------------------------------------------
    order_l = list(leak_table.index)
    y = np.arange(len(order_l))
    left_cum = np.zeros(len(order_l))
    for bucket in ("same_team", "wrong_team", "ordinary"):
        vals = leak_table[bucket].to_numpy()
        ax_l.barh(y, vals, left=left_cum, height=0.66, color=LEAK_COLOR[bucket],
                  label=bucket.replace("_", " "))
        left_cum += vals
    ax_l.set_yticks(y, [t.replace("_", " ") for t in order_l], fontsize=8)
    ax_l.invert_yaxis()
    for yi, (idx, row) in enumerate(leak_table.iterrows()):
        ax_l.text(row["n"] + leak_table["n"].max() * 0.012, yi,
                  f"n={int(row['n'])}\n{row['p']:.1%} [{row['lo']:.0%}-{row['hi']:.0%}] to ordinary",
                  va="center", fontsize=7.2, linespacing=1.35,
                  color=LEAK_COLOR["ordinary"] if row["p"] > 0.08 else "#444444")
    total = int(leak_table["n"].sum())
    leak_all = int(leak_table["ordinary"].sum())
    wrong_all = int(leak_table["wrong_team"].sum())
    lo_all, hi_all = wilson(leak_all, total)
    ax_l.set_xlim(0, leak_table["n"].max() * 1.52)
    ax_l.set_xlabel("test messages whose true intent is fraud-relevant")
    ax_l.set_title(f"Fraud leakage: {leak_all} of {total} risky test messages "
                   f"({leak_all / total:.1%}, 95% CI {lo_all:.1%}-{hi_all:.1%}) leave the fraud workflow\n"
                   f"another {wrong_all} ({wrong_all / total:.1%}) land in a different fraud team; "
                   f"intervals are Wilson, n is small",
                   loc="left", fontsize=10)
    ax_l.legend(loc="lower right", frameon=False, fontsize=7.6, title="predicted queue",
                title_fontsize=7.6)
    ax_l.grid(axis="x", alpha=0.25, lw=0.6)
    ax_l.set_axisbelow(True)

    fig.suptitle(
        f"B1  Intent confusion: TF-IDF + logistic regression   |   "
        f"test accuracy {metrics['accuracy']:.3f}, macro-F1 {metrics['macro_f1']:.3f} "
        f"(77 intents, {metrics['n_test']} held-out messages)   |   "
        f"fraud leakage {leak_all / total:.1%} [{lo_all:.1%}-{hi_all:.1%}] of risky messages",
        x=0.012, ha="left", fontsize=14, fontweight="bold",
    )
    return fig, pairs


def main() -> None:
    apply_style()
    clean, dim = load_data()
    train, test, pred, model = fit_classifier(clean)

    dim = dim.sort_values("intent_id").reset_index(drop=True)
    names = dim["intent"].tolist()
    stage_names = dim["journey_stage"].tolist()

    cm = confusion_matrix(test["label"], pred, labels=list(range(len(names))))
    off_diag = cm.sum(axis=1) - np.diag(cm)
    stage_order = sorted(set(stage_names))
    order = sorted(
        range(len(names)),
        key=lambda k: (stage_order.index(stage_names[k]),
                       RISK_RANK[dim["risk_level"][k]], -off_diag[k]),
    )

    metrics = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "accuracy": float((pred == test["label"].to_numpy()).mean()),
        "macro_f1": float(f1_score(test["label"], pred, average="macro")),
        "weighted_f1": float(f1_score(test["label"], pred, average="weighted")),
        "n_features": int(model.n_features_in_),
    }
    pairs = label_pairs(cm, names)
    pairs = pairs.merge(bootstrap_pairs(test, pred, len(names)), on=["i", "j"], how="left")
    leak_table, leak_detail = fraud_leakage(cm, dim)
    leak_table["p"] = leak_table["ordinary"] / leak_table["n"]
    bounds_ci = [wilson(int(r.ordinary), int(r.n)) for r in leak_table.itertuples()]
    leak_table["lo"] = [b[0] for b in bounds_ci]
    leak_table["hi"] = [b[1] for b in bounds_ci]

    fig, pairs = draw(pairs, cm, cm, names, dim, metrics, leak_table, order, stage_names)
    save_figure(fig, "B_confusion_heatmap.png")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT_DIR / "b1_top_pairs.csv", index=False)
    leak_table.to_csv(OUT_DIR / "b1_leakage_by_fraud_type.csv")
    leak_detail.sort_values("n", ascending=False).to_csv(
        OUT_DIR / "b1_leakage_detail.csv", index=False)
    (OUT_DIR / "b1_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(pairs[["a", "b", "count", "lo", "hi", "stab", "kind"]].to_string(index=False))
    print("\nleakage by fraud type:\n", leak_table.to_string())
    total = leak_table["n"].sum()
    print(f"\nrisky test messages {total}, to ordinary {leak_table['ordinary'].sum()} "
          f"({leak_table['ordinary'].sum() / total:.1%}), "
          f"wrong fraud team {leak_table['wrong_team'].sum()} "
          f"({leak_table['wrong_team'].sum() / total:.1%})")


if __name__ == "__main__":
    main()
