"""B2 - Words and phrases that mark each fraud type, and how far they carry.

Top panels: log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi &
Quinn, "Fightin' Words", 2008). Each fraud type is compared with the rest of the
training messages, so a term scores high only when that type over-uses it.

    delta_w = log((y_aw + a_w) / (n_a + a - y_aw - a_w))
            - log((y_bw + a_w) / (n_b + a - y_bw - a_w))
    var_w   = 1/(y_aw + a_w) + 1/(n_a + a - y_aw - a_w)
            + 1/(y_bw + a_w) + 1/(n_b + a - y_bw - a_w)
    z_w     = delta_w / sqrt(var_w)

Terms are unigrams and bigrams, greedily de-duplicated so "virtual card" cannot
push both "virtual" and "card" out of the list.

Bottom panel: can those terms act as a fraud rule on their own? The vocabulary and
the term lists are built on the `train` split only; the rule (a message is scored
for a type by the number of its terms it contains, and takes the best-scoring type)
is then run on the `test` split. Recall and precision are reported per fraud type.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from _common import OUT_DIR, RISK_COLOR, apply_style, load_data, save_figure

PRIOR = 0.5
TOP_WORDS = 10
MIN_DF = 5

# Drop function words but keep negations: "not recognised" and "didn't" carry the fraud signal.
KEEP = {"not", "no", "nor", "never", "cannot", "none", "nothing",
        "didn't", "don't", "doesn't", "isn't", "wasn't", "aren't", "weren't",
        "can't", "couldn't", "won't", "wouldn't", "shouldn't", "haven't",
        "hasn't", "hadn't", "unable", "without"}
# A few auxiliaries missing from sklearn's list would otherwise form junk bigrams ("did not").
EXTRA_STOP = {"did", "does", "having", "had", "been", "being"}
STOP_WORDS = sorted((set(ENGLISH_STOP_WORDS) - KEEP) | EXTRA_STOP)


def build_counts(clean: pd.DataFrame, dim: pd.DataFrame):
    """Vocabulary and counts fit on train only, then applied to every row."""
    text = clean["text"].str.replace("recognize", "recognise", regex=False)
    train_mask = (clean["split"] == "train").to_numpy()
    vectorizer = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-z][a-z'\-]+\b",
        min_df=MIN_DF,
        stop_words=STOP_WORDS,
        strip_accents="unicode",
    )
    vectorizer.fit(text[train_mask])
    matrix = vectorizer.transform(text)
    vocab = np.array(vectorizer.get_feature_names_out())
    ftype = dict(zip(dim["intent"], dim["fraud_type"]))
    doc_type = clean["intent"].map(ftype)
    return matrix, vocab, doc_type, train_mask


def fightin_words(y_a, n_a, y_b, n_b):
    total = y_a + y_b
    prior = PRIOR * total / (n_a + n_b)
    p_a = (y_a + prior) / (n_a + PRIOR - y_a - prior)
    p_b = (y_b + prior) / (n_b + PRIOR - y_b - prior)
    delta = np.log(p_a) - np.log(p_b)
    var = (1.0 / (y_a + prior) + 1.0 / (n_a + PRIOR - y_a - prior)
           + 1.0 / (y_b + prior) + 1.0 / (n_b + PRIOR - y_b - prior))
    return delta, delta / np.sqrt(var)


def top_words(matrix, vocab, doc_type, fraud_type, train_mask, top=TOP_WORDS) -> pd.DataFrame:
    counts = np.asarray(matrix[train_mask].sum(axis=0)).ravel()
    in_group = (doc_type == fraud_type).to_numpy() & train_mask
    y_a = np.asarray(matrix[in_group].sum(axis=0)).ravel()
    y_b = counts - y_a
    n_a, n_b = int(y_a.sum()), int(y_b.sum())

    delta, z = fightin_words(y_a, n_a, y_b, n_b)
    table = pd.DataFrame({"word": vocab, "count_in_type": y_a.astype(int),
                          "count_elsewhere": y_b.astype(int),
                          "delta": delta, "z": z})
    table["rate_in_type"] = table["count_in_type"] / n_a * 1000
    table["rate_elsewhere"] = table["count_elsewhere"] / n_b * 1000
    table = table.sort_values("z", ascending=False)

    # Greedy de-duplication: a selected term blocks later terms sharing a token.
    chosen, used = [], set()
    for _, row in table.iterrows():
        tokens = set(str(row["word"]).split())
        if tokens & used:
            continue
        chosen.append(row)
        used |= tokens
        if len(chosen) == top:
            break
    return pd.DataFrame(chosen).reset_index(drop=True)


def rule_eval(matrix, vocab, doc_type, results, train_mask) -> pd.DataFrame:
    """Score every test message with each type's term list; take the best-scoring type."""
    test_idx = np.where(~train_mask)[0]
    col = {w: i for i, w in enumerate(vocab)}
    types = list(results)

    score = np.zeros((len(test_idx), len(types)))
    for t, fraud_type in enumerate(types):
        cols = [col[w] for w in results[fraud_type]["word"] if w in col]
        hits = np.asarray((matrix[test_idx][:, cols] > 0).sum(axis=1)).ravel()
        score[:, t] = hits

    best = score.argmax(axis=1)
    pred = np.array(types, dtype=object)[best]
    pred[score.max(axis=1) == 0] = None
    true = doc_type.iloc[test_idx].to_numpy()

    rows = []
    for fraud_type in types:
        is_true = true == fraud_type
        is_pred = pred == fraud_type
        tp = int((is_true & is_pred).sum())
        rows.append({
            "fraud_type": fraud_type,
            "n_true": int(is_true.sum()),
            "n_pred": int(is_pred.sum()),
            "tp": tp,
            "recall": tp / max(int(is_true.sum()), 1),
            "precision": tp / max(int(is_pred.sum()), 1),
        })
    table = pd.DataFrame(rows).set_index("fraud_type")
    table.loc["ALL"] = {
        "n_true": int(pd.notna(true).sum()),
        "n_pred": int(pd.notna(pred).sum()),
        "tp": int((pd.notna(true) & (pred == true)).sum()),
        "recall": float((pred == true)[pd.notna(true)].mean()),
        "precision": float((pred == true)[pd.notna(pred)].mean()),
    }
    return table


def draw(results, sizes, dim, rule):
    risk = (dim.dropna(subset=["fraud_type"]).drop_duplicates("fraud_type")
            .set_index("fraud_type")["risk_level"].to_dict())
    order = sorted(results, key=lambda f: (-sizes[f], f))

    fig = plt.figure(figsize=(17.5, 13.2))
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.72], hspace=0.42, wspace=0.26,
                            left=0.055, right=0.985, top=0.90, bottom=0.075)
    axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(3)]
    ax_rule = fig.add_subplot(grid[2, :])

    for ax, fraud_type in zip(axes, order):
        table = results[fraud_type].iloc[::-1]
        color = RISK_COLOR[risk[fraud_type]]
        ax.barh(range(len(table)), table["z"], color=color, height=0.72, alpha=0.9)
        ax.set_yticks(range(len(table)), table["word"], fontsize=9)
        ax.set_xlim(0, max(table["z"].max() * 1.18, 2.4))
        ax.bar_label(ax.containers[0], labels=[f"{z:.0f}" for z in table["z"]],
                     padding=2, fontsize=7.4, color="#444444")
        ax.axvline(1.96, color="#888888", lw=0.8, ls=":")
        ax.set_title(f"{fraud_type.replace('_', ' ')}  "
                     f"({risk[fraud_type]} risk, {sizes[fraud_type]:,} train messages)",
                     loc="left", fontsize=10.5)
        ax.grid(axis="x", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)

    # ---- rule evaluation --------------------------------------------------------
    bars = rule.drop(index="ALL").sort_values("recall", ascending=False)
    y = np.arange(len(bars))
    ax_rule.barh(y - 0.19, bars["recall"], height=0.36, color="#2c6e9b",
                 label="recall (share of that type caught)")
    ax_rule.barh(y + 0.19, bars["precision"], height=0.36, color="#8fb8d0",
                 label="precision (share of flags that are that type)")
    ax_rule.set_yticks(y, [t.replace("_", " ") for t in bars.index], fontsize=9)
    ax_rule.invert_yaxis()
    ax_rule.set_xlim(0, 1.06)
    ax_rule.set_xlabel("share of test messages")
    for yi, (idx, row) in enumerate(bars.iterrows()):
        ax_rule.text(row["recall"] + 0.012, yi - 0.19, f"{row['recall']:.0%}",
                     va="center", fontsize=7.6, color="#2c6e9b")
        ax_rule.text(row["precision"] + 0.012, yi + 0.19, f"{row['precision']:.0%}",
                     va="center", fontsize=7.6, color="#5b7f95")
    all_row = rule.loc["ALL"]
    ax_rule.set_title(
        "If the top-10 terms per type were the whole detector (terms fit on train, "
        f"scored on test): overall recall {all_row['recall']:.0%}, "
        f"precision {all_row['precision']:.0%}, "
        f"{int(all_row['n_pred'])} test messages flagged",
        loc="left", fontsize=10,
    )
    ax_rule.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax_rule.grid(axis="x", alpha=0.25, lw=0.6)
    ax_rule.set_axisbelow(True)

    fig.suptitle(
        "B2  Words and phrases that mark each fraud type (log-odds ratio with Dirichlet prior)\n"
        "Top: each type against the rest of the train messages; dotted line = z 1.96 (~95%). "
        "Bottom: the same terms used as a standalone rule on test",
        x=0.012, y=0.985, ha="left", fontsize=13, fontweight="bold",
    )
    fig.legend(handles=[Patch(color=RISK_COLOR["high"], label="high risk fraud type"),
                        Patch(color=RISK_COLOR["medium"], label="medium risk fraud type")],
               loc="upper right", bbox_to_anchor=(0.995, 0.945), ncol=2,
               frameon=False, fontsize=9)
    fig.text(0.5, 0.012,
             "log-odds z-score of each term in that fraud type against the rest of the "
             "train messages; the x scale differs by panel",
             ha="center", fontsize=8.8, color="#444444")
    return fig


def main() -> None:
    apply_style()
    clean, dim = load_data()
    matrix, vocab, doc_type, train_mask = build_counts(clean, dim)
    sizes = doc_type[train_mask].dropna().value_counts().to_dict()

    results = {ft: top_words(matrix, vocab, doc_type, ft, train_mask) for ft in sizes}
    rule = rule_eval(matrix, vocab, doc_type, results, train_mask)

    fig = draw(results, sizes, dim, rule)
    save_figure(fig, "B_keywords_by_fraud_type.png")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([t.assign(fraud_type=ft) for ft, t in results.items()],
              ignore_index=True).to_csv(OUT_DIR / "b2_top_words.csv", index=False)
    rule.to_csv(OUT_DIR / "b2_keyword_rule_eval.csv")
    (OUT_DIR / "b2_group_sizes.json").write_text(
        json.dumps({k: int(v) for k, v in sorted(sizes.items())}, indent=2))

    print("train group sizes:", {k: int(v) for k, v in sorted(sizes.items())})
    for ft, table in results.items():
        print(f"\n{ft}")
        print(table[["word", "rate_in_type", "rate_elsewhere", "z"]]
              .to_string(index=False, float_format=lambda v: f"{v:0.2f}"))
    print("\nkeyword-rule evaluation on test:\n", rule.round(3).to_string())


if __name__ == "__main__":
    main()
