"""Ticket 1 diagnosis: why does the shipped baseline (pipeline/baseline.py) miss the
reference held-out F1 (0.7574221578566256 +/- 0.001, see configs/project_contract.json)?

This script is diagnostic only. It sweeps single settings and a few common combinations,
reporting dev *and* heldout F1 for each, so we can see which lever(s) close the gap.
It does NOT change the frozen baseline: pipeline/baseline.py stays pure-default and is
never re-picked using heldout.

Two passes:
  1. One-factor-at-a-time from the shipped baseline (TfidfVectorizer defaults +
     LogisticRegression(max_iter=1000, random_state=3102)), varying exactly one setting.
     Includes a genuinely-default LogisticRegression (max_iter=100) row to check whether
     non-convergence explains part of the gap (n_iter_ is reported for that reason).
  2. A grid over min_df x sublinear_tf x ngram_range x C (48 combinations), to see which
     land the heldout F1 inside the reference tolerance. Tokenizer/stopwords/sklearn-version
     differences are explicitly out of scope here -- left for Ticket 2 or as a limitation.

Output: results/ticket1_probe.csv
"""
import itertools
import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "train.csv"
SPLIT_JSON = ROOT / "data" / "split_indices.json"
CONTRACT_JSON = ROOT / "configs" / "project_contract.json"
OUT_CSV = ROOT / "results" / "ticket1_probe.csv"

RANDOM_STATE = 3102


def load_split(df: pd.DataFrame, ids: list) -> pd.DataFrame:
    split_df = df[df["id"].isin(ids)].copy()
    split_df = split_df.set_index("id").loc[ids].reset_index()
    return split_df


def run_variant(label, group, tfidf_kwargs, logreg_kwargs, train_df, dev_df, heldout_df, reference_f1, tolerance):
    vectorizer = TfidfVectorizer(**tfidf_kwargs)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_dev = vectorizer.transform(dev_df["text"])
    X_heldout = vectorizer.transform(heldout_df["text"])

    clf = LogisticRegression(random_state=RANDOM_STATE, **logreg_kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X_train, train_df["target"])
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    dev_f1 = f1_score(dev_df["target"], clf.predict(X_dev), pos_label=1)
    heldout_pred = clf.predict(X_heldout)
    heldout_f1 = f1_score(heldout_df["target"], heldout_pred, pos_label=1)
    gap = heldout_f1 - reference_f1

    return {
        "group": group,
        "label": label,
        "tfidf_params": tfidf_kwargs,
        "logreg_params": {"random_state": RANDOM_STATE, **logreg_kwargs},
        "n_iter": int(clf.n_iter_.max()),
        "converged": converged,
        "dev_f1_target_1": round(dev_f1, 4),
        "heldout_f1_target_1": round(heldout_f1, 4),
        "heldout_gap_vs_reference": round(gap, 4),
        "within_tolerance": abs(gap) <= tolerance,
    }


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split = json.load(f)
    with open(CONTRACT_JSON, "r", encoding="utf-8") as f:
        contract = json.load(f)

    reference_f1 = contract["reference_baseline_f1"]
    tolerance = contract["tolerance"]

    train_df = load_split(df, split["train_ids"])
    dev_df = load_split(df, split["dev_ids"])
    heldout_df = load_split(df, split["heldout_ids"])

    # Pass 1: one factor at a time, off the shipped baseline
    # (TfidfVectorizer() defaults + LogisticRegression(max_iter=1000, random_state=3102)).
    single_factor = [
        ("default (sklearn max_iter=100)", {}, {}),
        ("max_iter=1000  <- shipped baseline", {}, {"max_iter": 1000}),
        ("C=10", {}, {"max_iter": 1000, "C": 10}),
        ("sublinear_tf=True", {"sublinear_tf": True}, {"max_iter": 1000}),
        ("min_df=2", {"min_df": 2}, {"max_iter": 1000}),
        ("ngram_range=(1,2)", {"ngram_range": (1, 2)}, {"max_iter": 1000}),
    ]

    # Pass 2: grid over the levers that moved heldout F1 in pass 1.
    min_df_values = [2, 3, 5]
    sublinear_tf_values = [True, False]
    ngram_range_values = [(1, 1), (1, 2)]
    c_values = [1, 2, 4, 10]

    grid = []
    for min_df, sublinear_tf, ngram_range, c in itertools.product(
        min_df_values, sublinear_tf_values, ngram_range_values, c_values
    ):
        label = f"min_df={min_df},sublinear_tf={sublinear_tf},ngram={ngram_range},C={c}"
        tfidf_kwargs = {"min_df": min_df, "sublinear_tf": sublinear_tf, "ngram_range": ngram_range}
        logreg_kwargs = {"max_iter": 1000, "C": c}
        grid.append((label, tfidf_kwargs, logreg_kwargs))

    rows = []
    for label, tfidf_kwargs, logreg_kwargs in single_factor:
        rows.append(
            run_variant(
                label, "single_factor", tfidf_kwargs, logreg_kwargs, train_df, dev_df, heldout_df, reference_f1, tolerance
            )
        )
    for label, tfidf_kwargs, logreg_kwargs in grid:
        rows.append(
            run_variant(label, "grid", tfidf_kwargs, logreg_kwargs, train_df, dev_df, heldout_df, reference_f1, tolerance)
        )

    result_df = pd.DataFrame(rows)
    result_df["abs_gap"] = result_df["heldout_gap_vs_reference"].abs()

    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 60)
    print(f"reference heldout F1 = {reference_f1:.4f}  (tolerance +/-{tolerance})\n")
    print("--- pass 1: one factor at a time ---")
    print(
        result_df[result_df["group"] == "single_factor"][
            ["label", "n_iter", "converged", "dev_f1_target_1", "heldout_f1_target_1", "heldout_gap_vs_reference"]
        ].to_string(index=False)
    )

    grid_df = result_df[result_df["group"] == "grid"]
    print(f"\n--- pass 2: grid ({len(grid_df)} configs) ---")
    in_tolerance_df = grid_df[grid_df["within_tolerance"]].sort_values("abs_gap")
    print(f"\nWithin tolerance of reference ({len(in_tolerance_df)} configs):")
    if len(in_tolerance_df):
        print(
            in_tolerance_df[
                ["label", "dev_f1_target_1", "heldout_f1_target_1", "heldout_gap_vs_reference"]
            ].to_string(index=False)
        )
    else:
        print("  (none)")

    closest = grid_df.sort_values("abs_gap").iloc[0]
    print(f"\nClosest overall (grid): {closest['label']}")
    print(
        f"  dev F1={closest['dev_f1_target_1']:.4f}  heldout F1={closest['heldout_f1_target_1']:.4f}  "
        f"gap={closest['heldout_gap_vs_reference']:+.4f}"
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.drop(columns=["abs_gap"]).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(result_df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
