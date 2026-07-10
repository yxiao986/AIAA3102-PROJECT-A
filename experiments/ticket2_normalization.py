"""Ticket 2 diagnosis: does text normalization help, on dev only?

Starts from the Ticket 1 baseline (no cleaning at all: TfidfVectorizer defaults +
LogisticRegression(max_iter=1000, random_state=3102)) and adds exactly one normalization
step at a time from pipeline/normalize.py. All decisions here are made on dev; heldout is
never touched by this script.

For each variant we report:
  - dev F1 (target=1)
  - fixed_fp: baseline false positive, now correct
  - fixed_fn: baseline false negative, now correct
  - new_fp:   baseline correct negative, now a false positive
  - new_fn:   baseline correct positive, now a false negative

Outputs:
  - results/ticket2_dev.csv       comparison table, one row per variant
  - results/ticket2_examples.csv  a few example tweets per flip type per variant
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.normalize import normalize_series, vectorizer_kwargs  # noqa: E402

DATA_CSV = ROOT / "data" / "train.csv"
SPLIT_JSON = ROOT / "data" / "split_indices.json"
DEV_OUT_CSV = ROOT / "results" / "ticket2_dev.csv"
EXAMPLES_OUT_CSV = ROOT / "results" / "ticket2_examples.csv"

RANDOM_STATE = 3102
EXAMPLES_PER_FLIP_TYPE = 5

# (label, normalize_flags, preserve_case)
VARIANTS = [
    ("baseline_no_cleaning", {}, False),
    ("strip_urls", {"strip_urls": True}, False),
    ("strip_mentions", {"strip_mentions": True}, False),
    ("strip_hashtag_symbol", {"strip_hashtag_symbol": True}, False),
    ("unescape_html", {"unescape_html": True}, False),
    ("strip_emoji", {"strip_emoji": True}, False),
    ("preserve_case", {}, True),
]


def load_split(df: pd.DataFrame, ids: list) -> pd.DataFrame:
    split_df = df[df["id"].isin(ids)].copy()
    split_df = split_df.set_index("id").loc[ids].reset_index()
    return split_df


def fit_predict(train_df, dev_df, normalize_flags, preserve_case):
    train_text = normalize_series(train_df["text"], **normalize_flags)
    dev_text = normalize_series(dev_df["text"], **normalize_flags)

    vectorizer = TfidfVectorizer(**vectorizer_kwargs(preserve_case))
    X_train = vectorizer.fit_transform(train_text)
    X_dev = vectorizer.transform(dev_text)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, train_df["target"])
    return clf.predict(X_dev)


def flip_counts(y_true, baseline_pred, variant_pred):
    fixed_fp = int(((baseline_pred == 1) & (y_true == 0) & (variant_pred == 0)).sum())
    fixed_fn = int(((baseline_pred == 0) & (y_true == 1) & (variant_pred == 1)).sum())
    new_fp = int(((baseline_pred == 0) & (y_true == 0) & (variant_pred == 1)).sum())
    new_fn = int(((baseline_pred == 1) & (y_true == 1) & (variant_pred == 0)).sum())
    return fixed_fp, fixed_fn, new_fp, new_fn


def collect_examples(label, dev_df, y_true, baseline_pred, variant_pred):
    masks = {
        "fixed_fp": (baseline_pred == 1) & (y_true == 0) & (variant_pred == 0),
        "fixed_fn": (baseline_pred == 0) & (y_true == 1) & (variant_pred == 1),
        "new_fp": (baseline_pred == 0) & (y_true == 0) & (variant_pred == 1),
        "new_fn": (baseline_pred == 1) & (y_true == 1) & (variant_pred == 0),
    }
    rows = []
    for flip_type, mask in masks.items():
        # dev_df has a clean 0..n-1 RangeIndex (see load_split), matching the positional
        # order of baseline_pred/variant_pred, so the iterrows() index doubles as the
        # position into those arrays.
        flipped = dev_df[mask].head(EXAMPLES_PER_FLIP_TYPE)
        for pos, row in flipped.iterrows():
            rows.append(
                {
                    "variant": label,
                    "flip_type": flip_type,
                    "id": row["id"],
                    "text": row["text"],
                    "y_true": int(row["target"]),
                    "baseline_pred": int(baseline_pred[pos]),
                    "variant_pred": int(variant_pred[pos]),
                }
            )
    return rows


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_df = load_split(df, split["train_ids"])
    dev_df = load_split(df, split["dev_ids"])
    y_true = dev_df["target"].to_numpy()

    baseline_pred = None
    dev_rows = []
    example_rows = []

    for label, normalize_flags, preserve_case in VARIANTS:
        variant_pred = fit_predict(train_df, dev_df, normalize_flags, preserve_case)
        dev_f1 = f1_score(y_true, variant_pred, pos_label=1)

        if label == "baseline_no_cleaning":
            baseline_pred = variant_pred
            fixed_fp = fixed_fn = new_fp = new_fn = 0
        else:
            fixed_fp, fixed_fn, new_fp, new_fn = flip_counts(y_true, baseline_pred, variant_pred)
            example_rows.extend(collect_examples(label, dev_df, y_true, baseline_pred, variant_pred))

        dev_rows.append(
            {
                "variant": label,
                "normalize_flags": normalize_flags,
                "preserve_case": preserve_case,
                "dev_f1_target_1": round(dev_f1, 4),
                "fixed_fp": fixed_fp,
                "fixed_fn": fixed_fn,
                "new_fp": new_fp,
                "new_fn": new_fn,
                "net_flips": fixed_fp + fixed_fn - new_fp - new_fn,
            }
        )

    dev_result_df = pd.DataFrame(dev_rows)
    examples_df = pd.DataFrame(
        example_rows, columns=["variant", "flip_type", "id", "text", "y_true", "baseline_pred", "variant_pred"]
    )

    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 60)
    print(f"baseline (no cleaning) dev F1 (target=1) = {dev_result_df.iloc[0]['dev_f1_target_1']:.4f}\n")
    print(
        dev_result_df[
            ["variant", "dev_f1_target_1", "fixed_fp", "fixed_fn", "new_fp", "new_fn", "net_flips"]
        ].to_string(index=False)
    )

    DEV_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    dev_result_df.to_csv(DEV_OUT_CSV, index=False)
    examples_df.to_csv(EXAMPLES_OUT_CSV, index=False)
    print(f"\nwrote {len(dev_result_df)} rows to {DEV_OUT_CSV}")
    print(f"wrote {len(examples_df)} rows to {EXAMPLES_OUT_CSV}")


if __name__ == "__main__":
    main()
