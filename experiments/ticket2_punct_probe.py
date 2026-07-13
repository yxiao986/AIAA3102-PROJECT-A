"""Ticket 2 supplement: does keeping '!' and '?' as their own tokens help, on top of the
frozen pipeline (text-only + strip_urls)?

TfidfVectorizer's default token_pattern (r"(?u)\b\w\w+\b") only keeps alphanumeric runs of
length >= 2, so punctuation is silently dropped -- '!!!' and '?' never reach the vocabulary
today. This probe adds a custom token_pattern that keeps the default word tokens AND treats
each '!' or '?' as its own token (so '!!!' becomes three separate '!' tokens, same as most
punctuation-aware tokenizers).

Baseline = strip_urls text + TfidfVectorizer's default token_pattern (the frozen pipeline).
Variant  = strip_urls text + punctuation-preserving token_pattern.
Both trained on train, compared on dev only (diagnostic; does not touch the frozen
pipeline or heldout).

Output: results/ticket2_punct_probe.csv
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

from pipeline.normalize import normalize_series  # noqa: E402

DATA_CSV = ROOT / "data" / "train.csv"
SPLIT_JSON = ROOT / "data" / "split_indices.json"
OUT_CSV = ROOT / "results" / "ticket2_punct_probe.csv"

RANDOM_STATE = 3102
DEFAULT_TOKEN_PATTERN = r"(?u)\b\w\w+\b"
PUNCT_PRESERVING_TOKEN_PATTERN = r"(?u)\b\w\w+\b|[!?]"


def load_split(df: pd.DataFrame, ids: list) -> pd.DataFrame:
    split_df = df[df["id"].isin(ids)].copy()
    split_df = split_df.set_index("id").loc[ids].reset_index()
    return split_df


def fit_predict(train_text, train_y, dev_text, token_pattern):
    vectorizer = TfidfVectorizer(token_pattern=token_pattern)
    X_train = vectorizer.fit_transform(train_text)
    X_dev = vectorizer.transform(dev_text)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, train_y)
    return clf.predict(X_dev), len(vectorizer.vocabulary_)


def flip_counts(y_true, baseline_pred, variant_pred):
    fixed_fp = int(((baseline_pred == 1) & (y_true == 0) & (variant_pred == 0)).sum())
    fixed_fn = int(((baseline_pred == 0) & (y_true == 1) & (variant_pred == 1)).sum())
    new_fp = int(((baseline_pred == 0) & (y_true == 0) & (variant_pred == 1)).sum())
    new_fn = int(((baseline_pred == 1) & (y_true == 1) & (variant_pred == 0)).sum())
    return fixed_fp, fixed_fn, new_fp, new_fn


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_df = load_split(df, split["train_ids"])
    dev_df = load_split(df, split["dev_ids"])
    y_train = train_df["target"]
    y_dev = dev_df["target"].to_numpy()

    train_text = normalize_series(train_df["text"], strip_urls=True)
    dev_text = normalize_series(dev_df["text"], strip_urls=True)

    baseline_pred, baseline_vocab = fit_predict(train_text, y_train, dev_text, DEFAULT_TOKEN_PATTERN)
    variant_pred, variant_vocab = fit_predict(train_text, y_train, dev_text, PUNCT_PRESERVING_TOKEN_PATTERN)

    baseline_f1 = f1_score(y_dev, baseline_pred, pos_label=1)
    variant_f1 = f1_score(y_dev, variant_pred, pos_label=1)
    fixed_fp, fixed_fn, new_fp, new_fn = flip_counts(y_dev, baseline_pred, variant_pred)

    rows = [
        {
            "variant": "default_tokenizer",
            "token_pattern": DEFAULT_TOKEN_PATTERN,
            "vocab_size": baseline_vocab,
            "dev_f1_target_1": round(baseline_f1, 4),
            "fixed_fp": 0,
            "fixed_fn": 0,
            "new_fp": 0,
            "new_fn": 0,
        },
        {
            "variant": "punct_preserving_tokenizer",
            "token_pattern": PUNCT_PRESERVING_TOKEN_PATTERN,
            "vocab_size": variant_vocab,
            "dev_f1_target_1": round(variant_f1, 4),
            "fixed_fp": fixed_fp,
            "fixed_fn": fixed_fn,
            "new_fp": new_fp,
            "new_fn": new_fn,
        },
    ]
    result_df = pd.DataFrame(rows)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 40)
    print(f"baseline (default tokenizer, strip_urls) dev F1 = {baseline_f1:.4f}, vocab={baseline_vocab}")
    print(f"variant (punct-preserving tokenizer, strip_urls) dev F1 = {variant_f1:.4f}, vocab={variant_vocab}")
    print(f"fixed_fp={fixed_fp}  fixed_fn={fixed_fn}  new_fp={new_fp}  new_fn={new_fn}")
    print()
    print(result_df.to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(result_df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
