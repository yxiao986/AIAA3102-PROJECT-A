"""Ticket 3 audit: how much signal do shallow/shortcut features carry on their own?

Builds a family of LogisticRegression probes, each restricted to a different feature
slice (keyword identity, text length, URL presence, hashtag count, uppercase fraction,
location presence, and combinations), trained on train and scored on dev + heldout. This
is a diagnostic audit only -- it does not touch the frozen pipeline, which stays
text-only + strip_urls (Ticket 2's decision).

Shortcut features (all derived from raw `text` / `location`, no cleaning applied):
  - char_len:     len(text)
  - has_url:      1 if 'http' appears in text (case-insensitive), else 0
  - n_hashtags:   count of '#word' tokens in text
  - frac_upper:   fraction of alphabetic characters that are uppercase
  - has_location: 1 if `location` is non-null/non-blank, else 0
  - keyword:      the Kaggle `keyword` field, one-hot encoded (NaN -> its own category)

Output: results/ticket3_shortcuts.csv, plus the keyword_only model's top +/- 15 keyword
coefficients printed to stdout.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "train.csv"
SPLIT_JSON = ROOT / "data" / "split_indices.json"
OUT_CSV = ROOT / "results" / "ticket3_shortcuts.csv"

RANDOM_STATE = 3102
TOP_N_KEYWORDS = 15


def load_split(df: pd.DataFrame, ids: list) -> pd.DataFrame:
    split_df = df[df["id"].isin(ids)].copy()
    split_df = split_df.set_index("id").loc[ids].reset_index()
    return split_df


def build_metadata(df: pd.DataFrame) -> pd.DataFrame:
    text = df["text"]
    char_len = text.str.len()
    has_url = text.str.lower().str.contains("http", regex=False).astype(int)
    n_hashtags = text.str.count(r"#\w+")
    upper_count = text.apply(lambda t: sum(c.isupper() for c in t))
    alpha_count = text.apply(lambda t: sum(c.isalpha() for c in t))
    frac_upper = (upper_count / alpha_count.replace(0, np.nan)).fillna(0.0)
    has_location = (df["location"].notna() & (df["location"].astype(str).str.strip() != "")).astype(int)
    return pd.DataFrame(
        {
            "char_len": char_len,
            "has_url": has_url,
            "n_hashtags": n_hashtags,
            "frac_upper": frac_upper,
            "has_location": has_location,
        }
    )


NUMERIC_COLS = ["char_len", "has_url", "n_hashtags", "frac_upper", "has_location"]

# Not computed from data -- a domain read on what each feature *should* mean, for the
# reader to weigh against the actual F1 numbers below.
FEATURE_TYPE_NOTE = {
    "floor": "reference floor (no signal)",
    "keyword_only": "task info (curated Kaggle disaster keyword)",
    "char_len_only": "artifact (tweet length, not about disaster content)",
    "has_url_only": "artifact (URL presence, a posting-style proxy)",
    "n_hashtags_only": "artifact (hashtag usage style)",
    "frac_upper_only": "artifact (capitalization / shouting style)",
    "has_location_only": "mixed (free-text field; presence alone is a weak proxy)",
    "metadata_numeric_bundle": "artifact bundle (surface stats only)",
    "all_shortcuts": "mixed (task info + artifacts combined)",
    "text_only": "reference (full text signal)",
    "text_plus_all_shortcuts": "mixed (shortcuts stacked on full text)",
}


def fit_score(X_train, y_train, X_dev, y_dev, X_heldout, y_heldout, is_floor=False):
    if is_floor:
        clf = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    dev_f1 = f1_score(y_dev, clf.predict(X_dev), pos_label=1)
    heldout_f1 = f1_score(y_heldout, clf.predict(X_heldout), pos_label=1)
    return clf, dev_f1, heldout_f1


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_df = load_split(df, split["train_ids"])
    dev_df = load_split(df, split["dev_ids"])
    heldout_df = load_split(df, split["heldout_ids"])

    y_train, y_dev, y_heldout = train_df["target"], dev_df["target"], heldout_df["target"]

    meta_train = build_metadata(train_df)
    meta_dev = build_metadata(dev_df)
    meta_heldout = build_metadata(heldout_df)

    keyword_encoder = OneHotEncoder(handle_unknown="ignore")
    kw_train = keyword_encoder.fit_transform(train_df[["keyword"]].fillna("__missing__"))
    kw_dev = keyword_encoder.transform(dev_df[["keyword"]].fillna("__missing__"))
    kw_heldout = keyword_encoder.transform(heldout_df[["keyword"]].fillna("__missing__"))

    tfidf = TfidfVectorizer()
    text_train = tfidf.fit_transform(train_df["text"])
    text_dev = tfidf.transform(dev_df["text"])
    text_heldout = tfidf.transform(heldout_df["text"])

    meta_bundle_train = csr_matrix(meta_train[NUMERIC_COLS].to_numpy())
    meta_bundle_dev = csr_matrix(meta_dev[NUMERIC_COLS].to_numpy())
    meta_bundle_heldout = csr_matrix(meta_heldout[NUMERIC_COLS].to_numpy())

    all_shortcuts_train = hstack([meta_bundle_train, kw_train]).tocsr()
    all_shortcuts_dev = hstack([meta_bundle_dev, kw_dev]).tocsr()
    all_shortcuts_heldout = hstack([meta_bundle_heldout, kw_heldout]).tocsr()

    def single_col(meta_df, col):
        return meta_df[[col]].to_numpy()

    variants = {
        "floor": (
            np.zeros((len(train_df), 1)),
            np.zeros((len(dev_df), 1)),
            np.zeros((len(heldout_df), 1)),
            True,
        ),
        "keyword_only": (kw_train, kw_dev, kw_heldout, False),
        "char_len_only": (
            single_col(meta_train, "char_len"),
            single_col(meta_dev, "char_len"),
            single_col(meta_heldout, "char_len"),
            False,
        ),
        "has_url_only": (
            single_col(meta_train, "has_url"),
            single_col(meta_dev, "has_url"),
            single_col(meta_heldout, "has_url"),
            False,
        ),
        "n_hashtags_only": (
            single_col(meta_train, "n_hashtags"),
            single_col(meta_dev, "n_hashtags"),
            single_col(meta_heldout, "n_hashtags"),
            False,
        ),
        "frac_upper_only": (
            single_col(meta_train, "frac_upper"),
            single_col(meta_dev, "frac_upper"),
            single_col(meta_heldout, "frac_upper"),
            False,
        ),
        "has_location_only": (
            single_col(meta_train, "has_location"),
            single_col(meta_dev, "has_location"),
            single_col(meta_heldout, "has_location"),
            False,
        ),
        "metadata_numeric_bundle": (meta_bundle_train, meta_bundle_dev, meta_bundle_heldout, False),
        "all_shortcuts": (all_shortcuts_train, all_shortcuts_dev, all_shortcuts_heldout, False),
        "text_only": (text_train, text_dev, text_heldout, False),
        "text_plus_all_shortcuts": (
            hstack([text_train, all_shortcuts_train]).tocsr(),
            hstack([text_dev, all_shortcuts_dev]).tocsr(),
            hstack([text_heldout, all_shortcuts_heldout]).tocsr(),
            False,
        ),
    }

    rows = []
    keyword_clf = None
    for label, (X_train, X_dev, X_heldout, is_floor) in variants.items():
        clf, dev_f1, heldout_f1 = fit_score(X_train, y_train, X_dev, y_dev, X_heldout, y_heldout, is_floor)
        if label == "keyword_only":
            keyword_clf = clf
        rows.append(
            {
                "model": label,
                "n_features": X_train.shape[1],
                "dev_f1_target_1": round(dev_f1, 4),
                "heldout_f1_target_1": round(heldout_f1, 4),
                "feature_type_note": FEATURE_TYPE_NOTE[label],
            }
        )

    result_df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 60)
    print(result_df[["model", "n_features", "dev_f1_target_1", "heldout_f1_target_1"]].to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(result_df)} rows to {OUT_CSV}")

    # keyword_only coefficients: what did it actually learn?
    feature_names = keyword_encoder.get_feature_names_out(["keyword"])
    coefs = keyword_clf.coef_[0]
    order = np.argsort(coefs)
    top_negative = order[:TOP_N_KEYWORDS]
    top_positive = order[::-1][:TOP_N_KEYWORDS]

    print(f"\nkeyword_only: top {TOP_N_KEYWORDS} positive-weight keywords (push toward target=1):")
    for idx in top_positive:
        name = feature_names[idx].replace("keyword_", "")
        print(f"  {name:30s} {coefs[idx]:+.4f}")

    print(f"\nkeyword_only: top {TOP_N_KEYWORDS} negative-weight keywords (push toward target=0):")
    for idx in top_negative:
        name = feature_names[idx].replace("keyword_", "")
        print(f"  {name:30s} {coefs[idx]:+.4f}")


if __name__ == "__main__":
    main()
