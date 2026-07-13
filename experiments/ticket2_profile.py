"""Ticket 2 supplement: surface-feature profiling on the raw data/train.csv (all 7613
rows, not any one split -- this is a description of the dataset, not a modeling decision,
so it isn't split-sensitive the way training choices are).

For each shallow feature, reports how common it is and how it correlates with target:
positive rate among rows that have it vs rows that don't (and, for the length bins, vs
the dataset's overall positive rate). Purely descriptive -- does not touch the frozen
pipeline (text-only + strip_urls).

Features: has_url (contains 'http'), has_mention (@user), has_hashtag (#word), has_emoji,
has_exclamation ('!'), has_allcaps_word (a token of 2+ uppercase letters), has_digit, and
character-length bins.

Output: results/ticket2_profile.csv
  columns: feature, group, n, pct_of_total, pos_rate, pos_rate_vs_overall
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.normalize import EMOJI_RE, HASHTAG_RE, MENTION_RE, URL_RE  # noqa: E402

DATA_CSV = ROOT / "data" / "train.csv"
OUT_CSV = ROOT / "results" / "ticket2_profile.csv"

ALLCAPS_RE = re.compile(r"\b[A-Z]{2,}\b")
DIGIT_RE = re.compile(r"\d")
EXCLAIM_RE = re.compile(r"!")

LEN_BIN_EDGES = [0, 40, 80, 120, 160]
LEN_BIN_LABELS = ["0-40", "41-80", "81-120", "121-160"]


def flag(series: pd.Series, compiled_re: re.Pattern) -> pd.Series:
    return series.map(lambda t: bool(compiled_re.search(t)))


def summarize_group(rows: pd.DataFrame, label: str, n_total: int, overall_pos_rate: float) -> dict:
    n = len(rows)
    pos_rate = rows["target"].mean() if n else float("nan")
    return {
        "group": label,
        "n": n,
        "pct_of_total": round(100 * n / n_total, 2),
        "pos_rate": round(pos_rate, 4) if n else "",
        "pos_rate_vs_overall": round(pos_rate - overall_pos_rate, 4) if n else "",
    }


def profile_binary_feature(df: pd.DataFrame, feature_name: str, mask: pd.Series, n_total: int, overall_pos_rate: float) -> list:
    rows = []
    for label, sub in [("present", df[mask]), ("absent", df[~mask])]:
        row = {"feature": feature_name}
        row.update(summarize_group(sub, label, n_total, overall_pos_rate))
        rows.append(row)
    return rows


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    n_total = len(df)
    overall_pos_rate = df["target"].mean()

    rows = [
        {
            "feature": "overall",
            "group": "all",
            "n": n_total,
            "pct_of_total": 100.0,
            "pos_rate": round(overall_pos_rate, 4),
            "pos_rate_vs_overall": 0.0,
        }
    ]

    binary_features = {
        "has_url": flag(df["text"], URL_RE),
        "has_mention": flag(df["text"], MENTION_RE),
        "has_hashtag": flag(df["text"], HASHTAG_RE),
        "has_emoji": flag(df["text"], EMOJI_RE),
        "has_exclamation": flag(df["text"], EXCLAIM_RE),
        "has_allcaps_word": flag(df["text"], ALLCAPS_RE),
        "has_digit": flag(df["text"], DIGIT_RE),
    }
    for name, mask in binary_features.items():
        rows.extend(profile_binary_feature(df, name, mask, n_total, overall_pos_rate))

    char_len_bin = pd.cut(df["text"].str.len(), bins=LEN_BIN_EDGES, labels=LEN_BIN_LABELS, include_lowest=True)
    for label in LEN_BIN_LABELS:
        sub = df[char_len_bin == label]
        row = {"feature": "char_len_bin"}
        row.update(summarize_group(sub, label, n_total, overall_pos_rate))
        rows.append(row)

    profile_df = pd.DataFrame(rows, columns=["feature", "group", "n", "pct_of_total", "pos_rate", "pos_rate_vs_overall"])

    pd.set_option("display.width", 160)
    print(f"overall positive rate: {overall_pos_rate:.4f} (n={n_total})\n")
    print(profile_df.to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(profile_df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
