"""Ticket 5 audit: two systematic data-quality checks, output as an unreviewed candidate
table for manual disposition.

This script only performs SYSTEMATIC detection plus rule-based, defensible dispositions.
It does not fabricate a judgment about *why* a flagged row might be wrong (e.g. "sarcastic
use", "figurative use") -- that requires reading the tweet and is left to manual review.
Concretely:
  - duplicate_label_conflict rows get disposition=fix only when the duplicate group has a
    clear majority label AND the row is not in heldout; a true 50/50 tie is disposition=
    ambiguous; any heldout row is always disposition=keep_but_flag with no proposed_label,
    no matter how lopsided its group is.
  - likely_mislabel rows (model disagrees with high confidence) are always disposition=
    keep_but_flag with no proposed_label -- a single model's confidence is evidence to flag
    for a human, not grounds to auto-fix a label.
  - disposition=reject_false_positive is never emitted here; that verdict can only come
    from a human reading the flagged row.

Checks:
  1. duplicate_label_conflict: normalize text (strip + collapse internal whitespace +
     lowercase), group by normalized text, keep groups whose labels are not unique. Every
     row in such a group gets one output record.
  2. likely_mislabel: LogisticRegression(C=3, max_iter=1000, random_state=3102) trained on
     train using the Ticket 2/4 frozen pipeline (strip_urls text normalization + default
     TfidfVectorizer), scored on train+dev ONLY -- heldout is never scored by this check.
     Flags label=0 with p(target=1)>0.90, and label=1 with p(target=1)<0.10.

Output: results/data_quality_audit.csv
  columns: id, issue_type, evidence, original_label, proposed_label, disposition, confidence
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.normalize import normalize_series  # noqa: E402

DATA_CSV = ROOT / "data" / "train.csv"
SPLIT_JSON = ROOT / "data" / "split_indices.json"
OUT_CSV = ROOT / "results" / "data_quality_audit.csv"

RANDOM_STATE = 3102
HIGH_P = 0.90
LOW_P = 0.10
HIGH_CONFIDENCE_P = 0.95
LOW_CONFIDENCE_P = 0.05
SNIPPET_LEN = 80

WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_dedup(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip().lower())


def load_split(df: pd.DataFrame, ids: list) -> pd.DataFrame:
    split_df = df[df["id"].isin(ids)].copy()
    split_df = split_df.set_index("id").loc[ids].reset_index()
    return split_df


def find_duplicate_conflicts(df: pd.DataFrame) -> list:
    rows = []
    for norm_text, group in df.groupby("norm_text"):
        if group["target"].nunique() <= 1:
            continue

        group_sorted = group.sort_values("id")
        ids_list = group_sorted["id"].tolist()
        labels_list = group_sorted["target"].tolist()

        label_counts = group["target"].value_counts()
        is_tie = len(label_counts) == 2 and label_counts.iloc[0] == label_counts.iloc[1]
        majority_label = None if is_tie else int(label_counts.idxmax())
        confidence = "low" if is_tie else "high"

        snippet = norm_text[:SNIPPET_LEN] + ("..." if len(norm_text) > SNIPPET_LEN else "")
        evidence = f"dup group ids={ids_list} labels={labels_list} text='{snippet}'"

        for _, row in group_sorted.iterrows():
            original_label = int(row["target"])
            if row["split"] == "heldout":
                proposed_label = ""
                disposition = "keep_but_flag"
            elif is_tie:
                proposed_label = ""
                disposition = "ambiguous"
            else:
                proposed_label = majority_label
                disposition = "fix"

            rows.append(
                {
                    "id": int(row["id"]),
                    "issue_type": "duplicate_label_conflict",
                    "evidence": evidence,
                    "original_label": original_label,
                    "proposed_label": proposed_label,
                    "disposition": disposition,
                    "confidence": confidence,
                }
            )
    return rows


def score_and_flag(sub_df: pd.DataFrame, probs) -> list:
    rows = []
    for pos, row in enumerate(sub_df.itertuples()):
        p1 = probs[pos]
        label = int(row.target)
        if label == 0 and p1 > HIGH_P:
            direction = f"label=0 but p(target=1)={p1:.4f} > {HIGH_P}"
        elif label == 1 and p1 < LOW_P:
            direction = f"label=1 but p(target=1)={p1:.4f} < {LOW_P}"
        else:
            continue

        confidence = "high" if (p1 > HIGH_CONFIDENCE_P or p1 < LOW_CONFIDENCE_P) else "medium"
        keyword = row.keyword if pd.notna(row.keyword) else "(none)"
        evidence = f"LR(C=3) {direction}, keyword={keyword}, split={row.split}"

        rows.append(
            {
                "id": int(row.id),
                "issue_type": "likely_mislabel",
                "evidence": evidence,
                "original_label": label,
                "proposed_label": "",
                "disposition": "keep_but_flag",
                "confidence": confidence,
            }
        )
    return rows


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split = json.load(f)

    split_of = {}
    for sid in split["train_ids"]:
        split_of[sid] = "train"
    for sid in split["dev_ids"]:
        split_of[sid] = "dev"
    for sid in split["heldout_ids"]:
        split_of[sid] = "heldout"
    df["split"] = df["id"].map(split_of)
    assert df["split"].notna().all(), "found ids outside train/dev/heldout"

    # --- check 1: exact-duplicate text with conflicting labels (all splits) ---
    df["norm_text"] = df["text"].map(normalize_for_dedup)
    dup_rows = find_duplicate_conflicts(df)

    dup_groups = {r["evidence"] for r in dup_rows}
    print(f"check 1 (duplicate_label_conflict): {len(dup_groups)} conflicting groups, {len(dup_rows)} rows flagged")

    # --- check 2: high-confidence model disagreement (train+dev only) ---
    train_df = load_split(df, split["train_ids"])
    dev_df = load_split(df, split["dev_ids"])

    train_text = normalize_series(train_df["text"], strip_urls=True)
    dev_text = normalize_series(dev_df["text"], strip_urls=True)

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_text)
    X_dev = vectorizer.transform(dev_text)

    clf = LogisticRegression(C=3, max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, train_df["target"])

    train_probs = clf.predict_proba(X_train)[:, 1]
    dev_probs = clf.predict_proba(X_dev)[:, 1]

    mislabel_rows = score_and_flag(train_df, train_probs) + score_and_flag(dev_df, dev_probs)
    print(f"check 2 (likely_mislabel): {len(mislabel_rows)} rows flagged (train+dev only, heldout not scored)")

    audit_df = pd.DataFrame(
        dup_rows + mislabel_rows,
        columns=["id", "issue_type", "evidence", "original_label", "proposed_label", "disposition", "confidence"],
    )
    audit_df = audit_df.sort_values(["issue_type", "id"]).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(OUT_CSV, index=False)

    print("\nby issue_type x disposition:")
    print(audit_df.groupby(["issue_type", "disposition"]).size().to_string())
    print(f"\nwrote {len(audit_df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
