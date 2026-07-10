"""Ticket 5 manual review: apply the human-reviewed verdicts for the 22 likely_mislabel
candidates produced by experiments/ticket5_audit.py.

This is not a new systematic detector -- it encodes specific verdicts reached after
reading each flagged tweet. Re-running ticket5_audit.py regenerates the audit table from
scratch (every likely_mislabel row back to proposed_label blank, disposition=
keep_but_flag), so this script must be re-run afterward to re-apply the reviewed verdicts.

Duplicate_label_conflict rows and the heldout hard rule are untouched here.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AUDIT_CSV = ROOT / "results" / "data_quality_audit.csv"

FIX_IDS = [796, 805, 1040, 1051, 4711, 5461, 5559, 6325, 6407, 7457, 7722, 7844, 9276, 9738, 9780, 10823]
AMBIGUOUS_IDS = [467, 1836, 10318]
REJECT_IDS = [4395, 5330, 7761]


def main() -> None:
    df = pd.read_csv(AUDIT_CSV, dtype={"proposed_label": "Int64"})
    is_mislabel = df["issue_type"] == "likely_mislabel"

    fix_mask = is_mislabel & df["id"].isin(FIX_IDS)
    assert fix_mask.sum() == len(FIX_IDS), f"expected {len(FIX_IDS)} fix rows, found {fix_mask.sum()}"
    df.loc[fix_mask, "disposition"] = "fix"
    df.loc[fix_mask, "proposed_label"] = 0

    ambiguous_mask = is_mislabel & df["id"].isin(AMBIGUOUS_IDS)
    assert ambiguous_mask.sum() == len(AMBIGUOUS_IDS), (
        f"expected {len(AMBIGUOUS_IDS)} ambiguous rows, found {ambiguous_mask.sum()}"
    )
    df.loc[ambiguous_mask, "disposition"] = "ambiguous"
    df.loc[ambiguous_mask, "proposed_label"] = pd.NA

    reject_mask = is_mislabel & df["id"].isin(REJECT_IDS)
    assert reject_mask.sum() == len(REJECT_IDS), f"expected {len(REJECT_IDS)} reject rows, found {reject_mask.sum()}"
    df.loc[reject_mask, "disposition"] = "reject_false_positive"
    df.loc[reject_mask, "issue_type"] = "reject_false_positive"
    df.loc[reject_mask, "proposed_label"] = pd.NA

    reviewed = fix_mask | ambiguous_mask | reject_mask
    remaining = is_mislabel & ~reviewed
    if remaining.any():
        print(
            f"WARNING: {int(remaining.sum())} likely_mislabel rows were not covered by this review: "
            f"{df.loc[remaining, 'id'].tolist()}"
        )

    df.to_csv(AUDIT_CSV, index=False)
    print(f"updated {int(reviewed.sum())} rows in {AUDIT_CSV}")
    print(df.groupby(["issue_type", "disposition"]).size().to_string())


if __name__ == "__main__":
    main()
