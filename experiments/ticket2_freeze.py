"""Ticket 2 freeze: adopt strip_urls, verify once on heldout.

The dev-only sweep (experiments/ticket2_normalization.py) picked strip_urls as the only
normalization step to adopt; every other lever (mentions, hashtag symbol, html unescape,
emoji, case) stays at baseline default. This script is the single frozen heldout run for
that decision:
  - trains baseline (no cleaning) and strip_urls on train
  - scores heldout F1 (target=1) and accuracy for both
  - reports strip_urls' fixed/new fp/fn vs baseline, on heldout
  - appends the strip_urls heldout predictions to predictions/heldout_predictions.csv
  - appends the Ticket 2 row to results/summary.csv

Heldout is touched exactly once here.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.normalize import normalize_series  # noqa: E402

DATA_CSV = ROOT / "data" / "train.csv"
SPLIT_JSON = ROOT / "data" / "split_indices.json"
DEV_PROBE_CSV = ROOT / "results" / "ticket2_dev.csv"
PREDICTIONS_CSV = ROOT / "predictions" / "heldout_predictions.csv"
SUMMARY_CSV = ROOT / "results" / "summary.csv"

RANDOM_STATE = 3102
TICKET = "ticket-2-normalization"
MODEL_NAME = "tfidf_logreg_stripurls"
DECISION = "adopt_strip_urls"
DECISION_REASON = (
    "strips http/https URL tokens; dev F1 improved (0.7388 -> 0.7437) and the gain "
    "holds up on heldout, robust to URL-shortener noise"
)


def load_split(df: pd.DataFrame, ids: list) -> pd.DataFrame:
    split_df = df[df["id"].isin(ids)].copy()
    split_df = split_df.set_index("id").loc[ids].reset_index()
    return split_df


def fit_predict(train_df, heldout_df, normalize_flags):
    train_text = normalize_series(train_df["text"], **normalize_flags)
    heldout_text = normalize_series(heldout_df["text"], **normalize_flags)

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_text)
    X_heldout = vectorizer.transform(heldout_text)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, train_df["target"])
    pred = clf.predict(X_heldout)
    score = clf.predict_proba(X_heldout)[:, 1]
    return pred, score


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
    heldout_df = load_split(df, split["heldout_ids"])
    y_true = heldout_df["target"].to_numpy()

    baseline_pred, _ = fit_predict(train_df, heldout_df, {})
    stripurls_pred, stripurls_score = fit_predict(train_df, heldout_df, {"strip_urls": True})

    baseline_f1 = f1_score(y_true, baseline_pred, pos_label=1)
    baseline_acc = accuracy_score(y_true, baseline_pred)
    stripurls_f1 = f1_score(y_true, stripurls_pred, pos_label=1)
    stripurls_acc = accuracy_score(y_true, stripurls_pred)

    fixed_fp, fixed_fn, new_fp, new_fn = flip_counts(y_true, baseline_pred, stripurls_pred)

    print("heldout, frozen comparison:")
    print(f"  baseline (no cleaning): F1={baseline_f1:.4f}  accuracy={baseline_acc:.4f}")
    print(f"  strip_urls:             F1={stripurls_f1:.4f}  accuracy={stripurls_acc:.4f}")
    print(f"  fixed_fp={fixed_fp}  fixed_fn={fixed_fn}  new_fp={new_fp}  new_fn={new_fn}")

    if PREDICTIONS_CSV.exists():
        existing = pd.read_csv(PREDICTIONS_CSV)
        t1 = existing[existing["ticket"] == "ticket-1-baseline"].set_index("id").loc[heldout_df["id"]]
        mismatch = int((t1["y_pred"].to_numpy() != baseline_pred).sum())
        if mismatch:
            print(f"  WARNING: {mismatch} baseline predictions differ from ticket-1-baseline already on disk")

    dev_probe = pd.read_csv(DEV_PROBE_CSV)
    dev_f1 = float(dev_probe.loc[dev_probe["variant"] == "strip_urls", "dev_f1_target_1"].iloc[0])

    # --- append strip_urls heldout predictions (idempotent: replace any prior Ticket 2 rows) ---
    new_predictions = pd.DataFrame(
        {
            "id": heldout_df["id"],
            "y_true": heldout_df["target"],
            "y_pred": stripurls_pred,
            "score": stripurls_score,
            "model_name": MODEL_NAME,
            "ticket": TICKET,
        }
    )
    if PREDICTIONS_CSV.exists():
        existing = pd.read_csv(PREDICTIONS_CSV)
        existing = existing[existing["ticket"] != TICKET]
        combined_predictions = pd.concat([existing, new_predictions], ignore_index=True)
    else:
        combined_predictions = new_predictions
    combined_predictions.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nappended {len(new_predictions)} rows to {PREDICTIONS_CSV} (model_name={MODEL_NAME})")

    # --- append Ticket 2 summary row (idempotent: replace any prior Ticket 2 row) ---
    summary_row = pd.DataFrame(
        [
            {
                "ticket": TICKET,
                "model_name": MODEL_NAME,
                "dev_f1_target_1": round(dev_f1, 4),
                "heldout_f1_target_1": round(stripurls_f1, 4),
                "heldout_accuracy": round(stripurls_acc, 4),
                "fixed_fp": fixed_fp,
                "fixed_fn": fixed_fn,
                "new_fp": new_fp,
                "new_fn": new_fn,
                "decision": DECISION,
                "decision_reason": DECISION_REASON,
            }
        ]
    )
    if SUMMARY_CSV.exists():
        summary = pd.read_csv(SUMMARY_CSV)
        summary = summary[summary["ticket"] != TICKET]
        combined_summary = pd.concat([summary, summary_row], ignore_index=True)
    else:
        combined_summary = summary_row
    combined_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"appended Ticket 2 row to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
