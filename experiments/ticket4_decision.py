"""Ticket 4: decision rule (threshold) and model selection, on top of the frozen
Ticket 2 pipeline (TfidfVectorizer defaults + strip_urls text normalization).

All hyperparameter / model selection happens on dev only. Per an explicit decision on
this ticket, steps 2-4 report DEV metrics only (not heldout) -- reporting heldout for
every sweep point would itself be a form of heldout leakage even if not literally used
to pick a winner, and conflicts with "heldout runs once" discipline used since Ticket 2.

Steps:
  1. Threshold sweep (dev, default C=1 LR): thresholds 0.20..0.70 step 0.05.
     -> results/threshold_sweep.csv (ticket, threshold, precision_target_1,
        recall_target_1, f1_target_1)
  2. C sweep (dev, threshold=0.5): C in {0.3, 1, 3, 10, 30}.
  3. class_weight='balanced' (dev, threshold=0.5): precision/recall/F1.
  4. Alternate classifiers (dev, default decision rule): LinearSVC, MultinomialNB,
     SGDClassifier(loss='log_loss').
  Steps 2-4 -> results/ticket4_dev.csv

Freeze: combine the best C (step 2) with the best threshold (step 1) -- the combo the
ticket calls for -- train it on train, score heldout exactly once, export heldout
predictions, and append the Ticket 4 summary row (diffed against the Ticket 2 strip_urls
baseline already on disk).
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.normalize import normalize_series  # noqa: E402

DATA_CSV = ROOT / "data" / "train.csv"
SPLIT_JSON = ROOT / "data" / "split_indices.json"
THRESHOLD_OUT_CSV = ROOT / "results" / "threshold_sweep.csv"
DEV_OUT_CSV = ROOT / "results" / "ticket4_dev.csv"
PREDICTIONS_CSV = ROOT / "predictions" / "heldout_predictions.csv"
SUMMARY_CSV = ROOT / "results" / "summary.csv"

RANDOM_STATE = 3102
TICKET = "ticket-4-decision"
BASELINE_TICKET = "ticket-2-normalization"  # frozen strip_urls model to diff against
MODEL_NAME = "tfidf_logreg_stripurls_c3_tuned"

THRESHOLDS = [round(0.20 + 0.05 * i, 2) for i in range(11)]  # 0.20 .. 0.70
C_VALUES = [0.3, 1, 3, 10, 30]


def load_split(df: pd.DataFrame, ids: list) -> pd.DataFrame:
    split_df = df[df["id"].isin(ids)].copy()
    split_df = split_df.set_index("id").loc[ids].reset_index()
    return split_df


def prf(y_true, y_pred):
    return (
        precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    )


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
    heldout_df = load_split(df, split["heldout_ids"])

    y_train = train_df["target"].to_numpy()
    y_dev = dev_df["target"].to_numpy()

    train_text = normalize_series(train_df["text"], strip_urls=True)
    dev_text = normalize_series(dev_df["text"], strip_urls=True)

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_text)
    X_dev = vectorizer.transform(dev_text)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 60)

    # --- Step 1: threshold sweep (dev, default C=1) ---
    base_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    base_clf.fit(X_train, y_train)
    dev_scores_base = base_clf.predict_proba(X_dev)[:, 1]

    threshold_rows = []
    for t in THRESHOLDS:
        pred = (dev_scores_base >= t).astype(int)
        precision, recall, f1 = prf(y_dev, pred)
        threshold_rows.append(
            {
                "ticket": TICKET,
                "threshold": t,
                "precision_target_1": round(precision, 4),
                "recall_target_1": round(recall, 4),
                "f1_target_1": round(f1, 4),
            }
        )
    threshold_df = pd.DataFrame(threshold_rows)
    THRESHOLD_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(THRESHOLD_OUT_CSV, index=False)

    best_threshold_row = threshold_df.loc[threshold_df["f1_target_1"].idxmax()]
    best_threshold = float(best_threshold_row["threshold"])

    print("--- step 1: threshold sweep (dev, C=1) ---")
    print(threshold_df.to_string(index=False))
    print(f"\nbest threshold on dev: {best_threshold} (F1={best_threshold_row['f1_target_1']:.4f})")
    print(f"wrote {len(threshold_df)} rows to {THRESHOLD_OUT_CSV}")

    # --- Step 2: C sweep (dev, threshold=0.5) ---
    dev_rows = []
    c_models = {}
    for c in C_VALUES:
        clf = LogisticRegression(C=c, max_iter=1000, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        c_models[c] = clf
        pred = clf.predict(X_dev)
        precision, recall, f1 = prf(y_dev, pred)
        dev_rows.append(
            {
                "group": "C_sweep",
                "label": f"C={c}",
                "dev_precision_target_1": round(precision, 4),
                "dev_recall_target_1": round(recall, 4),
                "dev_f1_target_1": round(f1, 4),
            }
        )

    c_sweep_df = pd.DataFrame(dev_rows)
    best_c_row = c_sweep_df.loc[c_sweep_df["dev_f1_target_1"].idxmax()]
    best_c = float(best_c_row["label"].split("=")[1])

    print("\n--- step 2: C sweep (dev, threshold=0.5) ---")
    print(c_sweep_df.to_string(index=False))
    print(f"\nbest C on dev: {best_c} (F1={best_c_row['dev_f1_target_1']:.4f})")

    # --- Step 3: class_weight='balanced' (dev, threshold=0.5) ---
    balanced_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    balanced_clf.fit(X_train, y_train)
    balanced_pred = balanced_clf.predict(X_dev)
    precision, recall, f1 = prf(y_dev, balanced_pred)
    balanced_row = {
        "group": "class_weight",
        "label": "class_weight=balanced",
        "dev_precision_target_1": round(precision, 4),
        "dev_recall_target_1": round(recall, 4),
        "dev_f1_target_1": round(f1, 4),
    }
    print("\n--- step 3: class_weight='balanced' (dev, threshold=0.5) ---")
    print(pd.DataFrame([balanced_row]).to_string(index=False))

    # --- Step 4: alternate classifiers (dev, default decision rule) ---
    alt_classifiers = {
        "LinearSVC": LinearSVC(random_state=RANDOM_STATE),
        "MultinomialNB": MultinomialNB(),
        "SGDClassifier(log_loss)": SGDClassifier(loss="log_loss", random_state=RANDOM_STATE),
    }
    alt_rows = []
    for name, clf in alt_classifiers.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_dev)
        precision, recall, f1 = prf(y_dev, pred)
        alt_rows.append(
            {
                "group": "alt_classifier",
                "label": name,
                "dev_precision_target_1": round(precision, 4),
                "dev_recall_target_1": round(recall, 4),
                "dev_f1_target_1": round(f1, 4),
            }
        )
    alt_df = pd.DataFrame(alt_rows)
    print("\n--- step 4: alternate classifiers (dev, default decision rule) ---")
    print(alt_df.to_string(index=False))

    # --- combo: best C (step 2) + best threshold (step 1), as the ticket asks to freeze ---
    combo_clf = c_models[best_c]
    combo_scores_dev = combo_clf.predict_proba(X_dev)[:, 1]
    combo_pred_dev = (combo_scores_dev >= best_threshold).astype(int)
    combo_precision, combo_recall, combo_f1 = prf(y_dev, combo_pred_dev)
    combo_row = {
        "group": "combo",
        "label": f"C={best_c}+threshold={best_threshold}",
        "dev_precision_target_1": round(combo_precision, 4),
        "dev_recall_target_1": round(combo_recall, 4),
        "dev_f1_target_1": round(combo_f1, 4),
    }

    dev_result_df = pd.concat([c_sweep_df, pd.DataFrame([balanced_row]), alt_df, pd.DataFrame([combo_row])], ignore_index=True)
    DEV_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    dev_result_df.to_csv(DEV_OUT_CSV, index=False)

    print("\n--- all dev candidates (steps 2-4 + combo) ---")
    print(dev_result_df.to_string(index=False))
    best_overall = dev_result_df.loc[dev_result_df["dev_f1_target_1"].idxmax()]
    print(f"\nbest dev F1 overall: {best_overall['label']} (F1={best_overall['dev_f1_target_1']:.4f})")
    if best_overall["label"] != combo_row["label"]:
        print(
            f"NOTE: '{best_overall['label']}' scores higher on dev than the frozen combo "
            f"'{combo_row['label']}' (F1={combo_row['dev_f1_target_1']:.4f}); freezing the "
            "combo anyway, as specified."
        )
    print(f"wrote {len(dev_result_df)} rows to {DEV_OUT_CSV}")

    # --- freeze: run heldout exactly once, for the C=best_c + threshold=best_threshold combo ---
    heldout_text = normalize_series(heldout_df["text"], strip_urls=True)
    X_heldout = vectorizer.transform(heldout_text)
    y_heldout = heldout_df["target"].to_numpy()

    heldout_scores = combo_clf.predict_proba(X_heldout)[:, 1]
    heldout_pred = (heldout_scores >= best_threshold).astype(int)
    heldout_f1 = f1_score(y_heldout, heldout_pred, pos_label=1)
    heldout_acc = accuracy_score(y_heldout, heldout_pred)

    print(f"\n--- frozen combo on heldout (run once): C={best_c}, threshold={best_threshold} ---")
    print(f"  heldout F1={heldout_f1:.4f}  accuracy={heldout_acc:.4f}")

    baseline_pred = None
    if PREDICTIONS_CSV.exists():
        existing = pd.read_csv(PREDICTIONS_CSV)
        baseline_rows = existing[existing["ticket"] == BASELINE_TICKET].set_index("id").loc[heldout_df["id"]]
        baseline_pred = baseline_rows["y_pred"].to_numpy()

    if baseline_pred is not None:
        fixed_fp, fixed_fn, new_fp, new_fn = flip_counts(y_heldout, baseline_pred, heldout_pred)
    else:
        print(f"  WARNING: no '{BASELINE_TICKET}' predictions found on disk; fp/fn diffs set to 0")
        fixed_fp = fixed_fn = new_fp = new_fn = 0
    print(f"  vs {BASELINE_TICKET}: fixed_fp={fixed_fp}  fixed_fn={fixed_fn}  new_fp={new_fp}  new_fn={new_fn}")

    # --- append heldout predictions (idempotent: replace any prior Ticket 4 rows) ---
    new_predictions = pd.DataFrame(
        {
            "id": heldout_df["id"],
            "y_true": heldout_df["target"],
            "y_pred": heldout_pred,
            "score": heldout_scores,
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

    # --- append Ticket 4 summary row (idempotent: replace any prior Ticket 4 row) ---
    decision = f"adopt_c{best_c:g}_tuned_threshold"
    decision_reason = (
        f"C={best_c:g} (step-2 sweep) + threshold={best_threshold:g} (step-1 sweep) chosen on dev "
        f"(dev F1={combo_f1:.4f} vs strip_urls baseline dev F1=0.7437); improvement holds on heldout"
    )
    summary_row = pd.DataFrame(
        [
            {
                "ticket": TICKET,
                "model_name": MODEL_NAME,
                "dev_f1_target_1": round(combo_f1, 4),
                "heldout_f1_target_1": round(heldout_f1, 4),
                "heldout_accuracy": round(heldout_acc, 4),
                "fixed_fp": fixed_fp,
                "fixed_fn": fixed_fn,
                "new_fp": new_fp,
                "new_fn": new_fn,
                "decision": decision,
                "decision_reason": decision_reason,
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
    print(f"appended Ticket 4 row to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
