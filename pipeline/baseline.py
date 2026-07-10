"""Ticket 1 baseline: TF-IDF + Logistic Regression, default hyperparameters.

Loads the fixed split from data/split_indices.json (do not regenerate it),
trains on train, reports positive-class F1 on dev and held-out, and writes
predictions/heldout_predictions.csv.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "train.csv"
SPLIT_JSON = ROOT / "data" / "split_indices.json"
PREDICTIONS_CSV = ROOT / "predictions" / "heldout_predictions.csv"

RANDOM_STATE = 3102
MODEL_NAME = "tfidf_logreg_baseline"
TICKET = "ticket-1-baseline"


def load_split(df: pd.DataFrame, ids: list) -> pd.DataFrame:
    split_df = df[df["id"].isin(ids)].copy()
    split_df = split_df.set_index("id").loc[ids].reset_index()
    return split_df


def report_split(name: str, split_df: pd.DataFrame) -> None:
    n_pos = int((split_df["target"] == 1).sum())
    n_neg = int((split_df["target"] == 0).sum())
    print(f"{name}: {len(split_df)} rows ({n_pos} positive, {n_neg} negative)")


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_df = load_split(df, split["train_ids"])
    dev_df = load_split(df, split["dev_ids"])
    heldout_df = load_split(df, split["heldout_ids"])

    report_split("train", train_df)
    report_split("dev", dev_df)
    report_split("heldout", heldout_df)

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_df["text"])
    X_dev = vectorizer.transform(dev_df["text"])
    X_heldout = vectorizer.transform(heldout_df["text"])

    y_train = train_df["target"]

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    dev_pred = clf.predict(X_dev)
    heldout_pred = clf.predict(X_heldout)
    heldout_score = clf.predict_proba(X_heldout)[:, 1]

    dev_f1 = f1_score(dev_df["target"], dev_pred, pos_label=1)
    heldout_f1 = f1_score(heldout_df["target"], heldout_pred, pos_label=1)

    print(f"dev F1 (target=1): {dev_f1:.4f}")
    print(f"heldout F1 (target=1): {heldout_f1:.4f}")

    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(
        {
            "id": heldout_df["id"],
            "y_true": heldout_df["target"],
            "y_pred": heldout_pred,
            "score": heldout_score,
            "model_name": MODEL_NAME,
            "ticket": TICKET,
        }
    )
    out_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"wrote {len(out_df)} rows to {PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
