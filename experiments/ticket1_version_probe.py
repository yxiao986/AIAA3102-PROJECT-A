"""Ticket 1 version probe: does an older scikit-learn change the plain-default baseline's
heldout F1, and does it move it toward the reference contract value (0.7574221578566256)?

Read-only diagnostic: prints results and never writes to predictions/heldout_predictions.csv
or results/summary.csv. Intended to be run once per isolated conda environment (each with a
different scikit-learn version pinned); this script itself makes no environment changes.

Model: TfidfVectorizer() + LogisticRegression(max_iter=1000, random_state=3102), pure
defaults otherwise -- identical config to pipeline/baseline.py -- on the fixed split from
data/split_indices.json.

Reproduction note: when rebuilding a legacy-sklearn env on this machine (or any CPU with
AVX-512 fused off, e.g. consumer 12th-14th gen Intel), conda-forge's default MKL-backed
build hard-crashes (illegal instruction, exit 0xC06D007F) on any BLAS call -- even a plain
numpy matmul -- because MKL's CPU dispatch mis-detects AVX-512 support. Force OpenBLAS
instead, e.g.:
    conda create -n <env> -c conda-forge python=3.11 "scikit-learn=<ver>" pandas "libblas=*=*openblas" -y
(or `conda install -n <env> -c conda-forge "libblas=*=*openblas" -y` on an existing env).
"""
import json
import platform
import sys
from pathlib import Path

import numpy
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "train.csv"
SPLIT_JSON = ROOT / "data" / "split_indices.json"

RANDOM_STATE = 3102


def load_split(df: pd.DataFrame, ids: list) -> pd.DataFrame:
    split_df = df[df["id"].isin(ids)].copy()
    split_df = split_df.set_index("id").loc[ids].reset_index()
    return split_df


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_df = load_split(df, split["train_ids"])
    dev_df = load_split(df, split["dev_ids"])
    heldout_df = load_split(df, split["heldout_ids"])

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_df["text"])
    X_dev = vectorizer.transform(dev_df["text"])
    X_heldout = vectorizer.transform(heldout_df["text"])

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, train_df["target"])

    dev_f1 = f1_score(dev_df["target"], clf.predict(X_dev), pos_label=1)
    heldout_f1 = f1_score(heldout_df["target"], clf.predict(X_heldout), pos_label=1)

    print(f"python={platform.python_version()} sklearn={sklearn.__version__} "
          f"numpy={numpy.__version__} scipy={scipy.__version__} pandas={pd.__version__}")
    print(f"dev_f1_target_1={dev_f1:.4f}")
    print(f"heldout_f1_target_1={heldout_f1:.4f}")
    print(
        f"RESULT sklearn_version={sklearn.__version__} python_version={platform.python_version()} "
        f"dev_f1={dev_f1:.4f} heldout_f1={heldout_f1:.4f}"
    )


if __name__ == "__main__":
    main()
