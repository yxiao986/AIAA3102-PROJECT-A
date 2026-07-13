"""One-off compiler for results/ticket1_versions.csv: combines the per-environment
outputs from experiments/ticket1_version_probe.py (run manually in each isolated
scikit-learn version's conda env -- probes cannot cross environments in a single process)
against the reference contract value. Read-only w.r.t. the frozen pipeline artifacts.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_JSON = ROOT / "configs" / "project_contract.json"
OUT_CSV = ROOT / "results" / "ticket1_versions.csv"

with open(CONTRACT_JSON, "r", encoding="utf-8") as f:
    contract = json.load(f)
reference_f1 = contract["reference_baseline_f1"]
tolerance = contract["tolerance"]

# Captured from experiments/ticket1_version_probe.py, run once per isolated conda env
# (topica_sk1_3, topica_sk1_5, topica_sk1_7, and the project's topica env for 1.9.0).
# All envs: python 3.11.15, pandas 3.0.3, conda-forge, OpenBLAS-backed (the default
# MKL build in these conda-forge envs hard-crashes on this machine's CPU -- see notes).
rows = [
    {"sklearn_version": "1.3.2", "numpy_version": "1.26.4", "scipy_version": "1.17.1", "dev_f1_target_1": 0.7378, "heldout_f1_target_1": 0.7486, "note": "conda-forge topica_sk1_3 env; MKL build crashed on this CPU (AVX512 fused off on i9-14900HX), reinstalled with libblas=*openblas"},
    {"sklearn_version": "1.5.2", "numpy_version": "2.4.6", "scipy_version": "1.17.1", "dev_f1_target_1": 0.7388, "heldout_f1_target_1": 0.7492, "note": "conda-forge topica_sk1_5 env, openblas"},
    {"sklearn_version": "1.7.2", "numpy_version": "2.4.6", "scipy_version": "1.17.1", "dev_f1_target_1": 0.7388, "heldout_f1_target_1": 0.7492, "note": "conda-forge topica_sk1_7 env, openblas"},
    {"sklearn_version": "1.9.0", "numpy_version": "2.4.6", "scipy_version": "1.17.1", "dev_f1_target_1": 0.7388, "heldout_f1_target_1": 0.7492, "note": "project's pinned topica env (requirements.txt); current baseline"},
]

df = pd.DataFrame(rows)
df["heldout_gap_vs_reference"] = (df["heldout_f1_target_1"] - reference_f1).round(4)
df["within_tolerance"] = df["heldout_gap_vs_reference"].abs() <= tolerance
df["reference_f1"] = round(reference_f1, 4)

df = df[
    [
        "sklearn_version",
        "numpy_version",
        "scipy_version",
        "dev_f1_target_1",
        "heldout_f1_target_1",
        "reference_f1",
        "heldout_gap_vs_reference",
        "within_tolerance",
        "note",
    ]
]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)
print(df.drop(columns=["note"]).to_string(index=False))
print(f"\nwrote {len(df)} rows to {OUT_CSV}")
