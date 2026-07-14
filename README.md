# Topic A — Text Classification Pipeline Forensics

Diagnosing a noisy Kaggle Disaster Tweets classifier: reproduce a reference baseline,
then investigate which pipeline changes are genuinely trustworthy (not just higher-scoring).

Binary task: `target=1` = tweet describes a real disaster, `target=0` = it does not.
Metric of record: **F1 on the positive class** (`target=1`), reported on the held-out split.

## Environment

CPU-only. Managed with conda (env name `topica`, Python 3.11.15).

```powershell
conda create -n topica python=3.11 -y
conda activate topica
conda install pandas scikit-learn -y
# exact pinned versions are in requirements.txt
```

Package versions are pinned in `requirements.txt` and must match, because reproducing the
reference baseline metric is sensitive to the scikit-learn version.

## Data

Public source: Kaggle "Natural Language Processing with Disaster Tweets", mirrored at
`https://github.com/ucbrise/kaggle-nlp-disasters/tree/master/data`.

Only the full labeled `train.csv` is used. Download it and place it at `data/train.csv`
(it is git-ignored and not committed). The Kaggle `test.csv` / `sample_submission.csv` are
**not** used (their labels are hidden).

Columns: `id, keyword, location, text, target`.

## Fixed split (do not regenerate)

`data/split_indices.json` assigns every Kaggle `id` to one of three disjoint splits
(stratified, seed 3102):

| Split | Rows | Use |
|-------|------|-----|
| `train_ids`   | 4567 | fit final models |
| `dev_ids`     | 1523 | choose preprocessing, thresholds, model variants |
| `heldout_ids` | 1523 | report finished-ticket results **after** decisions are frozen |

Discipline: all tuning happens on **dev**. The **held-out** split is scored once per ticket,
after that ticket's decision is frozen. Do not move ids between splits.

## Reference contract

`configs/project_contract.json` gives the reference held-out metric:
`heldout_f1_target_1 = 0.7574221578566256` (tolerance `0.001`), floor F1 `0.0`.
Ticket 1 diagnoses whether this project's independently-built baseline reproduces that value,
and if not, why (split / seed / version / preprocessing).

## Repository layout

```
project_A/
├── report.pdf                     # final IEEE-format project report
├── README.md                      # this file: setup, data, reproduce commands
├── requirements.txt               # pinned CPU stack (scikit-learn 1.9.0, pandas 3.0.3)
│
├── data/
│   ├── split_indices.json         # fixed train/dev/heldout split by id (do not regenerate)
│   ├── README_DATA.md             # data source and split documentation
│   └── train.csv                  # Kaggle labeled data (download; git-ignored)
├── configs/
│   └── project_contract.json      # reference held-out F1 (0.7574) + tolerance
│
├── pipeline/
│   ├── baseline.py                # plain-default TF-IDF + Logistic Regression baseline
│   └── normalize.py               # composable text-normalization transforms (Ticket 2+)
├── experiments/                   # one or more scripts per ticket
│   ├── ticket1_probe.py           # T1: discrepancy probes (one-factor + grid)
│   ├── ticket1_version_probe.py   # T1: per-env version check  (+ ticket1_versions_compile.py)
│   ├── ticket2_profile.py         # T2: surface-feature profile
│   ├── ticket2_normalization.py   # T2: dev sweep of transforms  (+ ticket2_freeze.py, ticket2_punct_probe.py)
│   ├── ticket3_shortcuts.py       # T3: shallow-feature shortcut audit
│   ├── ticket4_decision.py        # T4: threshold / C / model sweep
│   └── ticket5_audit.py           # T5: data-quality detectors  (+ ticket5_manual_review.py)
│
├── tickets/                       # per-investigation write-ups
│   └── ticket-1-baseline.md ... ticket-5-data-quality.md
├── predictions/
│   └── heldout_predictions.csv    # one block per pipeline-changing ticket (T1, T2, T4)
├── results/                       # machine-checkable artifacts (all committed)
│   ├── summary.csv                # per-ticket metrics + prediction-change counts
│   ├── threshold_sweep.csv        # T4 precision–recall curve
│   ├── data_quality_audit.csv     # T5 audit (id, issue_type, evidence, disposition, confidence)
│   └── ticket{1,2,3}_*.csv        # per-ticket probe/profile tables
└── logs/
    └── chat.md                    # AI interaction log
```

## Reproduce

Run everything inside the `topica` env, from the project root. Scripts write their outputs to
`results/` and `predictions/`. Each ticket's script depends only on `data/` + the frozen decisions
of earlier tickets, so they can be run in order.

```powershell
conda activate topica

# Baseline (Ticket 1): plain-default TF-IDF + Logistic Regression.
# Prints split counts + dev/held-out F1(target=1); writes the Ticket 1 prediction block.
python pipeline/baseline.py

# Ticket 1 — baseline discrepancy diagnosis (one-factor + grid probes vs the reference 0.7574).
#   -> results/ticket1_probe.csv
python experiments/ticket1_probe.py

# Ticket 2 — text normalization.
#   Surface-feature profile that motivates the lever choices -> results/ticket2_profile.csv
python experiments/ticket2_profile.py
#   Dev sweep of URL/mention/hashtag/HTML/emoji/case transforms -> results/ticket2_dev.csv, ticket2_examples.csv
python experiments/ticket2_normalization.py
#   Freeze strip_urls and score held-out once
python experiments/ticket2_freeze.py
#   Supplementary: punctuation-preserving tokenizer probe (declined) -> results/ticket2_punct_probe.csv
python experiments/ticket2_punct_probe.py

# Ticket 3 — feature & shortcut audit. Shallow-feature-only baselines vs text-only.
#   -> results/ticket3_shortcuts.csv
python experiments/ticket3_shortcuts.py

# Ticket 4 — decision rule & model. Threshold sweep, C tuning, class weighting, alt classifiers
#   (all on dev); freeze C=3 + threshold=0.45 and score held-out once.
#   -> results/threshold_sweep.csv, results/ticket4_dev.csv
python experiments/ticket4_decision.py

# Ticket 5 — data-quality audit. Duplicate-conflict + confident-disagreement detection,
#   then apply the manual adjudication overlay.
#   -> results/data_quality_audit.csv
python experiments/ticket5_audit.py
python experiments/ticket5_manual_review.py
```

`pipeline/normalize.py` is the shared normalization module imported by the Ticket 2+ scripts.
`predictions/heldout_predictions.csv` and `results/summary.csv` accumulate one block/row per ticket
that changes the pipeline (Tickets 1, 2, 4); each script replaces only its own ticket's rows.
Held-out is scored exactly once per ticket, at that ticket's freeze step.

### Optional: version-sensitivity diagnostic (Ticket 1)

`results/ticket1_versions.csv` (does the reference gap come from the scikit-learn version?) is
produced by re-running the plain baseline under older scikit-learn in isolated environments. This
needs separate envs, not the `topica` env:

```powershell
# For each version, create an isolated env; force OpenBLAS — the conda-forge MKL build
# crashes (illegal instruction) on this CPU (i9-14900HX, AVX-512 fused off).
conda create -n topica_sk1_5 python=3.11 "scikit-learn=1.5" "libblas=*=*openblas" pandas -y
conda run -n topica_sk1_5 python experiments/ticket1_version_probe.py   # repeat per env
python experiments/ticket1_versions_compile.py   # aggregate -> results/ticket1_versions.csv
```

## Output artifacts (stable schemas)

- `predictions/heldout_predictions.csv` — `id, y_true, y_pred, score, model_name, ticket`
- `results/summary.csv` — `ticket, model_name, dev_f1_target_1, heldout_f1_target_1, heldout_accuracy, fixed_fp, fixed_fn, new_fp, new_fn, decision, decision_reason`
- `results/threshold_sweep.csv` — `ticket, threshold, precision_target_1, recall_target_1, f1_target_1`
- `results/data_quality_audit.csv` — `id, issue_type, evidence, disposition, confidence`
  (disposition ∈ `fix`, `keep_but_flag`, `ambiguous`, `reject_false_positive`)

## AI usage

AI tools were used during development; see `logs/chat.md`. All analysis, experiment design,
and conclusions were reviewed and verified by the author.
