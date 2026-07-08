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
├── data/
│   ├── split_indices.json      # fixed split (committed)
│   └── train.csv               # downloaded, git-ignored
├── configs/
│   └── project_contract.json   # reference baseline metric + tolerance
├── pipeline/                   # data loading, model, evaluation code
├── experiments/                # per-ticket experiment scripts
├── predictions/                # heldout_predictions.csv (per frozen ticket)
├── results/                    # summary.csv, threshold_sweep.csv, data_quality_audit.csv (git-ignored)
├── tickets/                    # ticket-1-baseline.md ... ticket-5-data-quality.md
├── logs/                       # chat.md (AI interaction log)
├── requirements.txt
└── README.md
```

## Reproduce

> Commands are added here as each stage lands. Run everything inside the `topica` env.

```powershell
conda activate topica

# 1. Baseline (Ticket 1): load split, train TF-IDF + Logistic Regression,
#    evaluate positive-class F1 on dev and held-out, export predictions.
python pipeline/baseline.py
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
