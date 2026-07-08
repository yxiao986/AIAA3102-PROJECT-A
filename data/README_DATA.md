# Real Starter Data

This starter repo uses the Kaggle Natural Language Processing with Disaster Tweets data from the public mirror `ucbrise/kaggle-nlp-disasters`. Students should download the public CSV files themselves.

The assignment uses `data/train.csv` for all supervised work. `data/test.csv` and `data/sample_submission.csv` are included only to preserve the Kaggle package context; they are not used by the reference contract.

## Columns

- `id`: unique integer tweet id.
- `keyword`: optional Kaggle keyword field.
- `location`: optional user-provided location.
- `text`: tweet text used by your models and audits.
- `target`: binary label, where `1` means a real disaster and `0` means not a real disaster.

## Split

`split_indices.json` contains three disjoint id lists:

- `train_ids`: 4567 rows used for fitting final models.
- `dev_ids`: 1523 rows used for choosing levers and hyperparameters.
- `heldout_ids`: 1523 rows used only for finished-ticket reporting.

Class balance in this fixed split:

- Full sample: 3271 positive, 4342 negative.
- Train split: 1962 positive, 2605 negative.
- Dev split: 655 positive, 868 negative.
- Held-out split: 654 positive, 869 negative.

The split is stratified and deterministic with seed `3102`. Do not regenerate it for submitted runs.
