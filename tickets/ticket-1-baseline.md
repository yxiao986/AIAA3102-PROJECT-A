# Ticket 1 — Baseline Discrepancy Diagnosis

## Question

Does an independently implemented TF-IDF + Logistic Regression baseline reproduce the
reference held-out metric in `configs/project_contract.json`
(`heldout_f1_target_1 = 0.7574221578566256`, tolerance `0.001`)? If not, is the gap caused
by the split, the random seed, the library version, or preprocessing?

## Hypothesis

A plain, default-configuration TF-IDF + Logistic Regression pipeline should land near the
reference. If it does not, the discrepancy is attributable to one identifiable factor rather
than a bug, and that factor can be isolated by changing one setting at a time.

## Canonical baseline (frozen)

The working baseline is deliberately the **plain scikit-learn default** — no vocabulary
pruning, no n-grams, no regularization tuning — so that vocabulary, n-grams, thresholds, and
model choices remain available as levers for later tickets.

- `TfidfVectorizer()` on the `text` column (defaults)
- `LogisticRegression(max_iter=1000, random_state=3102)`, 0.5 threshold
- scikit-learn 1.9.0, Python 3.11.15 (see `requirements.txt`)

| Split | Rows (pos/neg) | F1 (target=1) |
|-------|----------------|---------------|
| train | 4567 (1962/2605) | — |
| dev   | 1523 (655/868) | 0.7388 |
| held-out | 1523 (654/869) | **0.7492** |

Row and class counts match `data/README_DATA.md` exactly, confirming the fixed split was
loaded by id with no re-splitting.

**Result: the baseline does not reproduce the reference.** Held-out gap = `0.7492 - 0.7574 = -0.0082`,
well outside the `0.001` tolerance.

## Controlled experiment

Two diagnostic passes (`experiments/ticket1_probe.py`, results in `results/ticket1_probe.csv`).
These are **diagnosis only** — the canonical baseline above is not re-selected using held-out
scores.

### Pass 1 — one factor at a time (from the plain default)

| Variant | n_iter | converged | dev F1 | held-out F1 | gap vs ref |
|---------|:------:|:---------:|:------:|:-----------:|:----------:|
| default (`max_iter=100`) | 15 | yes | 0.7388 | 0.7492 | −0.0082 |
| `max_iter=1000` | 15 | yes | 0.7388 | 0.7492 | −0.0082 |
| `C=10` | 15 | yes | 0.7456 | 0.7623 | +0.0049 |
| `sublinear_tf=True` | 18 | yes | 0.7392 | 0.7518 | −0.0056 |
| `min_df=2` | 14 | yes | 0.7435 | 0.7552 | −0.0022 |
| `ngram_range=(1,2)` | 19 | yes | 0.7287 | 0.7270 | −0.0304 |

### Pass 2 — grid over standard TF-IDF/regularization settings

Grid: `min_df ∈ {2,3,5}` × `sublinear_tf ∈ {False,True}` × `ngram ∈ {(1,1),(1,2)}` × `C ∈ {1,2,4,10}`
(48 configs). Five land inside the `±0.001` tolerance band:

| Config | dev F1 | held-out F1 | gap |
|--------|:------:|:-----------:|:---:|
| min_df=5, sublinear_tf=True, ngram=(1,1), C=2 | 0.7549 | 0.7573 | −0.0001 |
| min_df=3, sublinear_tf=False, ngram=(1,2), C=4 | 0.7494 | 0.7571 | −0.0004 |
| min_df=2, sublinear_tf=True, ngram=(1,2), C=10 | 0.7566 | 0.7570 | −0.0005 |
| min_df=2, sublinear_tf=True, ngram=(1,2), C=2 | 0.7440 | 0.7582 | +0.0008 |
| min_df=3, sublinear_tf=True, ngram=(1,2), C=4 | 0.7469 | 0.7582 | +0.0008 |

## Findings

1. **Not the split.** Held-out row and class counts match the documented split exactly; ids
   were loaded directly, so no re-splitting occurred.
2. **Not the seed.** The solver is `lbfgs` (deterministic for this problem); the seed does not
   change the result.
3. **Not convergence.** Every variant converges in ≤19 iterations, so even scikit-learn's true
   default `max_iter=100` is enough. The `max_iter=1000` in an earlier draft baseline was
   unnecessary insurance, not a fix — it produces the identical 0.7492.
4. **Not the scikit-learn version.** The plain baseline was re-run under scikit-learn 1.3.2, 1.5.2,
   and 1.7.2 in isolated environments (`results/ticket1_versions.csv`). 1.5.2, 1.7.2, and 1.9.0 give
   *identical* held-out F1 (0.7492); 1.3.2 differs by only −0.0006 (an older lbfgs default), and
   none comes near the −0.0082 gap. Version is ruled out.
5. **Driven by TF-IDF vocabulary and regularization.** The levers that move held-out F1 toward
   the reference are vocabulary pruning (`min_df`) and weaker regularization (`C`); `sublinear_tf`
   helps mildly. Bigrams at `min_df=1` move it strongly the *wrong* way (−0.0304), so if the
   reference used bigrams it also pruned the vocabulary.
6. **The reference is reproducible within tolerance but not uniquely identifiable.** Five distinct
   standard configurations reach the tolerance band, and they do not share a single story (some use
   bigrams, some do not; `C` ranges 2–10). A single reported F1 therefore **under-determines the
   pipeline** — the contract is consistent with a conventional TF-IDF + LR baseline but cannot pin
   its exact configuration.

## Limitations

- With 48 configurations tested against a `±0.001` tolerance on a 1523-row held-out set,
  a small number of in-tolerance hits could arise partly by chance (multiple comparisons). The
  five hits are treated as evidence of *reproducibility and non-uniqueness*, not as identification
  of the true reference config.
- None of the five in-tolerance configs is also the dev-best configuration, so no single config is
  best on both splits — consistent with the metric being under-determined rather than pointing to
  one "correct" pipeline.
- After ruling out split, seed, convergence, and version, the two remaining candidates are a
  tokenizer/stopword/lowercasing difference (deferred to Ticket 2, text normalization) or a
  genuinely different train/dev/held-out assignment on the reference side. The residual is
  attributed to preprocessing rather than chased to an exact config, for the reasons in the
  limitations above.

## Decision

Adopt the plain-default pipeline as the canonical baseline (held-out F1 = 0.7492) and carry it
forward unchanged. Record in `results/summary.csv` with `decision = adopt_plain_default` and
`decision_reason = "reference reproducible within tolerance by standard TF-IDF/regularization
choices but non-unique; plain default kept so vocabulary/regularization remain Ticket 2–4 levers"`.
