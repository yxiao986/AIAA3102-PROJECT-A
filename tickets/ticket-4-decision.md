# Ticket 4 — Decision-Rule and Model Ticket

## Question

The first three tickets changed the model's *input* (cleaning, features). This one changes how
the model's probability output becomes a 0/1 decision, and whether a different CPU classifier
helps. Levers: decision threshold, regularization strength `C`, class weighting, and a second
classifier. The requirement is to explain the **precision–recall trade-off**, not just report the
best F1.

## Hypothesis

The default 0.5 threshold is not F1-optimal for this task: at 0.5 the model is precision-heavy
(cautious about predicting "disaster"), so lowering the threshold should trade precision for
recall and raise F1. Regularization `C` is a genuine lever (Ticket 1 hinted `C>1` helps). A second
linear classifier may match but is unlikely to beat a tuned Logistic Regression, since Ticket 3
showed the signal is mostly lexical and already captured.

## Method

Built on the frozen Ticket 2 pipeline (text + `strip_urls`). **Every choice is made on dev; the
held-out split is scored exactly once, on the final frozen configuration.** Alternate classifiers
were evaluated on dev only — their held-out scores were deliberately not computed, so selection
cannot be contaminated by held-out peeking (`experiments/ticket4_decision.py`, dev tables in
`results/ticket4_dev.csv` and `results/threshold_sweep.csv`).

Two search-design choices are deliberate. **`C` is searched on a logarithmic grid**
({0.3, 1, 3, 10, 30}) because regularization acts on an order-of-magnitude scale; the dev curve is
smooth and single-peaked, so the optimum is bracketed rather than stepped over, and a finer grid
would chase F1 differences within dev-set noise — the same overfitting-the-validation-set risk the
discipline guards against (cross-validation on train would be a more robust selector still).
**The candidate classifiers are those suited to high-dimensional sparse TF-IDF on CPU:** LinearSVC
(a margin/hinge-loss linear model — a genuinely different decision boundary from LR), MultinomialNB
(the classic probabilistic text classifier, a different modeling assumption), and SGD(log_loss)
(LR by another optimizer, a sanity check). Tree ensembles and kNN are ill-suited to ~15k sparse
lexical features, and neural models fall outside the CPU/simple-baseline scope.

## Evidence (dev)

### Threshold sweep (Logistic Regression, C=1) — the precision–recall curve

| threshold | precision | recall | F1 |
|:---------:|:---------:|:------:|:--:|
| 0.20 | 0.498 | 0.962 | 0.656 |
| 0.30 | 0.581 | 0.884 | 0.701 |
| 0.35 | 0.637 | 0.835 | 0.723 |
| 0.40 | 0.705 | 0.780 | 0.741 |
| **0.45** | 0.764 | 0.733 | **0.748** |
| 0.50 (default) | 0.824 | 0.678 | 0.744 |
| 0.60 | 0.895 | 0.536 | 0.671 |
| 0.70 | 0.962 | 0.391 | 0.556 |

The default 0.5 is precision-heavy (P=0.82, R=0.68) — it misses ~32% of real-disaster tweets.
Dev F1 peaks at **threshold 0.45** (0.748). The full curve shows the operating point is fully
dial-able: recall can be pushed from 0.39 up to 0.96 by lowering the threshold.

### Other levers (dev)

| Lever | dev P | dev R | dev F1 |
|-------|:-----:|:-----:|:------:|
| C=0.3 | 0.844 | 0.595 | 0.698 |
| C=1 (base) | 0.824 | 0.678 | 0.744 |
| **C=3** | 0.802 | 0.705 | **0.751** |
| C=10 | 0.787 | 0.705 | 0.744 |
| C=30 | 0.763 | 0.702 | 0.731 |
| class_weight=balanced | 0.759 | 0.737 | 0.748 |
| LinearSVC | 0.778 | 0.696 | 0.735 |
| MultinomialNB | 0.869 | 0.628 | 0.729 |
| SGDClassifier(log_loss) | 0.802 | 0.710 | 0.753 |
| **C=3 + threshold=0.45 (combo)** | 0.778 | 0.739 | **0.758** |

## Held-out (frozen combo, scored once)

Frozen decision: **C=3, threshold=0.45**.

| Pipeline | F1 (target=1) | accuracy |
|----------|:-------------:|:--------:|
| Ticket 2 (strip_urls, C=1, 0.5) | 0.7536 | 0.8063 |
| **Ticket 4 (C=3, threshold=0.45)** | **0.7653** | 0.8011 |

Flips vs the Ticket 2 pipeline: `fixed_fp=2, fixed_fn=43, new_fp=53, new_fn=0`.

## Findings

1. **The gain is almost entirely a recall move.** The frozen change recovers **43 missed
   real-disaster tweets** (`fixed_fn=43`) at the cost of **53 new false alarms** (`new_fp=53`),
   with essentially no movement in the other two cells. It is a pure operating-point shift toward
   recall, exactly what lowering the threshold from 0.5 to 0.45 predicts.
2. **F1 rose while accuracy fell — the point of not reporting one number.** The held-out confusion
   matrices (654 disasters, 869 non-disasters) make the trade concrete:

   | pipeline | TP | FP | FN | TN | precision | recall |
   |----------|:--:|:--:|:--:|:--:|:---------:|:------:|
   | Ticket 2 (thr 0.5) | 451 | 92 | 203 | 777 | 0.831 | 0.690 |
   | Ticket 4 (thr 0.45) | 494 | 143 | 160 | 726 | 0.776 | 0.755 |

   Lowering the threshold recovers 43 missed disasters (FN→TP) at the cost of 51 more false alarms
   (TN→FP). **Accuracy falls** (F1 0.7536→0.7653, acc 0.8063→0.8011) because it counts all 1523 rows
   equally — 43 gains vs 51 losses is a net −8 correct. **F1 rises** because its formula
   (`2TP/(2TP+FP+FN)`) has no TN term: it sees only the positive class, where recall (0.690→0.755)
   climbs far more than precision (0.831→0.776) dips. The two metrics diverge because they encode
   different values — accuracy treats a missed disaster and a false alarm as equally bad, F1 treats
   catching disasters as the goal. Reporting only accuracy would call this harmful; reporting only F1
   would hide the 51 new false alarms. The flip breakdown is what makes the trade-off legible.
3. **`C=3` is a real, dev-confirmed model improvement** (dev F1 0.744 → 0.751), independent of the
   threshold. Beyond C=3 it overfits (C=30 drops to 0.731).
4. **Class weighting and low thresholds are two routes to the same place.** `class_weight=balanced`
   up-weights the minority positive class during training — a *train-time* analogue of lowering the
   threshold — so it lands almost exactly where threshold 0.45 does (P=0.759, R=0.737), both
   rebalancing toward recall. They are not stacked; the explicit threshold is kept because it
   exposes the whole curve rather than a single re-weighted point.
5. **No second classifier beats the tuned Logistic Regression on dev.** LinearSVC (0.735) and
   MultinomialNB (0.729) are worse; SGDClassifier(log_loss) (0.753) essentially *is* logistic
   regression by another optimizer and does not beat the C=3 + threshold combo (0.758). Consistent
   with Ticket 3: the signal is lexical and already captured, so a different linear boundary adds
   nothing.

## Discipline note

The alternate classifiers were scored on **dev only**; their held-out F1 was never computed. This
is deliberate — model selection is the step where held-out peeking is most tempting, so the
candidates were ranked purely on dev and only the single dev-winning configuration touched
held-out. LinearSVC being dev-worst is enough to reject it; we did not need, and did not take, a
look at its held-out number.

## Operating-point choice (the trade-off, made explicitly)

We adopt the dev-F1-optimal threshold (0.45) because F1 is the contract metric. But the sweep
makes clear this is a *choice*, not a law: for a real disaster-alerting system a missed disaster
(false negative) is costlier than a false alarm (false positive), which would justify a *lower*
threshold (e.g. 0.35: recall 0.835) and a deliberately lower-precision operating point. The value
of the sweep is that it lets this decision be made on the cost structure of the application rather
than on a single F1 number.

## Limitations

- Threshold, `C`, and class weight all move the same precision–recall dimension, so their gains
  overlap; the combined dev F1 (0.758) is not the sum of individual gains.
- The threshold is a hyperparameter tuned on dev, which is legitimate, and it generalized *within*
  this dataset — the dev-optimal 0.45 also improved held-out (0.7536 → 0.7653), so it was not a
  dev-specific fluke. But it is calibrated to this data's class balance and score distribution: on a
  differently balanced stream the F1-optimal threshold would move, so 0.45 is not a universal
  constant. Crucially this affects only the operating point, not the model — the learned ranking
  (the PR curve's shape) transfers, and one would re-calibrate the threshold on a sample of the new
  distribution rather than retrain.

## Decision

Adopt **C=3 + threshold=0.45** on the strip_urls pipeline
(`decision = adopt_c3_tuned_threshold`). Reject the alternate classifiers and class weighting.
Held-out predictions written under `model_name = tfidf_logreg_stripurls_c3_tuned`; Ticket 4 row
appended to `results/summary.csv`.
