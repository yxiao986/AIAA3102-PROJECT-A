# Ticket 2 — Text Normalization Lever

## Question

Does normalizing the tweet text before TF-IDF (URLs, mentions, hashtags, HTML entities,
emoji, letter case) help or hurt the positive-class F1? For any change that moves the score,
*which* tweets flip, and does the mechanism match the hypothesis — genuine cleaning, or the
removal of a spurious shortcut?

## Hypothesis

The default `TfidfVectorizer` already lowercases and strips punctuation via its
`\b\w\w+\b` token pattern, so several "normalizations" should be near no-ops. The
transforms that can matter are the ones the default tokenizer handles poorly: URLs,
`@mentions`, HTML entities, and emoji. URL tokens in particular (`http`, `t`, `co`) carry no
disaster semantics, so removing them should reduce spurious signal without hurting — unless the
model is leaning on URL presence as a shortcut, in which case some borderline predictions will
flip both ways.

## Data profile (motivation, run after the fact)

The transforms below started as a conventional normalization checklist. A surface-feature profile
(`results/ticket2_profile.csv`) run afterwards shows the checklist was not principled and corrects
it in two directions: **emoji never occur** in the 7613 rows (so `strip_emoji` tests an empty set —
a redundant confirmatory null), and **punctuation** was *omitted* despite `!` correlating with the
label (−0.173), so a punctuation-preserving variant is added below. The lesson — profile before
choosing what to clean — is recorded rather than hidden. The profile also flags `has_url` (+0.245)
as the strongest positive correlate, which motivates the URL test.

## Lever

Text normalization applied to the raw `text` column before vectorization. Implemented as
independent, composable transforms in `pipeline/normalize.py`
(`strip_urls`, `strip_mentions`, `strip_hashtag_symbol`, `unescape_html`, `strip_emoji`) plus a
`preserve_case` switch that sets the vectorizer's `lowercase=False`. The `normalize()` composer
is a true no-op when no flag is set (verified: it reproduces the Ticket 1 dev F1 of 0.7388
exactly).

## Controlled experiment

Decisions are made on **dev**. Each transform is applied one at a time off the no-cleaning
baseline (`experiments/ticket2_normalization.py`, table in `results/ticket2_dev.csv`). Flip
counts are computed against the baseline's dev predictions:
`fixed_fp` = baseline false positive now correct, `fixed_fn` = baseline false negative now correct,
`new_fp` / `new_fn` = previously correct, now wrong.

### Dev evidence (one factor at a time)

| Variant | dev F1 | fixed_fp | fixed_fn | new_fp | new_fn | net flips |
|---------|:------:|:--------:|:--------:|:------:|:------:|:---------:|
| baseline (no cleaning) | 0.7388 | — | — | — | — | 0 |
| **strip_urls** | **0.7437** | 36 | 10 | 11 | 20 | **+15** |
| strip_mentions | 0.7366 | 0 | 1 | 2 | 2 | −3 |
| strip_hashtag_symbol | 0.7388 | 0 | 0 | 0 | 0 | 0 |
| unescape_html | 0.7384 | 2 | 0 | 1 | 1 | 0 |
| strip_emoji | 0.7388 | 0 | 0 | 0 | 0 | 0 |
| preserve_case | 0.7311 | 14 | 12 | 13 | 20 | −7 |

Only `strip_urls` clearly helps. It is frozen and scored **once** on held-out
(`experiments/ticket2_freeze.py`).

### Held-out evidence (frozen decision, scored once)

| Pipeline | F1 (target=1) | accuracy |
|----------|:-------------:|:--------:|
| baseline (no cleaning) | 0.7492 | 0.7978 |
| **strip_urls** | **0.7536** | **0.8063** |

Held-out flips vs baseline: `fixed_fp=29, fixed_fn=13, new_fp=7, new_fn=22` (net **+13**
corrected). The dev improvement holds on held-out and is dominated by false-positive cleanup.

## Findings

1. **`strip_urls` helps by removing a shortcut, not by "cleaning noise".** The 29/36 fixed
   false positives are non-disaster tweets whose only disaster-like feature was a link:
   - id 174 — *"…Financial Meltdown by David Wiedemer http http://t.co/…"* (a book title; note the
     bare `http http`), truth 0, baseline predicted 1.
   - id 386 — *"Please sign & RT to save #SaltRiverWildHorses http://t.co/…"*, truth 0.
   - id 971 — *"The mixtape is coming i promise… http://t.co/…"*, truth 0.
   - id 1888 — *"Kanger coils - burning out fast? … http://t.co/…"* (vaping; "burning" is a
     metaphor), truth 0.

   In training, links skew toward the disaster class (real-disaster news links to articles), so
   the token `http` became a weak positive cue. This is a **dataset shortcut**, not disaster
   semantics.
2. **The same shortcut also propped up some true positives.** The 20–22 new false negatives are
   genuine disasters that were only crossing the 0.5 threshold *because* of URL weight:
   - id 2193 / 2198 — *"Learning from the Legacy of a Catastrophic Eruption - The New Yorker
     http://t.co/…"*, truth 1.
   - id 2364 — *"Roof collapsed a bowling alley… http://t.co/…"*, truth 1.
   - id 2103 — *"Small casualty on the way to Colorado http://t.co/…"*, truth 1.

   Removing the shortcut costs these borderline cases. Net effect is still positive because
   false-positive cleanup outweighs the lost borderline positives.
3. **Case is not useful signal here.** `preserve_case` *lowers* F1 (0.7388 → 0.7311): keeping
   case fragments the vocabulary (`FIRE` / `Fire` / `fire` become distinct tokens), diluting
   signal more than any all-caps urgency cue adds. Default lowercasing is retained.
4. **Hashtag-symbol and emoji handling are no-ops.** Both produce *zero* flips. For hashtags the
   default `\b\w\w+\b` tokenizer already splits on `#` (so `#fire` was already `fire`); for emoji
   the reason is stronger — the profile shows the dataset contains **no emoji at all**, so
   `strip_emoji` was testing an empty set. Either way the characters were never seen, visible only
   because flips were measured rather than assumed.
5. **Mentions and HTML unescaping are rejected.** `strip_mentions` slightly hurts (−3 net);
   `unescape_html` is a wash (too few `&amp;`-style entities in this data to matter).
6. **A punctuation-preserving tokenizer helps slightly but is declined.** Keeping `!`/`?` as tokens
   (`results/ticket2_punct_probe.csv`) gives a small dev gain (0.7437 → 0.7483). It is not adopted:
   the profile shows `!` is a negatively-correlated *posting-style* feature, so the gain comes from
   a stylistic artifact — the mirror image of the shortcut `strip_urls` removes, and exactly the
   kind of fragile cue this project declines.

## Robustness note

The hidden stress variants perturb URLs. `strip_urls` makes the pipeline **invariant to URL
perturbation** because URLs are discarded before vectorization. So this lever is defensible on
two independent grounds — a modest, mechanism-explained held-out gain *and* robustness to the
URL-shortcut stress test — not on the +0.0044 F1 alone.

## Limitations

- The URL signal is entangled: `strip_urls` removes it as a false-positive source but also as a
  weak true-positive crutch, so the net gain (+13 held-out) understates how much genuine
  reshaping happened (42 corrected vs 29 broken). Whether URL *presence* is ever legitimate task
  information is not settled here — it is handed to Ticket 3 (shortcut audit).
- Most effects are measured under scikit-learn's default tokenizer. One tokenizer change was tested
  directly (the punctuation-preserving variant, finding 6); other tokenizer redefinitions
  (e.g. keeping single characters) remain out of scope, though the profile shows the highest-value
  candidates (emoji, punctuation) are already covered.

## Decision

Adopt `strip_urls` as the normalization step (`decision = adopt_strip_urls`). Reject mentions,
HTML unescape, hashtag-symbol, emoji, case preservation, and the punctuation-preserving tokenizer
(a stylistic-artifact gain, finding 6). Recorded in `results/summary.csv`; Ticket 2 held-out
predictions written to `predictions/heldout_predictions.csv` under
`model_name = tfidf_logreg_stripurls`.
