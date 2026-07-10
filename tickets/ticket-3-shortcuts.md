# Ticket 3 — Feature and Shortcut Audit

## Question

How much of the task is solvable by shallow surface features — the `keyword` field, text
length, URL presence, hashtag/mention counts, capitalization, `location` presence — without
reading the words? For each, is the signal legitimate task information, a dataset artifact, or a
mix? And does adding these features to the text model actually help?

## Hypothesis

A tweet's disaster label should depend on its *content*, so surface features ought to be weak.
If instead a single categorical field or a length statistic recovers most of the text model's
F1, that signal is at least partly a **collection artifact** rather than genuine understanding —
and the hidden stress variants (which perturb metadata-like shortcuts) would punish a model that
relies on it.

## Method

Shallow-feature-only Logistic Regression models, trained on train, scored on dev and held-out.
Each isolates one feature or a metadata bundle, with **no text tokens**, and is compared to the
text-only baseline (`experiments/ticket3_shortcuts.py`, table in
`results/ticket3_shortcuts.csv`). This is **diagnostic only** — the frozen pipeline stays
text-only + `strip_urls`; nothing here is adopted.

## Evidence

| Model | n features | dev F1 | held-out F1 |
|-------|:----------:|:------:|:-----------:|
| floor (majority class) | 1 | 0.0000 | 0.0000 |
| **keyword_only** (one-hot) | 222 | 0.6599 | **0.6905** |
| char_len_only | 1 | 0.4345 | 0.4315 |
| **has_url_only** | 1 | 0.5851 | **0.6110** |
| n_hashtags_only | 1 | 0.0350 | 0.0520 |
| frac_upper_only | 1 | 0.0000 | 0.0000 |
| has_location_only | 1 | 0.0000 | 0.0000 |
| metadata_numeric_bundle | 5 | 0.5127 | 0.5527 |
| all_shortcuts (metadata + keyword) | 227 | 0.6537 | 0.6853 |
| text_only (baseline) | 15269 | 0.7388 | 0.7492 |
| text + all_shortcuts | 15496 | 0.7323 | 0.7564 |

### Keyword-only, top weighted keywords

- **Positive (→ real disaster):** derailment, debris, rescuers, wreckage, oil spill, typhoon,
  outbreak, suicide bombing, bombing, wildfires, evacuated, forest fires, terrorist — concrete,
  literal disaster nouns.
- **Negative (→ not a disaster):** aftershock, body bags, bloody, blazing, ruin, wrecked,
  hellfire, panicking, explode, obliterate, blew up, electrocute, blizzard — disaster-adjacent
  words that in this dataset appear mostly in figurative / slang / hyperbolic tweets.

## Findings

1. **The shortcut ceiling is high.** The `keyword` field alone reaches held-out F1 0.69 — about
   92% of the text model's 0.749 — from one categorical column. A pure numeric metadata bundle
   (length, url flag, counts) reaches 0.55, and even a single "contains a link" bit reaches 0.61.
   Most of this task is solvable by surface features *without* reading the tweet. The dataset
   looks more interpretable than it is.
2. **Shortcuts are redundant with text, not additive.** Adding all shortcuts to the text model
   gives no dev-confirmed gain: dev F1 *drops* (0.7388 → 0.7323) while held-out drifts up
   (0.7492 → 0.7564). Because the two splits disagree and dev is the decision surface, this is
   treated as noise, not an improvement — the metadata is not adopted. The text model already
   encodes these cues (the keyword word and the `http` token are in the text), so the extra
   columns are duplicated signal.
3. **`keyword` is mixed — legitimate topic + dataset-specific memorization.** The positive
   weights are genuine disaster vocabulary (derailment, wreckage, typhoon), which is real task
   information. But the negative weights show the model learning per-keyword *base rates* in this
   dataset — that tweets tagged `blazing`, `bloody`, `explode`, `blew up` happened to be mostly
   non-disaster here. That is memorization of how Kaggle selected tweets per keyword, not language
   understanding, and it would not transfer to a new sample. Legitimate core, artifact-inflated
   strength.
4. **`has_url` is an artifact.** A single presence bit yielding 0.61 F1 is a posting-style
   correlate — accounts sharing real-disaster news links, not disaster content. This cross-checks
   Ticket 2: URL presence is a genuine label correlate in this data, and the text model was
   reaching it through URL tokens (`http`, `t`, `co`), which is why `strip_urls` shifted
   predictions. Our pipeline discards URLs, so it exploits neither.
5. **`char_len` is an artifact.** Length alone reaches 0.43 — a writing-style effect (real
   disaster reports are denser/more informative than casual tweets), with no causal link to
   whether a disaster occurred. Fragile signal.
6. **`frac_upper` and `has_location` carry no usable standalone signal.** Both collapse to F1 = 0
   at the 0.5 threshold — neither pushes any row to a positive prediction on its own.

## Robustness note

The hidden stress variants perturb metadata-like shortcuts (URLs, case, hashtags). The audit
shows why the frozen pipeline is reasonably safe: it uses none of these as explicit features, and
the one shortcut the text model *did* exploit — URL tokens — is already removed by `strip_urls`.
Case and hashtag symbols were shown in Ticket 2 to be no-ops under the default tokenizer. The main
residual exposure is lexical: the text model still benefits from the same disaster words that make
`keyword` predictive, which is legitimate content signal, not a metadata shortcut.

## Limitations

- "Legitimate vs artifact" for `keyword` is a judgment on a spectrum, not a measurement. The
  concrete-noun positive weights are defensible as task signal; the per-keyword base-rate
  memorization is the artifact side. The split is argued from the weight lists, not proven.
- Standalone F1 at a fixed 0.5 threshold understates features like `frac_upper`: they may carry
  ranking information that never crosses the threshold alone but could contribute in combination.
  The audit measures usable-in-isolation signal, not all signal.

## Decision

No shortcut or metadata feature is added to the pipeline (`decision = keep_text_only`). The frozen
pipeline remains text-only + `strip_urls`. The audit's value is diagnostic: it quantifies a high
surface-feature ceiling (~0.69 of 0.749 reachable without reading text), classifies `keyword` as
mixed, and `has_url` / `char_len` as artifacts, and confirms that removing URLs (Ticket 2) was the
right call for both accuracy and robustness.
