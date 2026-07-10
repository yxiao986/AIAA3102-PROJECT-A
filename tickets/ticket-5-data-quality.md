# Ticket 5 — Data-Quality and Error Ticket

## Question

Which duplicates, likely mislabels, and ambiguous tweets limit the achievable score, and how
should each be handled? The goal is judgment, not volume: separate what should be **fixed**, what
should be **kept but flagged**, what is a genuine **ambiguous** policy case, and which suspicious
candidates should be **rejected as audit false positives**.

## Hypothesis

A keyword-collected dataset will contain label noise concentrated where a disaster keyword is used
figuratively (a tweet tagged `hurricane` that is actually about sunburn). If so, some of the score
ceiling is imposed by the labels themselves, not the model — and not every model–label
disagreement is a mislabel, so the audit must also reject false positives.

## Method

Two systematic detectors, then manual adjudication into the four dispositions
(`experiments/ticket5_audit.py` + `experiments/ticket5_manual_review.py`, output
`results/data_quality_audit.csv`):

1. **Exact-duplicate label conflicts** — normalize text (whitespace, case), group, keep groups
   whose members do not share a single label.
2. **Confident model–label disagreements** — LR(C=3) on the strip_urls pipeline; surface
   `label=1, p<0.10` and `label=0, p>0.90` rows on train+dev.

**Hard rule enforced in code:** held-out rows are never assigned a corrected label. Any held-out
row is `keep_but_flag` with a blank `proposed_label`, regardless of how lopsided its duplicate
group is. Where a fix is proposed, both the original and proposed labels are retained.

## Evidence

`results/data_quality_audit.csv` — 78 rows:

| issue_type | disposition | count |
|------------|-------------|:-----:|
| duplicate_label_conflict | fix | 31 |
| duplicate_label_conflict | ambiguous | 15 |
| duplicate_label_conflict | keep_but_flag (held-out) | 10 |
| likely_mislabel | fix | 16 |
| likely_mislabel | ambiguous | 3 |
| reject_false_positive | reject_false_positive | 3 |

### Examples by disposition

- **fix (duplicate).** `cleared:incident with injury:i-495 inner loop exit 31…` appears three times
  labeled `[1,1,0]`; the lone `0` (id 6566) is corrected to `1`. `#allah describes piling up wealth…`
  appears `[0,0,1]`; the lone `1` (id 6123) is corrected to `0`.
- **fix (mislabel).** Figurative disaster-keyword tweets wrongly labeled `1`, corrected to `0`:
  id 6407 (`hurricane`) *"My back is so sunburned :("*; id 9276 (`sunk`) *"Aquarium Ornament … Sunk
  Ship … Fish Tank Decor - eBay"*; id 10823 (`wrecked`) *"Did you get wrecked again?"*; id 796
  (`battle`) a Star Wars toy listing; id 9780 (`trapped`) *"trapped in my room cuz my bathroom
  being remodeled"*.
- **ambiguous.** id 6012/6017 *"caution: breathing may be hazardous to your health."* labeled
  `[1,0]` — a genuine public-health-warning-vs-joke tie; id 467 (`armageddon`) is the bare word
  *"Armageddon"* plus a link, too little content to decide.
- **reject_false_positive.** id 5330 (`fire`) *"I'm On Fire."* labeled `0`, model p=0.92 — an idiom,
  the `0` is correct and the model over-triggers on `fire`; id 7761 (`police`) *"Police expand
  search for missing pregnant woman"* labeled `0`, model p=0.93 — a single missing-person search is
  not a "disaster" under this dataset's framing; id 4395 (`earthquake`) commentary on the
  `#Megaquake` story labeled `1`, model p=0.09 — the `1` is defensible, the model just misreads
  commentary. None of these three is changed.

## Findings

1. **Duplicate label conflicts impose an irreducible score ceiling.** 18 groups of identical text
   carry conflicting labels. Because identical text yields an identical prediction, any
   deterministic model must miss at least the minority members of each group — **22 rows are
   guaranteed wrong in the best case, 10 of them in held-out**. This is a floor on error that no
   model change (Tickets 1–4) can remove; it is a property of the labels.
2. **Mislabels are concentrated exactly where Ticket 3 predicted.** All 16 mislabel-fixes are
   `label=1` tweets using a disaster keyword figuratively (sunburn tagged `hurricane`, a toy
   `battle`, a fish-tank `sunk` ship). This is the same phenomenon as Ticket 3's negative-weight
   keywords (`blazing`, `bloody`, `explode`): the keyword-based collection labeled figurative uses
   as real disasters. The audit turns that Ticket-3 statistical signal into named rows.
3. **Not every disagreement is a mislabel — the rejects prove the audit has a brake.** Three
   confident model–label disagreements were inspected and *kept*: an idiom (`I'm On Fire`), a
   defensible negative (missing-person search), and a real-disaster commentary the model misread.
   Reporting these as `reject_false_positive` rather than flipping them is the difference between an
   audit and a list of everything the model disagreed with.
4. **The dev metric is measured against noisy labels.** 16 mislabels and 10 conflict rows fall in
   dev, all of them `label=1` cases the model correctly reads as non-disaster. So the model is
   penalized on dev for being right, which means the true operating point is slightly better than
   the reported dev F1 — data quality changes the *meaning* of the evaluation, not just the score.

## Impact / interventions

Two interventions are represented, both preserving original labels:
- **Relabel overlay (proposed, not applied to held-out):** 31 duplicate-fix + 16 mislabel-fix rows
  carry a `proposed_label`; retraining on the corrected *training* labels is the sanctioned
  experiment, with both labels kept in the audit table.
- **Keep-but-flag / reject:** duplicate-conflict held-out rows and the three false positives are
  documented but left unchanged, so the audit never silently edits the evaluation set.

The dominant lesson is the ceiling: the ~22-row guaranteed-error floor and the dev-side noise mean
the gap between the Ticket-4 held-out F1 (0.7653) and 1.0 is partly *unreachable* by construction.

## Limitations

- The two detectors find label noise that is either duplicated or confidently disagreed-with by a
  linear model; subtler mislabels that the model also gets wrong are invisible to this audit.
- The fix/ambiguous/reject calls on the 22 model-disagreement rows are human judgments on short,
  often figurative tweets; the three ambiguous and three reject rows are the least certain and are
  labeled `low`/`medium` confidence accordingly.

## Decision

Ship the audit as `results/data_quality_audit.csv` with dispositions and paired original/proposed
labels. Do not modify held-out labels or the frozen pipeline. The audit's role is to bound and
explain the score, not to raise it: 18 duplicate conflicts set a hard ceiling, mislabels are
concentrated in figurative-keyword positives (confirming Ticket 3), and three suspicious rows are
explicitly rejected to keep the audit honest.
