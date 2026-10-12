# Per-position IOI analysis: results on the 300-prompt positional data

Data: `results_positional_300/` (GPT-2 small, `gpt2-small-res-jb`, `blocks.8.hook_resid_pre`, shape 300 x 3 x 24,576).
Failures = 61, successes = 239. Everything below was computed by `src/analyze_positions_robust.py`
(commands at the end). Direction everywhere: **failure minus success** (d > 0 = higher on failures).

## 1. Data validation (all pass)
Shape, finiteness and non-negativity; ids/prompts identical to `data/tasks.jsonl`; `expected`, `predicted`,
`success`, `generated` identical to `results/predictions.csv`; stored last-3 token ids/indices equal the tail of each
prompt's full token list; mean over the 3 positions reproduces the original `results/activations.npy` (max |diff| 1.6e-6);
the pairing is example-specific (shifting rows by one breaks it). The existing `results_positional_300/position_feature_stats.csv`
agrees with this analysis (Cohen's d to 2e-15; Holm-significant counts 24 / 70 / 76 identical), so the positional
statistics already in the repo were correct.

## 2. What the three positions are
Positions are the last three tokens counted from the end, not semantic slots. In these 300 prompts:
position 1 is `" the"` (187 prompts, 29% fail) or `" a"` (113, 6% fail); position 2 is the object noun
(`" keys"` 42/45 fail; `" card"` 0/43; `" bag"` 0/36; the other five objects 6-16% fail); position 3 is `" to"` in all 300.
The article is determined by the object, so object identity fully determines position 1 and position 2.

## 3. Results (Welch + Holm over the complete 3 x 24,576 family; max-T permutation, 2000 shuffles)
| position | tested | Welch+Holm < .05 | max-T < .05 | |d| > 0.8 | **object-stratified max-T < .05** |
|---|---:|---:|---:|---:|---:|
| 1 (`the`/`a`) | 293 | 24 (3 higher in failure) | 24 | 0 | **0** |
| 2 (object noun) | 426 | 70 (39) | 72 | 57 | **0** |
| 3 (`to`) | 412 | 76 (30) | 84 | 79 | **0** |

* **Positions 2 and 3: the strongest features are keys detectors.** The top features (position 2: 17491, 21307, 10960,
  22272, 17139; position 3: 7536, 19149, 11132, 17491, 12273), d = +2.4 to +2.9, are active in exactly 42 failures and 3 successes, i.e. on the 45
  `"the keys"` prompts. Feature 17491 (the original "cryptographic keys" feature) is strong at both positions 2 and 3. The two
  positions use largely different features: only 4 of the top-20 by |d| overlap, and the rank correlation of d across features tested at both is
  rho = 0.44.
* **Position 1 effects are modest (|d| < 0.8)** and track `" the"` vs `" a"`: the top failure features are active in 187 = 54 + 133 prompts,
  exactly the number of `" the"` prompts.
* **Stratified test:** shuffling failure labels only within each object (and centring activations within each object), no feature at any
  position is significant. With keys excluded entirely (19 failures / 236 successes), only 2 features pass the unstratified max-T (5495 and 11082, position 2,
  both *higher in success*, so not failure features) and 1 passes the stratified one at position 1.
* **Averaging vs per-position:** averaging does not hide the keys signal (rank agreement of d with position 2: 0.89, with position 3: 0.90;
  156 averaged Holm hits vs 70 / 76). What averaging does is merge two different feature sets (object-token features at position 2, carried-forward features at
  position 3) and blur position 1. Ten features with a tested |d| >= 0.8 at one position have |d_avg| < 0.5 and four change sign across positions;
  these are lower-activity features and should be treated as leads, not findings.

## 4. The candidates in the progress report
| feature | position | d (F-S) | active fail / succ | p raw | p Holm | p max-T | p strat. max-T |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16421 | 2 | +0.443 | 9 / 11 | 0.049 | 1.0 | 1.0 | 0.86 |
| 22763 | 2 | -0.109 | 4 / 36 | 0.43 | 1.0 | 1.0 | 1.0 |
| 23952 | 1 | +0.370 | 3 / 3 | 0.14 | 1.0 | 1.0 | 1.0 |
The report's 16421 numbers (20 active prompts, 9 fail / 11 succeed, d 0.4432) reproduce exactly. But 16421 is **not** significant after any correction,
and 22763 is not associated with failure at all, so it is not a "matched" failure candidate in the statistical sense, only in activation magnitude on task 223.
**Interpretation caution on the ablations:** "identical predictions on all 300 prompts" is much less informative than it sounds. If an ablation changes the output on
at most one prompt (task 223), any two such ablations agree on the other 299 trivially. The only informative prompt is task 223; logits there (the report's next step 3) would be the real test.
This cannot be examined from the saved activations (needs the model).

## 5. Template effect (planned next step in the progress report)
Failure rate by template overall: 0.18 / 0.20 / 0.17 / 0.26 (chi-square p = 0.53, no overall effect), because keys prompts fail at 90%+ in every template.
Outside keys the picture differs: **13 of the 19 non-keys failures are in template 3** ("Once X and Y arrived at the park, X passed the O to"; 13/77 = 17%
vs 6/178 = 3% in the other templates; chi-square p = 0.001, one pre-specified-style test, small cells). Ten of those 13 predict the pronoun "her". This is the
most promising lead for a failure mechanism beyond keys, and is invisible to an analysis that averages over templates.

## 6. Limitations
* Only 19 failures remain outside keys, and only 3 successes inside keys, so the stratified test has low power. "No feature survives" means
  **no evidence of any failure association beyond object identity in this sample**, not evidence of absence.
* Welch p-values are anti-conservative for sparse features (see README); max-T permutation p-values are the more reliable column.
* One model, SAE layer, task and 300 prompts; no causal claims; the ablation results were not re-run here.

## Commands
```bash
python src/analyze_positions_robust.py --acts results_positional_300/positional_activations.npy \
  --meta results_positional_300/positional_metadata.csv --stratify-by object \
  --out-stats results_positional_300_robust/position_feature_stats_robust.csv \
  --out-top results_positional_300_robust/top_features_by_position_robust.md \
  --out-fig results_positional_300_robust/position_effects_robust.png \
  --out-compare results_positional_300_robust/position_vs_average_comparison.md
# same with:  --exclude-object "the keys" --suffix _no_keys
python -m pytest tests/test_positional_robust.py -v
```
