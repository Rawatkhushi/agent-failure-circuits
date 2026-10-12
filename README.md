# Agent-failure circuits — Reading task failure off SAE activations

A small, fully runnable pipeline that takes a task corpus, encodes the
target language model's residual stream through a sparse autoencoder
(SAE), and identifies which SAE features fire differently on failed
vs. successful runs.

Accompanies the IEEE Big Data 2026 submission _"Reading Task Failure
Off the Activations: A Sparse-Feature Audit of GPT-2 Small on Indirect
Object Identification"_.

The pipeline runs end-to-end on a laptop (Apple Silicon MPS or CPU) in
about 20 minutes. No GPU required for the GPT-2 small experiment in
this repository.

---

## TL;DR finding

- 300 Indirect Object Identification (IOI) prompts on GPT-2 small
- 239 successes, 61 failures (79.7% accuracy, matches Wang et al. 2022)
- 146 of the 24,576 SAE features (layer 8, `gpt2-small-res-jb`) clear a
  Holm-corrected significance threshold
- The single strongest predictor of failure is feature **17,491**
  (Cohen's _d_ = +2.93). Neuronpedia labels it a _"cryptographic keys"_
  detector
- Among the eight transferred-object choices in the prompts, `"the keys"`
  fails 93.3% of the time vs. 7.5% for the rest
  (Fisher exact _p_ < 10⁻⁵, odds ratio ≈ 174)
- Cause: surface-form interference. The IOI prompts using `"the keys"`
  trigger a feature aligned to a different concept ("cryptographic
  keys"), which disrupts the IOI computation downstream

After excluding the keys subset, the residual signal is real but
small. The finding tempers any claim that LLM task failures
decompose neatly into clean mechanistic causes — outside the
obvious lexical-interference case, the feature landscape is flatter.

---

## Repository layout

```
agent-failure-circuits/
├── src/
│   ├── build_task_set.py       generate the 300 IOI prompts
│   ├── run_experiment.py       load GPT-2 small + SAE, run, log activations
│   ├── analyze.py              per-feature stats: t-test, Cohen's d, Holm
│   └── analyze_conditional.py  same analysis on the keys-free subset
├── data/
│   └── tasks.jsonl             generated task corpus (seeded)
├── results/
│   ├── predictions.csv         per-task: prompt, expected, predicted, success
│   ├── activations.npy         (300, 24576) SAE feature activations
│   ├── feature_stats.csv       all 24,576 features ranked by |d|
│   ├── feature_stats_no_keys.csv conditional analysis output
│   ├── top_features.md         top 20 with Neuronpedia URLs
│   └── top_features_no_keys.md
├── figures/
│   └── effect_distribution.png
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
git clone https://github.com/mahdinaser/agent-failure-circuits
cd agent-failure-circuits

# 1. Install dependencies (recommend a venv)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate the task corpus (300 IOI prompts, seed 42)
python src/build_task_set.py --n 300

# 3. Run the model + log activations (~10 min on M3 Max MPS)
python src/run_experiment.py

# 4. Analyze: features that fire differently on fail vs. success
python src/analyze.py

# 5. Conditional analysis (excluding "the keys" subset)
python src/analyze_conditional.py
```

The expected output of step 3 ends with:

```
final acc: 239/300 = 0.797
```

The expected output of step 4 ends with:

```
Significant (Holm p < 0.05): 146 / 24576
|Cohen's d| > 0.5: 209 / 24576
|Cohen's d| > 0.8 (large): 105 / 24576
```

---

## Reproducing the keys-feature finding

Once `results/feature_stats.csv` exists:

```python
import pandas as pd
df = pd.read_csv("results/feature_stats.csv")
print(df.head(1)[["feature", "cohens_d", "mean_fail", "mean_succ", "p_holm"]])
# feature  cohens_d  mean_fail  mean_succ  p_holm
# 17491    +2.93     10.285     0.182      <1e-10
```

To verify the surface-form interference:

```python
import json, pandas as pd
preds = pd.read_csv("results/predictions.csv")
tasks = [json.loads(l) for l in open("data/tasks.jsonl")]
preds["object"] = [t["object"] for t in tasks]
print(preds.groupby("object").success.agg(["mean", "count"]))
# 'the keys' -> mean ≈ 0.067 (i.e., 93.3% fail), n=45
# others    -> mean ≈ 0.925
```

---

## Methodology notes

### Model and SAE

- **Model**: GPT-2 small (124M, 12 layers, 768-dim residual stream),
  loaded via TransformerLens.
- **SAE**: `gpt2-small-res-jb`, layer 8 (`blocks.8.hook_resid_pre`),
  loaded via `sae_lens`. d_sae = 24,576.

### Task: IOI (Wang et al. 2022)

- 4 sentence templates × 29 names × 8 places × 8 objects, seeded
  procedural generator
- Single-token completion with greedy decoding
- Success = the first generated alphanumeric token equals the
  expected indirect-object name (exact match, case-sensitive)

### Per-task feature logging

- Forward pass with `model.run_with_cache(prompt, names_filter=[hook])`
- Mean SAE encoding over the last 3 residual-stream tokens of the
  prompt (the tokens immediately before the predicted answer)
- One ℝ^24576 vector per task

### Per-feature statistics

- Welch's _t_-test, two-sided, conservative df = min(n_fail, n_succ) − 1
- Cohen's _d_ with pooled SD
- Holm-Bonferroni correction across all 24,576 features

---

## Limitations

1. **Single model, single SAE, single task.** GPT-2 small + layer-8
   res-jb + IOI. We do not claim generalization to other model sizes,
   SAE releases, or tasks. The pipeline is intentionally model-agnostic
   — point it at Gemma 2 + Gemma Scope or Llama 3 + Llama Scope and
   the same analysis runs.

2. **Single auditor in one pass.** Feature interpretation relies on
   Neuronpedia's auto-generated descriptions (GPT-3.5 / GPT-4o-mini
   labelers), which can be wrong. Verify against the actual
   top-activating-token list before drawing strong conclusions.

3. **Correlation, not causation.** Feature 17,491 correlates near-
   perfectly with failure on the keys subset. The causal test —
   ablating the feature and re-measuring IOI accuracy — is supported
   by `sae_lens` but not run in this repo.

4. **Surface-form contamination.** The `"the keys"` finding is, in
   one sense, an artifact of an IOI corpus that happened to contain a
   word that activates a homonym feature. In another sense, that's
   exactly the kind of thing this audit is for: any IOI result
   reported as a single average over the 8 objects is averaging over
   a 0% → 93% spread.

---

## Extension to larger models

The pipeline runs unchanged on any model + SAE pair supported by
`sae_lens`. To extend to instruction-tuned Gemma 2 with Gemma Scope:

```bash
python src/run_experiment.py \
  --model google/gemma-3-270m-it \
  --sae-release gemma-scope-2-270m-it-res \
  --sae-id layer_12_width_16k_l0_medium
```

(Gemma models are gated on Hugging Face; accept the license at
`https://huggingface.co/google/gemma-3-270m-it` first.)

For Llama 3 8B with publicly released SAEs from Goodfire or Llama
Scope, swap the release/id; everything else stays the same.

---

## Citation

```bibtex
@inproceedings{agent_failure_circuits_2026,
  title={Reading Task Failure Off the Activations: A Sparse-Feature
         Audit of GPT-2 Small on Indirect Object Identification},
  author={Anonymous},
  booktitle={IEEE International Conference on Big Data},
  year={2026}
}
```

## License

MIT.

## Per-position IOI analysis (complete pipeline)

Starts from the saved positional activations (`results_positional_300/`, shape 300 x 3 x 24,576) and needs no model or GPU:

```bash
python src/run_position_pipeline.py        # ~1 minute; writes results_positional_300_robust/
python -m pytest tests -q                  # 24 offline tests
```

| step                                 | script                                                       | output                                                                                                                                           |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1 verify data                        | `src/verify_positional_data.py`                              | 12 PASS/FAIL checks                                                                                                                              |
| 2 per-position stats                 | `src/analyze_positions_robust.py`                            | Welch + Holm (complete 3 x 24,576 family), max-T permutation, object-stratified; CSV, top-features md, figure, averaged-vs-positional comparison |
| 3 without keys / template-stratified | same script, `--exclude-object`, `--stratify-by template_id` | `*_no_keys`, `*_strat_template` files                                                                                                            |
| 4 template/object effects            | `src/analyze_templates.py`                                   | `template_analysis.md/json`                                                                                                                      |
| 5 stability                          | `src/stability_positions.py`                                 | bootstrap, split-half (with label-permuted baseline), leave-one-template-out                                                                     |
| 6 report                             | `src/run_position_pipeline.py`                               | `FINAL_REPORT.md` (every number computed from the outputs)                                                                                       |

To regenerate the activations themselves (needs the model): `python src/run_experiment_positional.py` (see its `--help`), then rerun the pipeline.
