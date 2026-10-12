# Top SAE features by token position (failure vs. success)

Model: gpt2-small | SAE: gpt2-small-res-jb/blocks.8.hook_resid_pre | N = 255 (19 failures, 236 successes, accuracy 0.925) | window K = 3

Prompts excluded by object: the keys.

**Direction:** `delta = mean(failure) - mean(success)`; `d > 0` means the feature is *higher on failures* at that position. **These are associations in a small sample, not evidence that a feature causes failure.** Tests are Welch two-sided; `p Holm` corrects over the complete family of 3 x 24,576 = 73,728 (feature, position) hypotheses (untested pairs count as p = 1, as in the original analysis); `p Holm (tested)` uses only the pairs actually tested (less conservative). `p max-T (strat)` (if `--stratify-by`) shuffles labels only within each level of the stratifying column, so it tests association beyond that column (e.g. token identity). `p max-T (perm)` is a single-step max-|t| permutation adjusted p over all tested pairs (no normality assumption; resolution 1/(B+1); NaN if not run). Welch p-values can be anti-conservative for sparse features, so prefer features where BOTH Holm and max-T are small. `p MWU` is an uncorrected Mann-Whitney check on the displayed rows (activations are sparse/non-normal). `d 95% CI` is a 1000-resample within-group bootstrap.

## Which tokens sit at each position?

Positions are counted from the end of the prompt, so the token at a given position differs across prompts. Failure rate per token identity shows how strongly token identity is confounded with the label.

**Position 1** (offset -3, absolute index 12-12):

| token | n | n fail | fail rate |
|:---|---:|---:|---:|
| `' the'` | 142 | 12 | 0.08 |
| `' a'` | 113 | 7 | 0.06 |

**Position 2** (offset -2, absolute index 13-13):

| token | n | n fail | fail rate |
|:---|---:|---:|---:|
| `' card'` | 43 | 0 | 0.00 |
| `' flowers'` | 39 | 4 | 0.10 |
| `' book'` | 37 | 5 | 0.14 |
| `' gift'` | 37 | 6 | 0.16 |
| `' bag'` | 36 | 0 | 0.00 |
| `' drink'` | 33 | 2 | 0.06 |
| `' ball'` | 30 | 2 | 0.07 |

**Position 3** (offset -1, absolute index 14-14):

| token | n | n fail | fail rate |
|:---|---:|---:|---:|
| `' to'` | 255 | 19 | 0.07 |

## Test family

999 of 73,728 (feature, position) pairs were tested. Excluded pairs: constant: 72,486, low_activity: 243

## Position 1 (offset -3)

Tested features: 287. Holm-significant (p < 0.05): 9 (0 higher in failure, 9 higher in success). max-T permutation p < 0.05: 0. Stratified (within-group) max-T p < 0.05: 1. |d| > 0.8: 12.

### Features higher in FAILURE (position 1), top 20 by |d|

| Rank | Feature | d | d 95% CI | Hedges g | mean fail | mean succ | active F/S | p raw | p Holm | p Holm (tested) | p max-T (perm) | p max-T (strat) | p MWU | Neuronpedia |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 23952 | +1.25 | [-0.08, +2.31] | +1.25 | 0.018 | 0.001 | 3/19, 3/236 | 0.105 | 1.000 | 1.000 | 1.000 | 1.000 | 5.23e-05 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/23952) |
| 2 | 6395 | +1.21 | [+0.56, +1.96] | +1.21 | 1.594 | 0.473 | 13/19, 63/236 | 0.002 | 1.000 | 1.000 | 1.000 | 0.936 | 9.19e-06 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/6395) |
| 3 | 6134 | +0.98 | [+0.32, +1.81] | +0.98 | 1.391 | 0.385 | 8/19, 35/236 | 0.021 | 1.000 | 1.000 | 1.000 | 1.000 | 5.50e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/6134) |
| 4 | 20989 | +0.97 | [+0.47, +1.49] | +0.97 | 17.093 | 7.228 | 17/19, 116/236 | 6.02e-04 | 1.000 | 0.522 | 1.000 | 0.776 | 6.61e-05 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/20989) |
| 5 | 14202 | +0.95 | [+0.29, +1.71] | +0.94 | 1.076 | 0.306 | 8/19, 36/236 | 0.022 | 1.000 | 1.000 | 1.000 | 1.000 | 9.41e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/14202) |
| 6 | 21443 | +0.93 | [+0.08, +2.07] | +0.93 | 0.097 | 0.011 | 5/19, 8/236 | 0.067 | 1.000 | 1.000 | 1.000 | 1.000 | 1.09e-05 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/21443) |
| 7 | 1922 | +0.92 | [+0.40, +1.49] | +0.92 | 1.086 | 0.481 | 16/19, 108/236 | 0.002 | 1.000 | 1.000 | 1.000 | 0.975 | 1.84e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/1922) |
| 8 | 5812 | +0.90 | [-0.01, +1.77] | +0.90 | 0.080 | 0.010 | 4/19, 9/236 | 0.174 | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/5812) |
| 9 | 12734 | +0.89 | [+0.25, +1.65] | +0.89 | 1.535 | 0.455 | 8/19, 35/236 | 0.026 | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/12734) |
| 10 | 15075 | +0.88 | [+0.31, +1.57] | +0.88 | 0.321 | 0.103 | 12/19, 59/236 | 0.014 | 1.000 | 1.000 | 1.000 | 1.000 | 1.44e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/15075) |
| 11 | 14502 | +0.82 | [+0.20, +1.58] | +0.82 | 0.403 | 0.136 | 8/19, 54/236 | 0.038 | 1.000 | 1.000 | 1.000 | 1.000 | 0.019 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/14502) |
| 12 | 21938 | +0.79 | [-0.12, +1.71] | +0.79 | 0.040 | 0.008 | 4/19, 18/236 | 0.146 | 1.000 | 1.000 | 1.000 | 1.000 | 0.034 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/21938) |
| 13 | 8463 | +0.77 | [+0.20, +1.48] | +0.77 | 2.168 | 0.733 | 8/19, 35/236 | 0.030 | 1.000 | 1.000 | 1.000 | 1.000 | 0.002 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/8463) |
| 14 | 2977 | +0.77 | [+0.11, +1.62] | +0.77 | 0.268 | 0.073 | 6/19, 27/236 | 0.068 | 1.000 | 1.000 | 1.000 | 1.000 | 0.008 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/2977) |
| 15 | 9951 | +0.76 | [+0.21, +1.37] | +0.76 | 1.319 | 0.514 | 13/19, 61/236 | 0.018 | 1.000 | 1.000 | 1.000 | 1.000 | 1.58e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/9951) |
| 16 | 535 | +0.75 | [-0.02, +1.77] | +0.75 | 0.100 | 0.018 | 4/19, 13/236 | 0.140 | 1.000 | 1.000 | 1.000 | 1.000 | 0.008 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/535) |
| 17 | 7072 | +0.74 | [+0.18, +1.40] | +0.74 | 0.520 | 0.179 | 9/19, 41/236 | 0.034 | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/7072) |
| 18 | 19909 | +0.74 | [-0.02, +1.63] | +0.73 | 0.074 | 0.016 | 5/19, 18/236 | 0.139 | 1.000 | 1.000 | 1.000 | 1.000 | 0.006 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/19909) |
| 19 | 13692 | +0.73 | [+0.22, +1.31] | +0.73 | 2.068 | 0.841 | 13/19, 64/236 | 0.019 | 1.000 | 1.000 | 1.000 | 1.000 | 2.30e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/13692) |
| 20 | 5634 | +0.73 | [+0.14, +1.45] | +0.73 | 0.421 | 0.133 | 8/19, 33/236 | 0.047 | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/5634) |

### Features higher in SUCCESS (position 1), top 10 by |d|

| Rank | Feature | d | d 95% CI | Hedges g | mean fail | mean succ | active F/S | p raw | p Holm | p Holm (tested) | p max-T (perm) | p max-T (strat) | p MWU | Neuronpedia |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 14403 | -0.91 | [-1.27, -0.50] | -0.91 | 4.207 | 13.576 | 6/19, 172/236 | 7.70e-05 | 1.000 | 0.071 | 0.988 | 0.477 | 3.04e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/14403) |
| 2 | 21905 | -0.66 | [-0.87, -0.35] | -0.65 | 0.509 | 2.054 | 7/19, 168/236 | 6.33e-05 | 1.000 | 0.058 | 0.996 | 0.724 | 3.92e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/21905) |
| 3 | 1607 | -0.64 | [-0.87, -0.34] | -0.64 | 0.153 | 0.542 | 5/19, 143/236 | 8.63e-05 | 1.000 | 0.079 | 0.998 | 0.408 | 0.003 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/1607) |
| 4 | 4790 | -0.64 | [-0.78, -0.44] | -0.64 | 0.023 | 0.197 | 2/19, 100/236 | 3.04e-08 | 0.002 | 2.99e-05 | 0.495 | 0.020 | 0.005 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/4790) |
| 5 | 15905 | -0.56 | [-0.69, -0.38] | -0.56 | 0.022 | 0.225 | 1/19, 73/236 | 3.54e-08 | 0.003 | 3.47e-05 | 0.633 | 0.218 | 0.015 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/15905) |
| 6 | 600 | -0.56 | [-0.64, -0.48] | -0.55 | 0.009 | 0.482 | 1/19, 82/236 | 2.00e-14 | 1.48e-09 | 1.99e-11 | 0.056 | 0.493 | 0.007 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/600) |
| 7 | 10352 | -0.55 | [-0.72, -0.33] | -0.55 | 0.087 | 0.426 | 3/19, 102/236 | 3.14e-05 | 1.000 | 0.029 | 0.993 | 0.588 | 0.014 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/10352) |
| 8 | 6406 | -0.55 | [-0.90, -0.09] | -0.55 | 0.334 | 0.750 | 5/19, 159/236 | 0.012 | 1.000 | 1.000 | 1.000 | 0.984 | 0.005 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/6406) |
| 9 | 9926 | -0.54 | [-0.94, -0.10] | -0.54 | 3.766 | 4.074 | 19/19, 236/236 | 0.020 | 1.000 | 1.000 | 1.000 | 1.000 | 0.031 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/9926) |
| 10 | 7969 | -0.54 | [-0.71, -0.28] | -0.54 | 0.058 | 0.384 | 1/19, 69/236 | 4.27e-05 | 1.000 | 0.040 | 0.996 | 0.559 | 0.023 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/7969) |

## Position 2 (offset -2)

Tested features: 372. Holm-significant (p < 0.05): 18 (0 higher in failure, 18 higher in success). max-T permutation p < 0.05: 2. Stratified (within-group) max-T p < 0.05: 0. |d| > 0.8: 9.

### Features higher in FAILURE (position 2), top 20 by |d|

| Rank | Feature | d | d 95% CI | Hedges g | mean fail | mean succ | active F/S | p raw | p Holm | p Holm (tested) | p max-T (perm) | p max-T (strat) | p MWU | Neuronpedia |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 16421 | +1.40 | [+0.49, +2.56] | +1.39 | 0.256 | 0.020 | 6/19, 11/236 | 0.036 | 1.000 | 1.000 | 1.000 | 1.000 | 3.07e-06 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/16421) |
| 2 | 4858 | +1.03 | [-0.11, +2.16] | +1.03 | 0.042 | 0.002 | 2/19, 3/236 | 0.180 | 1.000 | 1.000 | 1.000 | 1.000 | 0.005 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/4858) |
| 3 | 24301 | +0.98 | [-0.13, +1.85] | +0.98 | 0.055 | 0.002 | 2/19, 3/236 | 0.231 | 1.000 | 1.000 | 1.000 | 1.000 | 0.005 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/24301) |
| 4 | 17373 | +0.93 | [+0.25, +1.82] | +0.93 | 0.247 | 0.057 | 9/19, 28/236 | 0.023 | 1.000 | 1.000 | 1.000 | 1.000 | 1.98e-05 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/17373) |
| 5 | 10247 | +0.81 | [+0.01, +1.82] | +0.80 | 0.327 | 0.071 | 6/19, 31/236 | 0.092 | 1.000 | 1.000 | 1.000 | 1.000 | 0.016 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/10247) |
| 6 | 23668 | +0.80 | [-0.13, +1.86] | +0.80 | 0.547 | 0.075 | 4/19, 9/236 | 0.130 | 1.000 | 1.000 | 1.000 | 1.000 | 9.11e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/23668) |
| 7 | 2258 | +0.80 | [+0.05, +1.79] | +0.80 | 0.736 | 0.216 | 5/19, 51/236 | 0.094 | 1.000 | 1.000 | 1.000 | 1.000 | 0.334 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/2258) |
| 8 | 23778 | +0.79 | [+0.24, +1.44] | +0.79 | 1.073 | 0.459 | 13/19, 102/236 | 0.016 | 1.000 | 1.000 | 1.000 | 0.973 | 0.005 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/23778) |
| 9 | 12910 | +0.77 | [-0.02, +1.65] | +0.77 | 0.300 | 0.065 | 4/19, 20/236 | 0.122 | 1.000 | 1.000 | 1.000 | 1.000 | 0.048 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/12910) |
| 10 | 17411 | +0.77 | [-0.02, +1.77] | +0.76 | 0.248 | 0.054 | 6/19, 22/236 | 0.121 | 1.000 | 1.000 | 1.000 | 1.000 | 0.002 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/17411) |
| 11 | 4572 | +0.76 | [+0.32, +1.26] | +0.76 | 1.223 | 0.707 | 18/19, 171/236 | 0.007 | 1.000 | 1.000 | 1.000 | 0.858 | 0.002 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/4572) |
| 12 | 23235 | +0.76 | [-0.11, +1.81] | +0.75 | 0.283 | 0.042 | 4/19, 8/236 | 0.131 | 1.000 | 1.000 | 1.000 | 1.000 | 4.92e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/23235) |
| 13 | 16729 | +0.74 | [-0.12, +1.57] | +0.74 | 0.012 | 0.001 | 1/19, 2/236 | 0.352 | 1.000 | 1.000 | 1.000 | 1.000 | 0.085 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/16729) |
| 14 | 21204 | +0.74 | [-0.14, +1.71] | +0.74 | 0.101 | 0.017 | 4/19, 18/236 | 0.201 | 1.000 | 1.000 | 1.000 | 1.000 | 0.036 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/21204) |
| 15 | 1354 | +0.72 | [-0.14, +1.58] | +0.72 | 0.048 | 0.003 | 1/19, 2/236 | 0.359 | 1.000 | 1.000 | 1.000 | 1.000 | 0.085 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/1354) |
| 16 | 22271 | +0.71 | [-0.05, +1.63] | +0.71 | 0.187 | 0.041 | 4/19, 20/236 | 0.140 | 1.000 | 1.000 | 1.000 | 1.000 | 0.054 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/22271) |
| 17 | 7740 | +0.70 | [-0.10, +1.84] | +0.69 | 0.072 | 0.006 | 2/19, 2/236 | 0.212 | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/7740) |
| 18 | 19497 | +0.68 | [-0.07, +1.74] | +0.68 | 0.060 | 0.004 | 2/19, 1/236 | 0.273 | 1.000 | 1.000 | 1.000 | 1.000 | 9.81e-05 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/19497) |
| 19 | 22078 | +0.67 | [-0.19, +1.50] | +0.67 | 0.083 | 0.008 | 1/19, 5/236 | 0.378 | 1.000 | 1.000 | 1.000 | 1.000 | 0.372 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/22078) |
| 20 | 2598 | +0.67 | [+0.09, +1.34] | +0.66 | 1.124 | 0.463 | 9/19, 58/236 | 0.061 | 1.000 | 1.000 | 1.000 | 1.000 | 0.022 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/2598) |

### Features higher in SUCCESS (position 2), top 10 by |d|

| Rank | Feature | d | d 95% CI | Hedges g | mean fail | mean succ | active F/S | p raw | p Holm | p Holm (tested) | p max-T (perm) | p max-T (strat) | p MWU | Neuronpedia |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 10036 | -0.99 | [-1.42, -0.59] | -0.99 | 2.772 | 6.008 | 15/19, 222/236 | 8.40e-05 | 1.000 | 0.077 | 0.988 | 0.827 | 1.01e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/10036) |
| 2 | 4154 | -0.90 | [-1.30, -0.56] | -0.89 | 1.105 | 2.093 | 14/19, 224/236 | 1.14e-04 | 1.000 | 0.104 | 0.996 | 1.000 | 2.41e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/4154) |
| 3 | 5495 | -0.67 | [-0.76, -0.58] | -0.67 | 0.043 | 0.837 | 2/19, 119/236 | 8.98e-18 | 6.62e-13 | 8.96e-15 | 0.012 | 1.000 | 6.03e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/5495) |
| 4 | 11082 | -0.64 | [-0.72, -0.56] | -0.64 | 0.000 | 0.384 | 0/19, 83/236 | 3.85e-18 | 2.84e-13 | 3.84e-15 | 0.010 | 1.000 | 0.002 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/11082) |
| 5 | 12748 | -0.57 | [-0.78, -0.37] | -0.57 | 0.161 | 0.529 | 7/19, 125/236 | 7.92e-06 | 0.584 | 0.008 | 0.961 | 1.000 | 0.037 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/12748) |
| 6 | 23710 | -0.56 | [-0.71, -0.38] | -0.56 | 0.115 | 0.692 | 3/19, 89/236 | 2.12e-07 | 0.016 | 2.07e-04 | 0.739 | 1.000 | 0.028 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/23710) |
| 7 | 6121 | -0.52 | [-0.64, -0.39] | -0.52 | 0.252 | 2.172 | 3/19, 89/236 | 2.68e-09 | 1.97e-04 | 2.64e-06 | 0.495 | 1.000 | 0.031 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/6121) |
| 8 | 14403 | -0.51 | [-0.69, -0.29] | -0.51 | 0.353 | 1.673 | 2/19, 109/236 | 1.02e-04 | 1.000 | 0.093 | 1.000 | 1.000 | 0.005 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/14403) |
| 9 | 12246 | -0.49 | [-0.57, -0.41] | -0.49 | 0.000 | 7.748 | 0/19, 43/236 | 6.79e-12 | 5.01e-07 | 6.77e-09 | 0.192 | 1.000 | 0.043 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/12246) |
| 10 | 18780 | -0.48 | [-0.55, -0.41] | -0.48 | 0.000 | 0.664 | 0/19, 58/236 | 1.35e-11 | 9.93e-07 | 1.34e-08 | 0.234 | 1.000 | 0.015 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/18780) |

## Position 3 (offset -1)

Tested features: 340. Holm-significant (p < 0.05): 6 (0 higher in failure, 6 higher in success). max-T permutation p < 0.05: 0. Stratified (within-group) max-T p < 0.05: 0. |d| > 0.8: 12.

### Features higher in FAILURE (position 3), top 20 by |d|

| Rank | Feature | d | d 95% CI | Hedges g | mean fail | mean succ | active F/S | p raw | p Holm | p Holm (tested) | p max-T (perm) | p max-T (strat) | p MWU | Neuronpedia |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 16919 | +1.29 | [+0.62, +2.13] | +1.29 | 1.149 | 0.292 | 14/19, 65/236 | 0.002 | 1.000 | 1.000 | 1.000 | 0.974 | 2.23e-06 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/16919) |
| 2 | 15509 | +1.04 | [-0.05, +2.08] | +1.04 | 0.088 | 0.007 | 4/19, 6/236 | 0.106 | 1.000 | 1.000 | 1.000 | 1.000 | 5.97e-05 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/15509) |
| 3 | 24544 | +0.98 | [-0.11, +1.96] | +0.97 | 0.029 | 0.001 | 2/19, 2/236 | 0.185 | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/24544) |
| 4 | 18157 | +0.96 | [+0.26, +1.67] | +0.96 | 0.516 | 0.152 | 10/19, 56/236 | 0.017 | 1.000 | 1.000 | 1.000 | 0.993 | 0.001 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/18157) |
| 5 | 4768 | +0.95 | [+0.12, +2.01] | +0.95 | 0.103 | 0.017 | 5/19, 18/236 | 0.093 | 1.000 | 1.000 | 1.000 | 1.000 | 0.004 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/4768) |
| 6 | 18242 | +0.93 | [+0.27, +1.69] | +0.93 | 0.765 | 0.273 | 13/19, 97/236 | 0.019 | 1.000 | 1.000 | 1.000 | 1.000 | 0.002 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/18242) |
| 7 | 20989 | +0.93 | [+0.38, +1.46] | +0.92 | 4.166 | 1.614 | 13/19, 64/236 | 0.002 | 1.000 | 1.000 | 1.000 | 0.974 | 1.50e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/20989) |
| 8 | 11847 | +0.83 | [+0.32, +1.37] | +0.83 | 2.475 | 1.007 | 15/19, 85/236 | 0.005 | 1.000 | 1.000 | 1.000 | 0.974 | 1.55e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/11847) |
| 9 | 17949 | +0.80 | [+0.21, +1.54] | +0.80 | 0.472 | 0.134 | 10/19, 50/236 | 0.027 | 1.000 | 1.000 | 1.000 | 1.000 | 4.46e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/17949) |
| 10 | 5346 | +0.79 | [-0.16, +1.68] | +0.78 | 0.105 | 0.010 | 2/19, 8/236 | 0.256 | 1.000 | 1.000 | 1.000 | 1.000 | 0.112 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/5346) |
| 11 | 5345 | +0.79 | [-0.13, +1.86] | +0.78 | 0.068 | 0.008 | 3/19, 5/236 | 0.146 | 1.000 | 1.000 | 1.000 | 1.000 | 9.94e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/5345) |
| 12 | 8180 | +0.77 | [-0.14, +1.84] | +0.77 | 0.053 | 0.005 | 2/19, 4/236 | 0.216 | 1.000 | 1.000 | 1.000 | 1.000 | 0.014 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/8180) |
| 13 | 24179 | +0.76 | [+0.31, +1.21] | +0.76 | 6.372 | 3.540 | 18/19, 168/236 | 0.003 | 1.000 | 1.000 | 1.000 | 0.637 | 0.001 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/24179) |
| 14 | 19955 | +0.72 | [+0.08, +1.52] | +0.72 | 0.198 | 0.055 | 8/19, 27/236 | 0.051 | 1.000 | 1.000 | 1.000 | 1.000 | 2.00e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/19955) |
| 15 | 6258 | +0.69 | [-0.15, +1.57] | +0.69 | 0.010 | 0.001 | 1/19, 3/236 | 0.366 | 1.000 | 1.000 | 1.000 | 1.000 | 0.174 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/6258) |
| 16 | 20375 | +0.66 | [-0.12, +1.75] | +0.66 | 0.040 | 0.003 | 2/19, 2/236 | 0.257 | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/20375) |
| 17 | 8461 | +0.64 | [-0.15, +1.76] | +0.64 | 0.017 | 0.002 | 2/19, 6/236 | 0.228 | 1.000 | 1.000 | 1.000 | 1.000 | 0.051 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/8461) |
| 18 | 4137 | +0.60 | [+0.11, +1.11] | +0.60 | 2.210 | 1.487 | 18/19, 210/236 | 0.033 | 1.000 | 1.000 | 1.000 | 0.795 | 0.022 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/4137) |
| 19 | 8545 | +0.60 | [-0.02, +1.29] | +0.60 | 0.220 | 0.090 | 9/19, 60/236 | 0.091 | 1.000 | 1.000 | 1.000 | 1.000 | 0.023 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/8545) |
| 20 | 20290 | +0.58 | [+0.02, +1.19] | +0.58 | 0.258 | 0.101 | 10/19, 49/236 | 0.070 | 1.000 | 1.000 | 1.000 | 1.000 | 0.002 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/20290) |

### Features higher in SUCCESS (position 3), top 10 by |d|

| Rank | Feature | d | d 95% CI | Hedges g | mean fail | mean succ | active F/S | p raw | p Holm | p Holm (tested) | p max-T (perm) | p max-T (strat) | p MWU | Neuronpedia |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 23420 | -0.85 | [-1.27, -0.50] | -0.85 | 10.306 | 12.641 | 19/19, 236/236 | 3.51e-04 | 1.000 | 0.310 | 1.000 | 0.887 | 6.59e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/23420) |
| 2 | 6090 | -0.84 | [-1.18, -0.51] | -0.83 | 1.500 | 2.351 | 19/19, 236/236 | 8.45e-05 | 1.000 | 0.077 | 0.992 | 0.972 | 3.83e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/6090) |
| 3 | 10036 | -0.83 | [-1.16, -0.50] | -0.83 | 0.717 | 2.196 | 6/19, 173/236 | 5.20e-05 | 1.000 | 0.048 | 0.986 | 0.730 | 3.17e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/10036) |
| 4 | 15536 | -0.69 | [-1.17, -0.24] | -0.69 | 0.894 | 1.363 | 18/19, 232/236 | 0.008 | 1.000 | 1.000 | 1.000 | 1.000 | 0.008 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/15536) |
| 5 | 21228 | -0.66 | [-0.98, -0.36] | -0.66 | 3.495 | 5.214 | 19/19, 234/236 | 5.70e-04 | 1.000 | 0.495 | 1.000 | 0.728 | 0.003 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/21228) |
| 6 | 14403 | -0.63 | [-0.84, -0.37] | -0.63 | 0.282 | 1.200 | 4/19, 129/236 | 5.87e-05 | 1.000 | 0.054 | 0.996 | 0.446 | 0.004 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/14403) |
| 7 | 4996 | -0.54 | [-0.92, -0.08] | -0.53 | 1.067 | 2.199 | 8/19, 177/236 | 0.023 | 1.000 | 1.000 | 1.000 | 0.998 | 0.005 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/4996) |
| 8 | 22488 | -0.52 | [-0.77, -0.11] | -0.52 | 0.118 | 0.456 | 2/19, 111/236 | 0.009 | 1.000 | 1.000 | 1.000 | 1.000 | 0.004 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/22488) |
| 9 | 9926 | -0.52 | [-0.87, -0.16] | -0.51 | 3.764 | 4.290 | 19/19, 236/236 | 0.009 | 1.000 | 1.000 | 1.000 | 1.000 | 0.023 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/9926) |
| 10 | 7341 | -0.50 | [-0.93, -0.07] | -0.50 | 1.323 | 1.831 | 18/19, 234/236 | 0.027 | 1.000 | 1.000 | 1.000 | 1.000 | 0.035 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/7341) |

