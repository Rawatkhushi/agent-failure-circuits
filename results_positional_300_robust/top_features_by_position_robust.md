# Top SAE features by token position (failure vs. success)

Model: gpt2-small | SAE: gpt2-small-res-jb/blocks.8.hook_resid_pre | N = 300 (61 failures, 239 successes, accuracy 0.797) | window K = 3

**Direction:** `delta = mean(failure) - mean(success)`; `d > 0` means the feature is *higher on failures* at that position. **These are associations in a small sample, not evidence that a feature causes failure.** Tests are Welch two-sided; `p Holm` corrects over the complete family of 3 x 24,576 = 73,728 (feature, position) hypotheses (untested pairs count as p = 1, as in the original analysis); `p Holm (tested)` uses only the pairs actually tested (less conservative). `p max-T (strat)` (if `--stratify-by`) shuffles labels only within each level of the stratifying column, so it tests association beyond that column (e.g. token identity). `p max-T (perm)` is a single-step max-|t| permutation adjusted p over all tested pairs (no normality assumption; resolution 1/(B+1); NaN if not run). Welch p-values can be anti-conservative for sparse features, so prefer features where BOTH Holm and max-T are small. `p MWU` is an uncorrected Mann-Whitney check on the displayed rows (activations are sparse/non-normal). `d 95% CI` is a 1000-resample within-group bootstrap.

## Which tokens sit at each position?

Positions are counted from the end of the prompt, so the token at a given position differs across prompts. Failure rate per token identity shows how strongly token identity is confounded with the label.

**Position 1** (offset -3, absolute index 12-12):

| token | n | n fail | fail rate |
|:---|---:|---:|---:|
| `' the'` | 187 | 54 | 0.29 |
| `' a'` | 113 | 7 | 0.06 |

**Position 2** (offset -2, absolute index 13-13):

| token | n | n fail | fail rate |
|:---|---:|---:|---:|
| `' keys'` | 45 | 42 | 0.93 |
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
| `' to'` | 300 | 61 | 0.20 |

## Test family

1,131 of 73,728 (feature, position) pairs were tested. Excluded pairs: constant: 72,349, low_activity: 248

## Position 1 (offset -3)

Tested features: 293. Holm-significant (p < 0.05): 24 (3 higher in failure, 21 higher in success). max-T permutation p < 0.05: 24. Stratified (within-group) max-T p < 0.05: 0. |d| > 0.8: 0.

### Features higher in FAILURE (position 1), top 20 by |d|

| Rank | Feature | d | d 95% CI | Hedges g | mean fail | mean succ | active F/S | p raw | p Holm | p Holm (tested) | p max-T (perm) | p max-T (strat) | p MWU | Neuronpedia |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 21103 | +0.68 | [+0.44, +0.93] | +0.68 | 6.125 | 3.858 | 54/61, 133/239 | 2.04e-08 | 0.001 | 2.04e-05 | 0.003 | 1.000 | 9.00e-05 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/21103) |
| 2 | 6283 | +0.67 | [+0.43, +0.93] | +0.67 | 3.240 | 2.022 | 54/61, 133/239 | 1.76e-07 | 0.013 | 1.73e-04 | 0.015 | 1.000 | 5.91e-06 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/6283) |
| 3 | 20089 | +0.66 | [+0.42, +0.91] | +0.66 | 3.153 | 1.988 | 54/61, 133/239 | 8.65e-08 | 0.006 | 8.54e-05 | 0.013 | 1.000 | 5.00e-05 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/20089) |
| 4 | 11030 | +0.62 | [+0.35, +0.89] | +0.62 | 4.328 | 2.692 | 54/61, 133/239 | 5.15e-06 | 0.378 | 0.005 | 0.115 | 1.000 | 4.02e-05 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/11030) |
| 5 | 495 | +0.54 | [+0.29, +0.80] | +0.54 | 2.936 | 1.944 | 54/61, 133/239 | 3.97e-05 | 1.000 | 0.036 | 0.362 | 1.000 | 1.81e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/495) |
| 6 | 24394 | +0.52 | [+0.14, +0.85] | +0.51 | 0.029 | 0.004 | 6/61, 7/239 | 0.045 | 1.000 | 1.000 | 1.000 | 1.000 | 0.015 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/24394) |
| 7 | 18055 | +0.49 | [+0.22, +0.80] | +0.49 | 2.306 | 1.398 | 53/61, 130/239 | 9.68e-04 | 1.000 | 0.782 | 0.959 | 1.000 | 9.37e-06 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/18055) |
| 8 | 16021 | +0.47 | [+0.17, +0.77] | +0.47 | 2.387 | 2.046 | 61/61, 239/239 | 0.002 | 1.000 | 1.000 | 0.996 | 0.976 | 0.003 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/16021) |
| 9 | 23774 | +0.46 | [+0.19, +0.75] | +0.46 | 14.499 | 9.598 | 54/61, 133/239 | 0.002 | 1.000 | 1.000 | 0.984 | 0.998 | 1.04e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/23774) |
| 10 | 14202 | +0.45 | [+0.11, +0.81] | +0.45 | 0.684 | 0.302 | 17/61, 36/239 | 0.015 | 1.000 | 1.000 | 1.000 | 0.972 | 0.008 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/14202) |
| 11 | 3481 | +0.45 | [+0.13, +0.78] | +0.45 | 0.549 | 0.305 | 30/61, 80/239 | 0.008 | 1.000 | 1.000 | 1.000 | 1.000 | 0.006 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/3481) |
| 12 | 676 | +0.44 | [+0.10, +0.81] | +0.44 | 0.072 | 0.027 | 19/61, 34/239 | 0.023 | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/676) |
| 13 | 14502 | +0.44 | [+0.11, +0.78] | +0.44 | 0.281 | 0.135 | 24/61, 54/239 | 0.015 | 1.000 | 1.000 | 1.000 | 0.999 | 0.005 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/14502) |
| 14 | 6134 | +0.44 | [+0.10, +0.79] | +0.43 | 0.841 | 0.380 | 17/61, 35/239 | 0.019 | 1.000 | 1.000 | 1.000 | 0.951 | 0.007 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/6134) |
| 15 | 2977 | +0.43 | [+0.08, +0.76] | +0.43 | 0.191 | 0.072 | 14/61, 27/239 | 0.032 | 1.000 | 1.000 | 1.000 | 1.000 | 0.014 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/2977) |
| 16 | 12734 | +0.43 | [+0.10, +0.78] | +0.43 | 0.990 | 0.449 | 17/61, 35/239 | 0.019 | 1.000 | 1.000 | 1.000 | 0.989 | 0.008 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/12734) |
| 17 | 19244 | +0.43 | [+0.11, +0.78] | +0.43 | 0.119 | 0.051 | 24/61, 38/239 | 0.021 | 1.000 | 1.000 | 1.000 | 1.000 | 1.02e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/19244) |
| 18 | 12436 | +0.41 | [+0.16, +0.70] | +0.41 | 4.058 | 2.897 | 55/61, 159/239 | 0.002 | 1.000 | 1.000 | 0.996 | 1.000 | 4.38e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/12436) |
| 19 | 15536 | +0.41 | [+0.14, +0.68] | +0.41 | 1.620 | 1.261 | 55/61, 199/239 | 0.001 | 1.000 | 1.000 | 0.980 | 1.000 | 0.012 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/15536) |
| 20 | 8461 | +0.40 | [+0.04, +0.76] | +0.40 | 0.096 | 0.034 | 13/61, 29/239 | 0.056 | 1.000 | 1.000 | 1.000 | 1.000 | 0.047 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/8461) |

### Features higher in SUCCESS (position 1), top 10 by |d|

| Rank | Feature | d | d 95% CI | Hedges g | mean fail | mean succ | active F/S | p raw | p Holm | p Holm (tested) | p max-T (perm) | p max-T (strat) | p MWU | Neuronpedia |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 23051 | -0.73 | [-0.92, -0.53] | -0.72 | 0.575 | 2.761 | 7/61, 106/239 | 9.35e-12 | 6.88e-07 | 9.94e-09 | 5.00e-04 | 1.000 | 9.35e-07 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/23051) |
| 2 | 5625 | -0.72 | [-0.88, -0.54] | -0.72 | 0.472 | 3.213 | 7/61, 106/239 | 1.50e-13 | 1.10e-08 | 1.63e-10 | 5.00e-04 | 1.000 | 1.18e-06 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/5625) |
| 3 | 22903 | -0.68 | [-0.92, -0.46] | -0.68 | 1.267 | 4.760 | 7/61, 106/239 | 1.49e-08 | 0.001 | 1.50e-05 | 0.003 | 1.000 | 7.49e-06 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/22903) |
| 4 | 24412 | -0.66 | [-0.90, -0.42] | -0.66 | 0.618 | 2.130 | 7/61, 106/239 | 1.48e-07 | 0.011 | 1.45e-04 | 0.015 | 0.981 | 3.82e-05 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/24412) |
| 5 | 12200 | -0.62 | [-0.74, -0.48] | -0.62 | 0.076 | 0.863 | 2/61, 77/239 | 3.84e-12 | 2.83e-07 | 4.11e-09 | 5.00e-04 | 1.000 | 5.86e-06 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/12200) |
| 6 | 19954 | -0.60 | [-0.84, -0.38] | -0.60 | 0.704 | 2.295 | 7/61, 106/239 | 1.23e-06 | 0.090 | 0.001 | 0.058 | 1.000 | 1.94e-05 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/19954) |
| 7 | 23092 | -0.59 | [-0.74, -0.42] | -0.59 | 0.051 | 0.229 | 9/61, 102/239 | 8.57e-10 | 6.31e-05 | 8.86e-07 | 1.00e-03 | 1.000 | 2.36e-05 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/23092) |
| 8 | 7924 | -0.58 | [-0.72, -0.42] | -0.58 | 0.020 | 0.225 | 1/61, 74/239 | 5.80e-10 | 4.27e-05 | 6.02e-07 | 1.00e-03 | 1.000 | 4.60e-06 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/7924) |
| 9 | 10577 | -0.55 | [-0.68, -0.39] | -0.55 | 0.077 | 0.414 | 16/61, 110/239 | 2.43e-09 | 1.79e-04 | 2.48e-06 | 0.002 | 1.000 | 5.32e-04 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/10577) |
| 10 | 18854 | -0.53 | [-0.66, -0.37] | -0.53 | 0.042 | 0.375 | 2/61, 73/239 | 4.63e-09 | 3.41e-04 | 4.73e-06 | 0.003 | 1.000 | 1.66e-05 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/18854) |

## Position 2 (offset -2)

Tested features: 426. Holm-significant (p < 0.05): 70 (39 higher in failure, 31 higher in success). max-T permutation p < 0.05: 72. Stratified (within-group) max-T p < 0.05: 0. |d| > 0.8: 57.

### Features higher in FAILURE (position 2), top 20 by |d|

| Rank | Feature | d | d 95% CI | Hedges g | mean fail | mean succ | active F/S | p raw | p Holm | p Holm (tested) | p max-T (perm) | p max-T (strat) | p MWU | Neuronpedia |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 17491 | +2.92 | [+2.22, +4.02] | +2.92 | 26.514 | 0.472 | 42/61, 3/239 | 1.58e-16 | 1.16e-11 | 1.77e-13 | 5.00e-04 | 1.000 | 1.15e-39 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/17491) |
| 2 | 21307 | +2.79 | [+2.15, +3.72] | +2.79 | 3.892 | 0.070 | 42/61, 3/239 | 1.32e-15 | 9.71e-11 | 1.46e-12 | 5.00e-04 | 1.000 | 2.24e-39 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/21307) |
| 3 | 10960 | +2.78 | [+2.17, +3.65] | +2.78 | 2.059 | 0.031 | 42/61, 3/239 | 4.86e-15 | 3.58e-10 | 5.36e-12 | 5.00e-04 | 1.000 | 9.60e-40 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/10960) |
| 4 | 22272 | +2.66 | [+2.04, +3.56] | +2.65 | 1.826 | 0.040 | 42/61, 8/239 | 3.19e-15 | 2.35e-10 | 3.52e-12 | 5.00e-04 | 1.000 | 3.23e-35 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/22272) |
| 5 | 17139 | +2.47 | [+1.89, +3.24] | +2.46 | 3.152 | 0.069 | 42/61, 3/239 | 5.29e-14 | 3.90e-09 | 5.78e-11 | 5.00e-04 | 1.000 | 4.84e-39 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/17139) |
| 6 | 8480 | +2.45 | [+1.87, +3.20] | +2.44 | 0.908 | 0.063 | 43/61, 28/239 | 6.88e-15 | 5.07e-10 | 7.58e-12 | 5.00e-04 | 1.000 | 2.43e-26 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/8480) |
| 7 | 14538 | +2.31 | [+1.76, +3.05] | +2.31 | 1.801 | 0.055 | 42/61, 14/239 | 9.18e-13 | 6.76e-08 | 9.90e-10 | 5.00e-04 | 1.000 | 2.22e-31 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/14538) |
| 8 | 2598 | +2.29 | [+1.81, +2.92] | +2.28 | 3.267 | 0.509 | 51/61, 61/239 | 1.02e-18 | 7.48e-14 | 1.14e-15 | 5.00e-04 | 1.000 | 2.83e-25 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/2598) |
| 9 | 8545 | +2.27 | [+1.77, +2.89] | +2.26 | 0.553 | 0.041 | 45/61, 30/239 | 2.27e-13 | 1.67e-08 | 2.46e-10 | 5.00e-04 | 1.000 | 3.86e-27 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/8545) |
| 10 | 10211 | +2.15 | [+1.65, +2.74] | +2.14 | 0.641 | 0.013 | 39/61, 3/239 | 3.74e-11 | 2.76e-06 | 3.96e-08 | 5.00e-04 | 1.000 | 5.51e-36 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/10211) |
| 11 | 11132 | +2.09 | [+1.72, +2.61] | +2.09 | 0.937 | 0.012 | 39/61, 3/239 | 3.22e-10 | 2.37e-05 | 3.35e-07 | 5.00e-04 | 1.000 | 2.02e-36 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/11132) |
| 12 | 15473 | +2.08 | [+1.59, +2.67] | +2.08 | 0.853 | 0.010 | 35/61, 2/239 | 1.94e-10 | 1.43e-05 | 2.03e-07 | 5.00e-04 | 1.000 | 6.86e-33 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/15473) |
| 13 | 13466 | +2.08 | [+1.61, +2.67] | +2.07 | 1.991 | 0.150 | 45/61, 46/239 | 1.85e-11 | 1.36e-06 | 1.96e-08 | 5.00e-04 | 1.000 | 6.18e-22 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/13466) |
| 14 | 3019 | +2.01 | [+1.55, +2.61] | +2.01 | 5.041 | 0.779 | 46/61, 38/239 | 3.43e-16 | 2.53e-11 | 3.84e-13 | 5.00e-04 | 1.000 | 4.00e-26 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/3019) |
| 15 | 7747 | +1.91 | [+1.48, +2.42] | +1.90 | 1.072 | 0.013 | 34/61, 3/239 | 3.44e-09 | 2.53e-04 | 3.52e-06 | 5.00e-04 | 1.000 | 7.00e-31 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/7747) |
| 16 | 23092 | +1.89 | [+1.42, +2.55] | +1.88 | 0.910 | 0.111 | 47/61, 42/239 | 3.86e-13 | 2.84e-08 | 4.18e-10 | 5.00e-04 | 1.000 | 1.70e-23 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/23092) |
| 17 | 12325 | +1.84 | [+1.39, +2.39] | +1.83 | 1.824 | 0.306 | 46/61, 38/239 | 3.61e-15 | 2.66e-10 | 3.99e-12 | 5.00e-04 | 1.000 | 5.62e-24 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/12325) |
| 18 | 607 | +1.82 | [+1.38, +2.36] | +1.82 | 10.354 | 2.010 | 46/61, 102/239 | 7.06e-16 | 5.20e-11 | 7.87e-13 | 5.00e-04 | 1.000 | 4.97e-16 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/607) |
| 19 | 3251 | +1.80 | [+1.40, +2.32] | +1.80 | 0.576 | 0.020 | 32/61, 15/239 | 1.28e-08 | 9.39e-04 | 1.29e-05 | 5.00e-04 | 1.000 | 1.47e-20 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/3251) |
| 20 | 11919 | +1.80 | [+1.35, +2.34] | +1.80 | 1.339 | 0.311 | 52/61, 124/239 | 1.26e-12 | 9.31e-08 | 1.36e-09 | 5.00e-04 | 1.000 | 2.79e-15 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/11919) |

### Features higher in SUCCESS (position 2), top 10 by |d|

| Rank | Feature | d | d 95% CI | Hedges g | mean fail | mean succ | active F/S | p raw | p Holm | p Holm (tested) | p max-T (perm) | p max-T (strat) | p MWU | Neuronpedia |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 11675 | -0.88 | [-1.13, -0.61] | -0.88 | 0.699 | 2.289 | 17/61, 211/239 | 4.28e-11 | 3.15e-06 | 4.52e-08 | 5.00e-04 | 1.000 | 8.65e-14 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/11675) |
| 2 | 11331 | -0.80 | [-1.05, -0.55] | -0.80 | 0.411 | 1.595 | 7/61, 150/239 | 9.76e-10 | 7.19e-05 | 1.01e-06 | 5.00e-04 | 1.000 | 1.56e-09 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/11331) |
| 3 | 12748 | -0.79 | [-0.92, -0.66] | -0.79 | 0.051 | 0.522 | 8/61, 125/239 | 3.12e-20 | 2.30e-15 | 3.51e-17 | 5.00e-04 | 1.000 | 9.03e-09 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/12748) |
| 4 | 6988 | -0.72 | [-0.88, -0.54] | -0.72 | 0.330 | 2.195 | 5/61, 103/239 | 2.08e-12 | 1.53e-07 | 2.23e-09 | 5.00e-04 | 1.000 | 4.95e-07 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/6988) |
| 5 | 10102 | -0.70 | [-0.90, -0.50] | -0.70 | 0.539 | 1.876 | 37/61, 147/239 | 1.05e-10 | 7.71e-06 | 1.10e-07 | 5.00e-04 | 1.000 | 0.002 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/10102) |
| 6 | 23710 | -0.68 | [-0.78, -0.58] | -0.68 | 0.036 | 0.683 | 3/61, 89/239 | 5.21e-17 | 3.84e-12 | 5.84e-14 | 5.00e-04 | 1.000 | 7.13e-07 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/23710) |
| 7 | 11131 | -0.66 | [-0.87, -0.45] | -0.66 | 1.005 | 2.221 | 44/61, 200/239 | 1.28e-08 | 9.45e-04 | 1.30e-05 | 0.003 | 1.000 | 9.55e-06 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/11131) |
| 8 | 11082 | -0.65 | [-0.75, -0.55] | -0.65 | 0.020 | 0.385 | 3/61, 84/239 | 8.50e-16 | 6.27e-11 | 9.46e-13 | 5.00e-04 | 1.000 | 2.11e-06 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/11082) |
| 9 | 545 | -0.65 | [-0.76, -0.51] | -0.64 | 0.106 | 1.122 | 5/61, 96/239 | 8.43e-14 | 6.21e-09 | 9.18e-11 | 5.00e-04 | 0.987 | 1.39e-06 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/545) |
| 10 | 19995 | -0.63 | [-0.85, -0.39] | -0.62 | 0.373 | 1.089 | 16/61, 156/239 | 5.01e-07 | 0.037 | 4.83e-04 | 0.037 | 1.000 | 1.73e-07 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/19995) |

## Position 3 (offset -1)

Tested features: 412. Holm-significant (p < 0.05): 76 (30 higher in failure, 46 higher in success). max-T permutation p < 0.05: 84. Stratified (within-group) max-T p < 0.05: 0. |d| > 0.8: 79.

### Features higher in FAILURE (position 3), top 20 by |d|

| Rank | Feature | d | d 95% CI | Hedges g | mean fail | mean succ | active F/S | p raw | p Holm | p Holm (tested) | p max-T (perm) | p max-T (strat) | p MWU | Neuronpedia |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 7536 | +2.90 | [+2.25, +3.79] | +2.89 | 2.272 | 0.031 | 42/61, 3/239 | 1.25e-15 | 9.21e-11 | 1.39e-12 | 5.00e-04 | 1.000 | 5.08e-40 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/7536) |
| 2 | 19149 | +2.89 | [+2.22, +3.85] | +2.88 | 3.397 | 0.058 | 42/61, 3/239 | 4.14e-16 | 3.05e-11 | 4.63e-13 | 5.00e-04 | 1.000 | 1.23e-39 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/19149) |
| 3 | 11132 | +2.85 | [+2.23, +3.69] | +2.84 | 2.062 | 0.024 | 42/61, 3/239 | 4.70e-15 | 3.47e-10 | 5.19e-12 | 5.00e-04 | 1.000 | 4.57e-40 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/11132) |
| 4 | 17491 | +2.84 | [+2.16, +3.77] | +2.84 | 4.341 | 0.073 | 42/61, 3/239 | 7.65e-16 | 5.64e-11 | 8.51e-13 | 5.00e-04 | 1.000 | 1.81e-39 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/17491) |
| 5 | 12273 | +2.71 | [+2.12, +3.46] | +2.70 | 1.782 | 0.025 | 42/61, 3/239 | 1.98e-14 | 1.46e-09 | 2.17e-11 | 5.00e-04 | 1.000 | 1.07e-39 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/12273) |
| 6 | 10357 | +2.70 | [+2.09, +3.61] | +2.69 | 2.297 | 0.048 | 42/61, 7/239 | 2.20e-15 | 1.62e-10 | 2.43e-12 | 5.00e-04 | 1.000 | 5.69e-36 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/10357) |
| 7 | 9647 | +2.66 | [+2.10, +3.37] | +2.65 | 1.453 | 0.022 | 42/61, 8/239 | 5.53e-14 | 4.07e-09 | 6.03e-11 | 5.00e-04 | 1.000 | 6.14e-36 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/9647) |
| 8 | 15638 | +2.65 | [+2.06, +3.48] | +2.64 | 0.605 | 0.008 | 42/61, 3/239 | 2.01e-14 | 1.48e-09 | 2.20e-11 | 5.00e-04 | 1.000 | 1.57e-39 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/15638) |
| 9 | 7230 | +2.60 | [+2.03, +3.34] | +2.59 | 1.736 | 0.129 | 46/61, 48/239 | 1.55e-15 | 1.14e-10 | 1.72e-12 | 5.00e-04 | 1.000 | 4.79e-24 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/7230) |
| 10 | 2078 | +2.58 | [+2.01, +3.29] | +2.57 | 1.832 | 0.033 | 42/61, 3/239 | 5.52e-14 | 4.07e-09 | 6.03e-11 | 5.00e-04 | 1.000 | 1.69e-39 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/2078) |
| 11 | 23457 | +2.51 | [+1.97, +3.15] | +2.50 | 1.870 | 0.039 | 42/61, 14/239 | 4.35e-13 | 3.20e-08 | 4.70e-10 | 5.00e-04 | 1.000 | 2.89e-32 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/23457) |
| 12 | 20280 | +2.37 | [+1.85, +3.05] | +2.37 | 0.695 | 0.071 | 46/61, 70/239 | 2.36e-14 | 1.74e-09 | 2.59e-11 | 5.00e-04 | 1.000 | 1.78e-19 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/20280) |
| 13 | 8480 | +2.31 | [+1.78, +3.03] | +2.31 | 1.071 | 0.122 | 48/61, 38/239 | 3.00e-16 | 2.21e-11 | 3.36e-13 | 5.00e-04 | 1.000 | 7.72e-28 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/8480) |
| 14 | 8545 | +2.22 | [+1.75, +2.87] | +2.21 | 0.799 | 0.096 | 50/61, 62/239 | 2.05e-14 | 1.51e-09 | 2.25e-11 | 5.00e-04 | 1.000 | 7.31e-23 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/8545) |
| 15 | 22593 | +2.18 | [+1.81, +2.63] | +2.17 | 2.350 | 0.027 | 42/61, 3/239 | 1.01e-10 | 7.42e-06 | 1.06e-07 | 5.00e-04 | 1.000 | 1.52e-39 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/22593) |
| 16 | 2073 | +2.14 | [+1.65, +2.75] | +2.14 | 1.421 | 0.036 | 41/61, 3/239 | 9.25e-12 | 6.82e-07 | 9.85e-09 | 5.00e-04 | 1.000 | 9.38e-38 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/2073) |
| 17 | 16758 | +2.09 | [+1.62, +2.69] | +2.08 | 0.489 | 0.010 | 37/61, 8/239 | 1.48e-10 | 1.09e-05 | 1.55e-07 | 5.00e-04 | 1.000 | 5.40e-30 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/16758) |
| 18 | 20852 | +2.05 | [+1.67, +2.53] | +2.04 | 0.915 | 0.003 | 38/61, 2/239 | 1.21e-09 | 8.89e-05 | 1.24e-06 | 5.00e-04 | 1.000 | 1.54e-36 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/20852) |
| 19 | 19091 | +2.02 | [+1.55, +2.55] | +2.02 | 0.931 | 0.051 | 40/61, 19/239 | 5.93e-11 | 4.36e-06 | 6.24e-08 | 5.00e-04 | 1.000 | 3.16e-26 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/19091) |
| 20 | 21101 | +1.98 | [+1.47, +2.55] | +1.97 | 0.700 | 0.099 | 44/61, 66/239 | 1.08e-12 | 7.94e-08 | 1.16e-09 | 5.00e-04 | 1.000 | 1.56e-17 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/21101) |

### Features higher in SUCCESS (position 3), top 10 by |d|

| Rank | Feature | d | d 95% CI | Hedges g | mean fail | mean succ | active F/S | p raw | p Holm | p Holm (tested) | p max-T (perm) | p max-T (strat) | p MWU | Neuronpedia |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 21066 | -1.96 | [-2.47, -1.52] | -1.95 | 1.401 | 4.471 | 19/61, 236/239 | 1.81e-16 | 1.34e-11 | 2.03e-13 | 5.00e-04 | 1.000 | 1.01e-16 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/21066) |
| 2 | 6090 | -1.82 | [-2.15, -1.56] | -1.82 | 0.469 | 2.322 | 22/61, 237/239 | 4.17e-29 | 3.08e-24 | 4.72e-26 | 5.00e-04 | 0.566 | 1.41e-23 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/6090) |
| 3 | 23420 | -1.74 | [-2.06, -1.46] | -1.74 | 7.753 | 12.584 | 61/61, 239/239 | 1.70e-21 | 1.25e-16 | 1.92e-18 | 5.00e-04 | 0.716 | 2.47e-22 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/23420) |
| 4 | 6269 | -1.50 | [-1.86, -1.16] | -1.49 | 0.511 | 2.112 | 17/61, 219/239 | 1.55e-19 | 1.14e-14 | 1.75e-16 | 5.00e-04 | 1.000 | 2.28e-18 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/6269) |
| 5 | 6406 | -1.46 | [-1.75, -1.21] | -1.45 | 0.688 | 2.375 | 28/61, 235/239 | 2.23e-21 | 1.65e-16 | 2.52e-18 | 5.00e-04 | 1.000 | 7.41e-19 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/6406) |
| 6 | 21228 | -1.31 | [-1.56, -1.09] | -1.31 | 1.906 | 5.174 | 51/61, 237/239 | 2.17e-24 | 1.60e-19 | 2.46e-21 | 5.00e-04 | 0.351 | 4.96e-17 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/21228) |
| 7 | 7341 | -1.31 | [-1.59, -1.05] | -1.31 | 0.542 | 1.817 | 33/61, 237/239 | 3.33e-19 | 2.45e-14 | 3.74e-16 | 5.00e-04 | 1.000 | 2.61e-18 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/7341) |
| 8 | 14075 | -1.24 | [-1.57, -0.94] | -1.24 | 0.571 | 1.365 | 45/61, 233/239 | 1.16e-13 | 8.53e-09 | 1.26e-10 | 5.00e-04 | 1.000 | 2.18e-14 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/14075) |
| 9 | 4671 | -1.04 | [-1.33, -0.74] | -1.03 | 1.219 | 2.369 | 54/61, 238/239 | 9.18e-11 | 6.76e-06 | 9.66e-08 | 5.00e-04 | 1.000 | 1.94e-12 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/4671) |
| 10 | 4079 | -0.97 | [-1.24, -0.69] | -0.97 | 0.374 | 1.032 | 26/61, 216/239 | 1.16e-11 | 8.52e-07 | 1.23e-08 | 5.00e-04 | 1.000 | 6.49e-12 | [link](https://www.neuronpedia.org/gpt2-small/8-res-jb/4079) |

