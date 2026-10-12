# Stability of the per-position findings

N = 300 (61 failures); top-K = 20; 500 bootstrap resamples; 300 split-half repeats. Features are those tested at each position (non-constant, active in >= 3 prompts).

## Position 1 (offset -3)

* Bootstrap: of the 20 strongest features by |d|, mean recurrence in the top-20 = 0.65; 8 recur in >= 80% of resamples; 20 have a 95% CI for d excluding 0.
* Split-half, failure-side top-20 chosen on one half: held-out mean d = +0.27 (83% keep the sign), versus +0.01 (43%) when labels are permuted (selection-bias baseline).
* Success-side: held-out mean d = -0.51 (99% keep sign) vs +0.00 under permuted labels.

| feature | d (all) | min d leaving one template out | within-template d (templates 0..3) |
|---:|---:|---:|:---|
| 21103 | +0.68 | +0.60 | [0.62, 0.98, 0.81, 0.43] |
| 6283 | +0.67 | +0.58 | [0.77, 0.98, 0.99, 0.35] |
| 20089 | +0.66 | +0.57 | [0.61, 1.08, 0.69, 0.46] |
| 11030 | +0.62 | +0.58 | [0.58, 0.89, 0.73, 0.46] |
| 495 | +0.54 | +0.38 | [0.65, 1.05, 0.68, 0.27] |

## Position 2 (offset -2)

* Bootstrap: of the 20 strongest features by |d|, mean recurrence in the top-20 = 0.90; 14 recur in >= 80% of resamples; 20 have a 95% CI for d excluding 0.
* Split-half, failure-side top-20 chosen on one half: held-out mean d = +2.19 (100% keep the sign), versus +0.00 (40%) when labels are permuted (selection-bias baseline).
* Success-side: held-out mean d = -0.59 (100% keep sign) vs -0.00 under permuted labels.

| feature | d (all) | min d leaving one template out | within-template d (templates 0..3) |
|---:|---:|---:|:---|
| 17491 | +2.92 | +2.43 | [8.08, 5.59, 2.26, 1.63] |
| 21307 | +2.79 | +2.37 | [7.44, 5.77, 2.36, 1.62] |
| 10960 | +2.78 | +2.34 | [5.98, 6.03, 2.1, 1.57] |
| 22272 | +2.66 | +2.26 | [7.04, 5.63, 2.23, 1.62] |
| 17139 | +2.47 | +2.09 | [5.79, 4.0, 2.25, 1.56] |

## Position 3 (offset -1)

* Bootstrap: of the 20 strongest features by |d|, mean recurrence in the top-20 = 0.93; 17 recur in >= 80% of resamples; 20 have a 95% CI for d excluding 0.
* Split-half, failure-side top-20 chosen on one half: held-out mean d = +2.46 (100% keep the sign), versus +0.00 (41%) when labels are permuted (selection-bias baseline).
* Success-side: held-out mean d = -1.12 (100% keep sign) vs +0.01 under permuted labels.

| feature | d (all) | min d leaving one template out | within-template d (templates 0..3) |
|---:|---:|---:|:---|
| 7536 | +2.90 | +2.44 | [6.77, 5.81, 2.29, 1.61] |
| 19149 | +2.89 | +2.44 | [7.62, 5.69, 2.3, 1.62] |
| 11132 | +2.85 | +2.42 | [6.32, 5.64, 2.33, 1.58] |
| 17491 | +2.84 | +2.35 | [7.63, 5.16, 2.39, 1.61] |
| 12273 | +2.71 | +2.29 | [6.8, 5.97, 2.34, 1.6] |

