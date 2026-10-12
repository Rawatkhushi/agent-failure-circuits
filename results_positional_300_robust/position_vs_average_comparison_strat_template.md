# Averaged vs. per-position analysis (same examples, same statistics code)

N = 300 (61 failures). `avg` = mean of the 3 positions' SAE activations (what src/analyze.py does, recomputed here with this script's Welch test so that the two analyses are directly comparable). Holm uses the complete family in both cases: 24,576 hypotheses for the averaged analysis, 73,728 (3 x 24,576) for the per-position analysis, so the per-position correction is stricter by construction. Observations below are descriptive; no outcome is presumed.

## Counts

| analysis | tested | Holm p<0.05 | \|d\|>0.8 |
|:---|---:|---:|---:|
| average | 886 | 156 | 105 |
| position 1 | 293 | 24 | 0 |
| position 2 | 426 | 70 | 57 |
| position 3 | 412 | 76 | 79 |

## Overlap of the top-20 features by |d|

| | average | pos1 | pos2 | pos3 |
|:---|---:|---:|---:|---:|
| average | 20 | 0 | 10 | 14 |
| pos1 | 0 | 20 | 1 | 0 |
| pos2 | 10 | 1 | 20 | 4 |
| pos3 | 14 | 0 | 4 | 20 |

## Top-10 averaged features: where does the effect live?

| feature | d avg | d pos1 | d pos2 | d pos3 | strongest position |
|---:|---:|---:|---:|---:|:---|
| 17491 | +2.93 | n/a | +2.92 | +2.84 | pos 2 |
| 7536 | +2.90 | -0.07 | n/a | +2.90 | pos 3 |
| 19149 | +2.89 | n/a | n/a | +2.89 | pos 3 |
| 21307 | +2.79 | n/a | +2.79 | n/a | pos 2 |
| 10960 | +2.78 | n/a | +2.78 | n/a | pos 2 |
| 12273 | +2.71 | n/a | n/a | +2.71 | pos 3 |
| 11132 | +2.68 | -0.00 | +2.09 | +2.85 | pos 3 |
| 10357 | +2.66 | n/a | +0.31 | +2.70 | pos 3 |
| 22272 | +2.66 | n/a | +2.66 | n/a | pos 2 |
| 9647 | +2.66 | n/a | n/a | +2.66 | pos 3 |

## Possible effects of averaging

* Features with a *tested* |d| >= 0.8 at some position but |d_avg| < 0.5 (or untested after averaging): **10**
* Features with d >= +0.5 at one position and d <= -0.5 at another (sign cancellation): **4**

Largest 'hidden by averaging' cases (by position |d| minus |d_avg|):

| feature | d pos1 | d pos2 | d pos3 | d avg |
|---:|---:|---:|---:|---:|
| 20280 | n/a | -0.15 | +2.37 | +0.36 |
| 11647 | -0.12 | -0.25 | +0.95 | -0.08 |
| 21708 | +0.05 | +1.06 | +0.09 | +0.25 |
| 4392 | n/a | -0.33 | +1.13 | +0.33 |
| 14287 | -0.16 | -0.47 | +1.21 | +0.44 |
| 9926 | +0.16 | +0.77 | -0.90 | +0.12 |
| 11096 | -0.09 | +1.11 | n/a | +0.35 |
| 4671 | +0.36 | +0.34 | -1.04 | -0.34 |
| 22852 | +0.28 | +1.11 | -0.47 | +0.44 |
| 14403 | -0.32 | +0.51 | -0.81 | -0.23 |

## Rank agreement of d across analyses (Spearman, features tested in both)

| pair | n features | Spearman rho |
|:---|---:|---:|
| pos1 vs pos2 | 118 | 0.135 |
| pos1 vs pos3 | 138 | -0.040 |
| pos2 vs pos3 | 155 | 0.440 |
| pos1 vs average | 348 | 0.649 |
| pos2 vs average | 520 | 0.894 |
| pos3 vs average | 511 | 0.896 |

## Context: original analysis (results/feature_stats.csv)

The original file was produced from the full 300-prompt corpus with src/analyze.py (conservative df = min(n)-1 for p-values). It is only directly comparable to this analysis when the positional run covers the same examples; otherwise it is shown for reference only.

| feature | d (original) | mean fail | mean succ | p Holm (original) |
|---:|---:|---:|---:|---:|
| 17491 | +2.93 | 10.285 | 0.182 | 5.46e-12 |
| 7536 | +2.90 | 0.757 | 0.011 | 3.82e-11 |
| 19149 | +2.89 | 1.132 | 0.019 | 1.64e-11 |
| 21307 | +2.79 | 1.297 | 0.023 | 4.36e-11 |
| 10960 | +2.78 | 0.686 | 0.010 | 1.47e-10 |

