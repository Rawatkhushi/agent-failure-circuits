# Averaged vs. per-position analysis (same examples, same statistics code)

N = 255 (19 failures). `avg` = mean of the 3 positions' SAE activations (what src/analyze.py does, recomputed here with this script's Welch test so that the two analyses are directly comparable). Holm uses the complete family in both cases: 24,576 hypotheses for the averaged analysis, 73,728 (3 x 24,576) for the per-position analysis, so the per-position correction is stricter by construction. Observations below are descriptive; no outcome is presumed.

## Counts

| analysis | tested | Holm p<0.05 | \|d\|>0.8 |
|:---|---:|---:|---:|
| average | 786 | 32 | 26 |
| position 1 | 287 | 9 | 12 |
| position 2 | 372 | 18 | 9 |
| position 3 | 340 | 6 | 12 |

## Overlap of the top-20 features by |d|

| | average | pos1 | pos2 | pos3 |
|:---|---:|---:|---:|---:|
| average | 20 | 10 | 5 | 7 |
| pos1 | 10 | 20 | 0 | 1 |
| pos2 | 5 | 0 | 20 | 1 |
| pos3 | 7 | 1 | 1 | 20 |

## Top-10 averaged features: where does the effect live?

| feature | d avg | d pos1 | d pos2 | d pos3 | strongest position |
|---:|---:|---:|---:|---:|:---|
| 16421 | +1.40 | n/a | +1.40 | n/a | pos 2 |
| 16919 | +1.29 | n/a | n/a | +1.29 | pos 3 |
| 23952 | +1.25 | +1.25 | n/a | n/a | pos 1 |
| 6395 | +1.21 | +1.21 | n/a | n/a | pos 1 |
| 15509 | +1.04 | n/a | n/a | +1.04 | pos 3 |
| 4858 | +1.03 | n/a | +1.03 | n/a | pos 2 |
| 10036 | -1.01 | +0.08 | -0.99 | -0.83 | pos 2 |
| 6134 | +0.98 | +0.98 | n/a | n/a | pos 1 |
| 24544 | +0.98 | n/a | n/a | +0.98 | pos 3 |
| 14202 | +0.95 | +0.95 | n/a | n/a | pos 1 |

## Possible effects of averaging

* Features with a *tested* |d| >= 0.8 at some position but |d_avg| < 0.5 (or untested after averaging): **0**
* Features with d >= +0.5 at one position and d <= -0.5 at another (sign cancellation): **0**

## Rank agreement of d across analyses (Spearman, features tested in both)

| pair | n features | Spearman rho |
|:---|---:|---:|
| pos1 vs pos2 | 111 | 0.258 |
| pos1 vs pos3 | 125 | 0.348 |
| pos2 vs pos3 | 126 | 0.426 |
| pos1 vs average | 343 | 0.840 |
| pos2 vs average | 459 | 0.869 |
| pos3 vs average | 440 | 0.883 |

## Context: original analysis (results/feature_stats.csv)

The original file was produced from the full 300-prompt corpus with src/analyze.py (conservative df = min(n)-1 for p-values). It is only directly comparable to this analysis when the positional run covers the same examples; otherwise it is shown for reference only.

| feature | d (original) | mean fail | mean succ | p Holm (original) |
|---:|---:|---:|---:|---:|
| 17491 | +2.93 | 10.285 | 0.182 | 5.46e-12 |
| 7536 | +2.90 | 0.757 | 0.011 | 3.82e-11 |
| 19149 | +2.89 | 1.132 | 0.019 | 1.64e-11 |
| 21307 | +2.79 | 1.297 | 0.023 | 4.36e-11 |
| 10960 | +2.78 | 0.686 | 0.010 | 1.47e-10 |

