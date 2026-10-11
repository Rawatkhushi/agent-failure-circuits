# Category-adjusted positional SAE analysis

Examples: 300
Positions: 3
Features per position: 24576
Covariate: categorical position_2_token (8 levels)
Residual degrees of freedom: 291
Outcome: feature activation
Predictor of interest: failure (1) vs success (0)
Correction: Holm across every feature-position test
Interpretation: associations, not causal effects

Significant tests (Holm p < 0.05): 2 / 73728

## Position 1

| Feature | Adjusted failure − success | p | Holm p |
|---:|---:|---:|---:|
| 23952 | 0.0147148 | 6.282e-07 | 0.04632 |
| 6395 | 1.05697 | 9.26e-07 | 0.06827 |
| 0 | 0 | 1 | 1 |
| 1 | 0 | 1 | 1 |
| 2 | 0 | 1 | 1 |
| 3 | 0 | 1 | 1 |
| 4 | 0 | 1 | 1 |
| 5 | 0 | 1 | 1 |
| 6 | 0 | 1 | 1 |
| 7 | 0 | 1 | 1 |

## Position 2

| Feature | Adjusted failure − success | p | Holm p |
|---:|---:|---:|---:|
| 16421 | 0.195469 | 4.142e-08 | 0.003054 |
| 4858 | 0.0376817 | 3.551e-06 | 0.2618 |
| 0 | 0 | 1 | 1 |
| 1 | 0 | 1 | 1 |
| 2 | 0 | 1 | 1 |
| 3 | 0 | 1 | 1 |
| 4 | 0 | 1 | 1 |
| 5 | 0 | 1 | 1 |
| 6 | 0 | 1 | 1 |
| 7 | -0.00469545 | 0.5254 | 1 |

## Position 3

| Feature | Adjusted failure − success | p | Holm p |
|---:|---:|---:|---:|
| 12461 | 0.00709578 | 7.833e-07 | 0.05775 |
| 18157 | 0.362112 | 2.372e-06 | 0.1749 |
| 16919 | 0.621067 | 2.404e-06 | 0.1772 |
| 21228 | -1.75634 | 1.193e-05 | 0.8794 |
| 0 | 0 | 1 | 1 |
| 1 | 0 | 1 | 1 |
| 2 | 0 | 1 | 1 |
| 3 | 0 | 1 | 1 |
| 4 | 0 | 1 | 1 |
| 5 | 0 | 1 | 1 |

