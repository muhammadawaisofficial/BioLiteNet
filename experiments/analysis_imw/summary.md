# Results (30 runs, folds [0, 1, 2, 3, 4])

## Main table (mean ± std over folds; %, except hazard_cost)

| config | n_folds | params | acc | macro_f1 | sharps_f1 | hazard_cost | critical_miss_rate | under_seg_rate |
|---|---|---|---|---|---|---|---|---|
| mnv3_ce | 5 | 2982523 | 94.50 ± 1.31 | 94.40 ± 1.37 | 98.12 ± 1.70 | 0.0219 ± 0.0112 | 1.00 ± 0.71 | 0.71 ± 0.48 |
| mnv3_wce | 5 | 2982523 | 94.50 ± 2.46 | 94.60 ± 2.17 | 98.25 ± 1.24 | 0.0321 ± 0.0175 | 1.41 ± 0.95 | 1.02 ± 0.69 |
| mnv3_fl | 5 | 2982523 | 93.94 ± 1.85 | 93.93 ± 1.71 | 95.28 ± 3.36 | 0.0399 ± 0.0138 | 2.09 ± 0.99 | 1.59 ± 0.71 |
| mnv3_hafl | 5 | 2982523 | 94.16 ± 1.55 | 94.08 ± 1.37 | 98.20 ± 0.58 | 0.0334 ± 0.0157 | 1.45 ± 0.80 | 1.08 ± 0.62 |
| cbam_hafl | 5 | 2984903 | 92.31 ± 3.04 | 92.33 ± 2.76 | 95.65 ± 3.48 | 0.0471 ± 0.0213 | 2.34 ± 1.76 | 1.74 ± 1.31 |
| mnv3_hafl_steep | 5 | 2982523 | 93.75 ± 2.64 | 93.70 ± 2.42 | 97.54 ± 3.13 | 0.0355 ± 0.0108 | 1.63 ± 0.58 | 1.17 ± 0.39 |


## Pairwise tests (pooled predictions; McNemar exact, Wilcoxon over folds, bootstrap CI)

| A | B | acc_diff_pp | 95%CI_pp | A_only_right | B_only_right | mcnemar_p | wilcoxon_macroF1_p |
|---|---|---|---|---|---|---|---|
| mnv3_ce | mnv3_wce | -0.03 | [-0.55, +0.47] | 122 | 124 | 0.9492 | 0.8125 |
| mnv3_ce | mnv3_fl | +0.53 | [-0.03, +1.07] | 162 | 130 | 0.06947 | 1 |
| mnv3_ce | mnv3_hafl | +0.33 | [-0.20, +0.84] | 142 | 122 | 0.2422 | 0.125 |
| mnv3_ce | cbam_hafl | +2.14 | [+1.55, +2.72] | 222 | 94 | 4.338e-13 | 0.0625 |
| mnv3_ce | mnv3_hafl_steep | +0.73 | [+0.17, +1.24] | 167 | 123 | 0.01144 | 0.4375 |
| mnv3_wce | mnv3_fl | +0.57 | [+0.03, +1.12] | 149 | 115 | 0.04205 | 0.625 |
| mnv3_wce | mnv3_hafl | +0.37 | [-0.15, +0.90] | 135 | 113 | 0.1823 | 0.625 |
| mnv3_wce | cbam_hafl | +2.17 | [+1.60, +2.72] | 208 | 78 | 8.135e-15 | 0.0625 |
| mnv3_wce | mnv3_hafl_steep | +0.77 | [+0.25, +1.27] | 144 | 98 | 0.00373 | 0.3125 |
| mnv3_fl | mnv3_hafl | -0.20 | [-0.79, +0.37] | 149 | 161 | 0.5322 | 0.8125 |
| mnv3_fl | cbam_hafl | +1.60 | [+0.99, +2.19] | 213 | 117 | 1.398e-07 | 0.1875 |
| mnv3_fl | mnv3_hafl_steep | +0.20 | [-0.33, +0.73] | 146 | 134 | 0.511 | 0.8125 |
| mnv3_hafl | cbam_hafl | +1.80 | [+1.20, +2.41] | 219 | 111 | 2.811e-09 | 0.3125 |
| mnv3_hafl | mnv3_hafl_steep | +0.40 | [-0.15, +0.90] | 153 | 129 | 0.1707 | 0.625 |
| cbam_hafl | mnv3_hafl_steep | -1.40 | [-1.99, -0.85] | 115 | 199 | 2.469e-06 | 0.3125 |


## Training-time (HAFL) vs decision-time cost sensitivity (pooled)

| config | rule | acc | macro_f1 | sharps_f1 | hazard_cost | critical_miss_rate | under_seg_rate |
|---|---|---|---|---|---|---|---|
| mnv3_ce | argmax | 94.50 | 94.40 | 98.10 | 0.0221 | 0.99 | 0.72 |
| mnv3_ce | min-expected-cost | 94.30 | 94.20 | 97.89 | 0.0190 | 0.74 | 0.53 |
| mnv3_wce | argmax | 94.54 | 94.62 | 98.23 | 0.0321 | 1.39 | 1.02 |
| mnv3_wce | min-expected-cost | 94.35 | 94.44 | 98.01 | 0.0266 | 1.02 | 0.73 |
| mnv3_fl | argmax | 93.97 | 93.89 | 95.13 | 0.0400 | 2.08 | 1.59 |
| mnv3_fl | min-expected-cost | 93.79 | 93.71 | 95.32 | 0.0290 | 1.20 | 0.90 |
| mnv3_hafl | argmax | 94.17 | 94.13 | 98.22 | 0.0335 | 1.43 | 1.09 |
| mnv3_hafl | min-expected-cost | 94.15 | 94.04 | 98.12 | 0.0199 | 0.65 | 0.48 |
| cbam_hafl | argmax | 92.37 | 92.36 | 95.59 | 0.0468 | 2.31 | 1.72 |
| cbam_hafl | min-expected-cost | 92.28 | 92.30 | 94.79 | 0.0325 | 1.30 | 0.95 |
| mnv3_hafl_steep | argmax | 93.77 | 93.72 | 97.54 | 0.0356 | 1.62 | 1.17 |
| mnv3_hafl_steep | min-expected-cost | 93.62 | 93.54 | 96.57 | 0.0229 | 0.79 | 0.57 |


## Residual leakage: accuracy by nearest-training-image similarity

| config | acc_with_train_neighbour>=0.9 | acc_without |
|---|---|---|
| mnv3_ce | 98.61 (n=2814) | 90.86 (n=3173) |
| mnv3_wce | 98.83 (n=2814) | 90.73 (n=3173) |
| mnv3_fl | 97.51 (n=2814) | 90.83 (n=3173) |
| mnv3_hafl | 98.29 (n=2814) | 90.51 (n=3173) |
| cbam_hafl | 97.23 (n=2814) | 88.06 (n=3173) |
| mnv3_hafl_steep | 98.05 (n=2814) | 89.98 (n=3173) |


## Top confusions (cbam_hafl, pooled)

- 36× SDB → CLW
- 32× SDB → PhW
- 26× PPE → SDB
- 22× CLW → BBP
- 21× SDB → CAW
- 21× MW → BBP
- 19× CBL → PPE
- 17× PW → DW
- 15× BBP → MW
- 14× PPE → BBP
