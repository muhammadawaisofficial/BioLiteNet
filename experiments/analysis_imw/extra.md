# Extra analyses — imw (5 folds, pooled out-of-fold predictions)


## 1. Min-expected-cost rule: critical misses (tier>=4 sent to a lower tier)

| config | n_critical_items | crit_misses_argmax | crit_misses_rule | to_general_argmax | to_general_rule | fixed | newly_missed | mcnemar_p | rel_reduction_% | 95%CI_group_boot | acc_change_pp | over_seg_%_argmax | over_seg_%_rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mnv3_ce | 4323 | 43 | 32 | 19 | 15 | 11 | 0 | 0.000977 | 26 | [11, 44] | -0.20 | 1.09 | 1.37 |
| mnv3_wce | 4323 | 60 | 44 | 41 | 30 | 16 | 0 | 3.05e-05 | 27 | [15, 41] | -0.18 | 0.99 | 1.30 |
| mnv3_fl | 4323 | 90 | 52 | 30 | 16 | 38 | 0 | 7.28e-12 | 42 | [30, 55] | -0.18 | 1.32 | 2.12 |
| mnv3_hafl | 4323 | 62 | 28 | 38 | 12 | 34 | 0 | 1.16e-10 | 55 | [27, 73] | -0.02 | 1.27 | 1.65 |
| mnv3_hafl_steep | 4323 | 70 | 34 | 39 | 13 | 36 | 0 | 2.91e-11 | 51 | [35, 66] | -0.15 | 1.44 | 2.14 |
| cbam_hafl | 4323 | 100 | 56 | 36 | 19 | 44 | 0 | 1.14e-13 | 44 | [28, 61] | -0.08 | 1.55 | 2.25 |


### Paired comparison of critical misses between training losses

| A | B | rule | A_misses | B_misses | only_A_missed | only_B_missed | mcnemar_p | A_acc | B_acc | acc_mcnemar_p |
|---|---|---|---|---|---|---|---|---|---|---|
| mnv3_hafl | mnv3_ce | argmax | 62 | 43 | 43 | 24 | 0.0271 | 94.17 | 94.50 | 0.242 |
| mnv3_hafl | mnv3_ce | cost-rule | 28 | 32 | 22 | 26 | 0.665 | 94.15 | 94.30 | 0.631 |
| mnv3_hafl | mnv3_wce | argmax | 62 | 60 | 34 | 32 | 0.902 | 94.17 | 94.54 | 0.182 |
| mnv3_hafl | mnv3_wce | cost-rule | 28 | 44 | 15 | 31 | 0.0259 | 94.15 | 94.35 | 0.502 |
| mnv3_hafl | mnv3_fl | argmax | 62 | 90 | 33 | 61 | 0.00508 | 94.17 | 93.97 | 0.532 |
| mnv3_hafl | mnv3_fl | cost-rule | 28 | 52 | 16 | 40 | 0.00184 | 94.15 | 93.79 | 0.249 |
| mnv3_wce | mnv3_ce | argmax | 60 | 43 | 39 | 22 | 0.0396 | 94.54 | 94.50 | 0.949 |
| mnv3_wce | mnv3_ce | cost-rule | 44 | 32 | 29 | 17 | 0.104 | 94.35 | 94.30 | 0.9 |
| mnv3_fl | mnv3_ce | argmax | 90 | 43 | 66 | 19 | 3.04e-07 | 93.97 | 94.50 | 0.0695 |
| mnv3_fl | mnv3_ce | cost-rule | 52 | 32 | 38 | 18 | 0.0105 | 93.79 | 94.30 | 0.0826 |
| mnv3_wce | mnv3_fl | argmax | 60 | 90 | 29 | 59 | 0.00182 | 94.54 | 93.97 | 0.0421 |
| mnv3_wce | mnv3_fl | cost-rule | 44 | 52 | 30 | 38 | 0.396 | 94.35 | 93.79 | 0.0476 |
| mnv3_hafl_steep | mnv3_hafl | argmax | 70 | 62 | 46 | 38 | 0.445 | 93.77 | 94.17 | 0.171 |
| mnv3_hafl_steep | mnv3_hafl | cost-rule | 34 | 28 | 24 | 18 | 0.441 | 93.62 | 94.15 | 0.0801 |
| mnv3_hafl_steep | mnv3_fl | argmax | 70 | 90 | 38 | 58 | 0.0519 | 93.77 | 93.97 | 0.511 |
| mnv3_hafl_steep | mnv3_fl | cost-rule | 34 | 52 | 19 | 37 | 0.0222 | 93.62 | 93.79 | 0.583 |
| mnv3_hafl_steep | mnv3_ce | argmax | 70 | 43 | 45 | 18 | 0.000898 | 93.77 | 94.50 | 0.0114 |
| mnv3_hafl_steep | mnv3_ce | cost-rule | 34 | 32 | 20 | 18 | 0.871 | 93.62 | 94.30 | 0.021 |


## 2. Operating curve: accuracy / critical-miss rate (%) as the cost scale λ grows

| config | λ=0 | λ=0.25 | λ=0.5 | λ=1 | λ=2 | λ=4 | λ=8 | λ=16 |
|---|---|---|---|---|---|---|---|---|
| mnv3_ce | 94.50 / 0.99 | 94.37 / 0.86 | 94.30 / 0.81 | 94.30 / 0.74 | 94.22 / 0.74 | 94.19 / 0.65 | 94.17 / 0.65 | 94.12 / 0.65 |
| mnv3_wce | 94.54 / 1.39 | 94.44 / 1.25 | 94.44 / 1.09 | 94.35 / 1.02 | 94.30 / 0.97 | 94.27 / 0.88 | 94.24 / 0.79 | 94.22 / 0.79 |
| mnv3_fl | 93.97 / 2.08 | 94.04 / 1.53 | 93.85 / 1.41 | 93.79 / 1.20 | 93.72 / 1.06 | 93.62 / 0.97 | 93.62 / 0.95 | 93.62 / 0.90 |
| mnv3_hafl | 94.17 / 1.43 | 94.15 / 0.99 | 94.19 / 0.69 | 94.15 / 0.65 | 94.12 / 0.53 | 94.04 / 0.44 | 93.92 / 0.39 | 93.89 / 0.37 |
| mnv3_hafl_steep | 93.77 / 1.62 | 93.75 / 1.20 | 93.72 / 0.97 | 93.62 / 0.79 | 93.54 / 0.58 | 93.37 / 0.46 | 93.28 / 0.44 | 93.25 / 0.42 |
| cbam_hafl | 92.37 / 2.31 | 92.40 / 1.80 | 92.43 / 1.53 | 92.28 / 1.29 | 92.22 / 1.11 | 92.08 / 1.04 | 91.88 / 1.02 | 91.92 / 0.93 |


## 3. Calibration (15-bin ECE) and how often the rule changes the decision

| config | ECE_% | NLL | mean_conf_% | acc_% | rule_changes_% |
|---|---|---|---|---|---|
| mnv3_ce | 3.24 | 0.281 | 97.74 | 94.50 | 0.43 |
| mnv3_wce | 3.67 | 0.321 | 98.18 | 94.54 | 0.58 |
| mnv3_fl | 1.28 | 0.223 | 95.14 | 93.97 | 1.50 |
| mnv3_hafl | 1.40 | 0.227 | 95.55 | 94.17 | 0.97 |
| mnv3_hafl_steep | 1.83 | 0.237 | 95.43 | 93.77 | 1.37 |
| cbam_hafl | 2.06 | 0.285 | 94.43 | 92.37 | 1.42 |


## 4. Sensitivity to the rule's cost matrix (over-segregation price, tier-gap shape)

| config | over_cost | gap | acc | crit_misses | over_seg_% |
|---|---|---|---|---|---|
| mnv3_ce | 0.1 | linear | 94.22 | 32 (argmax 43) | 1.49 |
| mnv3_ce | 0.1 | squared | 94.12 | 27 (argmax 43) | 1.64 |
| mnv3_ce | 0.2 | linear | 94.30 | 32 (argmax 43) | 1.37 |
| mnv3_ce | 0.2 | squared | 94.19 | 32 (argmax 43) | 1.55 |
| mnv3_ce | 0.5 | linear | 94.32 | 37 (argmax 43) | 1.29 |
| mnv3_ce | 0.5 | squared | 94.20 | 38 (argmax 43) | 1.42 |
| mnv3_ce | 1.0 | linear | 94.44 | 43 (argmax 43) | 1.12 |
| mnv3_ce | 1.0 | squared | 94.25 | 45 (argmax 43) | 1.27 |
| mnv3_hafl | 0.1 | linear | 94.14 | 26 (argmax 62) | 1.69 |
| mnv3_hafl | 0.1 | squared | 93.85 | 18 (argmax 62) | 2.10 |
| mnv3_hafl | 0.2 | linear | 94.15 | 28 (argmax 62) | 1.65 |
| mnv3_hafl | 0.2 | squared | 93.94 | 23 (argmax 62) | 1.94 |
| mnv3_hafl | 0.5 | linear | 94.27 | 33 (argmax 62) | 1.44 |
| mnv3_hafl | 0.5 | squared | 94.00 | 40 (argmax 62) | 1.62 |
| mnv3_hafl | 1.0 | linear | 94.20 | 54 (argmax 62) | 1.24 |
| mnv3_hafl | 1.0 | squared | 93.82 | 67 (argmax 62) | 1.49 |


## 5. Critical misses left when the 5% / 10% least certain items are referred to a human

(lowconf = lowest max-probability; highrisk = highest expected cost of the chosen bin)

| config | rule | lowconf@95% | lowconf@90% | highrisk@95% | highrisk@90% | all@100% |
|---|---|---|---|---|---|---|
| mnv3_ce | argmax | 21 | 9 | 14 | 8 | 43 |
| mnv3_ce | cost-rule | 21 | 9 | 14 | 8 | 32 |
| mnv3_wce | argmax | 25 | 9 | 19 | 7 | 60 |
| mnv3_wce | cost-rule | 25 | 9 | 19 | 7 | 44 |
| mnv3_fl | argmax | 51 | 32 | 33 | 19 | 90 |
| mnv3_fl | cost-rule | 49 | 32 | 33 | 19 | 52 |
| mnv3_hafl | argmax | 23 | 11 | 11 | 6 | 62 |
| mnv3_hafl | cost-rule | 23 | 11 | 11 | 6 | 28 |
| mnv3_hafl_steep | argmax | 25 | 6 | 9 | 6 | 70 |
| mnv3_hafl_steep | cost-rule | 20 | 6 | 9 | 6 | 34 |
| cbam_hafl | argmax | 56 | 32 | 33 | 20 | 100 |
| cbam_hafl | cost-rule | 49 | 32 | 33 | 20 | 56 |


## 6. Temperature scaling (fitted on other folds, in-domain) before the rule

| config | T_mean | ECE_none | rule_acc_none | rule_crit_misses_none | ECE_T | rule_acc_T | rule_crit_misses_T |
|---|---|---|---|---|---|---|---|
| mnv3_ce | 2.07 | 3.24 | 94.30 | 32 | 1.04 | 93.89 | 16 |
| mnv3_wce | 2.32 | 3.67 | 94.35 | 44 | 0.78 | 94.09 | 23 |
| mnv3_fl | 1.23 | 1.28 | 93.79 | 52 | 1.35 | 93.64 | 43 |
| mnv3_hafl | 1.30 | 1.40 | 94.15 | 28 | 0.93 | 94.04 | 22 |
| mnv3_hafl_steep | 1.33 | 1.83 | 93.62 | 34 | 1.32 | 93.34 | 27 |
| cbam_hafl | 1.33 | 2.06 | 92.28 | 56 | 1.46 | 92.03 | 45 |


### Paired critical misses after the rule: recalibrated CE vs HAFL

| A | B | A_misses | B_misses | only_A_missed | only_B_missed | mcnemar_p |
|---|---|---|---|---|---|---|
| mnv3_ce+T+rule | mnv3_hafl+rule | 16 | 28 | 11 | 23 | 0.0576 |
| mnv3_ce+T+rule | mnv3_hafl+T+rule | 16 | 22 | 11 | 17 | 0.345 |


## 7. Error severity at the WHO-bin level

| config | rule | acc | bin_acc | errors | same_bin_% | safer_bin | less_safe_bin | crit_misses_per_1000 |
|---|---|---|---|---|---|---|---|---|
| mnv3_ce | argmax | 94.50 | 98.20 | 329 | 67 | 65 | 43 | 9.9 |
| mnv3_ce | cost-rule | 94.30 | 98.10 | 341 | 67 | 82 | 32 | 7.4 |
| mnv3_wce | argmax | 94.54 | 98.00 | 327 | 63 | 59 | 61 | 13.9 |
| mnv3_wce | cost-rule | 94.35 | 97.96 | 338 | 64 | 78 | 44 | 10.2 |
| mnv3_fl | argmax | 93.97 | 97.09 | 361 | 52 | 79 | 95 | 20.8 |
| mnv3_fl | cost-rule | 93.79 | 96.98 | 372 | 51 | 127 | 54 | 12.0 |
| mnv3_hafl | argmax | 94.17 | 97.64 | 349 | 60 | 76 | 65 | 14.3 |
| mnv3_hafl | cost-rule | 94.15 | 97.86 | 350 | 63 | 99 | 29 | 6.5 |
| mnv3_hafl_steep | argmax | 93.77 | 97.39 | 373 | 58 | 86 | 70 | 16.2 |
| mnv3_hafl_steep | cost-rule | 93.62 | 97.29 | 382 | 58 | 128 | 34 | 7.9 |
| cbam_hafl | argmax | 92.37 | 96.73 | 457 | 57 | 93 | 103 | 23.1 |
| cbam_hafl | cost-rule | 92.28 | 96.79 | 462 | 58 | 135 | 57 | 13.0 |


### Remaining critical misses, mnv3_hafl + rule

- 8× SDB (tier 4) → PPE (tier 1)
- 4× DW (tier 4) → PhW (tier 2)
- 4× SDB (tier 4) → PhW (tier 2)
- 3× SW (tier 5) → CLW (tier 4)
- 2× BBP (tier 4) → PhW (tier 2)
- 1× BBP (tier 4) → PPE (tier 1)
- 1× CLW (tier 4) → PhW (tier 2)
- 1× CLW (tier 4) → CBL (tier 1)
