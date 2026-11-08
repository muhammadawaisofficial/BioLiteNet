# Extra analyses — bmw (5 folds, pooled out-of-fold predictions)


## 1. Min-expected-cost rule: critical misses (tier>=4 sent to a lower tier)

| config | n_critical_items | crit_misses_argmax | crit_misses_rule | to_general_argmax | to_general_rule | fixed | newly_missed | mcnemar_p | rel_reduction_% | 95%CI_group_boot | acc_change_pp | over_seg_%_argmax | over_seg_%_rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mnv3_ce | 6400 | 43 | 37 | 22 | 15 | 6 | 0 | 0.0312 | 14 | [4, 26] | -0.09 | 0.94 | 1.17 |
| mnv3_wce | 6400 | 58 | 46 | 39 | 27 | 12 | 0 | 0.000488 | 21 | [11, 32] | -0.06 | 0.92 | 1.08 |
| mnv3_fl | 6400 | 50 | 32 | 31 | 13 | 18 | 0 | 7.63e-06 | 36 | [22, 50] | -0.17 | 1.00 | 1.48 |
| mnv3_hafl | 6400 | 57 | 39 | 34 | 18 | 18 | 0 | 7.63e-06 | 32 | [20, 44] | -0.12 | 0.96 | 1.35 |
| mnv3_hafl_steep | 6400 | 50 | 33 | 29 | 14 | 17 | 0 | 1.53e-05 | 34 | [21, 48] | -0.20 | 0.99 | 1.42 |
| cbam_fl | 6400 | 83 | 53 | 47 | 25 | 30 | 0 | 1.86e-09 | 36 | [26, 47] | -0.16 | 1.24 | 1.77 |
| cbam_hafl | 6400 | 84 | 46 | 52 | 25 | 38 | 0 | 7.28e-12 | 45 | [34, 56] | -0.12 | 1.18 | 1.67 |
| r50_ce | 6400 | 61 | 41 | 34 | 22 | 20 | 0 | 1.91e-06 | 33 | [20, 45] | -0.04 | 1.13 | 1.39 |


### Paired comparison of critical misses between training losses

| A | B | rule | A_misses | B_misses | only_A_missed | only_B_missed | mcnemar_p | A_acc | B_acc | acc_mcnemar_p |
|---|---|---|---|---|---|---|---|---|---|---|
| mnv3_hafl | mnv3_ce | argmax | 57 | 43 | 26 | 12 | 0.0336 | 97.60 | 97.73 | 0.207 |
| mnv3_hafl | mnv3_ce | cost-rule | 39 | 37 | 17 | 15 | 0.86 | 97.48 | 97.64 | 0.142 |
| mnv3_hafl | mnv3_wce | argmax | 57 | 58 | 20 | 21 | 1 | 97.60 | 97.71 | 0.28 |
| mnv3_hafl | mnv3_wce | cost-rule | 39 | 46 | 13 | 20 | 0.296 | 97.48 | 97.65 | 0.106 |
| mnv3_hafl | mnv3_fl | argmax | 57 | 50 | 21 | 14 | 0.311 | 97.60 | 97.58 | 0.906 |
| mnv3_hafl | mnv3_fl | cost-rule | 39 | 32 | 15 | 8 | 0.21 | 97.48 | 97.40 | 0.521 |
| mnv3_wce | mnv3_ce | argmax | 58 | 43 | 28 | 13 | 0.0275 | 97.71 | 97.73 | 0.9 |
| mnv3_wce | mnv3_ce | cost-rule | 46 | 37 | 20 | 11 | 0.15 | 97.65 | 97.64 | 0.949 |
| mnv3_fl | mnv3_ce | argmax | 50 | 43 | 25 | 18 | 0.36 | 97.58 | 97.73 | 0.16 |
| mnv3_fl | mnv3_ce | cost-rule | 32 | 37 | 11 | 16 | 0.442 | 97.40 | 97.64 | 0.0322 |
| mnv3_wce | mnv3_fl | argmax | 58 | 50 | 24 | 16 | 0.268 | 97.71 | 97.58 | 0.229 |
| mnv3_wce | mnv3_fl | cost-rule | 46 | 32 | 25 | 11 | 0.0288 | 97.65 | 97.40 | 0.0252 |
| mnv3_hafl_steep | mnv3_hafl | argmax | 50 | 57 | 21 | 28 | 0.392 | 97.50 | 97.60 | 0.405 |
| mnv3_hafl_steep | mnv3_hafl | cost-rule | 33 | 39 | 10 | 16 | 0.327 | 97.30 | 97.48 | 0.111 |
| mnv3_hafl_steep | mnv3_fl | argmax | 50 | 50 | 19 | 19 | 1 | 97.50 | 97.58 | 0.53 |
| mnv3_hafl_steep | mnv3_fl | cost-rule | 33 | 32 | 10 | 9 | 1 | 97.30 | 97.40 | 0.403 |
| mnv3_hafl_steep | mnv3_ce | argmax | 50 | 43 | 27 | 20 | 0.382 | 97.50 | 97.73 | 0.0376 |
| mnv3_hafl_steep | mnv3_ce | cost-rule | 33 | 37 | 14 | 18 | 0.597 | 97.30 | 97.64 | 0.00225 |


## 2. Operating curve: accuracy / critical-miss rate (%) as the cost scale λ grows

| config | λ=0 | λ=0.25 | λ=0.5 | λ=1 | λ=2 | λ=4 | λ=8 | λ=16 |
|---|---|---|---|---|---|---|---|---|
| mnv3_ce | 97.73 / 0.67 | 97.69 / 0.66 | 97.67 / 0.59 | 97.64 / 0.58 | 97.60 / 0.56 | 97.59 / 0.52 | 97.56 / 0.50 | 97.55 / 0.50 |
| mnv3_wce | 97.71 / 0.91 | 97.69 / 0.83 | 97.66 / 0.78 | 97.65 / 0.72 | 97.62 / 0.69 | 97.60 / 0.69 | 97.56 / 0.64 | 97.53 / 0.62 |
| mnv3_fl | 97.58 / 0.78 | 97.60 / 0.61 | 97.53 / 0.55 | 97.40 / 0.50 | 97.31 / 0.50 | 97.20 / 0.48 | 97.11 / 0.42 | 97.03 / 0.41 |
| mnv3_hafl | 97.60 / 0.89 | 97.56 / 0.78 | 97.55 / 0.72 | 97.48 / 0.61 | 97.42 / 0.53 | 97.38 / 0.48 | 97.33 / 0.45 | 97.28 / 0.45 |
| mnv3_hafl_steep | 97.50 / 0.78 | 97.40 / 0.69 | 97.35 / 0.59 | 97.30 / 0.52 | 97.22 / 0.44 | 97.11 / 0.42 | 97.05 / 0.41 | 97.02 / 0.41 |
| cbam_fl | 96.70 / 1.30 | 96.65 / 1.00 | 96.60 / 0.95 | 96.54 / 0.83 | 96.48 / 0.78 | 96.39 / 0.73 | 96.22 / 0.70 | 96.14 / 0.69 |
| cbam_hafl | 96.86 / 1.31 | 96.92 / 0.97 | 96.85 / 0.80 | 96.74 / 0.72 | 96.60 / 0.61 | 96.48 / 0.58 | 96.39 / 0.58 | 96.30 / 0.55 |
| r50_ce | 97.46 / 0.95 | 97.47 / 0.77 | 97.45 / 0.72 | 97.42 / 0.64 | 97.39 / 0.59 | 97.37 / 0.53 | 97.31 / 0.52 | 97.27 / 0.52 |


## 3. Calibration (15-bin ECE) and how often the rule changes the decision

| config | ECE_% | NLL | mean_conf_% | acc_% | rule_changes_% |
|---|---|---|---|---|---|
| mnv3_ce | 1.34 | 0.107 | 99.07 | 97.73 | 0.33 |
| mnv3_wce | 1.45 | 0.121 | 99.13 | 97.71 | 0.27 |
| mnv3_fl | 0.54 | 0.088 | 97.17 | 97.58 | 0.72 |
| mnv3_hafl | 0.39 | 0.084 | 97.94 | 97.60 | 0.62 |
| mnv3_hafl_steep | 0.39 | 0.088 | 97.85 | 97.50 | 0.64 |
| cbam_fl | 0.58 | 0.119 | 96.50 | 96.70 | 0.88 |
| cbam_hafl | 0.30 | 0.108 | 97.04 | 96.86 | 0.88 |
| r50_ce | 1.20 | 0.106 | 98.62 | 97.46 | 0.53 |


## 4. Sensitivity to the rule's cost matrix (over-segregation price, tier-gap shape)

| config | over_cost | gap | acc | crit_misses | over_seg_% |
|---|---|---|---|---|---|
| mnv3_ce | 0.1 | linear | 97.61 | 37 (argmax 43) | 1.20 |
| mnv3_ce | 0.1 | squared | 97.56 | 37 (argmax 43) | 1.27 |
| mnv3_ce | 0.2 | linear | 97.64 | 37 (argmax 43) | 1.17 |
| mnv3_ce | 0.2 | squared | 97.58 | 41 (argmax 43) | 1.21 |
| mnv3_ce | 0.5 | linear | 97.69 | 42 (argmax 43) | 1.06 |
| mnv3_ce | 0.5 | squared | 97.59 | 49 (argmax 43) | 1.12 |
| mnv3_ce | 1.0 | linear | 97.71 | 46 (argmax 43) | 0.96 |
| mnv3_ce | 1.0 | squared | 97.58 | 57 (argmax 43) | 1.03 |
| mnv3_hafl | 0.1 | linear | 97.44 | 35 (argmax 57) | 1.43 |
| mnv3_hafl | 0.1 | squared | 97.31 | 38 (argmax 57) | 1.58 |
| mnv3_hafl | 0.2 | linear | 97.48 | 39 (argmax 57) | 1.35 |
| mnv3_hafl | 0.2 | squared | 97.31 | 51 (argmax 57) | 1.48 |
| mnv3_hafl | 0.5 | linear | 97.55 | 52 (argmax 57) | 1.16 |
| mnv3_hafl | 0.5 | squared | 97.33 | 71 (argmax 57) | 1.29 |
| mnv3_hafl | 1.0 | linear | 97.56 | 63 (argmax 57) | 0.98 |
| mnv3_hafl | 1.0 | squared | 97.30 | 93 (argmax 57) | 1.10 |


## 5. Critical misses left when the 5% / 10% least certain items are referred to a human

(lowconf = lowest max-probability; highrisk = highest expected cost of the chosen bin)

| config | rule | lowconf@95% | lowconf@90% | highrisk@95% | highrisk@90% | all@100% |
|---|---|---|---|---|---|---|
| mnv3_ce | argmax | 8 | 2 | 8 | 1 | 43 |
| mnv3_ce | cost-rule | 8 | 2 | 8 | 1 | 37 |
| mnv3_wce | argmax | 11 | 4 | 11 | 2 | 58 |
| mnv3_wce | cost-rule | 11 | 4 | 11 | 2 | 46 |
| mnv3_fl | argmax | 17 | 10 | 13 | 5 | 50 |
| mnv3_fl | cost-rule | 17 | 10 | 13 | 5 | 32 |
| mnv3_hafl | argmax | 12 | 4 | 8 | 3 | 57 |
| mnv3_hafl | cost-rule | 12 | 4 | 8 | 3 | 39 |
| mnv3_hafl_steep | argmax | 12 | 0 | 9 | 1 | 50 |
| mnv3_hafl_steep | cost-rule | 12 | 0 | 9 | 1 | 33 |
| cbam_fl | argmax | 30 | 9 | 15 | 8 | 83 |
| cbam_fl | cost-rule | 30 | 9 | 15 | 8 | 53 |
| cbam_hafl | argmax | 23 | 11 | 16 | 8 | 84 |
| cbam_hafl | cost-rule | 23 | 11 | 16 | 8 | 46 |
| r50_ce | argmax | 13 | 3 | 13 | 1 | 61 |
| r50_ce | cost-rule | 13 | 3 | 13 | 1 | 41 |


## 6. Temperature scaling (fitted on other folds, in-domain) before the rule

| config | T_mean | ECE_none | rule_acc_none | rule_crit_misses_none | ECE_T | rule_acc_T | rule_crit_misses_T |
|---|---|---|---|---|---|---|---|
| mnv3_ce | 1.86 | 1.34 | 97.64 | 37 | 0.33 | 97.56 | 29 |
| mnv3_wce | 1.98 | 1.45 | 97.65 | 46 | 0.36 | 97.54 | 37 |
| mnv3_fl | 0.99 | 0.54 | 97.40 | 32 | 0.49 | 97.42 | 33 |
| mnv3_hafl | 1.15 | 0.39 | 97.48 | 39 | 0.38 | 97.46 | 37 |
| mnv3_hafl_steep | 1.15 | 0.39 | 97.30 | 33 | 0.46 | 97.28 | 30 |
| cbam_fl | 1.03 | 0.58 | 96.54 | 53 | 0.64 | 96.53 | 53 |
| cbam_hafl | 1.08 | 0.30 | 96.74 | 46 | 0.46 | 96.71 | 46 |
| r50_ce | 1.66 | 1.20 | 97.42 | 41 | 0.21 | 97.30 | 32 |


### Paired critical misses after the rule: recalibrated CE vs HAFL

| A | B | A_misses | B_misses | only_A_missed | only_B_missed | mcnemar_p |
|---|---|---|---|---|---|---|
| mnv3_ce+T+rule | mnv3_hafl+rule | 29 | 39 | 10 | 20 | 0.0987 |
| mnv3_ce+T+rule | mnv3_hafl+T+rule | 29 | 37 | 10 | 18 | 0.185 |


## 7. Error severity at the WHO-bin level

| config | rule | acc | bin_acc | errors | same_bin_% | safer_bin | less_safe_bin | crit_misses_per_1000 |
|---|---|---|---|---|---|---|---|---|
| mnv3_ce | argmax | 97.73 | 98.55 | 354 | 36 | 147 | 79 | 6.7 |
| mnv3_ce | cost-rule | 97.64 | 98.45 | 368 | 34 | 182 | 60 | 5.8 |
| mnv3_wce | argmax | 97.71 | 98.49 | 357 | 34 | 144 | 92 | 9.1 |
| mnv3_wce | cost-rule | 97.65 | 98.44 | 366 | 34 | 168 | 75 | 7.2 |
| mnv3_fl | argmax | 97.58 | 98.37 | 378 | 33 | 156 | 98 | 7.8 |
| mnv3_fl | cost-rule | 97.40 | 98.15 | 405 | 29 | 231 | 57 | 5.0 |
| mnv3_hafl | argmax | 97.60 | 98.44 | 375 | 35 | 149 | 94 | 8.9 |
| mnv3_hafl | cost-rule | 97.48 | 98.29 | 393 | 32 | 210 | 57 | 6.1 |
| mnv3_hafl_steep | argmax | 97.50 | 98.41 | 390 | 36 | 154 | 94 | 7.8 |
| mnv3_hafl_steep | cost-rule | 97.30 | 98.19 | 421 | 33 | 221 | 62 | 5.2 |
| cbam_fl | argmax | 96.70 | 97.78 | 515 | 33 | 193 | 154 | 13.0 |
| cbam_fl | cost-rule | 96.54 | 97.60 | 540 | 31 | 276 | 99 | 8.3 |
| cbam_hafl | argmax | 96.86 | 97.96 | 490 | 35 | 184 | 134 | 13.1 |
| cbam_hafl | cost-rule | 96.74 | 97.86 | 508 | 34 | 260 | 74 | 7.2 |
| r50_ce | argmax | 97.46 | 98.22 | 396 | 30 | 177 | 100 | 9.5 |
| r50_ce | cost-rule | 97.42 | 98.22 | 402 | 31 | 217 | 60 | 6.4 |


### Remaining critical misses, mnv3_hafl + rule

- 11× tweezers (tier 5) → human_organs (tier 4)
- 5× used_syringes (tier 5) → human_organs (tier 4)
- 3× blood_soaked_bandages (tier 4) → used_medical_paper (tier 1)
- 3× human_organs (tier 4) → general_organic_waste (tier 1)
- 3× tweezers (tier 5) → used_medical_paper (tier 1)
- 2× blood_soaked_bandages (tier 4) → general_organic_waste (tier 1)
- 2× blood_soaked_bandages (tier 4) → waterbottles (tier 1)
- 2× blood_soaked_bandages (tier 4) → used_medical_gloves (tier 1)

## 1. Min-expected-cost rule: critical misses (tier>=4 sent to a lower tier) — RAW-background images

| config | n_critical_items | crit_misses_argmax | crit_misses_rule | to_general_argmax | to_general_rule | fixed | newly_missed | mcnemar_p | rel_reduction_% | 95%CI_group_boot | acc_change_pp | over_seg_%_argmax | over_seg_%_rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mnv3_ce | 6400 | 378 | 255 | 262 | 164 | 123 | 0 | 1.88e-37 | 33 | [27, 39] | +0.26 | 1.40 | 1.68 |
| mnv3_wce | 6400 | 454 | 345 | 321 | 231 | 109 | 0 | 3.08e-33 | 24 | [20, 29] | +0.20 | 1.31 | 1.54 |
| mnv3_fl | 6400 | 268 | 118 | 168 | 56 | 150 | 0 | 1.4e-45 | 56 | [49, 63] | +0.06 | 1.40 | 2.06 |
| mnv3_hafl | 6400 | 307 | 174 | 196 | 83 | 133 | 0 | 1.84e-40 | 43 | [37, 50] | -0.01 | 1.46 | 2.07 |
| mnv3_hafl_steep | 6400 | 413 | 257 | 292 | 143 | 156 | 0 | 2.19e-47 | 38 | [33, 43] | +0.16 | 1.37 | 1.93 |
| cbam_fl | 6400 | 402 | 253 | 252 | 117 | 149 | 0 | 2.8e-45 | 37 | [30, 45] | +0.15 | 2.33 | 3.33 |
| cbam_hafl | 6400 | 298 | 187 | 154 | 77 | 111 | 0 | 7.7e-34 | 37 | [30, 45] | -0.06 | 2.56 | 3.56 |
| r50_ce | 6400 | 371 | 266 | 253 | 167 | 105 | 0 | 4.93e-32 | 28 | [22, 36] | +0.15 | 1.43 | 1.72 |


### Paired comparison of critical misses between training losses — RAW-background images

| A | B | rule | A_misses | B_misses | only_A_missed | only_B_missed | mcnemar_p | A_acc | B_acc | acc_mcnemar_p |
|---|---|---|---|---|---|---|---|---|---|---|
| mnv3_hafl | mnv3_ce | argmax | 307 | 378 | 183 | 254 | 0.000793 | 88.40 | 87.32 | 3.5e-06 |
| mnv3_hafl | mnv3_ce | cost-rule | 174 | 255 | 108 | 189 | 3.02e-06 | 88.39 | 87.58 | 0.000522 |
| mnv3_hafl | mnv3_wce | argmax | 307 | 454 | 180 | 327 | 6.61e-11 | 88.40 | 88.34 | 0.799 |
| mnv3_hafl | mnv3_wce | cost-rule | 174 | 345 | 103 | 274 | 4.93e-19 | 88.39 | 88.54 | 0.533 |
| mnv3_hafl | mnv3_fl | argmax | 307 | 268 | 196 | 157 | 0.043 | 88.40 | 87.28 | 1.55e-06 |
| mnv3_hafl | mnv3_fl | cost-rule | 174 | 118 | 121 | 65 | 4.88e-05 | 88.39 | 87.34 | 6.3e-06 |
| mnv3_wce | mnv3_ce | argmax | 454 | 378 | 304 | 228 | 0.00113 | 88.34 | 87.32 | 1.17e-05 |
| mnv3_wce | mnv3_ce | cost-rule | 345 | 255 | 245 | 155 | 7.9e-06 | 88.54 | 87.58 | 3.93e-05 |
| mnv3_fl | mnv3_ce | argmax | 268 | 378 | 159 | 269 | 1.18e-07 | 87.28 | 87.32 | 0.892 |
| mnv3_fl | mnv3_ce | cost-rule | 118 | 255 | 64 | 201 | 1.18e-17 | 87.34 | 87.58 | 0.327 |
| mnv3_wce | mnv3_fl | argmax | 454 | 268 | 331 | 145 | 8.63e-18 | 88.34 | 87.28 | 4.55e-06 |
| mnv3_wce | mnv3_fl | cost-rule | 345 | 118 | 282 | 55 | 6.87e-38 | 88.54 | 87.34 | 2.73e-07 |
| mnv3_hafl_steep | mnv3_hafl | argmax | 413 | 307 | 281 | 175 | 7.88e-07 | 86.88 | 88.40 | 6.79e-11 |
| mnv3_hafl_steep | mnv3_hafl | cost-rule | 257 | 174 | 190 | 107 | 1.69e-06 | 87.04 | 88.39 | 7.58e-09 |
| mnv3_hafl_steep | mnv3_fl | argmax | 413 | 268 | 301 | 156 | 1.1e-11 | 86.88 | 87.28 | 0.0949 |
| mnv3_hafl_steep | mnv3_fl | cost-rule | 257 | 118 | 201 | 62 | 2.68e-18 | 87.04 | 87.34 | 0.215 |
| mnv3_hafl_steep | mnv3_ce | argmax | 413 | 378 | 259 | 224 | 0.122 | 86.88 | 87.32 | 0.0661 |
| mnv3_hafl_steep | mnv3_ce | cost-rule | 257 | 255 | 164 | 162 | 0.956 | 87.04 | 87.58 | 0.0239 |


## 2. Operating curve: accuracy / critical-miss rate (%) as the cost scale λ grows — RAW-background images

| config | λ=0 | λ=0.25 | λ=0.5 | λ=1 | λ=2 | λ=4 | λ=8 | λ=16 |
|---|---|---|---|---|---|---|---|---|
| mnv3_ce | 87.32 / 5.91 | 87.50 / 4.91 | 87.54 / 4.48 | 87.58 / 3.98 | 87.61 / 3.70 | 87.65 / 3.38 | 87.58 / 3.23 | 87.54 / 3.20 |
| mnv3_wce | 88.34 / 7.09 | 88.49 / 6.34 | 88.54 / 5.92 | 88.54 / 5.39 | 88.59 / 5.06 | 88.58 / 4.86 | 88.59 / 4.69 | 88.53 / 4.66 |
| mnv3_fl | 87.28 / 4.19 | 87.37 / 2.75 | 87.33 / 2.34 | 87.34 / 1.84 | 87.28 / 1.67 | 87.13 / 1.53 | 87.00 / 1.27 | 86.88 / 1.22 |
| mnv3_hafl | 88.40 / 4.80 | 88.51 / 3.67 | 88.45 / 3.23 | 88.39 / 2.72 | 88.28 / 2.30 | 88.20 / 2.02 | 88.11 / 1.88 | 88.04 / 1.83 |
| mnv3_hafl_steep | 86.89 / 6.45 | 86.97 / 5.08 | 87.00 / 4.55 | 87.05 / 4.02 | 86.98 / 3.48 | 86.94 / 3.08 | 86.82 / 2.92 | 86.78 / 2.84 |
| cbam_fl | 80.44 / 6.28 | 80.55 / 4.91 | 80.61 / 4.33 | 80.60 / 3.95 | 80.44 / 3.67 | 80.32 / 3.36 | 80.24 / 3.20 | 80.16 / 3.09 |
| cbam_hafl | 82.23 / 4.66 | 82.33 / 3.62 | 82.25 / 3.22 | 82.17 / 2.92 | 82.03 / 2.62 | 81.80 / 2.45 | 81.55 / 2.42 | 81.48 / 2.39 |
| r50_ce | 87.43 / 5.80 | 87.54 / 4.84 | 87.58 / 4.47 | 87.58 / 4.16 | 87.59 / 3.84 | 87.58 / 3.64 | 87.56 / 3.55 | 87.55 / 3.41 |


## 3. Calibration (15-bin ECE) and how often the rule changes the decision — RAW-background images

| config | ECE_% | NLL | mean_conf_% | acc_% | rule_changes_% |
|---|---|---|---|---|---|
| mnv3_ce | 8.25 | 0.593 | 95.56 | 87.32 | 1.28 |
| mnv3_wce | 7.71 | 0.605 | 96.05 | 88.34 | 1.09 |
| mnv3_fl | 3.23 | 0.417 | 90.50 | 87.28 | 1.92 |
| mnv3_hafl | 3.92 | 0.380 | 92.32 | 88.40 | 1.73 |
| mnv3_hafl_steep | 4.95 | 0.467 | 91.83 | 86.88 | 1.91 |
| cbam_fl | 6.70 | 0.652 | 87.14 | 80.44 | 2.71 |
| cbam_hafl | 7.88 | 0.639 | 90.11 | 82.23 | 2.24 |
| r50_ce | 8.24 | 0.684 | 95.67 | 87.43 | 1.21 |


## 4. Sensitivity to the rule's cost matrix (over-segregation price, tier-gap shape) — RAW-background images

| config | over_cost | gap | acc | crit_misses | over_seg_% |
|---|---|---|---|---|---|
| mnv3_ce | 0.1 | linear | 87.62 | 228 (argmax 378) | 1.73 |
| mnv3_ce | 0.1 | squared | 87.41 | 264 (argmax 378) | 1.87 |
| mnv3_ce | 0.2 | linear | 87.58 | 255 (argmax 378) | 1.68 |
| mnv3_ce | 0.2 | squared | 87.26 | 322 (argmax 378) | 1.79 |
| mnv3_ce | 0.5 | linear | 87.51 | 306 (argmax 378) | 1.51 |
| mnv3_ce | 0.5 | squared | 87.03 | 411 (argmax 378) | 1.70 |
| mnv3_ce | 1.0 | linear | 87.34 | 370 (argmax 378) | 1.42 |
| mnv3_ce | 1.0 | squared | 86.79 | 503 (argmax 378) | 1.63 |
| mnv3_hafl | 0.1 | linear | 88.34 | 152 (argmax 307) | 2.21 |
| mnv3_hafl | 0.1 | squared | 87.78 | 182 (argmax 307) | 2.79 |
| mnv3_hafl | 0.2 | linear | 88.39 | 174 (argmax 307) | 2.07 |
| mnv3_hafl | 0.2 | squared | 87.76 | 250 (argmax 307) | 2.58 |
| mnv3_hafl | 0.5 | linear | 88.53 | 227 (argmax 307) | 1.73 |
| mnv3_hafl | 0.5 | squared | 87.63 | 366 (argmax 307) | 2.21 |
| mnv3_hafl | 1.0 | linear | 88.46 | 286 (argmax 307) | 1.43 |
| mnv3_hafl | 1.0 | squared | 87.15 | 505 (argmax 307) | 1.96 |


## 5. Critical misses left when the 5% / 10% least certain items are referred to a human — RAW-background images

(lowconf = lowest max-probability; highrisk = highest expected cost of the chosen bin)

| config | rule | lowconf@95% | lowconf@90% | highrisk@95% | highrisk@90% | all@100% |
|---|---|---|---|---|---|---|
| mnv3_ce | argmax | 219 | 141 | 136 | 96 | 378 |
| mnv3_ce | cost-rule | 205 | 141 | 136 | 96 | 255 |
| mnv3_wce | argmax | 298 | 185 | 208 | 136 | 454 |
| mnv3_wce | cost-rule | 288 | 185 | 208 | 136 | 345 |
| mnv3_fl | argmax | 160 | 103 | 63 | 45 | 268 |
| mnv3_fl | cost-rule | 101 | 93 | 63 | 45 | 118 |
| mnv3_hafl | argmax | 207 | 136 | 84 | 61 | 307 |
| mnv3_hafl | cost-rule | 158 | 127 | 83 | 61 | 174 |
| mnv3_hafl_steep | argmax | 261 | 184 | 126 | 79 | 413 |
| mnv3_hafl_steep | cost-rule | 209 | 170 | 125 | 79 | 257 |
| cbam_fl | argmax | 295 | 247 | 158 | 119 | 402 |
| cbam_fl | cost-rule | 220 | 207 | 156 | 119 | 253 |
| cbam_hafl | argmax | 196 | 145 | 108 | 80 | 298 |
| cbam_hafl | cost-rule | 155 | 132 | 108 | 80 | 187 |
| r50_ce | argmax | 223 | 135 | 140 | 96 | 371 |
| r50_ce | cost-rule | 214 | 135 | 140 | 96 | 266 |


## 6. Temperature scaling (fitted on other folds, in-domain) before the rule — RAW-background images

| config | T_mean | ECE_none | rule_acc_none | rule_crit_misses_none | ECE_T | rule_acc_T | rule_crit_misses_T |
|---|---|---|---|---|---|---|---|
| mnv3_ce | 1.86 | 8.25 | 87.58 | 255 | 3.57 | 87.60 | 161 |
| mnv3_wce | 1.98 | 7.71 | 88.54 | 345 | 3.03 | 88.58 | 229 |
| mnv3_fl | 0.99 | 3.23 | 87.34 | 118 | 3.33 | 87.34 | 120 |
| mnv3_hafl | 1.15 | 3.92 | 88.39 | 174 | 2.47 | 88.33 | 155 |
| mnv3_hafl_steep | 1.15 | 4.95 | 87.04 | 257 | 3.44 | 87.02 | 226 |
| cbam_fl | 1.03 | 6.70 | 80.60 | 253 | 6.26 | 80.59 | 251 |
| cbam_hafl | 1.08 | 7.88 | 82.17 | 187 | 6.96 | 82.12 | 178 |
| r50_ce | 1.66 | 8.24 | 87.58 | 266 | 4.77 | 87.65 | 198 |


### Paired critical misses after the rule: recalibrated CE vs HAFL — RAW-background images

| A | B | A_misses | B_misses | only_A_missed | only_B_missed | mcnemar_p |
|---|---|---|---|---|---|---|
| mnv3_ce+T+rule | mnv3_hafl+rule | 161 | 174 | 107 | 120 | 0.426 |
| mnv3_ce+T+rule | mnv3_hafl+T+rule | 161 | 155 | 111 | 105 | 0.734 |


## 7. Error severity at the WHO-bin level — RAW-background images

| config | rule | acc | bin_acc | errors | same_bin_% | safer_bin | less_safe_bin | crit_misses_per_1000 |
|---|---|---|---|---|---|---|---|---|
| mnv3_ce | argmax | 87.32 | 95.62 | 1978 | 65 | 218 | 465 | 59.1 |
| mnv3_ce | cost-rule | 87.58 | 96.21 | 1938 | 70 | 262 | 329 | 39.8 |
| mnv3_wce | argmax | 88.34 | 95.22 | 1819 | 59 | 204 | 542 | 70.9 |
| mnv3_wce | cost-rule | 88.54 | 95.78 | 1788 | 63 | 240 | 418 | 53.9 |
| mnv3_fl | argmax | 87.28 | 96.29 | 1984 | 71 | 219 | 360 | 41.9 |
| mnv3_fl | cost-rule | 87.34 | 96.81 | 1975 | 75 | 322 | 175 | 18.4 |
| mnv3_hafl | argmax | 88.40 | 96.08 | 1809 | 66 | 227 | 385 | 48.0 |
| mnv3_hafl | cost-rule | 88.39 | 96.51 | 1811 | 70 | 323 | 222 | 27.2 |
| mnv3_hafl_steep | argmax | 86.88 | 95.36 | 2046 | 65 | 214 | 510 | 64.5 |
| mnv3_hafl_steep | cost-rule | 87.04 | 95.96 | 2021 | 69 | 301 | 330 | 40.2 |
| cbam_fl | argmax | 80.44 | 93.32 | 3051 | 66 | 363 | 679 | 62.8 |
| cbam_fl | cost-rule | 80.60 | 93.87 | 3027 | 68 | 519 | 437 | 39.5 |
| cbam_hafl | argmax | 82.23 | 94.27 | 2772 | 68 | 399 | 495 | 46.6 |
| cbam_hafl | cost-rule | 82.17 | 94.43 | 2782 | 69 | 555 | 314 | 29.2 |
| r50_ce | argmax | 87.43 | 95.61 | 1961 | 65 | 223 | 462 | 58.0 |
| r50_ce | cost-rule | 87.58 | 96.09 | 1937 | 69 | 269 | 341 | 41.6 |


### Remaining critical misses, mnv3_hafl + rule — RAW-background images

- 24× stitch_removal_scissors (tier 5) → uncategorized_or_overlapping (tier 1)
- 16× scalpels (tier 5) → human_organs (tier 4)
- 14× episiotomy_scissors (tier 5) → uncategorized_or_overlapping (tier 1)
- 9× forceps (tier 5) → human_organs (tier 4)
- 9× used_syringes (tier 5) → expired_tablets (tier 2)
- 8× blood_soaked_bandages (tier 4) → used_medical_paper (tier 1)
- 8× used_syringes (tier 5) → used_medical_paper (tier 1)
- 8× used_syringes (tier 5) → human_organs (tier 4)
