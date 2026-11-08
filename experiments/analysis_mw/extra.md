# Extra analyses — mw (5 folds, pooled out-of-fold predictions)


## 1. Min-expected-cost rule: critical misses (tier>=4 sent to a lower tier)

| config | n_critical_items | crit_misses_argmax | crit_misses_rule | to_general_argmax | to_general_rule | fixed | newly_missed | mcnemar_p | rel_reduction_% | 95%CI_group_boot | acc_change_pp | over_seg_%_argmax | over_seg_%_rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mnv3_ce | 3117 | 17 | 15 | 17 | 15 | 2 | 0 | 0.5 | 12 | [0, 43] | -0.01 | 0.27 | 0.31 |
| mnv3_wce | 3117 | 12 | 9 | 10 | 7 | 3 | 0 | 0.25 | 25 | [0, 100] | +0.00 | 0.29 | 0.33 |
| mnv3_fl | 3117 | 29 | 20 | 24 | 15 | 9 | 0 | 0.00391 | 31 | [9, 64] | -0.20 | 0.47 | 0.80 |
| mnv3_hafl | 3117 | 19 | 11 | 18 | 10 | 8 | 0 | 0.00781 | 42 | [7, 75] | -0.13 | 0.30 | 0.54 |
| cbam_hafl | 3117 | 22 | 16 | 16 | 10 | 6 | 0 | 0.0312 | 27 | [8, 67] | -0.06 | 0.34 | 0.49 |


### Paired comparison of critical misses between training losses

| A | B | rule | A_misses | B_misses | only_A_missed | only_B_missed | mcnemar_p | A_acc | B_acc | acc_mcnemar_p |
|---|---|---|---|---|---|---|---|---|---|---|
| mnv3_hafl | mnv3_ce | argmax | 19 | 17 | 7 | 5 | 0.774 | 97.82 | 98.06 | 0.119 |
| mnv3_hafl | mnv3_ce | cost-rule | 11 | 15 | 2 | 6 | 0.289 | 97.69 | 98.05 | 0.0168 |
| mnv3_hafl | mnv3_wce | argmax | 19 | 12 | 10 | 3 | 0.0923 | 97.82 | 98.18 | 0.0137 |
| mnv3_hafl | mnv3_wce | cost-rule | 11 | 9 | 4 | 2 | 0.688 | 97.69 | 98.18 | 0.000953 |
| mnv3_hafl | mnv3_fl | argmax | 19 | 29 | 5 | 15 | 0.0414 | 97.82 | 97.54 | 0.0814 |
| mnv3_hafl | mnv3_fl | cost-rule | 11 | 20 | 2 | 11 | 0.0225 | 97.69 | 97.34 | 0.0366 |
| mnv3_wce | mnv3_ce | argmax | 12 | 17 | 4 | 9 | 0.267 | 98.18 | 98.06 | 0.415 |
| mnv3_wce | mnv3_ce | cost-rule | 9 | 15 | 2 | 8 | 0.109 | 98.18 | 98.05 | 0.375 |
| mnv3_fl | mnv3_ce | argmax | 29 | 17 | 17 | 5 | 0.0169 | 97.54 | 98.06 | 0.00113 |
| mnv3_fl | mnv3_ce | cost-rule | 20 | 15 | 9 | 4 | 0.267 | 97.34 | 98.05 | 1.04e-05 |
| mnv3_wce | mnv3_fl | argmax | 12 | 29 | 1 | 18 | 7.63e-05 | 98.18 | 97.54 | 2.95e-05 |
| mnv3_wce | mnv3_fl | cost-rule | 9 | 20 | 0 | 11 | 0.000977 | 98.18 | 97.34 | 1.2e-07 |


## 2. Operating curve: accuracy / critical-miss rate (%) as the cost scale λ grows

| config | λ=0 | λ=0.25 | λ=0.5 | λ=1 | λ=2 | λ=4 | λ=8 | λ=16 |
|---|---|---|---|---|---|---|---|---|
| mnv3_ce | 98.06 / 0.55 | 98.08 / 0.48 | 98.06 / 0.48 | 98.05 / 0.48 | 98.05 / 0.45 | 98.03 / 0.45 | 98.04 / 0.42 | 98.03 / 0.42 |
| mnv3_wce | 98.18 / 0.39 | 98.16 / 0.35 | 98.17 / 0.32 | 98.18 / 0.29 | 98.17 / 0.29 | 98.14 / 0.29 | 98.12 / 0.29 | 98.12 / 0.29 |
| mnv3_fl | 97.54 / 0.93 | 97.48 / 0.80 | 97.43 / 0.74 | 97.34 / 0.64 | 97.25 / 0.58 | 97.19 / 0.55 | 97.16 / 0.55 | 97.16 / 0.51 |
| mnv3_hafl | 97.82 / 0.61 | 97.77 / 0.48 | 97.74 / 0.45 | 97.69 / 0.35 | 97.59 / 0.29 | 97.54 / 0.29 | 97.52 / 0.29 | 97.53 / 0.26 |
| cbam_hafl | 97.88 / 0.71 | 97.84 / 0.61 | 97.85 / 0.51 | 97.82 / 0.51 | 97.77 / 0.51 | 97.72 / 0.51 | 97.69 / 0.48 | 97.65 / 0.45 |


## 3. Calibration (15-bin ECE) and how often the rule changes the decision

| config | ECE_% | NLL | mean_conf_% | acc_% | rule_changes_% |
|---|---|---|---|---|---|
| mnv3_ce | 1.10 | 0.096 | 99.12 | 98.06 | 0.06 |
| mnv3_wce | 1.32 | 0.104 | 99.35 | 98.18 | 0.07 |
| mnv3_fl | 0.43 | 0.087 | 97.17 | 97.54 | 0.43 |
| mnv3_hafl | 0.68 | 0.081 | 97.29 | 97.82 | 0.32 |
| cbam_hafl | 1.20 | 0.083 | 96.80 | 97.88 | 0.21 |


## 4. Sensitivity to the rule's cost matrix (over-segregation price, tier-gap shape)

| config | over_cost | gap | acc | crit_misses | over_seg_% |
|---|---|---|---|---|---|
| mnv3_ce | 0.1 | linear | 98.04 | 14 (argmax 17) | 0.37 |
| mnv3_ce | 0.1 | squared | 98.04 | 12 (argmax 17) | 0.42 |
| mnv3_ce | 0.2 | linear | 98.05 | 15 (argmax 17) | 0.31 |
| mnv3_ce | 0.2 | squared | 98.03 | 15 (argmax 17) | 0.37 |
| mnv3_ce | 0.5 | linear | 98.07 | 15 (argmax 17) | 0.29 |
| mnv3_ce | 0.5 | squared | 98.04 | 17 (argmax 17) | 0.30 |
| mnv3_ce | 1.0 | linear | 98.04 | 19 (argmax 17) | 0.26 |
| mnv3_ce | 1.0 | squared | 98.04 | 19 (argmax 17) | 0.27 |
| mnv3_hafl | 0.1 | linear | 97.59 | 10 (argmax 19) | 0.65 |
| mnv3_hafl | 0.1 | squared | 97.47 | 8 (argmax 19) | 0.82 |
| mnv3_hafl | 0.2 | linear | 97.69 | 11 (argmax 19) | 0.54 |
| mnv3_hafl | 0.2 | squared | 97.57 | 9 (argmax 19) | 0.69 |
| mnv3_hafl | 0.5 | linear | 97.77 | 16 (argmax 19) | 0.39 |
| mnv3_hafl | 0.5 | squared | 97.75 | 14 (argmax 19) | 0.44 |
| mnv3_hafl | 1.0 | linear | 97.86 | 18 (argmax 19) | 0.25 |
| mnv3_hafl | 1.0 | squared | 97.86 | 17 (argmax 19) | 0.27 |


## 5. Critical misses left when the 5% / 10% least certain items are referred to a human

(lowconf = lowest max-probability; highrisk = highest expected cost of the chosen bin)

| config | rule | lowconf@95% | lowconf@90% | highrisk@95% | highrisk@90% | all@100% |
|---|---|---|---|---|---|---|
| mnv3_ce | argmax | 11 | 9 | 9 | 8 | 17 |
| mnv3_ce | cost-rule | 11 | 9 | 9 | 8 | 15 |
| mnv3_wce | argmax | 7 | 7 | 7 | 7 | 12 |
| mnv3_wce | cost-rule | 7 | 7 | 7 | 7 | 9 |
| mnv3_fl | argmax | 17 | 11 | 15 | 10 | 29 |
| mnv3_fl | cost-rule | 17 | 11 | 15 | 10 | 20 |
| mnv3_hafl | argmax | 7 | 7 | 7 | 7 | 19 |
| mnv3_hafl | cost-rule | 7 | 7 | 7 | 7 | 11 |
| cbam_hafl | argmax | 15 | 8 | 11 | 8 | 22 |
| cbam_hafl | cost-rule | 15 | 8 | 11 | 8 | 16 |


## 6. Temperature scaling (fitted on other folds, in-domain) before the rule

| config | T_mean | ECE_none | rule_acc_none | rule_crit_misses_none | ECE_T | rule_acc_T | rule_crit_misses_T |
|---|---|---|---|---|---|---|---|
| mnv3_ce | 1.72 | 1.10 | 98.05 | 15 | 0.53 | 98.04 | 12 |
| mnv3_wce | 1.91 | 1.32 | 98.18 | 9 | 0.38 | 98.11 | 9 |
| mnv3_fl | 1.01 | 0.43 | 97.34 | 20 | 0.45 | 97.34 | 20 |
| mnv3_hafl | 1.03 | 0.68 | 97.69 | 11 | 0.84 | 97.70 | 10 |
| cbam_hafl | 0.94 | 1.20 | 97.82 | 16 | 0.95 | 97.82 | 16 |


### Paired critical misses after the rule: recalibrated CE vs HAFL

| A | B | A_misses | B_misses | only_A_missed | only_B_missed | mcnemar_p |
|---|---|---|---|---|---|---|
| mnv3_ce+T+rule | mnv3_hafl+rule | 12 | 11 | 3 | 2 | 1 |
| mnv3_ce+T+rule | mnv3_hafl+T+rule | 12 | 10 | 3 | 1 | 0.625 |


## 7. Error severity at the WHO-bin level

| config | rule | acc | bin_acc | errors | same_bin_% | safer_bin | less_safe_bin | crit_misses_per_1000 |
|---|---|---|---|---|---|---|---|---|
| mnv3_ce | argmax | 98.06 | 99.55 | 187 | 77 | 26 | 17 | 5.5 |
| mnv3_ce | cost-rule | 98.05 | 99.53 | 188 | 76 | 30 | 15 | 4.8 |
| mnv3_wce | argmax | 98.18 | 99.58 | 175 | 77 | 28 | 12 | 3.8 |
| mnv3_wce | cost-rule | 98.18 | 99.57 | 175 | 77 | 32 | 9 | 2.9 |
| mnv3_fl | argmax | 97.54 | 99.23 | 237 | 69 | 45 | 29 | 9.3 |
| mnv3_fl | cost-rule | 97.34 | 98.99 | 256 | 62 | 77 | 20 | 6.4 |
| mnv3_hafl | argmax | 97.82 | 99.50 | 210 | 77 | 29 | 19 | 6.1 |
| mnv3_hafl | cost-rule | 97.69 | 99.35 | 223 | 72 | 52 | 11 | 3.5 |
| cbam_hafl | argmax | 97.88 | 99.43 | 204 | 73 | 33 | 22 | 7.1 |
| cbam_hafl | cost-rule | 97.82 | 99.35 | 210 | 70 | 47 | 16 | 5.1 |


### Remaining critical misses, mnv3_hafl + rule

- 7× gauze (tier 4) → paper_equipment_packaging (tier 1)
- 1× gauze (tier 4) → syringe (tier 1)
- 1× syringe_needle (tier 5) → test_tube (tier 4)
- 1× urine_bag (tier 4) → glove_pair_nitrile (tier 1)
- 1× urine_bag (tier 4) → medical_glasses (tier 1)
