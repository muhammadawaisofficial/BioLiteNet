# Results (25 runs, folds [0, 1, 2, 3, 4])

## Main table (mean ± std over folds; %, except hazard_cost)

| config | n_folds | params | acc | macro_f1 | sharps_f1 | hazard_cost | critical_miss_rate | under_seg_rate |
|---|---|---|---|---|---|---|---|---|
| mnv3_ce | 5 | 2994055 | 98.06 ± 0.40 | 97.75 ± 0.49 | 99.52 ± 0.60 | 0.0071 ± 0.0077 | 0.56 ± 0.82 | 0.17 ± 0.25 |
| mnv3_wce | 5 | 2994055 | 98.19 ± 0.46 | 97.95 ± 0.55 | 99.81 ± 0.21 | 0.0051 ± 0.0028 | 0.38 ± 0.43 | 0.12 ± 0.13 |
| mnv3_fl | 5 | 2994055 | 97.55 ± 0.84 | 97.34 ± 0.94 | 99.40 ± 1.24 | 0.0108 ± 0.0084 | 0.94 ± 0.70 | 0.30 ± 0.21 |
| mnv3_hafl | 5 | 2994055 | 97.83 ± 0.70 | 97.50 ± 0.77 | 99.73 ± 0.29 | 0.0076 ± 0.0057 | 0.61 ± 0.66 | 0.20 ± 0.20 |
| cbam_hafl | 5 | 2996435 | 97.89 ± 0.61 | 97.56 ± 0.81 | 98.86 ± 1.29 | 0.0082 ± 0.0067 | 0.71 ± 0.62 | 0.23 ± 0.19 |


## Pairwise tests (pooled predictions; McNemar exact, Wilcoxon over folds, bootstrap CI)

| A | B | acc_diff_pp | 95%CI_pp | A_only_right | B_only_right | mcnemar_p | wilcoxon_macroF1_p |
|---|---|---|---|---|---|---|---|
| mnv3_ce | mnv3_wce | -0.12 | [-0.40, +0.15] | 85 | 97 | 0.4149 | 0.8125 |
| mnv3_ce | mnv3_fl | +0.52 | [+0.22, +0.83] | 139 | 89 | 0.001128 | 0.625 |
| mnv3_ce | mnv3_hafl | +0.24 | [-0.06, +0.53] | 111 | 88 | 0.1186 | 0.625 |
| mnv3_ce | cbam_hafl | +0.18 | [-0.10, +0.47] | 110 | 93 | 0.2614 | 0.8125 |
| mnv3_wce | mnv3_fl | +0.64 | [+0.35, +0.93] | 139 | 77 | 2.95e-05 | 0.1875 |
| mnv3_wce | mnv3_hafl | +0.36 | [+0.09, +0.64] | 113 | 78 | 0.01368 | 0.1875 |
| mnv3_wce | cbam_hafl | +0.30 | [+0.01, +0.60] | 119 | 90 | 0.05251 | 0.3125 |
| mnv3_fl | mnv3_hafl | -0.28 | [-0.58, +0.01] | 98 | 125 | 0.08144 | 0.625 |
| mnv3_fl | cbam_hafl | -0.34 | [-0.65, -0.03] | 101 | 134 | 0.03662 | 0.3125 |
| mnv3_hafl | cbam_hafl | -0.06 | [-0.36, +0.24] | 109 | 115 | 0.7384 | 0.8125 |


## Training-time (HAFL) vs decision-time cost sensitivity (pooled)

| config | rule | acc | macro_f1 | sharps_f1 | hazard_cost | critical_miss_rate | under_seg_rate |
|---|---|---|---|---|---|---|---|
| mnv3_ce | argmax | 98.06 | 97.77 | 99.55 | 0.0071 | 0.55 | 0.18 |
| mnv3_ce | min-expected-cost | 98.05 | 97.77 | 99.55 | 0.0067 | 0.48 | 0.16 |
| mnv3_wce | argmax | 98.18 | 97.97 | 99.82 | 0.0051 | 0.38 | 0.12 |
| mnv3_wce | min-expected-cost | 98.18 | 97.97 | 99.77 | 0.0044 | 0.29 | 0.09 |
| mnv3_fl | argmax | 97.54 | 97.30 | 99.47 | 0.0109 | 0.93 | 0.30 |
| mnv3_fl | min-expected-cost | 97.34 | 97.17 | 99.33 | 0.0102 | 0.64 | 0.21 |
| mnv3_hafl | argmax | 97.82 | 97.50 | 99.73 | 0.0076 | 0.61 | 0.20 |
| mnv3_hafl | min-expected-cost | 97.69 | 97.40 | 99.41 | 0.0067 | 0.35 | 0.11 |
| cbam_hafl | argmax | 97.88 | 97.60 | 98.92 | 0.0083 | 0.71 | 0.23 |
| cbam_hafl | min-expected-cost | 97.82 | 97.58 | 98.92 | 0.0071 | 0.51 | 0.17 |


## Residual leakage: accuracy by nearest-training-image similarity

| config | acc_with_train_neighbour>=0.99 | acc_without |
|---|---|---|
| mnv3_ce | 99.46 (n=551) | 97.97 (n=9083) |
| mnv3_wce | 99.64 (n=551) | 98.10 (n=9083) |
| mnv3_fl | 99.46 (n=551) | 97.42 (n=9083) |
| mnv3_hafl | 99.46 (n=551) | 97.72 (n=9083) |
| cbam_hafl | 99.27 (n=551) | 97.80 (n=9083) |


## Top confusions (cbam_hafl, pooled)

- 28× glove_single_latex → glove_pair_latex
- 23× glove_pair_latex → glove_single_latex
- 19× glove_single_nitrile → glove_pair_nitrile
- 18× glove_pair_surgery → glove_single_surgery
- 18× glove_pair_nitrile → glove_single_nitrile
- 11× shoe_cover_pair → shoe_cover_single
- 9× glass_equipment_packaging → syringe_needle
- 7× syringe → paper_equipment_packaging
- 7× paper_equipment_packaging → gauze
- 7× glove_single_surgery → glove_pair_surgery
