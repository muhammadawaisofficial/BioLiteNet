# Results (40 runs, folds [0, 1, 2, 3, 4])

## Main table (mean ± std over folds; %, except hazard_cost)

| config | n_folds | params | acc | macro_f1 | sharps_f1 | hazard_cost | critical_miss_rate | under_seg_rate |
|---|---|---|---|---|---|---|---|---|
| mnv3_ce | 5 | 2994055 | 97.73 ± 0.24 | 97.59 ± 0.26 | 98.42 ± 0.17 | 0.0132 ± 0.0033 | 0.68 ± 0.28 | 0.51 ± 0.17 |
| mnv3_wce | 5 | 2994055 | 97.72 ± 0.38 | 97.57 ± 0.43 | 98.56 ± 0.29 | 0.0167 ± 0.0049 | 0.90 ± 0.35 | 0.59 ± 0.22 |
| mnv3_fl | 5 | 2994055 | 97.58 ± 0.61 | 97.43 ± 0.67 | 98.56 ± 0.43 | 0.0164 ± 0.0051 | 0.78 ± 0.35 | 0.63 ± 0.27 |
| mnv3_hafl | 5 | 2994055 | 97.60 ± 0.41 | 97.45 ± 0.45 | 98.44 ± 0.27 | 0.0159 ± 0.0033 | 0.89 ± 0.36 | 0.60 ± 0.15 |
| cbam_fl | 5 | 2996435 | 96.70 ± 0.26 | 96.49 ± 0.31 | 97.84 ± 0.51 | 0.0239 ± 0.0059 | 1.31 ± 0.43 | 0.99 ± 0.22 |
| cbam_hafl | 5 | 2996435 | 96.86 ± 0.41 | 96.66 ± 0.46 | 97.73 ± 0.34 | 0.0231 ± 0.0027 | 1.32 ± 0.27 | 0.86 ± 0.23 |
| r50_ce | 5 | 23555159 | 97.46 ± 0.28 | 97.31 ± 0.32 | 98.52 ± 0.43 | 0.0176 ± 0.0021 | 0.95 ± 0.27 | 0.64 ± 0.18 |
| mnv3_hafl_steep | 5 | 2994055 | 97.50 ± 0.20 | 97.36 ± 0.25 | 98.40 ± 0.24 | 0.0158 ± 0.0024 | 0.78 ± 0.34 | 0.60 ± 0.17 |


## Pairwise tests (pooled predictions; McNemar exact, Wilcoxon over folds, bootstrap CI)

| A | B | acc_diff_pp | 95%CI_pp | A_only_right | B_only_right | mcnemar_p | wilcoxon_macroF1_p |
|---|---|---|---|---|---|---|---|
| mnv3_ce | mnv3_wce | +0.02 | [-0.17, +0.22] | 128 | 125 | 0.9 | 1 |
| mnv3_ce | mnv3_fl | +0.15 | [-0.05, +0.37] | 146 | 122 | 0.1599 | 0.625 |
| mnv3_ce | mnv3_hafl | +0.13 | [-0.07, +0.33] | 136 | 115 | 0.2067 | 0.1875 |
| mnv3_ce | cbam_fl | +1.03 | [+0.79, +1.28] | 274 | 113 | 1.518e-16 | 0.0625 |
| mnv3_ce | cbam_hafl | +0.87 | [+0.65, +1.11] | 249 | 113 | 6.951e-13 | 0.0625 |
| mnv3_ce | r50_ce | +0.27 | [+0.05, +0.49] | 177 | 135 | 0.02013 | 0.0625 |
| mnv3_ce | mnv3_hafl_steep | +0.23 | [+0.02, +0.46] | 160 | 124 | 0.03763 | 0.125 |
| mnv3_wce | mnv3_fl | +0.13 | [-0.08, +0.35] | 149 | 128 | 0.2294 | 0.625 |
| mnv3_wce | mnv3_hafl | +0.12 | [-0.08, +0.31] | 133 | 115 | 0.2803 | 0.3125 |
| mnv3_wce | cbam_fl | +1.01 | [+0.78, +1.26] | 264 | 106 | 1.12e-16 | 0.0625 |
| mnv3_wce | cbam_hafl | +0.85 | [+0.62, +1.10] | 242 | 109 | 9.93e-13 | 0.0625 |
| mnv3_wce | r50_ce | +0.25 | [+0.04, +0.47] | 180 | 141 | 0.03376 | 0.125 |
| mnv3_wce | mnv3_hafl_steep | +0.21 | [+0.01, +0.42] | 152 | 119 | 0.05171 | 0.125 |
| mnv3_fl | mnv3_hafl | -0.02 | [-0.23, +0.20] | 141 | 144 | 0.9057 | 1 |
| mnv3_fl | cbam_fl | +0.88 | [+0.65, +1.13] | 259 | 122 | 1.875e-12 | 0.0625 |
| mnv3_fl | cbam_hafl | +0.72 | [+0.47, +0.97] | 240 | 128 | 5.533e-09 | 0.0625 |
| mnv3_fl | r50_ce | +0.12 | [-0.12, +0.35] | 177 | 159 | 0.3537 | 1 |
| mnv3_fl | mnv3_hafl_steep | +0.08 | [-0.15, +0.30] | 159 | 147 | 0.5295 | 0.625 |
| mnv3_hafl | cbam_fl | +0.90 | [+0.65, +1.15] | 271 | 131 | 2.498e-12 | 0.0625 |
| mnv3_hafl | cbam_hafl | +0.74 | [+0.50, +0.98] | 238 | 123 | 1.446e-09 | 0.0625 |
| mnv3_hafl | r50_ce | +0.13 | [-0.10, +0.37] | 174 | 153 | 0.2687 | 0.625 |
| mnv3_hafl | mnv3_hafl_steep | +0.10 | [-0.12, +0.31] | 149 | 134 | 0.4053 | 1 |
| cbam_fl | cbam_hafl | -0.16 | [-0.40, +0.08] | 176 | 201 | 0.2164 | 0.4375 |
| cbam_fl | r50_ce | -0.76 | [-1.02, -0.51] | 149 | 268 | 5.991e-09 | 0.0625 |
| cbam_fl | mnv3_hafl_steep | -0.80 | [-1.04, -0.55] | 136 | 261 | 3.508e-10 | 0.0625 |
| cbam_hafl | r50_ce | -0.60 | [-0.86, -0.35] | 152 | 246 | 2.837e-06 | 0.0625 |
| cbam_hafl | mnv3_hafl_steep | -0.64 | [-0.88, -0.40] | 128 | 228 | 1.288e-07 | 0.0625 |
| r50_ce | mnv3_hafl_steep | -0.04 | [-0.28, +0.19] | 166 | 172 | 0.7857 | 1 |


## Training-time (HAFL) vs decision-time cost sensitivity (pooled)

| config | rule | acc | macro_f1 | sharps_f1 | hazard_cost | critical_miss_rate | under_seg_rate |
|---|---|---|---|---|---|---|---|
| mnv3_ce | argmax | 97.73 | 97.60 | 98.43 | 0.0131 | 0.67 | 0.51 |
| mnv3_ce | min-expected-cost | 97.64 | 97.51 | 98.34 | 0.0116 | 0.58 | 0.38 |
| mnv3_wce | argmax | 97.71 | 97.58 | 98.58 | 0.0168 | 0.91 | 0.59 |
| mnv3_wce | min-expected-cost | 97.65 | 97.52 | 98.51 | 0.0143 | 0.72 | 0.48 |
| mnv3_fl | argmax | 97.58 | 97.44 | 98.58 | 0.0164 | 0.78 | 0.63 |
| mnv3_fl | min-expected-cost | 97.40 | 97.26 | 98.36 | 0.0125 | 0.50 | 0.37 |
| mnv3_hafl | argmax | 97.60 | 97.45 | 98.46 | 0.0159 | 0.89 | 0.60 |
| mnv3_hafl | min-expected-cost | 97.48 | 97.34 | 98.36 | 0.0122 | 0.61 | 0.37 |
| cbam_fl | argmax | 96.70 | 96.51 | 97.89 | 0.0238 | 1.30 | 0.99 |
| cbam_fl | min-expected-cost | 96.54 | 96.34 | 97.70 | 0.0184 | 0.83 | 0.63 |
| cbam_hafl | argmax | 96.86 | 96.68 | 97.78 | 0.0231 | 1.31 | 0.86 |
| cbam_hafl | min-expected-cost | 96.74 | 96.57 | 97.72 | 0.0163 | 0.72 | 0.47 |
| r50_ce | argmax | 97.46 | 97.32 | 98.53 | 0.0176 | 0.95 | 0.64 |
| r50_ce | min-expected-cost | 97.42 | 97.28 | 98.39 | 0.0134 | 0.64 | 0.38 |
| mnv3_hafl_steep | argmax | 97.50 | 97.36 | 98.40 | 0.0158 | 0.78 | 0.60 |
| mnv3_hafl_steep | min-expected-cost | 97.30 | 97.15 | 98.32 | 0.0127 | 0.52 | 0.40 |


## Raw (original background) images of the test folds (pooled)

| config | acc | macro_f1 | sharps_f1 | hazard_cost | critical_miss_rate | under_seg_rate |
|---|---|---|---|---|---|---|
| mnv3_ce | 87.32 | 85.92 | 70.59 | 0.0938 | 5.91 | 2.98 |
| mnv3_wce | 88.34 | 87.30 | 74.76 | 0.1092 | 7.09 | 3.47 |
| mnv3_fl | 87.28 | 86.12 | 70.52 | 0.0665 | 4.19 | 2.31 |
| mnv3_hafl | 88.40 | 87.48 | 74.13 | 0.0746 | 4.80 | 2.47 |
| cbam_fl | 80.44 | 78.78 | 58.66 | 0.1179 | 6.28 | 4.35 |
| cbam_hafl | 82.23 | 81.11 | 64.25 | 0.0864 | 4.66 | 3.17 |
| r50_ce | 87.43 | 86.30 | 72.24 | 0.0934 | 5.80 | 2.96 |
| mnv3_hafl_steep | 86.88 | 85.51 | 70.28 | 0.1022 | 6.45 | 3.27 |


## Residual leakage: accuracy by nearest-training-image similarity

| config | acc_with_train_neighbour>=0.95 | acc_without |
|---|---|---|
| mnv3_ce | 99.53 (n=7256) | 96.16 (n=8344) |
| mnv3_wce | 99.59 (n=7256) | 96.08 (n=8344) |
| mnv3_fl | 99.52 (n=7256) | 95.89 (n=8344) |
| mnv3_hafl | 99.61 (n=7256) | 95.84 (n=8344) |
| cbam_fl | 99.41 (n=7256) | 94.34 (n=8344) |
| cbam_hafl | 99.50 (n=7256) | 94.56 (n=8344) |
| r50_ce | 99.60 (n=7256) | 95.60 (n=8344) |
| mnv3_hafl_steep | 99.59 (n=7256) | 95.69 (n=8344) |


## Top confusions (cbam_hafl, pooled)

- 26× ampoules_full → ampuoles_broken
- 17× ampuoles_broken → ampoules_full
- 13× tweezers → human_organs
- 12× waterbottles → expired_tablets
- 11× tweezers → used_medical_paper
- 10× used_medical_paper → blood_soaked_bandages
- 10× human_organs → general_organic_waste
- 10× expired_tablets → disinfectant_bottles
- 9× vaccine_or_medicine_vials → ampoules_full
- 9× used_medical_paper → human_organs
