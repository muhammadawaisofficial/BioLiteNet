# BioLiteNet

Code, data splits and saved predictions for the paper

> M. Awais, R. Nadeem, J. U. Rehman and S. Asad, "BioLiteNet: Hazard-Aware Training and Decision Rules for Lightweight Biomedical Waste Classification," International Conference on Frontiers of Information Technology (FIT), 2026 (to appear).

We train MobileNetV3-Large on three public biomedical-waste datasets with duplicate-grouped five-fold cross-validation, compare four training losses, and then choose the final class with a minimum-expected-cost rule over WHO hazard tiers. Items with the highest expected cost can be sent to a person instead.

The code of the version first submitted to FIT is kept under the git tag `submitted-version`. Some of its results were wrong (see the corrections in the paper), so please use the code on the main branch.

## What is in this repository

```
model.py                 MobileNetV3-Large backbone, optional CBAM attention
experiments/
  manifest.csv           D1 split: class, file, duplicate group, test fold, validation flags
  manifest_mw.csv        D2 split
  manifest_imw.csv       D3 split
  run_bln.py             training of all configurations and folds (written for Kaggle T4 GPUs)
  embed_images.py        image embeddings for finding near-duplicates (D1)
  prep_mw.py             the same for D2 and D3
  build_folds.py         duplicate groups and the five folds
  analyze.py             accuracy, macro-F1, critical-miss rate, significance tests
  extra_analysis.py      decision rule, calibration, temperature scaling, referral
  tier_sensitivity.py    other tier mappings
  diagnostics.py         ROC / PR for critical items, training curves
  per_class.py           per-class F1
  failure_figure.py      randomly drawn critical misses with Grad-CAM
  bench.py               parameters, MACs and CPU latency
  paper_assets.py        tables and figures of the paper
  xdata/                 cross-dataset test (train on one dataset, test on another)
  results*/              saved test predictions and training logs of all 95 models
  analysis*/             the numbers reported in the paper
```

`results` is D1, `results_mw` is D2 and `results_imw` is D3. Each run folder (`<config>_f<fold>`) holds `metrics.json`, `log.csv` and the per-image test predictions (`test_preds.npz`; for D1 also `raw_preds.npz`, the same photos with their original backgrounds). The checkpoints are not in the repository because of their size. They are attached to the GitHub release:

- `checkpoints_ce_hafl.zip`: MobileNetV3-Large with CE and with HAFL, 5 folds x 3 datasets (364 MB)
- `checkpoints_all.zip`: all 95 models (1.6 GB)
- `embeddings.zip`: the image embeddings used to find duplicates (112 MB)

## Datasets

| | Kaggle dataset | Images | Classes |
|---|---|---|---|
| D1 | [mario78/biomedical-waste-classification-images-dataset](https://www.kaggle.com/datasets/mario78/biomedical-waste-classification-images-dataset) | 15,600 | 23 |
| D2 | [sumn2u/medical-waste](https://www.kaggle.com/datasets/sumn2u/medical-waste) | 9,634 | 23 |
| D3 | [engineeringubu/infectious-municipal-waste](https://www.kaggle.com/datasets/engineeringubu/infectious-municipal-waste) | 5,987 | 11 |

For D1 we train on the background-removed images and use the original photos only as a second test set. In D3 the two folders with soiled dressings are merged and 112 images whose near-copies have a different label are removed. The manifests list exactly which files are used.

## Setup

```
pip install -r requirements.txt
```

Python 3.10 or newer and PyTorch 2.x. Training ran on Kaggle (2 x T4); everything else runs on a CPU.

## Reproducing the tables without training

All numbers in the paper are computed from the saved predictions, so no GPU is needed:

```
cd experiments
python analyze.py results              # D1 -> analysis/
python analyze.py results_mw mw        # D2 -> analysis_mw/
python analyze.py results_imw imw      # D3 -> analysis_imw/
python extra_analysis.py results bmw
python extra_analysis.py results_mw mw
python extra_analysis.py results_imw imw
python tier_sensitivity.py results_mw mw syringe=4
python tier_sensitivity.py results_imw imw PPE=4 CBL=4
python xdata/analyze_xdata.py
```

One part of `analysis/summary.md`, accuracy by similarity to the nearest training image, also needs the embeddings: unzip `embeddings.zip` into `experiments/` first. `failure_figure.py` needs the checkpoints and the original images.

## Training

`run_bln.py` trains every configuration on every fold with the same settings (30 epochs, AdamW, lr 1e-3, cosine schedule, batch 32). On Kaggle, add the dataset and a private dataset with this code, then run

```
!python /kaggle/input/<your-code-dataset>/run_bln.py --probe     # checks the data and times one epoch
!python /kaggle/input/<your-code-dataset>/run_bln.py             # all jobs, one worker per GPU
```

For D2 and D3 set `BLN_DATA=mw` or `BLN_DATA=imw` before the command. Finished jobs are skipped, so a run stopped by the 12-hour limit can be started again. The configurations are `mnv3_ce`, `mnv3_wce`, `mnv3_fl`, `mnv3_hafl`, `cbam_fl`, `cbam_hafl`, `r50_ce` and `mnv3_hafl_steep`.

To rebuild the splits instead of using the manifests:

```
python build_folds.py dedup_embeddings.npz --threshold 0.985 --out manifest.csv
python build_folds.py mw_embeddings.npz --threshold 0.995 --stem --out manifest_mw.csv
python build_folds.py imw_embeddings.npz --threshold 0.95 --merge SoB=SDB --drop-conflicts 0.99 --out manifest_imw.csv
```

## Main results

Accuracy (%, mean over 5 folds) and critical-miss rate (%, argmax -> decision rule). A critical miss is a sharp or infectious item put into a less safe bin.

| Training | D1 acc. | D1 CMR | D2 acc. | D2 CMR | D3 acc. | D3 CMR |
|---|---|---|---|---|---|---|
| CE | 97.73 | 0.67 -> 0.58 | 98.06 | 0.55 -> 0.48 | 94.50 | 0.99 -> 0.74 |
| HAFL | 97.60 | 0.89 -> 0.61 | 97.83 | 0.61 -> 0.35 | 94.16 | 1.43 -> 0.65 |

The full tables, including weighted CE, focal loss, CBAM, ResNet-50 and the cross-dataset test, are in `experiments/analysis*/` and in the paper.

## Citation

```
@inproceedings{awais2026biolitenet,
  author    = {Muhammad Awais and Romina Nadeem and Jawad Ur Rehman and Sabahat Asad},
  title     = {{BioLiteNet}: Hazard-Aware Training and Decision Rules for Lightweight Biomedical Waste Classification},
  booktitle = {International Conference on Frontiers of Information Technology (FIT)},
  year      = {2026}
}
```

## License

MIT, see `LICENSE`. The datasets keep their own licenses.
