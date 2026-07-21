
import json, os

def cell(cell_type, source):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

cells = []

# ── Title ──────────────────────────────────────────────────────────────
cells.append(cell("markdown", [
    "# BioLiteNet — Biomedical Waste Classification\n",
    "**MobileNetV3-Large + CBAM + Hazard-Aware Focal Loss (HAFL)**\n\n",
    "This notebook contains the complete pipeline:\n",
    "1. Configuration\n",
    "2. Dataset extraction & splitting\n",
    "3. Model definition (CBAM + BioLiteNet)\n",
    "4. Loss functions (HAFL & Focal Loss)\n",
    "5. Trainer (train / evaluate / test)\n",
    "6. Run training (ablation variants)\n",
    "7. Results table generator\n",
]))

# ── Cell 0: Install deps ───────────────────────────────────────────────
cells.append(cell("markdown", ["## 0. Install Dependencies"]))
cells.append(cell("code", [
    "# Run this cell only if packages are not yet installed\n",
    "# !pip install torch torchvision scikit-learn pillow numpy\n",
]))

# ── Cell 1: CONFIG ─────────────────────────────────────────────────────
cells.append(cell("markdown", ["## 1. Configuration"]))

config_src = open(r"e:\- CS\6th Semester\ML Lab\Project\BioLiteNet\config.py", encoding="utf-8").read()
# Remove the `import os` line since it's cleaner to keep it at top of config block
cells.append(cell("code", [config_src]))

# ── Cell 2: DATASET ────────────────────────────────────────────────────
cells.append(cell("markdown", ["## 2. Dataset — Extraction & Splitting"]))

dataset_src = open(r"e:\- CS\6th Semester\ML Lab\Project\BioLiteNet\dataset.py", encoding="utf-8").read()
# Remove relative imports that won't work in notebook
dataset_src = dataset_src.replace(
    "from config import (DATASET_ZIP, DATASET_DIR, OUTPUT_DIR,\n"
    "                    USE_BACKGROUND_REMOVED, TRAIN_RATIO, VAL_RATIO,\n"
    "                    SEED, HAZARD_WEIGHTS)",
    "# config variables imported from Cell 1 (already defined above)"
)
cells.append(cell("code", [dataset_src]))

# ── Cell 3: MODEL ──────────────────────────────────────────────────────
cells.append(cell("markdown", ["## 3. Model — CBAM + BioLiteNet"]))

model_src = open(r"e:\- CS\6th Semester\ML Lab\Project\BioLiteNet\model.py", encoding="utf-8").read()
# Strip the __main__ block
if 'if __name__ == "__main__":' in model_src:
    model_src = model_src[:model_src.index('if __name__ == "__main__":')].rstrip() + "\n"
cells.append(cell("code", [model_src]))

# ── Cell 4: LOSS ───────────────────────────────────────────────────────
cells.append(cell("markdown", ["## 4. Loss Functions — HAFL & Focal Loss"]))

loss_src = open(r"e:\- CS\6th Semester\ML Lab\Project\BioLiteNet\loss.py", encoding="utf-8").read()
if 'if __name__ == "__main__":' in loss_src:
    loss_src = loss_src[:loss_src.index('if __name__ == "__main__":')].rstrip() + "\n"
cells.append(cell("code", [loss_src]))

# ── Cell 5: TRAINER ────────────────────────────────────────────────────
cells.append(cell("markdown", ["## 5. Trainer"]))

trainer_src = open(r"e:\- CS\6th Semester\ML Lab\Project\BioLiteNet\trainer.py", encoding="utf-8").read()
trainer_src = trainer_src.replace(
    "from config import (\n"
    "    OUTPUT_DIR, IMG_SIZE, BATCH_SIZE, NUM_EPOCHS, LR, LR_MIN,\n"
    "    WEIGHT_DECAY, DROPOUT_RATE, NUM_WORKERS, SEED, LOG_INTERVAL,\n"
    "    FOCAL_GAMMA, HAZARD_WEIGHTS, CBAM_STAGES, CBAM_REDUCTION,\n"
    "    USE_CBAM, USE_HAFL,\n"
    ")",
    "# config variables already defined in Cell 1"
).replace(
    "from model import BioLiteNet\n",
    "# BioLiteNet already defined in Cell 3\n"
).replace(
    "from loss  import build_hafl, FocalLoss\n",
    "# build_hafl, FocalLoss already defined in Cell 4\n"
)
cells.append(cell("code", [trainer_src]))

# ── Cell 6: MAIN / RUN ─────────────────────────────────────────────────
cells.append(cell("markdown", ["## 6. Run Training\n",
    "Edit `variant` to one of: `'A'` (baseline), `'B'` (CBAM only), `'C'` (HAFL only), `'D'` (BioLiteNet full).\n",
    "Set `run_all=True` to train all four variants for the ablation study."
]))

main_src = open(r"e:\- CS\6th Semester\ML Lab\Project\BioLiteNet\main.py", encoding="utf-8").read()
# Remove imports that are already defined, and __main__ guard; expose as plain runnable code
main_src = (main_src
    .replace("import os, sys, argparse\n", "import os\n")
    .replace("from dataset import extract_dataset, build_splits\n", "")
    .replace("from trainer import train\n", "")
    .replace("from config  import OUTPUT_DIR\n", "")
)
# Remove parse_args function and replace main() call with direct execution
run_cell = """ABLATION_VARIANTS = {
    "A": dict(name="A_baseline",   use_cbam=False, use_hafl=False),
    "B": dict(name="B_cbam_only",  use_cbam=True,  use_hafl=False),
    "C": dict(name="C_hafl_only",  use_cbam=False, use_hafl=True),
    "D": dict(name="D_BioLiteNet", use_cbam=True,  use_hafl=True),
}

# ── Choose what to run ──────────────────────────────
variant     = "D"    # single variant key: A / B / C / D
run_all     = False  # set True to train all 4 variants
skip_extract = False # set True if dataset already extracted

# ── Setup ──────────────────────────────────────────
if not skip_extract:
    extract_dataset()
split_root, classes = build_splits()
print(f"\\n[INFO] Dataset ready. Classes: {len(classes)}")

# ── Train ──────────────────────────────────────────
variants_to_run = list(ABLATION_VARIANTS.keys()) if run_all else [variant]

results = {}
for key in variants_to_run:
    cfg = ABLATION_VARIANTS[key]
    print(f"\\n{'#'*50}")
    print(f"  VARIANT {key}: {cfg['name']}")
    print(f"{'#'*50}")
    result = train(
        split_root   = split_root,
        variant_name = cfg["name"],
        use_cbam     = cfg["use_cbam"],
        use_hafl     = cfg["use_hafl"],
    )
    if result:
        results[cfg["name"]] = result

if results:
    print(f"\\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    for name, res in results.items():
        print(f"  {name:<25} Acc={res['accuracy']*100:.2f}%  "
              f"F1={res['macro_f1']:.4f}  SharpsF1={res['sharps_f1']:.4f}")

print(f"\\n[DONE] All outputs in: {OUTPUT_DIR}")
"""
cells.append(cell("code", [run_cell]))

# ── Cell 7: RESULTS TABLE ──────────────────────────────────────────────
cells.append(cell("markdown", ["## 7. Results Table (run after training all variants)"]))

rt_src = open(r"e:\- CS\6th Semester\ML Lab\Project\BioLiteNet\results_table.py", encoding="utf-8").read()
# Remove docstring at top and strip __main__ block; also fix imports
rt_src = rt_src.replace(
    '"""\nBioLiteNet — Results Table Generator\nReads per_class_f1.json from all 4 ablation variants and prints:\n  Table 1 : Overall metrics comparison (Accuracy, Macro-F1, Sharps-F1, Params, Speed)\n  Table 2 : Per-class F1 across all variants  ← paste directly into paper\nAlso saves both tables as CSV files.\n"""\n\n',
    ""
).replace(
    "from config import OUTPUT_DIR, CBAM_STAGES, CBAM_REDUCTION, DROPOUT_RATE\n",
    "# OUTPUT_DIR, CBAM_STAGES, CBAM_REDUCTION, DROPOUT_RATE already defined in Cell 1\n"
).replace(
    "from model import BioLiteNet\n",
    "# BioLiteNet already defined in Cell 3\n"
)
if 'if __name__ == "__main__":' in rt_src:
    rt_body = rt_src[:rt_src.index('if __name__ == "__main__":')].rstrip()
    rt_run  = (
        "\n# ── Run results table ──\n"
        "results = load_results()\n"
        "if not results:\n"
        "    print('[ERROR] No results found. Train at least one variant first.')\n"
        "else:\n"
        "    print_table1(results)\n"
        "    print_table2(results)\n"
    )
    rt_src = rt_body + rt_run
cells.append(cell("code", [rt_src]))

# ── Assemble notebook ──────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells,
}

out_path = r"e:\- CS\6th Semester\ML Lab\Project\BioLiteNet\BioLiteNet.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook created: {out_path}")
print(f"Total cells: {len(cells)}")
