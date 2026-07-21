"""
BioLiteNet — Results Table Generator
Reads per_class_f1.json from all 4 ablation variants and prints:
  Table 1 : Overall metrics comparison (Accuracy, Macro-F1, Sharps-F1, Params, Speed)
  Table 2 : Per-class F1 across all variants  ← paste directly into paper
Also saves both tables as CSV files.
"""

import os, json
import numpy as np
import torch, time
from torchvision import transforms
from PIL import Image
from config import OUTPUT_DIR, CBAM_STAGES, CBAM_REDUCTION, DROPOUT_RATE

VARIANTS = ["A_baseline", "B_cbam_only", "C_hafl_only", "D_BioLiteNet"]

def load_results():
    results = {}
    for v in VARIANTS:
        path = os.path.join(OUTPUT_DIR, v, "per_class_f1.json")
        if os.path.exists(path):
            with open(path) as f:
                results[v] = json.load(f)
        else:
            print(f"[WARN] Missing results for variant {v}: {path}")
    return results

def measure_inference_speed(variant_dir: str, num_classes: int):
    """Measure avg inference time on CPU (paper's efficiency claim)."""
    try:
        from model import BioLiteNet
        use_cbam = "cbam" in variant_dir.lower() or "biolite" in variant_dir.lower()
        model = BioLiteNet(num_classes=num_classes, use_cbam=use_cbam,
                           cbam_stages=CBAM_STAGES, pretrained=False)
        ckpt_path = os.path.join(variant_dir, "checkpoints", "best_model.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model_state"])
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                model(dummy)
        # Time 50 passes
        times = []
        with torch.no_grad():
            for _ in range(50):
                t0 = time.perf_counter()
                model(dummy)
                times.append(time.perf_counter() - t0)
        return round(np.mean(times) * 1000, 2)   # ms per image
    except Exception as e:
        return f"N/A ({e})"

def print_table1(results: dict):
    """Table 1: Main comparison table for the paper."""
    print("\n" + "=" * 100)
    print("TABLE 1 — BioLiteNet Ablation Study & Performance Comparison")
    print("=" * 100)
    header = f"{'Variant':<20} {'CBAM':^6} {'HAFL':^6} {'Accuracy':^10} {'Macro-F1':^10} {'Sharps-F1':^11} {'Inf. (ms)':^11}"
    print(header)
    print("-" * 100)

    rows = []
    for v in VARIANTS:
        if v not in results:
            continue
        r        = results[v]
        use_cbam = "cbam" in v.lower() or "biolite" in v.lower()
        use_hafl = "hafl" in v.lower() or "biolite" in v.lower()
        spd      = measure_inference_speed(os.path.join(OUTPUT_DIR, v), len(r["per_class"]))
        acc_str  = f"{r['accuracy']*100:.2f}%"
        f1_str   = f"{r['macro_f1']:.4f}"
        sf1_str  = f"{r['sharps_f1']:.4f}"
        spd_str  = f"{spd} ms" if isinstance(spd, float) else spd
        label    = v.replace("_", " ")
        cbam_s   = "✓" if use_cbam else "✗"
        hafl_s   = "✓" if use_hafl else "✗"
        row_str  = f"{label:<20} {cbam_s:^6} {hafl_s:^6} {acc_str:^10} {f1_str:^10} {sf1_str:^11} {spd_str:^11}"
        print(row_str)
        rows.append([label, cbam_s, hafl_s, acc_str, f1_str, sf1_str, spd_str])

    print("=" * 100)

    # Save as CSV
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "table1_comparison.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Variant", "CBAM", "HAFL", "Accuracy", "Macro-F1", "Sharps-F1", "Inf_ms"])
        w.writerows(rows)
    print(f"\n[CSV] Table 1 saved → {csv_path}")

def print_table2(results: dict):
    """Table 2: Per-class F1 across all variants — paste into paper appendix."""
    if not results:
        return
    # Use class list from first available result
    first = next(iter(results.values()))
    class_names = sorted(first["per_class"].keys())

    print("\n" + "=" * 110)
    print("TABLE 2 — Per-Class F1 Score Across Ablation Variants")
    print("=" * 110)
    col_w = 12
    header = f"{'Class':<38}" + "".join(f"{v:<{col_w}}" for v in VARIANTS if v in results)
    print(header)
    print("-" * 110)

    rows = []
    for cls in class_names:
        row = [cls]
        line = f"{cls:<38}"
        for v in VARIANTS:
            if v not in results:
                continue
            f1 = results[v]["per_class"].get(cls, 0.0)
            line += f"{f1:<{col_w}.4f}"
            row.append(f"{f1:.4f}")
        print(line)
        rows.append(row)

    print("=" * 110)

    import csv
    csv_path = os.path.join(OUTPUT_DIR, "table2_perclass_f1.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Class"] + [v for v in VARIANTS if v in results])
        w.writerows(rows)
    print(f"\n[CSV] Table 2 saved → {csv_path}")

if __name__ == "__main__":
    results = load_results()
    if not results:
        print("[ERROR] No results found. Train at least one variant first.")
    else:
        print_table1(results)
        print_table2(results)
