import os, sys, argparse
from dataset import extract_dataset, build_splits
from trainer import train
from config  import OUTPUT_DIR

ABLATION_VARIANTS = {
    "A": dict(name="A_baseline",   use_cbam=False, use_hafl=False),
    "B": dict(name="B_cbam_only",  use_cbam=True,  use_hafl=False),
    "C": dict(name="C_hafl_only",  use_cbam=False, use_hafl=True),
    "D": dict(name="D_BioLiteNet", use_cbam=True,  use_hafl=True),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant",      nargs="+", choices=["A", "B", "C", "D"], default=["D"])
    p.add_argument("--all-variants", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.skip_extract:
        extract_dataset()
    split_root, classes = build_splits()
    print(f"\n[INFO] Dataset ready. Classes: {len(classes)}")

    variants_to_run = list(ABLATION_VARIANTS.keys()) if args.all_variants else args.variant

    results = {}
    for key in variants_to_run:
        cfg = ABLATION_VARIANTS[key]
        print(f"\n{'#'*50}")
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
        print(f"\n{'='*55}")
        print("  SUMMARY")
        print(f"{'='*55}")
        for name, res in results.items():
            print(f"  {name:<25} Acc={res['accuracy']*100:.2f}%  "
                  f"F1={res['macro_f1']:.4f}  SharpsF1={res['sharps_f1']:.4f}")

    print(f"\n[DONE] All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
