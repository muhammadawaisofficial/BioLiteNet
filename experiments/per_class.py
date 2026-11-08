"""
Per-class F1 of one configuration from the pooled out-of-fold predictions.

    python per_class.py results bmw [mnv3_hafl]     # -> analysis/per_class_f1_mnv3_hafl.csv
"""
import os, sys
import numpy as np
from sklearn.metrics import f1_score
from analyze import TIER, SUFFIX, HERE, load, pooled, write_csv

CONFIG = sys.argv[3] if len(sys.argv) > 3 else "mnv3_hafl"


def main(results_dir):
    runs = load(results_dir)
    folds = sorted({k[1] for k in runs if k[0] == CONFIG})
    classes = runs[(CONFIG, folds[0])]["classes"]
    _, y, p = pooled(runs, CONFIG, folds)
    f1 = f1_score(y, p.argmax(1), average=None, labels=range(len(classes)), zero_division=0)
    rows = [{"class": c, "tier": TIER[c], "n": int((y == i).sum()), "f1": f"{f1[i] * 100:.1f}"}
            for i, c in enumerate(classes)]
    rows.sort(key=lambda r: (-r["tier"], r["class"]))
    write_csv(os.path.join(HERE, "analysis" + SUFFIX, f"per_class_f1_{CONFIG}.csv"), rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main(os.path.join(HERE, sys.argv[1]))
