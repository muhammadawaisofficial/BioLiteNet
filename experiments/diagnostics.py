"""
Data for the diagnostic figures, from the pooled out-of-fold predictions and training logs.

    python diagnostics.py results bmw         # D1 and D1 with original backgrounds
    python diagnostics.py results_mw mw
    python diagnostics.py results_imw imw     # -> analysis<suffix>/diag_<config>.json

ROC / PR: detecting critical items (tier >= 4); the score is the probability mass on critical classes.
Confusion matrix (counts) and training curves (mean over folds) are saved for D1 only.
"""
import csv, glob, json, os, sys
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from analyze import TIER, DATA, SUFFIX, HERE, load, pooled

CONFIG = "mnv3_hafl"


def thin(a, b, n=300):
    keep = np.unique(np.linspace(0, len(a) - 1, min(n, len(a))).astype(int))
    return [round(float(v), 4) for v in a[keep]], [round(float(v), 4) for v in b[keep]]


def main(results_dir):
    runs = load(results_dir)
    folds = sorted({k[1] for k in runs if k[0] == CONFIG})
    classes = runs[(CONFIG, folds[0])]["classes"]
    crit = np.array([TIER[c] >= 4 for c in classes])
    out = {"classes": classes, "tiers": [TIER[c] for c in classes], "curves": {}}
    doms = [("test", None)] + ([("raw", "raw")] if all("raw" in runs[(CONFIG, f)] for f in folds) else [])
    for dom, key in doms:
        if key is None:
            _, y, p = pooled(runs, CONFIG, folds)
        else:
            y = np.concatenate([runs[(CONFIG, f)]["raw"][1] for f in folds])
            p = np.concatenate([runs[(CONFIG, f)]["raw"][2] for f in folds])
        t, s = crit[y], p[:, crit].sum(1).astype(float)
        fpr, tpr, _ = roc_curve(t, s); prec, rec, _ = precision_recall_curve(t, s)
        out["curves"][dom] = {"fpr_tpr": thin(fpr, tpr), "rec_prec": thin(rec[::-1], prec[::-1]),
                              "auc": round(float(roc_auc_score(t, s)), 4), "ap": round(float(average_precision_score(t, s)), 4),
                              "n_critical": int(t.sum()), "n": int(len(t))}
        if dom == "test" and DATA == "bmw":
            cm = np.zeros((len(classes), len(classes)), int)
            np.add.at(cm, (y, p.argmax(1)), 1)
            out["confusion"] = cm.tolist()
    if DATA == "bmw":
        logs = [list(csv.DictReader(open(f))) for f in
                sorted(glob.glob(os.path.join(results_dir, "**", f"{CONFIG}_f[0-9]", "log.csv"), recursive=True))]
        out["train"] = {k: np.mean([[float(r[k]) for r in log] for log in logs], 0).round(4).tolist()
                        for k in ("train_loss", "val_acc", "val_macro_f1")}
        out["train"]["n_folds"] = len(logs)
    path = os.path.join(HERE, "analysis" + SUFFIX, f"diag_{CONFIG}.json")
    json.dump(out, open(path, "w"))
    print(path, {d: (c["auc"], c["ap"]) for d, c in out["curves"].items()})


if __name__ == "__main__":
    main(os.path.join(HERE, sys.argv[1]))
