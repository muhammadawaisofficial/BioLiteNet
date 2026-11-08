"""
Analyse the cross-validation runs produced by run_bln.py.

    python analyze.py results/            # writes analysis/*.csv and analysis/summary.md
    python analyze.py results_mw/ mw      # external dataset -> analysis_mw/

Everything is computed from the saved per-image test predictions, so every
number in the paper can be regenerated from results/ + manifest.csv.
"""
import csv, glob, json, os, sys
from itertools import combinations
import numpy as np
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import f1_score

if len(sys.argv) > 2:
    os.environ["BLN_DATA"] = sys.argv[2]
from run_bln import TIER, SHARPS, CORE, DATA, SUFFIX, MANIFEST

HERE = os.path.dirname(os.path.abspath(__file__))
OVER_COST = 0.2      # cost per tier of over-segregation (item put in a more hazardous stream)
EMB_FILE = "dedup_embeddings.npz" if DATA == "bmw" else f"{DATA}_embeddings.npz"
NEAR_SIM = {"bmw": 0.95, "mw": 0.99, "imw": 0.90}[DATA]   # below each grouping threshold (0.985, 0.995, 0.95); mw: shared white background inflates similarity
N_BOOT = 2000


def load(results_dir):
    runs = {}
    for m in glob.glob(os.path.join(results_dir, "**", "metrics.json"), recursive=True):
        d = os.path.dirname(m)
        meta = json.load(open(m))
        p = np.load(os.path.join(d, "test_preds.npz"))
        run = {"meta": meta, "idx": p["idx"], "y": p["y"], "prob": p["prob"].astype(np.float32),
               "classes": list(p["classes"])}
        rp = os.path.join(d, "raw_preds.npz")
        if os.path.exists(rp):
            r = np.load(rp)
            run["raw"] = (r["idx"], r["y"], r["prob"].astype(np.float32))
        runs[(meta["config"], meta["fold"])] = run
    return runs


def cost_matrix(classes):
    t = np.array([TIER[c] for c in classes], float)
    gap = t[:, None] - t[None, :]                       # true tier - predicted tier
    return np.where(gap > 0, gap, -OVER_COST * gap)     # under-segregation costs the full tier gap


def hazard_metrics(y, yhat, classes, C):
    t = np.array([TIER[c] for c in classes])
    crit = t[y] >= 4                                     # sharps + pathological
    return {"hazard_cost": float(C[y, yhat].mean()),
            "critical_miss_rate": float((t[yhat][crit] < t[y][crit]).mean()),
            "under_seg_rate": float((t[yhat] < t[y]).mean())}


def scores(y, yhat, classes):
    f1 = f1_score(y, yhat, average=None, labels=range(len(classes)), zero_division=0)
    sh = [classes.index(c) for c in SHARPS]
    return {"acc": float((y == yhat).mean()), "macro_f1": float(f1.mean()), "sharps_f1": float(f1[sh].mean())}


def pooled(runs, cfg, folds, field="prob"):
    rs = [runs[(cfg, f)] for f in folds]
    idx = np.concatenate([r["idx"] for r in rs]); y = np.concatenate([r["y"] for r in rs])
    prob = np.concatenate([r[field] for r in rs])
    o = np.argsort(idx)
    return idx[o], y[o], prob[o]


def mcnemar(correct_a, correct_b):
    b = int((correct_a & ~correct_b).sum()); c = int((~correct_a & correct_b).sum())
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)


def main(results_dir):
    runs = load(results_dir)
    out_dir = os.path.join(HERE, "analysis" + SUFFIX); os.makedirs(out_dir, exist_ok=True)
    cfgs = [c for c in CORE + sorted({k[0] for k in runs} - set(CORE)) if any(k[0] == c for k in runs)]
    folds = sorted({k[1] for k in runs})
    complete = [c for c in cfgs if all((c, f) in runs for f in folds)]
    classes = runs[next(iter(runs))]["classes"]
    C = cost_matrix(classes)
    lines = [f"# Results ({len(runs)} runs, folds {folds})\n"]
    missing = [(c, f) for c in cfgs for f in folds if (c, f) not in runs]
    if missing:
        lines.append(f"**Missing runs:** {missing}\n")

    # 1) Per-fold metrics -> mean ± std, plus pooled hazard metrics
    rows = []
    for c in cfgs:
        per = []
        for f in folds:
            if (c, f) not in runs: continue
            r = runs[(c, f)]; yhat = r["prob"].argmax(1)
            per.append({**scores(r["y"], yhat, classes), **hazard_metrics(r["y"], yhat, classes, C)})
        row = {"config": c, "n_folds": len(per), "params": runs[(c, folds[0])]["meta"]["params"]
               if (c, folds[0]) in runs else ""}
        for k in per[0]:
            v = np.array([p[k] for p in per]) * (100 if k in ("acc", "macro_f1", "sharps_f1", "critical_miss_rate",
                                                              "under_seg_rate") else 1)
            dp = 4 if k == "hazard_cost" else 2
            row[k] = f"{v.mean():.{dp}f} ± {v.std(ddof=1) if len(v) > 1 else 0:.{dp}f}"
        rows.append(row)
    write_csv(os.path.join(out_dir, "main_table.csv"), rows)
    lines += ["## Main table (mean ± std over folds; %, except hazard_cost)\n", md_table(rows)]

    # 2) Pairwise significance on pooled out-of-fold predictions
    sig = []
    for a, b in combinations(complete, 2):
        _, ya, pa = pooled(runs, a, folds); _, yb, pb = pooled(runs, b, folds)
        ca, cb = pa.argmax(1) == ya, pb.argmax(1) == yb
        n01, n10, p = mcnemar(ca, cb)
        fa = [scores(runs[(a, f)]["y"], runs[(a, f)]["prob"].argmax(1), classes)["macro_f1"] for f in folds]
        fb = [scores(runs[(b, f)]["y"], runs[(b, f)]["prob"].argmax(1), classes)["macro_f1"] for f in folds]
        wp = wilcoxon(fa, fb).pvalue if len(folds) >= 5 and np.any(np.array(fa) != np.array(fb)) else float("nan")
        rng = np.random.default_rng(0); d = []
        for _ in range(N_BOOT):
            s = rng.integers(0, len(ya), len(ya)); d.append(ca[s].mean() - cb[s].mean())
        lo, hi = np.percentile(d, [2.5, 97.5]) * 100
        sig.append({"A": a, "B": b, "acc_diff_pp": f"{(ca.mean()-cb.mean())*100:+.2f}",
                    "95%CI_pp": f"[{lo:+.2f}, {hi:+.2f}]", "A_only_right": n01, "B_only_right": n10,
                    "mcnemar_p": f"{p:.4g}", "wilcoxon_macroF1_p": f"{wp:.4g}"})
    write_csv(os.path.join(out_dir, "significance.csv"), sig)
    lines += ["\n## Pairwise tests (pooled predictions; McNemar exact, Wilcoxon over folds, bootstrap CI)\n",
              md_table(sig)]

    # 3) Decision-time cost-sensitive rule: argmin_k sum_j p_j * (1[j!=k] + C[j,k])
    D = (1 - np.eye(len(classes))) + C
    dec = []
    for c in complete:
        _, y, p = pooled(runs, c, folds)
        for rule, yhat in (("argmax", p.argmax(1)), ("min-expected-cost", (p @ D).argmin(1))):
            dec.append({"config": c, "rule": rule, **{k: f"{v*100:.2f}" if k != "hazard_cost" else f"{v:.4f}"
                        for k, v in {**scores(y, yhat, classes), **hazard_metrics(y, yhat, classes, C)}.items()}})
    write_csv(os.path.join(out_dir, "decision_rule.csv"), dec)
    lines += ["\n## Training-time (HAFL) vs decision-time cost sensitivity (pooled)\n", md_table(dec)]

    # 4) Raw-background domain shift
    raw = []
    for c in complete:
        if all("raw" in runs[(c, f)] for f in folds):
            y = np.concatenate([runs[(c, f)]["raw"][1] for f in folds])
            p = np.concatenate([runs[(c, f)]["raw"][2] for f in folds])
            raw.append({"config": c, **{k: f"{v*100:.2f}" if k != "hazard_cost" else f"{v:.4f}" for k, v in
                        {**scores(y, p.argmax(1), classes), **hazard_metrics(y, p.argmax(1), classes, C)}.items()}})
    if raw:
        write_csv(os.path.join(out_dir, "raw_domain.csv"), raw)
        lines += ["\n## Raw (original background) images of the test folds (pooled)\n", md_table(raw)]

    # 5) Residual near-duplicate check
    emb_path = os.path.join(HERE, EMB_FILE)
    if os.path.exists(emb_path) and complete:
        lines += ["\n## Residual leakage: accuracy by nearest-training-image similarity\n", residual(runs, complete, folds)]

    # 6) Confusion matrix of the proposed model (pooled)
    for c in ("cbam_hafl",) if "cbam_hafl" in complete else ():
        _, y, p = pooled(runs, c, folds)
        cm = np.zeros((len(classes), len(classes)), int); np.add.at(cm, (y, p.argmax(1)), 1)
        np.save(os.path.join(out_dir, f"confusion_{c}.npy"), cm)
        off = sorted(((cm[i, j], classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes))
                      if i != j and cm[i, j]), reverse=True)[:10]
        lines += [f"\n## Top confusions ({c}, pooled)\n"] + [f"- {n}× {a} → {b}" for n, a, b in off]

    open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def residual(runs, cfgs, folds):
    import csv as _csv
    man = list(_csv.DictReader(open(os.path.join(HERE, MANIFEST), encoding="utf-8")))
    e = np.load(os.path.join(HERE, EMB_FILE))
    # imw files carry their folder (unique on their own); its labels were remapped (SoB=SDB) after embedding
    key = (lambda c, f: f) if DATA == "imw" else (lambda c, f: (c, f))
    pos = {key(m[1], m[2]): i for i, m in enumerate(e["meta"])}
    emb = e["emb"][[pos[key(r["class"], r["file"])] for r in man]]
    fold = np.array([int(r["fold"]) for r in man]); cls = np.array([r["class"] for r in man])
    near = np.zeros(len(man), bool)
    for f in folds:
        te = np.where(fold == f)[0]
        for c in np.unique(cls[te]):
            a = te[cls[te] == c]; b = np.where((fold != f) & (cls == c))[0]
            near[a] = (emb[a] @ emb[b].T).max(1) >= NEAR_SIM
    rows = []
    for c in cfgs:
        idx, y, p = pooled(runs, c, folds); ok = p.argmax(1) == y
        rows.append({"config": c, f"acc_with_train_neighbour>={NEAR_SIM}": f"{ok[near[idx]].mean()*100:.2f} (n={near[idx].sum()})",
                     "acc_without": f"{ok[~near[idx]].mean()*100:.2f} (n={(~near[idx]).sum()})"})
    return md_table(rows)


def write_csv(path, rows):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def md_table(rows):
    if not rows: return "(none)\n"
    k = list(rows[0])
    return "\n".join(["| " + " | ".join(k) + " |", "|" + "---|" * len(k)] +
                     ["| " + " | ".join(str(r[x]) for x in k) + " |" for r in rows]) + "\n"


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results"))
