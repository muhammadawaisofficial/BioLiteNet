"""
Cross-dataset results at the WHO-bin level, from the Kaggle output of xdata.py.

    python xdata/analyze_xdata.py        # needs xdata/xdata_probs.npz and xdata/xdata_checks.json
    -> analysis/xdata.csv (mean and s.d. over the five fold models) + printed summary

For a model trained on dataset A and tested on dataset B, a prediction is a class of A; its WHO tier is
compared with the tier of the true class of B. Critical items: tier >= 4.
"""
import csv, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
NAMES = {"bmw": "D1", "mw": "D2", "imw": "D3"}
RESULTS = {"bmw": "results", "mw": "results_mw", "imw": "results_imw"}
OVER, REFER = 0.2, 10


def cost(t):
    gap = t[:, None] - t[None, :]
    return np.where(gap > 0, gap, OVER * (-gap)).astype(float)


def evaluate(p, ts, tt):
    """p: probabilities over source classes, ts: tiers of source classes, tt: true tiers of target items"""
    D = (1 - np.eye(len(ts))) + cost(ts)
    crit = tt >= 4
    out = {}
    for rule, yh in (("argmax", p.argmax(1)), ("rule", (p @ D).argmin(1))):
        miss = crit & (ts[yh] < tt)
        out[f"bin_acc_{rule}"] = (ts[yh] == tt).mean() * 100
        out[f"cmr_{rule}"] = miss.sum() / crit.sum() * 100
        out[f"over_{rule}"] = (ts[yh] > tt).mean() * 100
        out[f"_miss_{rule}"] = miss
        if rule == "rule":                          # refer the items with the highest expected cost
            risk = (p @ D)[np.arange(len(p)), yh]
            keep = risk <= np.percentile(risk, 100 - REFER)
            out["cmr_rule_referral"] = (miss & keep).sum() / crit.sum() * 100
    out["new_misses"] = int((out["_miss_rule"] & ~out["_miss_argmax"]).sum())
    out["n_critical"] = int(crit.sum())
    return out


def main():
    z = np.load(os.path.join(HERE, "xdata_probs.npz"), allow_pickle=False)
    tiers = json.load(open(os.path.join(HERE, "tiers.json")))

    # check: own-fold accuracy on Kaggle against the accuracy saved at training time
    saved = {}
    for d, folder in RESULTS.items():
        for m in glob.glob(os.path.join(REV, folder, "**", "metrics.json"), recursive=True):
            j = json.load(open(m))
            saved[(d, j["config"], j["fold"])] = round(j["test_acc"] * 100, 2)
    worst = 0
    for c in json.load(open(os.path.join(HERE, "xdata_checks.json"))):
        s = saved.get((c["source"], c["config"], c["fold"]))
        if s is not None:
            worst = max(worst, abs(s - c["own_fold_acc"]))
        print(f"check {c['source']:4s} {c['config']:10s} f{c['fold']}: Kaggle {c['own_fold_acc']:.2f}  saved {s}")
    print(f"largest difference to the saved accuracy: {worst:.2f} points\n")

    rows = []
    for key in sorted({k.rsplit("__", 2)[0] for k in z.files if k.count("__") == 3}):
        src, tgt = key.split("__")
        ts = np.array([tiers[src][c] for c in z[f"classes__{src}"]])
        tt = np.array([tiers[tgt][c] for c in z[f"labels__{tgt}"]])
        for cfg in ("mnv3_ce", "mnv3_hafl"):
            per = [evaluate(z[f"{src}__{tgt}__{cfg}__f{f}"].astype(np.float64), ts, tt) for f in range(5)]
            row = {"direction": f"{NAMES[src]} -> {NAMES[tgt]}", "config": cfg, "n_critical": per[0]["n_critical"]}
            for k in ("bin_acc_argmax", "bin_acc_rule", "cmr_argmax", "cmr_rule", "cmr_rule_referral",
                      "over_argmax", "over_rule"):
                v = np.array([r[k] for r in per])
                row[k] = f"{v.mean():.1f} +- {v.std(ddof=1):.1f}"
            row["new_misses_total"] = sum(r["new_misses"] for r in per)
            rows.append(row)
            print(row)
    out = os.path.join(REV, "analysis", "xdata.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
