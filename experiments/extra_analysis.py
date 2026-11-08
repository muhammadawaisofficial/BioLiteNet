"""
Camera-ready follow-up analyses, all computed from the saved test predictions (no retraining).

    python extra_analysis.py results bmw
    python extra_analysis.py results_mw mw
    python extra_analysis.py results_imw imw          # each writes analysis<suffix>/extra.md + csv files

1. Effect of the min-expected-cost rule: absolute counts, exact McNemar test on the
   critical items it fixes vs breaks, duplicate-group bootstrap CI, and its price
   (over-segregation).
2. Operating curves: rule with cost matrix scaled by lambda (lambda = 0 is argmax).
   Does hazard-aware training give a better accuracy / critical-miss trade-off than
   applying the same rule to a plain cross-entropy model?
3. Calibration (ECE, NLL, mean confidence), which the rule relies on.
4. Sensitivity to the cost matrix the rule uses (over-segregation price, tier-gap shape).
5. Referral to a human for the least certain items (coverage 100/95/90 %).
6. Temperature scaling: does a recalibrated CE model match hazard-aware training under the rule?
   T is fitted on the in-domain test predictions of the other four folds (cross-fitted).
"""
import csv, os, sys
import numpy as np
import analyze as A                                   # sets BLN_DATA from argv[2] before importing run_bln
from analyze import TIER, DATA, SUFFIX, MANIFEST, HERE, load, pooled, scores, md_table, write_csv

LAMBDAS = [0, 0.25, 0.5, 1, 2, 4, 8, 16]
N_BOOT = 2000


def tiers(classes):
    return np.array([TIER[c] for c in classes])


def cost(classes, over=A.OVER_COST, power=1):
    t = tiers(classes).astype(float)
    gap = t[:, None] - t[None, :]
    return np.where(gap > 0, gap ** power, over * (-gap) ** power)


def decide(p, classes, lam=1.0, **kw):
    return p.argmax(1) if lam == 0 else (p @ ((1 - np.eye(len(classes))) + lam * cost(classes, **kw))).argmin(1)


def misses(y, yhat, t):
    """critical-item mask, critical-miss indicator, over-segregation indicator"""
    crit = t[y] >= 4
    return crit, crit & (t[yhat] < t[y]), t[yhat] > t[y]


def group_boot_ci(groups, crit, m0, m1):
    """95% CI of the relative reduction in critical-miss rate, resampling duplicate groups"""
    u, g = np.unique(groups, return_inverse=True)
    c, a, b = (np.bincount(g, weights=v.astype(float), minlength=len(u)) for v in (crit, m0, m1))
    rng = np.random.default_rng(0); red = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        r0, r1 = (w * a).sum() / (w * c).sum(), (w * b).sum() / (w * c).sum()
        red.append(1 - r1 / r0 if r0 else np.nan)
    return np.nanpercentile(red, [2.5, 97.5]) * 100


def temper(p, T):
    z = np.log(np.clip(p, 1e-8, 1)) / T
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def fit_T(p, y):
    grid = np.arange(0.5, 5.01, 0.05)
    nll = [-np.log(np.clip(temper(p, T)[np.arange(len(y)), y], 1e-12, 1)).mean() for T in grid]
    return grid[int(np.argmin(nll))]


def ece(p, y, bins=15):
    conf, ok = p.max(1), p.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(ok[s].mean() - conf[s].mean())
    return e


def main(results_dir):
    runs = load(results_dir)
    out_dir = os.path.join(HERE, "analysis" + SUFFIX)
    folds = sorted({k[1] for k in runs})
    cfgs = [c for c in dict.fromkeys(k[0] for k in sorted(runs)) if all((c, f) in runs for f in folds)]
    order = ["mnv3_ce", "mnv3_wce", "mnv3_fl", "mnv3_hafl", "mnv3_hafl_steep", "cbam_fl", "cbam_hafl", "r50_ce"]
    cfgs = [c for c in order if c in cfgs]
    classes = runs[next(iter(runs))]["classes"]; t = tiers(classes)
    man = list(csv.DictReader(open(os.path.join(HERE, MANIFEST), encoding="utf-8")))
    group_of = np.array([int(r["group"]) for r in man])
    domains = [("test", "prob")]
    if all("raw" in runs[(c, f)] for c in cfgs for f in folds):
        for c in cfgs:
            for f in folds:
                runs[(c, f)]["rawprob"] = runs[(c, f)]["raw"][2]
                runs[(c, f)]["rawidx"], runs[(c, f)]["rawy"] = runs[(c, f)]["raw"][0], runs[(c, f)]["raw"][1]
        domains.append(("raw", "rawprob"))
    lines = [f"# Extra analyses — {DATA} ({len(folds)} folds, pooled out-of-fold predictions)\n"]

    for dom, field in domains:
        data = {}
        for c in cfgs:
            if field == "prob":
                idx, y, p = pooled(runs, c, folds)
            else:
                idx = np.concatenate([runs[(c, f)]["rawidx"] for f in folds])
                y = np.concatenate([runs[(c, f)]["rawy"] for f in folds])
                p = np.concatenate([runs[(c, f)]["rawprob"] for f in folds])
            data[c] = (idx, y, p.astype(np.float64))
        tag = "" if dom == "test" else " — RAW-background images"

        # 1) rule effect with counts, significance, price
        rows = []
        for c in cfgs:
            idx, y, p = data[c]
            y0, y1 = decide(p, classes, 0), decide(p, classes, 1)
            crit, m0, o0 = misses(y, y0, t); _, m1, o1 = misses(y, y1, t)
            fixed, broke = int((m0 & ~m1).sum()), int((~m0 & m1).sum())
            pv = A.binomtest(fixed, fixed + broke, 0.5).pvalue if fixed + broke else 1.0
            lo, hi = group_boot_ci(group_of[idx], crit, m0, m1)
            rows.append({"config": c, "n_critical_items": int(crit.sum()),
                         "crit_misses_argmax": int(m0.sum()), "crit_misses_rule": int(m1.sum()),
                         "to_general_argmax": int((crit & (t[y0] == 1)).sum()),
                         "to_general_rule": int((crit & (t[y1] == 1)).sum()),
                         "fixed": fixed, "newly_missed": broke, "mcnemar_p": f"{pv:.3g}",
                         "rel_reduction_%": f"{(1 - m1.sum() / m0.sum()) * 100:.0f}" if m0.sum() else "-",
                         "95%CI_group_boot": f"[{lo:.0f}, {hi:.0f}]",
                         "acc_change_pp": f"{((y1 == y).mean() - (y0 == y).mean()) * 100:+.2f}",
                         "over_seg_%_argmax": f"{o0.mean() * 100:.2f}", "over_seg_%_rule": f"{o1.mean() * 100:.2f}"})
        write_csv(os.path.join(out_dir, f"extra_rule_{dom}.csv"), rows)
        lines += [f"\n## 1. Min-expected-cost rule: critical misses (tier>=4 sent to a lower tier){tag}\n", md_table(rows)]

        # pairwise after the rule: critical-miss indicators on the same critical items (paired)
        pair = []
        for a, b in [("mnv3_hafl", "mnv3_ce"), ("mnv3_hafl", "mnv3_wce"), ("mnv3_hafl", "mnv3_fl"),
                     ("mnv3_wce", "mnv3_ce"), ("mnv3_fl", "mnv3_ce"), ("mnv3_wce", "mnv3_fl"),
                     ("mnv3_hafl_steep", "mnv3_hafl"), ("mnv3_hafl_steep", "mnv3_fl"), ("mnv3_hafl_steep", "mnv3_ce")]:
            if a not in data or b not in data: continue
            (ia, ya, pa), (ib, yb, pb) = data[a], data[b]
            oa, ob = np.argsort(ia, kind="stable"), np.argsort(ib, kind="stable")
            assert (ia[oa] == ib[ob]).all()
            for lam in (0, 1):
                ha, hb = decide(pa[oa], classes, lam), decide(pb[ob], classes, lam)
                ca, ma, _ = misses(ya[oa], ha, t)
                _, mb, _ = misses(yb[ob], hb, t)
                n_a, n_b = int((ma & ~mb).sum()), int((~ma & mb).sum())
                pv = A.binomtest(n_a, n_a + n_b, 0.5).pvalue if n_a + n_b else 1.0
                okA, okB = ha == ya[oa], hb == yb[ob]
                pair.append({"A": a, "B": b, "rule": "argmax" if lam == 0 else "cost-rule",
                             "A_misses": int(ma.sum()), "B_misses": int(mb.sum()),
                             "only_A_missed": n_a, "only_B_missed": n_b, "mcnemar_p": f"{pv:.3g}",
                             "A_acc": f"{okA.mean() * 100:.2f}", "B_acc": f"{okB.mean() * 100:.2f}",
                             "acc_mcnemar_p": f"{A.mcnemar(okA, okB)[2]:.3g}"})
        write_csv(os.path.join(out_dir, f"extra_pairs_{dom}.csv"), pair)
        lines += [f"\n### Paired comparison of critical misses between training losses{tag}\n", md_table(pair)]

        # 2) operating curves over lambda
        curve = []
        for c in cfgs:
            idx, y, p = data[c]
            for lam in LAMBDAS:
                yh = decide(p, classes, lam); crit, m, o = misses(y, yh, t)
                curve.append({"config": c, "lambda": lam, "acc": round((yh == y).mean() * 100, 3),
                              "crit_miss_rate": round(m.sum() / crit.sum() * 100, 3),
                              "over_seg": round(o.mean() * 100, 3)})
        write_csv(os.path.join(out_dir, f"extra_curve_{dom}.csv"), curve)
        piv = [{"config": c, **{f"λ={l}": next(f"{r['acc']:.2f} / {r['crit_miss_rate']:.2f}" for r in curve
                                               if r["config"] == c and r["lambda"] == l) for l in LAMBDAS}}
               for c in cfgs]
        lines += [f"\n## 2. Operating curve: accuracy / critical-miss rate (%) as the cost scale λ grows{tag}\n",
                  md_table(piv)]

        # 3) calibration
        cal = []
        for c in cfgs:
            idx, y, p = data[c]
            cal.append({"config": c, "ECE_%": f"{ece(p, y) * 100:.2f}",
                        "NLL": f"{-np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1)).mean():.3f}",
                        "mean_conf_%": f"{p.max(1).mean() * 100:.2f}", "acc_%": f"{(p.argmax(1) == y).mean() * 100:.2f}",
                        "rule_changes_%": f"{(decide(p, classes, 1) != p.argmax(1)).mean() * 100:.2f}"})
        write_csv(os.path.join(out_dir, f"extra_calibration_{dom}.csv"), cal)
        lines += [f"\n## 3. Calibration (15-bin ECE) and how often the rule changes the decision{tag}\n", md_table(cal)]

        # 4) sensitivity to the cost matrix used by the rule
        sens = []
        for c in [c for c in ("mnv3_ce", "mnv3_hafl") if c in data]:
            idx, y, p = data[c]
            crit, m0, o0 = misses(y, p.argmax(1), t)
            for over in (0.1, 0.2, 0.5, 1.0):
                for power in (1, 2):
                    yh = decide(p, classes, 1, over=over, power=power); _, m, o = misses(y, yh, t)
                    sens.append({"config": c, "over_cost": over, "gap": "linear" if power == 1 else "squared",
                                 "acc": f"{(yh == y).mean() * 100:.2f}",
                                 "crit_misses": f"{int(m.sum())} (argmax {int(m0.sum())})",
                                 "over_seg_%": f"{o.mean() * 100:.2f}"})
        write_csv(os.path.join(out_dir, f"extra_costsens_{dom}.csv"), sens)
        lines += [f"\n## 4. Sensitivity to the rule's cost matrix (over-segregation price, tier-gap shape){tag}\n",
                  md_table(sens)]

        # 5) referral of uncertain items to a human (assumed correct)
        ref = []
        for c in cfgs:
            idx, y, p = data[c]
            for rule, lam in (("argmax", 0), ("cost-rule", 1)):
                yh = decide(p, classes, lam)
                risk = (p @ ((1 - np.eye(len(classes))) + cost(classes)))[np.arange(len(y)), yh]
                crit, m, _ = misses(y, yh, t)
                row = {"config": c, "rule": rule}
                for name, score in (("lowconf", -p.max(1)), ("highrisk", risk)):
                    for cov in (95, 90):
                        keep = score <= np.percentile(score, cov)
                        row[f"{name}@{cov}%"] = int((m & keep).sum())
                row["all@100%"] = int(m.sum())
                ref.append(row)
        write_csv(os.path.join(out_dir, f"extra_referral_{dom}.csv"), ref)
        lines += [f"\n## 5. Critical misses left when the 5% / 10% least certain items are referred to a human{tag}\n",
                  "(lowconf = lowest max-probability; highrisk = highest expected cost of the chosen bin)\n",
                  md_table(ref)]

        # 6) temperature scaling, cross-fitted on the other folds' in-domain predictions
        ts, cal = [], {}
        for c in cfgs:
            yy, ii, pp = [], [], {"none": [], "T": []}; Ts = []
            for f in folds:
                others = [g for g in folds if g != f]
                T = fit_T(np.concatenate([runs[(c, g)]["prob"] for g in others]).astype(np.float64),
                          np.concatenate([runs[(c, g)]["y"] for g in others])); Ts.append(T)
                p = runs[(c, f)][field].astype(np.float64)
                yy.append(runs[(c, f)]["y" if field == "prob" else "rawy"])
                ii.append(runs[(c, f)]["idx" if field == "prob" else "rawidx"])
                pp["none"].append(p); pp["T"].append(temper(p, T))
            y = np.concatenate(yy)
            o = np.argsort(np.concatenate(ii), kind="stable")
            cal[c] = (y[o], {k: np.concatenate(v)[o] for k, v in pp.items()})
            row = {"config": c, "T_mean": f"{np.mean(Ts):.2f}"}
            for k in ("none", "T"):
                p = np.concatenate(pp[k]); yh = decide(p, classes, 1); crit, m, _ = misses(y, yh, t)
                row[f"ECE_{k}"] = f"{ece(p, y) * 100:.2f}"
                row[f"rule_acc_{k}"] = f"{(yh == y).mean() * 100:.2f}"
                row[f"rule_crit_misses_{k}"] = int(m.sum())
            ts.append(row)
        write_csv(os.path.join(out_dir, f"extra_temperature_{dom}.csv"), ts)
        lines += [f"\n## 6. Temperature scaling (fitted on other folds, in-domain) before the rule{tag}\n", md_table(ts)]

        # paired critical misses after the rule: recalibrated CE vs hazard-aware training
        tp = []
        for (a, ka), (b, kb) in [(("mnv3_ce", "T"), ("mnv3_hafl", "none")), (("mnv3_ce", "T"), ("mnv3_hafl", "T"))]:
            if a not in cal or b not in cal: continue
            assert (cal[a][0] == cal[b][0]).all()
            y = cal[a][0]
            _, ma, _ = misses(y, decide(cal[a][1][ka], classes, 1), t)
            _, mb, _ = misses(y, decide(cal[b][1][kb], classes, 1), t)
            n_a, n_b = int((ma & ~mb).sum()), int((~ma & mb).sum())
            pv = A.binomtest(n_a, n_a + n_b, 0.5).pvalue if n_a + n_b else 1.0
            tp.append({"A": a + "+T" * (ka == "T") + "+rule", "B": b + "+T" * (kb == "T") + "+rule",
                       "A_misses": int(ma.sum()), "B_misses": int(mb.sum()),
                       "only_A_missed": n_a, "only_B_missed": n_b, "mcnemar_p": f"{pv:.3g}"})
        write_csv(os.path.join(out_dir, f"extra_temperature_pairs_{dom}.csv"), tp)
        lines += [f"\n### Paired critical misses after the rule: recalibrated CE vs HAFL{tag}\n", md_table(tp)]

        # 7) error severity: does a wrong class still land in the right WHO bin?
        sev = []
        for c in cfgs:
            idx, y, p = data[c]
            for rule, lam in (("argmax", 0), ("cost-rule", 1)):
                yh = decide(p, classes, lam); wrong = yh != y; crit, m, _ = misses(y, yh, t)
                sev.append({"config": c, "rule": rule, "acc": f"{(~wrong).mean() * 100:.2f}",
                            "bin_acc": f"{(t[yh] == t[y]).mean() * 100:.2f}", "errors": int(wrong.sum()),
                            "same_bin_%": f"{(wrong & (t[yh] == t[y])).sum() / wrong.sum() * 100:.0f}",
                            "safer_bin": int((t[yh] > t[y]).sum()), "less_safe_bin": int((t[yh] < t[y]).sum()),
                            "crit_misses_per_1000": f"{m.sum() / crit.sum() * 1000:.1f}"})
        write_csv(os.path.join(out_dir, f"extra_bins_{dom}.csv"), sev)
        lines += [f"\n## 7. Error severity at the WHO-bin level{tag}\n", md_table(sev)]

        # remaining critical confusions for HAFL + rule
        if "mnv3_hafl" in data:
            idx, y, p = data["mnv3_hafl"]; yh = decide(p, classes, 1); crit, m, _ = misses(y, yh, t)
            pairs = {}
            for a, b in zip(y[m], yh[m]):
                pairs[(classes[a], classes[b])] = pairs.get((classes[a], classes[b]), 0) + 1
            top = sorted(pairs.items(), key=lambda kv: -kv[1])[:8]
            lines += [f"\n### Remaining critical misses, mnv3_hafl + rule{tag}\n"] + \
                     [f"- {n}× {a} (tier {TIER[a]}) → {b} (tier {TIER[b]})" for (a, b), n in top]

    open(os.path.join(out_dir, "extra.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(os.path.join(HERE, sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "results"))
