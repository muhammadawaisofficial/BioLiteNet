"""
Sensitivity of the hazard evaluation to debatable tier assignments (no retraining).

    python tier_sensitivity.py results_mw mw syringe=4          # needle-free syringes counted as infectious
    python tier_sensitivity.py results_imw imw PPE=4 CBL=4      # contaminated PPE and bedding counted as infectious

Writes analysis<suffix>/tier_sensitivity.csv: critical misses with argmax and with the cost rule,
under the paper's tiers and under the alternative tiers (the rule then uses the alternative cost matrix).
"""
import os, sys
import extra_analysis as E                            # analyze reads BLN_DATA from argv[2]
from analyze import TIER, load, write_csv, md_table, binomtest

CONFIGS = ["mnv3_ce", "mnv3_wce", "mnv3_fl", "mnv3_hafl", "mnv3_hafl_steep", "cbam_hafl"]


def main(results_dir, changes):
    runs = load(results_dir); folds = sorted({k[1] for k in runs})
    classes = runs[next(iter(runs))]["classes"]
    cfgs = [c for c in CONFIGS if all((c, f) in runs for f in folds)]
    rows = []
    for name in ("paper", "alternative"):
        if name == "alternative":
            TIER.update(changes)                      # E.tiers / E.cost read TIER at call time
        t = E.tiers(classes)
        for c in cfgs:
            idx, y, p = E.pooled(runs, c, folds); p = p.astype(float)
            crit, m0, _ = E.misses(y, E.decide(p, classes, 0), t)
            _, m1, _ = E.misses(y, E.decide(p, classes, 1), t)
            fixed, broke = int((m0 & ~m1).sum()), int((~m0 & m1).sum())
            rows.append({"tiers": name if name == "paper" else " ".join(f"{k}={v}" for k, v in changes.items()),
                         "config": c, "n_critical_items": int(crit.sum()), "crit_misses_argmax": int(m0.sum()),
                         "crit_misses_rule": int(m1.sum()), "newly_missed": broke,
                         "mcnemar_p": f"{binomtest(fixed, fixed + broke, 0.5).pvalue if fixed + broke else 1.0:.3g}"})
    write_csv(os.path.join(E.HERE, "analysis" + E.SUFFIX, "tier_sensitivity.csv"), rows)
    print(md_table(rows))


if __name__ == "__main__":
    main(os.path.join(E.HERE, sys.argv[1]), {k: int(v) for k, v in (a.split("=") for a in sys.argv[3:])})
