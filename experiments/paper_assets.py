"""
Tables and figures of the camera-ready paper, generated from the analysis CSV files
(run analyze.py and extra_analysis.py on results, results_mw and results_imw first).

    python paper_assets.py        # -> paper/tables/*.tex (table bodies), paper/figures/*.pdf + .png
"""
import csv, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "paper")
DOMS = [("D1", "analysis", "test"), ("D1-raw", "analysis", "raw"), ("D2", "analysis_mw", "test"),
        ("D3", "analysis_imw", "test")]
TITLES = {"D1": "D1 (clean background)", "D1-raw": "D1 (original background)", "D2": "D2", "D3": "D3"}
MODELS = [("mnv3_ce", "CE"), ("mnv3_wce", "wCE"), ("mnv3_fl", "FL"), ("mnv3_hafl", "HAFL"),
          ("mnv3_hafl_steep", "HAFL, steep $h$"), ("cbam_fl", "CBAM + FL"), ("cbam_hafl", "CBAM + HAFL"),
          ("r50_ce", "ResNet-50, CE")]


def read(path):
    with open(os.path.join(HERE, path), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def by_config(path):
    return {r["config"]: r for r in read(path)}


def write(name, lines):
    os.makedirs(os.path.join(PAPER, "tables"), exist_ok=True)
    with open(os.path.join(PAPER, "tables", name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def datasets():
    lines = []
    for label, folder, man, cite in (("D1", "analysis", "manifest.csv", "d1"),
                                     ("D2", "analysis_mw", "manifest_mw.csv", "d2"),
                                     ("D3", "analysis_imw", "manifest_imw.csv", "d3")):
        rows = read(man); groups = Counter(r["group"] for r in rows)
        multi = sum(v for v in groups.values() if v > 1)
        crit = int(read(f"{folder}/extra_rule_test.csv")[0]["n_critical_items"])
        lines.append(f"{label}~\\cite{{{cite}}} & {len(rows):,} & {len({r['class'] for r in rows})} & "
                     f"{len(groups):,} & {multi:,} ({100 * multi / len(rows):.0f}\\%) & {crit:,} \\\\")
    write("datasets.tex", lines)


def main_results():
    acc = {"D1": by_config("analysis/main_table.csv"), "D1-raw": by_config("analysis/raw_domain.csv"),
           "D2": by_config("analysis_mw/main_table.csv"), "D3": by_config("analysis_imw/main_table.csv")}
    rule = {d: by_config(f"{a}/extra_rule_{dom}.csv") for d, a, dom in DOMS}
    lines = []
    for c, name in MODELS:
        cells = [name]
        for d, _, _ in DOMS:
            if c not in rule[d]:
                cells += ["--", "--"]; continue
            r = rule[d][c]; n = int(r["n_critical_items"])
            cells += [acc[d][c]["acc"].replace(" ± ", " $\\pm$ "),
                      f"{100 * int(r['crit_misses_argmax']) / n:.2f} $\\to$ {100 * int(r['crit_misses_rule']) / n:.2f}"]
        lines.append(" & ".join(cells) + r" \\")
        if c in ("mnv3_hafl_steep", "cbam_hafl"):
            lines.append(r"\midrule")
    write("main.tex", lines)


def calibration():
    T = {d: by_config(f"{a}/extra_temperature_{dom}.csv") for d, a, dom in DOMS}
    lines = []
    for name, c, k in (("CE", "mnv3_ce", "none"), ("CE + $T$", "mnv3_ce", "T"), ("FL", "mnv3_fl", "none"),
                       ("HAFL", "mnv3_hafl", "none"), ("HAFL + $T$", "mnv3_hafl", "T")):
        lines.append(" & ".join([name] + [f"{T[d][c]['ECE_' + k]} / {T[d][c]['rule_crit_misses_' + k]}"
                                          for d, _, _ in DOMS]) + r" \\")
    write("calibration.tex", lines)


def efficiency():
    params = by_config("analysis/main_table.csv")      # exact counts (efficiency.csv is rounded to 3 decimals)
    cfg = {"MobileNetV3-L": "mnv3_ce", "MobileNetV3-L + CBAM": "cbam_fl", "ResNet-50": "r50_ce"}
    lines = [f"{r['model'].replace('MobileNetV3-L + CBAM', '+ CBAM')} & {int(params[cfg[r['model']]]['params']) / 1e6:.2f} & "
             f"{float(r['GMACs']):.2f} & {float(r['ms_median_1t']):.1f} & {float(r['ms_median_4t']):.1f} \\\\"
             for r in read("analysis/efficiency.csv")]
    write("efficiency.tex", lines)


def ablation():
    """CBAM and HAFL switched on and off on D1 (fold mean and s.d.), with CE and ResNet-50 for reference"""
    m = by_config("analysis/main_table.csv")
    rows = [("mnv3_ce", "CE"), ("mnv3_fl", "FL"), ("cbam_fl", "FL + CBAM"), ("mnv3_hafl", "HAFL"),
            ("cbam_hafl", "HAFL + CBAM"), ("r50_ce", "ResNet-50, CE")]
    lines = []
    for c, name in rows:
        if c == "r50_ce":
            lines.append(r"\midrule")
        lines.append(" & ".join([name] + [m[c][k].replace(" ± ", " $\\pm$ ") for k in ("acc", "macro_f1", "sharps_f1")])
                     + r" \\")
    write("ablation.tex", lines)


def xdata():
    """models trained on one dataset, tested on another (WHO-bin level; written by xdata/analyze_xdata.py)"""
    import json
    tiers = json.load(open(os.path.join(HERE, "xdata", "tiers.json")))
    majority = {}
    for d, name, man in (("bmw", "D1", "manifest.csv"), ("mw", "D2", "manifest_mw.csv"), ("imw", "D3", "manifest_imw.csv")):
        n = Counter(tiers[d][r["class"]] for r in read(man))
        majority[name] = 100 * max(n.values()) / sum(n.values())
    x = {(r["direction"], r["config"]): r for r in read("analysis/xdata.csv")}
    mean = lambda s: s.split(" +- ")[0]
    lines = []
    for src, tgt in (("D1", "D2"), ("D1", "D3"), ("D2", "D1"), ("D3", "D1")):
        ce, hafl = x[(f"{src} -> {tgt}", "mnv3_ce")], x[(f"{src} -> {tgt}", "mnv3_hafl")]
        lines.append(f"{src} $\\to$ {tgt} & {majority[tgt]:.1f} & {mean(ce['bin_acc_argmax'])} & {mean(hafl['bin_acc_argmax'])} & "
                     + " & ".join(f"{mean(r['cmr_argmax'])} $\\to$ {mean(r['cmr_rule'])}" for r in (ce, hafl)) + r" \\")
    write("xdata.tex", lines)


D1_NAMES = {"used_syringes": "syringes", "episiotomy_scissors": "episiotomy scissors", "mayo_scissors": "Mayo scissors",
            "stitch_removal_scissors": "stitch scissors", "blood_soaked_bandages": "blood-soaked bandages",
            "human_organs": "organs", "ampoules_full": "full ampoules", "ampuoles_broken": "broken ampoules",
            "disinfectant_bottles": "disinfectants", "expired_tablets": "expired tablets", "iv_bottles": "IV bottles",
            "syrup_bottles": "syrup bottles", "vaccine_or_medicine_vials": "vials",
            "general_organic_waste": "organic waste", "uncategorized_or_overlapping": "mixed items",
            "used_masks": "masks", "used_medical_gloves": "gloves", "used_medical_paper": "medical paper",
            "waterbottles": "water bottles"}


def per_class():
    """per-class F1 of HAFL on D1 in two side-by-side halves (written by per_class.py)"""
    rows = [(D1_NAMES.get(r["class"], r["class"]), r["tier"], r["f1"]) for r in read("analysis/per_class_f1_mnv3_hafl.csv")]
    rows.sort(key=lambda r: (-int(r[1]), r[0].lower()))
    cells = [f"{n} & {t} & {f}" for n, t, f in rows]
    half = (len(cells) + 1) // 2
    left, right = cells[:half], cells[half:] + [" & & "] * (2 * half - len(cells))
    write("perclass.tex", [f"{a} & {b} \\\\" for a, b in zip(left, right)])


def save(fig, name):
    import matplotlib
    matplotlib.rcParams["pdf.fonttype"] = 42            # TrueType, not Type 3 (IEEE PDF eXpress)
    os.makedirs(os.path.join(PAPER, "figures"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PAPER, "figures", f"{name}.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.02)


def curves():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.5})
    style = {"mnv3_ce": ("CE", "tab:blue", "o"), "mnv3_fl": ("FL", "tab:orange", "s"),
             "mnv3_hafl": ("HAFL", "tab:green", "^"), "mnv3_hafl_steep": ("HAFL, steep $h$", "tab:red", "v")}
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.55))
    for ax, (d, a, dom) in zip(axes, DOMS):
        cur = read(f"{a}/extra_curve_{dom}.csv")
        for c, (label, color, marker) in style.items():
            pts = sorted((float(r["lambda"]), float(r["crit_miss_rate"]), float(r["acc"])) for r in cur
                         if r["config"] == c)
            if not pts:
                continue
            lam, x, y = zip(*pts)
            ax.plot(x, y, "-", color=color, lw=0.9, label=label)
            ax.plot(x[0], y[0], marker, color=color, mfc="white", ms=3.5, mew=0.8)     # argmax (lambda = 0)
            i = lam.index(1.0)
            ax.plot(x[i], y[i], marker, color=color, ms=3.5)                          # default rule (lambda = 1)
        ax.set_title(TITLES[d], fontsize=7, pad=3)
        ax.set_xlabel("critical-miss rate (%)", labelpad=1)
        ax.tick_params(width=0.5, length=2, pad=1.5)
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel("accuracy (%)", labelpad=1)
    axes[0].legend(frameon=False, fontsize=6, handlelength=1.4, loc="lower right")
    fig.tight_layout(pad=0.3, w_pad=0.6)
    save(fig, "curves")


def pipeline():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(figsize=(3.5, 0.95))
    ax.set_xlim(0, 100); ax.set_ylim(0, 29); ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc=color, ec="0.35", lw=0.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.2, linespacing=1.2)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.25",
                                                            shrinkA=0, shrinkB=0, mutation_scale=6))

    box(0.5, 17, 11, 11, "waste\nimage", "#eeeeee")
    box(14.5, 17, 21, 11, "MobileNetV3-L\n(any loss)", "#dbe9f6")
    box(38.5, 17, 17, 11, "class\nprobabilities", "#eeeeee")
    box(59, 17, 20, 11, "min-expected-\ncost decision", "#fde3c8")
    box(82.5, 17, 17, 11, "WHO\nhazard bin", "#d9f0d3")
    box(59, 1, 20, 10, "WHO-tier\ncost matrix $C$", "#fff7e6")
    box(82.5, 1, 17, 10, "refer to\na person", "#f6d5d5")
    arrow(11.8, 22.5, 14.2, 22.5); arrow(35.8, 22.5, 38.2, 22.5); arrow(55.8, 22.5, 58.7, 22.5)
    arrow(79.3, 22.5, 82.2, 22.5); arrow(69, 11.3, 69, 16.7); arrow(79.3, 18.5, 82.2, 9.5)
    save(fig, "pipeline")


def architecture():
    """MobileNetV3-Large backbone (optional CBAM) followed by the decision rule and referral"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix"})
    fig = plt.figure(figsize=(3.5, 1.04))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 3.5); ax.set_ylim(0, 1.04); ax.axis("off")
    cols = ((0.0, 0.70), (0.88, 1.30), (2.36, 1.14))       # (left, width) in inches
    rows = (0.64, 0.04)                                    # bottom of each row, boxes 0.36 in high

    def box(c, r, text, fill="white"):
        x, w = cols[c]
        ax.add_patch(Rectangle((x, rows[r]), w, 0.36, fc=fill, ec="black", lw=0.6))
        ax.text(x + w / 2, rows[r] + 0.18, text, ha="center", va="center", fontsize=6, linespacing=1.2)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black",
                                                            shrinkA=0, shrinkB=0, mutation_scale=6))

    box(0, 0, "input image\n224$\\times$224$\\times$3")
    box(1, 0, "MobileNetV3-Large (ImageNet)\n15 inverted-residual blocks\nCBAM after 3, 6, 12 (variant)")
    box(2, 0, "avg. pool, dropout,\nlinear, softmax $p$")
    box(0, 1, "cost matrix $C$\nEq. (1)", "0.92")
    box(1, 1, "minimum-expected-cost\ndecision $\\hat{k}$, Eq. (2)", "0.92")
    box(2, 1, "WHO bin of $\\hat{k}$, or refer\nto a person (top 10%\nexpected cost)", "0.92")
    for c in (0, 1):                                       # left to right in both rows
        x = cols[c][0] + cols[c][1]
        arrow(x, 0.82, cols[c + 1][0], 0.82); arrow(x, 0.22, cols[c + 1][0], 0.22)
    xp, xd = cols[2][0] + cols[2][1] / 2, cols[1][0] + cols[1][1] / 2
    ax.plot([xp, xp, xd], [0.64, 0.52, 0.52], color="black", lw=0.6)   # p goes down to the decision
    arrow(xd, 0.52, xd, 0.40)
    save(fig, "architecture")


def diagnostics():
    """(a) training curves, (b) ROC and (c) PR for detecting critical items, HAFL, pooled out-of-fold"""
    import json, numpy as np, matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.5})
    d1 = json.load(open(os.path.join(HERE, "analysis/diag_mnv3_hafl.json")))
    sets = [("D1", d1["curves"]["test"], "tab:blue"), ("D1, orig. bg.", d1["curves"]["raw"], "tab:red"),
            ("D2", json.load(open(os.path.join(HERE, "analysis_mw/diag_mnv3_hafl.json")))["curves"]["test"], "tab:green"),
            ("D3", json.load(open(os.path.join(HERE, "analysis_imw/diag_mnv3_hafl.json")))["curves"]["test"], "tab:orange")]
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 1.55))
    tr = d1["train"]; ep = np.arange(1, len(tr["train_loss"]) + 1)
    ax[0].plot(ep, tr["train_loss"], color="0.3", lw=0.9, label="training loss")
    ax[0].set_xlabel("epoch", labelpad=1); ax[0].set_ylabel("training loss", labelpad=1)
    a2 = ax[0].twinx()
    a2.plot(ep, np.array(tr["val_acc"]) * 100, color="tab:blue", lw=0.9, label="val. accuracy")
    a2.plot(ep, np.array(tr["val_macro_f1"]) * 100, color="tab:green", lw=0.9, ls="--", label="val. macro-F1")
    a2.set_ylabel("validation (%)", labelpad=1); a2.set_ylim(60, 100); a2.tick_params(width=0.5, length=2, pad=1.5)
    h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, frameon=False, fontsize=5.5, loc="center right")
    ax[0].set_title("(a) training, D1 (mean of 5 folds)", fontsize=7, pad=3)
    for name, c, color in sets:
        x, y = c["fpr_tpr"]
        ax[1].plot(np.maximum(x, 1e-4), y, color=color, lw=0.9)
        r, p = c["rec_prec"]
        ax[2].plot(r, p, color=color, lw=0.9, label=f"{name} (AUC {c['auc']:.4f}, AP {c['ap']:.4f})")
    ax[1].set_xscale("log"); ax[1].set_xlim(1e-4, 1); ax[1].set_ylim(0.6, 1.005)
    ax[1].set_xlabel("false-positive rate (log scale)", labelpad=1); ax[1].set_ylabel("true-positive rate", labelpad=1)
    ax[1].set_title("(b) ROC, critical items (legend in c)", fontsize=7, pad=3)
    ax[2].set_xlim(0.8, 1.002); ax[2].set_ylim(0.8, 1.005)
    ax[2].set_xlabel("recall", labelpad=1); ax[2].set_ylabel("precision", labelpad=1)
    ax[2].set_title("(c) precision-recall, critical items", fontsize=7, pad=3)
    for a in ax:
        a.tick_params(width=0.5, length=2, pad=1.5); a.grid(alpha=0.3, lw=0.4)
    ax[2].legend(frameon=False, fontsize=5.5, loc="lower left")
    fig.tight_layout(pad=0.3, w_pad=0.8)
    save(fig, "diagnostics")


def confusion():
    """row-normalised confusion matrix of HAFL on D1, classes ordered by tier, per-class F1 in the labels"""
    import json, numpy as np, matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.rcParams.update({"font.family": "serif", "font.size": 5.5, "axes.linewidth": 0.5})
    d = json.load(open(os.path.join(HERE, "analysis/diag_mnv3_hafl.json")))
    cm = np.array(d["confusion"], float); tiers = np.array(d["tiers"])
    f1 = {r["class"]: r["f1"] for r in read("analysis/per_class_f1_mnv3_hafl.csv")}
    names = [D1_NAMES.get(c, c) for c in d["classes"]]
    order = sorted(range(len(names)), key=lambda i: (-tiers[i], names[i].lower()))
    cm = cm[np.ix_(order, order)]; pct = 100 * cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    im = ax.imshow(np.where(pct > 0, pct, np.nan), cmap="Blues", norm=LogNorm(0.1, 100))
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=4.8)
    ax.set_yticklabels([f"{names[i]} ({f1[d['classes'][i]]})" for i in order], fontsize=4.8)
    ax.set_xlabel("predicted class", labelpad=1); ax.set_ylabel("true class (F1, %)", labelpad=1)
    t = tiers[order]
    for b in np.where(np.diff(t) != 0)[0]:                 # WHO tier boundaries
        ax.axhline(b + 0.5, color="0.4", lw=0.4); ax.axvline(b + 0.5, color="0.4", lw=0.4)
    ax.tick_params(width=0.4, length=1.5, pad=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("% of true class (log scale)", fontsize=5)
    cb.ax.tick_params(labelsize=4.8, width=0.4, length=1.5)
    fig.tight_layout(pad=0.2)
    save(fig, "confusion")


NAMES = {"used_medical_paper": "medical paper", "human_organs": "organs", "general_organic_waste": "organic waste",
         "ampoules_full": "full ampoules", "uncategorized_or_overlapping": "mixed items",
         "stitch_removal_scissors": "stitch scissors", "scalpels": "scalpel", "SDB": "soiled dressing",
         "MW": "microbiological", "DW": "dental waste", "PhW": "pharmaceutical"}


def failures():
    """first random draw of each set (fixed / remaining) per test set, saved by failure_figure.py"""
    import json
    import numpy as np, matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"]})
    panels = []
    for folder, groups in (("analysis", (("test", "D1, clean background"), ("raw", "D1, original background"))),
                           ("analysis_imw", (("test", "D3"),))):
        meta = json.load(open(os.path.join(HERE, folder, "failures_mnv3_hafl.json")))
        arr = np.load(os.path.join(HERE, folder, "failures_mnv3_hafl.npz"))
        for dom, title in groups:
            for s in ("fixed", "remaining"):
                i = next(j for j, m in enumerate(meta) if m["dom"] == dom and m["set"] == s)
                panels.append((title, meta[i], arr["img"][i], arr["cam"][i].astype(float)))
    name = lambda c: NAMES.get(c, c.replace("_", " "))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.16, 1.75))
    for n, (ax, (title, m, img, cam)) in enumerate(zip(axes, panels)):
        ax.imshow(img); ax.imshow(cam, cmap="jet", alpha=0.4); ax.axis("off")
        ax.set_title(f"({'abcdef'[n]}) {name(m['true'])}\nargmax: {name(m['argmax'])} ({m['p_argmax']:.2f})\n"
                     f"rule: {name(m['rule'])} ({'safe' if m['rule_safe'] else 'unsafe'})",
                     fontsize=6.5, pad=2, linespacing=1.15)
    fig.tight_layout(pad=0.2, w_pad=0.5, rect=(0, 0, 1, 0.9))
    for n in range(0, len(panels), 2):                 # one label over each pair of panels
        left, right = axes[n].get_position(), axes[n + 1].get_position()
        fig.text((left.x0 + right.x1) / 2, 0.99, panels[n][0], ha="center", va="top", fontsize=7, fontweight="bold")
    save(fig, "failures")


if __name__ == "__main__":
    datasets(); main_results(); calibration(); efficiency(); ablation(); per_class(); xdata(); curves(); architecture(); diagnostics(); confusion()
    failures()
    print("wrote", os.path.join(PAPER, "tables"), "and", os.path.join(PAPER, "figures"))
