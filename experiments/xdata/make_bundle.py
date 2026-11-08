"""
Builds xdata/bln_xdata.zip for the cross-dataset test on Kaggle: xdata.py, model.py, the three
manifests, tiers.json and the CE / HAFL checkpoints of all folds (ckpt/<dataset>/<config>_f<fold>.pt).

    python xdata/make_bundle.py
"""
import glob, json, os, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
RESULTS = {"bmw": "results", "mw": "results_mw", "imw": "results_imw"}
CONFIGS = ("mnv3_ce", "mnv3_hafl")


def main():
    files = {"xdata.py": os.path.join(HERE, "xdata.py"), "tiers.json": os.path.join(HERE, "tiers.json"),
             "model.py": os.path.join(os.path.dirname(REV), "model.py"),
             "manifest.csv": os.path.join(REV, "manifest.csv"),
             "manifest_mw.csv": os.path.join(REV, "manifest_mw.csv"),
             "manifest_imw.csv": os.path.join(REV, "manifest_imw.csv")}
    for d, folder in RESULTS.items():
        for m in glob.glob(os.path.join(REV, folder, "**", "metrics.json"), recursive=True):
            j = json.load(open(m))
            if j["config"] in CONFIGS:
                files[f"ckpt/{d}/{j['config']}_f{j['fold']}.pt"] = os.path.join(os.path.dirname(m), "best.pt")
    n_ckpt = sum(k.startswith("ckpt/") for k in files)
    assert n_ckpt == 30, f"expected 30 checkpoints, found {n_ckpt}"
    out = os.path.join(HERE, "bln_xdata.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for arc, src in sorted(files.items()):
            z.write(src, arc)
    print(f"wrote {out}: {len(files)} files, {os.path.getsize(out) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
