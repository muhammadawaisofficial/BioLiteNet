"""
Real dangerous misses of the paper's model, with Grad-CAM, from the saved out-of-fold predictions.

    python failure_figure.py results bmw                          # no images needed: lists the files to fetch
    python failure_figure.py results bmw E:/datasets/picks/bmw    # root containing Baackground_Removed/ and Raw/
    python failure_figure.py results_imw imw E:/datasets/picks/imw   # root containing the "<name> (CODE)" folders

Examples are drawn at random (fixed seed) from two sets, so they are not hand-picked:
  fixed     - critical item (tier >= 4) that argmax sends to a lower tier and the cost rule sends to a safe bin
  remaining - critical item still sent to a lower tier after the rule
Grad-CAM is computed with the checkpoint of the fold in which the image was a test image, and the
local prediction is checked against the saved one (a wrong file, checkpoint or preprocessing shows up).
"""
import csv, glob, json, os, sys
import numpy as np

import extra_analysis as E                            # analyze reads BLN_DATA from argv[2]
from analyze import TIER, DATA, MANIFEST, HERE, load

CONFIG, PER_SET, SEED = "mnv3_hafl", 2, 0
BIN = {5: "sharps", 4: "infectious/patho.", 3: "chemical", 2: "pharmaceutical", 1: "general"}


def checkpoint_dirs(results_dir):
    out = {}
    for m in glob.glob(os.path.join(results_dir, "**", "metrics.json"), recursive=True):
        j = json.load(open(m)); out[(j["config"], j["fold"])] = os.path.dirname(m)
    return out


def pick(runs, classes):
    """random critical misses, per domain and set; needs only the saved predictions"""
    t = E.tiers(classes); folds = sorted({k[1] for k in runs})
    rng = np.random.default_rng(SEED); picks = []
    for dom in ("test", "raw") if DATA == "bmw" else ("test",):
        if dom == "test":
            idx, y, p = E.pooled(runs, CONFIG, folds)
        else:
            idx, y, p = (np.concatenate([runs[(CONFIG, f)]["raw"][k] for f in folds]) for k in range(3))
        p = p.astype(float); y0, y1 = E.decide(p, classes, 0), E.decide(p, classes, 1)
        crit = t[y] >= 4
        sets = {"fixed": crit & (t[y0] < t[y]) & (t[y1] >= t[y]), "remaining": crit & (t[y1] < t[y])}
        for name, mask in sets.items():
            for j in rng.choice(np.where(mask)[0], min(PER_SET, mask.sum()), replace=False):
                picks.append({"dom": dom, "set": name, "i": int(idx[j]), "y": int(y[j]),
                              "y0": int(y0[j]), "y1": int(y1[j]), "p": p[j]})
    return picks


def needed(k, row):
    """(folder searched for on Kaggle, file inside it) for one pick; raw images are matched by stem"""
    if DATA == "imw":
        return "<imw root>", row["file"]
    if k["dom"] == "raw":
        return "Raw", f"{row['class']}/{os.path.splitext(row['file'])[0]}.*"
    return "Baackground_Removed", f"{row['class']}/{row['file']}"


def gradcam(model, x, k):
    """Grad-CAM of class k at the last backbone stage, and the softmax output"""
    import torch, torch.nn.functional as F
    store = {}

    def hook(m, i, o):                                # must return None, or it replaces the stage output
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))

    h = model.stages[-1].register_forward_hook(hook)
    model.zero_grad(); out = model(x); out[0, k].backward(); h.remove()
    cam = F.relu((store["g"].mean((2, 3), keepdim=True) * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return (cam / (cam.max() + 1e-8)).detach().numpy(), torch.softmax(out, 1)[0].detach().numpy()


def main(results_dir, data_root=None):
    runs = load(results_dir); ckpt = checkpoint_dirs(results_dir)
    classes = runs[next(iter(runs))]["classes"]
    man = list(csv.DictReader(open(os.path.join(HERE, MANIFEST), encoding="utf-8")))
    fold_of = {int(i): f for (c, f), r in runs.items() if c == CONFIG for i in r["idx"]}
    picks = pick(runs, classes)

    if data_root is None:
        print(f"# {DATA}: {len(picks)} images ({CONFIG}, seed {SEED})")
        for k in picks:
            print(f'    ("{DATA}", "{needed(k, man[k["i"]])[0]}", "{needed(k, man[k["i"]])[1]}"),'
                  f'   # {k["dom"]} {k["set"]}: {classes[k["y"]]} -> {classes[k["y0"]]} -> {classes[k["y1"]]}')
        return

    import torch, matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    sys.path[:0] = [os.path.dirname(HERE)]
    from model import BioLiteNet

    def image(dom, row):                              # same decoding and resizing as run_bln.py
        if DATA == "imw":
            im = Image.open(os.path.join(data_root, row["file"])); im.draft("RGB", (512, 512))
        elif dom == "raw":
            im = Image.open(sorted(glob.glob(os.path.join(data_root, "Raw", row["class"],
                                                          os.path.splitext(row["file"])[0] + ".*")))[0])
        else:
            im = Image.open(os.path.join(data_root, "Baackground_Removed", row["class"], row["file"]))
        return np.asarray(im.convert("RGB").resize((224, 224), Image.BILINEAR))

    use_cbam = CONFIG.startswith("cbam")
    fig, ax = plt.subplots(2, len(picks), figsize=(2.1 * len(picks), 4.9), squeeze=False)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    imgs, cams, meta = [], [], []                     # kept for the paper figure (paper_assets.py)
    for c, k in enumerate(picks):
        f = fold_of[k["i"]]
        model = BioLiteNet(num_classes=len(classes), use_cbam=use_cbam, pretrained=False)
        model.load_state_dict(torch.load(os.path.join(ckpt[(CONFIG, f)], "best.pt"), map_location="cpu",
                                         weights_only=True))
        model.eval()
        img = image(k["dom"], man[k["i"]])
        x = torch.from_numpy(((img / 255.0 - mean) / std).astype(np.float32)).permute(2, 0, 1)[None]
        cam, q = gradcam(model, x, k["y0"])
        ok = "OK" if q.argmax() == k["y0"] else "MISMATCH"
        print(k["dom"], k["set"], man[k["i"]]["file"], "fold", f, "|", classes[k["y"]], "->", classes[k["y0"]],
              "->", classes[k["y1"]], f"| local vs saved prob: max diff {np.abs(q - k['p']).max():.3f} {ok}")
        ax[0, c].imshow(img); ax[1, c].imshow(img); ax[1, c].imshow(cam, cmap="jet", alpha=0.45)
        y, y0, y1 = classes[k["y"]], classes[k["y0"]], classes[k["y1"]]
        tick = "safe" if TIER[y1] >= TIER[y] else "still unsafe"
        imgs.append(img); cams.append(cam)
        meta.append({"dom": k["dom"], "set": k["set"], "file": man[k["i"]]["file"], "fold": int(f),
                     "true": y, "argmax": y0, "rule": y1, "p_argmax": round(float(k["p"][k["y0"]]), 2),
                     "rule_safe": bool(TIER[y1] >= TIER[y])})
        ax[0, c].set_title(f"{'Raw bg' if k['dom'] == 'raw' else 'Clean bg'} · {k['set']}\n"
                           f"true: {y} ({BIN[TIER[y]]})\n"
                           f"argmax: {y0} ({k['p'][k['y0']]:.2f})\nrule: {y1} [{tick}]", fontsize=6.5)
        for a in ax[:, c]:
            a.axis("off")
    plt.tight_layout()
    out = os.path.join(HERE, "analysis" + E.SUFFIX, f"failures_{CONFIG}.png")
    plt.savefig(out, dpi=300); plt.savefig(out.replace(".png", ".pdf"))
    np.savez_compressed(out.replace(".png", ".npz"), img=np.stack(imgs), cam=np.stack(cams).astype(np.float16))
    json.dump(meta, open(out.replace(".png", ".json"), "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(os.path.join(HERE, sys.argv[1]), sys.argv[3] if len(sys.argv) > 3 else None)
