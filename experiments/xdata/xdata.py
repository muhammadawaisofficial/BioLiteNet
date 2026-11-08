"""
Cross-dataset test on Kaggle: models trained on one dataset classify the images of another one.
Inference only, with the same decoding and resizing as run_bln.py.

Kaggle notebook (GPU T4), inputs:
    mario78/biomedical-waste-classification-images-dataset      (D1)
    sumn2u/medical-waste                                       (D2)
    engineeringubu/infectious-municipal-waste                  (D3)
    your private dataset made from bln_xdata.zip (this script, model.py, manifests, tiers.json, ckpt/)
then run one cell:
    !python $(find /kaggle/input -name xdata.py | head -1)
Output: /kaggle/working/xdata_probs.npz  (download it and put it in revision/xdata/)

Directions: D1 -> D2, D1 -> D3, D2 -> D1, D3 -> D1 (D2 and D3 share photos, so they are not tested on
each other). Every model is also run on its own test fold as a check against the saved accuracy.
"""
import csv, glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATASETS = ("bmw", "mw", "imw")
CONFIGS = ("mnv3_ce", "mnv3_hafl")
TARGETS = {"bmw": ("mw", "imw"), "mw": ("bmw",), "imw": ("bmw",)}


def walk_root(test):
    for root, dirs, _ in os.walk("/kaggle/input"):
        if test(root, dirs):
            return root
    raise FileNotFoundError("image folder not found in /kaggle/input")


def image_root(d):
    if d == "bmw":
        return os.path.join(walk_root(lambda r, ds: "Baackground_Removed" in ds), "Baackground_Removed")
    marker = "syringe_needle" if d == "mw" else "Sharps Waste (SW)"
    return walk_root(lambda r, ds: marker in ds)


def manifest(d):
    name = "manifest.csv" if d == "bmw" else f"manifest_{d}.csv"
    rows = list(csv.DictReader(open(os.path.join(HERE, name), newline="", encoding="utf-8")))
    return rows, sorted({r["class"] for r in rows})


def decode(d, rows):
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    root = image_root(d)

    def one(r):
        path = os.path.join(root, r["file"]) if d == "imw" else os.path.join(root, r["class"], r["file"])
        im = Image.open(path)
        if d == "imw":
            im.draft("RGB", (512, 512))
        return np.asarray(im.convert("RGB").resize((224, 224), Image.BILINEAR))

    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        arr = np.stack(list(ex.map(one, rows)))
    print(f"[{d}] decoded {len(rows)} images from {root} in {time.time() - t0:.0f}s", flush=True)
    return arr


def predict(model, arr, device, bs=256):
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad(), torch.autocast("cuda", enabled=device == "cuda"):
        for i in range(0, len(arr), bs):
            x = torch.from_numpy(arr[i:i + bs]).to(device).permute(0, 3, 1, 2).float().div(255)
            out.append(torch.softmax(model((x - mean) / std).float(), 1).cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def main():
    import torch
    from model import BioLiteNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {}
    for d in DATASETS:
        rows, classes = manifest(d)
        data[d] = (rows, classes, decode(d, rows))
    save, checks = {}, []
    for src in DATASETS:
        rows_s, classes_s, _ = data[src]
        for cfg in CONFIGS:
            for f in range(5):
                model = BioLiteNet(num_classes=len(classes_s), use_cbam=False, pretrained=False)
                state = torch.load(os.path.join(HERE, "ckpt", src, f"{cfg}_f{f}.pt"), map_location="cpu",
                                   weights_only=True)
                model.load_state_dict(state)
                model.to(device).eval()
                # check: accuracy on the model's own test fold
                own = np.array([int(r["fold"]) == f for r in rows_s])
                y = np.array([classes_s.index(r["class"]) for r in rows_s])[own]
                acc = (predict(model, data[src][2][own], device).argmax(1) == y).mean() * 100
                checks.append({"source": src, "config": cfg, "fold": f, "own_fold_acc": round(float(acc), 2)})
                print(f"[check] {src} {cfg} f{f}: own test fold accuracy {acc:.2f}%", flush=True)
                for tgt in TARGETS[src]:
                    save[f"{src}__{tgt}__{cfg}__f{f}"] = predict(model, data[tgt][2], device)
    for d in DATASETS:
        save[f"labels__{d}"] = np.array([r["class"] for r in data[d][0]])
        save[f"classes__{d}"] = np.array(data[d][1])
    np.savez_compressed("/kaggle/working/xdata_probs.npz", **save)
    json.dump(checks, open("/kaggle/working/xdata_checks.json", "w"), indent=1)
    print("wrote /kaggle/working/xdata_probs.npz and xdata_checks.json")


if __name__ == "__main__":
    main()
