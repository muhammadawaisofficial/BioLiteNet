"""
Prepare an external dataset for duplicate grouping.
    BLN_DATA=mw  (default)  Kaggle sumn2u/medical-waste
    BLN_DATA=imw            Kaggle engineeringubu/infectious-municipal-waste

Kaggle (attach the dataset + bln-code, GPU on):
    !BLN_DATA=imw python /kaggle/input/<code-dataset>/prep_mw.py
Prints file-name / image-format samples per class and writes to /kaggle/working:
    <data>_embeddings.npz   MD5 + flip/rotation-invariant grayscale embeddings (input to build_folds.py)
    <data>_pairs.png        sample same-class pairs per similarity band, to choose the grouping threshold
"""
import collections, hashlib, os, re, sys
import numpy as np, torch, torchvision.models as tvm
from PIL import Image, ImageDraw

DATA = os.environ.get("BLN_DATA", "mw")
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")       # .heic is skipped (PIL cannot decode it)
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
MARKER = {"mw": "syringe_needle", "imw": "Sharps Waste (SW)"}[DATA]


def find_root():
    for root, dirs, _ in os.walk("/kaggle/input"):
        if MARKER in dirs:
            return root
    sys.exit(f"{DATA} dataset is not attached")


root = find_root()
# items: (original split, class label, file as stored in the manifest; relative to root for imw)
items = []
for d in sorted(os.listdir(root)):
    if not os.path.isdir(os.path.join(root, d)):
        continue
    if DATA == "mw":
        items += [("all", d, f) for f in sorted(os.listdir(os.path.join(root, d))) if f.lower().endswith(EXT)]
    else:
        label = re.search(r"\((\w+)\)", d).group(1)
        for sub, split in (("Train Images", "train"), ("Test Images", "test")):
            p = os.path.join(root, d, sub)
            items += [(split, label, f"{d}/{sub}/{f}") for f in sorted(os.listdir(p)) if f.lower().endswith(EXT)]
path = lambda it: os.path.join(root, it[2]) if DATA == "imw" else os.path.join(root, it[1], it[2])
classes = sorted({it[1] for it in items})
print(f"{root}: {len(classes)} classes, {len(items)} images")

for c in classes:
    by_ext = collections.defaultdict(list)
    for it in items:
        if it[1] == c:
            by_ext[(it[0], os.path.splitext(it[2])[1].lower())].append(it)
    print(f"\n{c}")
    for (sp, e), its in by_ext.items():
        im = Image.open(path(its[0]))
        names = [os.path.basename(it[2]) for it in its]
        print(f"  {sp} {e} {len(its)}  mode {im.mode} size {im.size}  e.g. {names[:3]} ... {names[-2:]}")

# Grayscale, whole-image, 8 dihedral views averaged -> invariant to flips, rotations and colour.
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1).eval().to(dev)
feat = torch.nn.Sequential(net.features, net.avgpool, torch.nn.Flatten())
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def d8(x):
    v = [x, x.flip(-1)]
    for k in (1, 2, 3):
        v += [torch.rot90(x, k, (-2, -1)), torch.rot90(x, k, (-2, -1)).flip(-1)]
    return torch.cat(v)


embs, md5s, buf = [], [], []
def flush():
    x = torch.stack(buf).to(dev)
    with torch.no_grad():
        f = feat((d8(x) - mean) / std).view(8, len(buf), -1).mean(0)
    embs.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy()); buf.clear()

for i, it in enumerate(items):
    p = path(it)
    md5s.append(hashlib.md5(open(p, "rb").read()).hexdigest())
    im = Image.open(p).convert("L").resize((160, 160)).convert("RGB")
    buf.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1))
    if len(buf) == 64:
        flush()
    if i % 2000 == 0:
        print("embedded", i, flush=True)
if buf:
    flush()
emb = np.concatenate(embs)
meta = np.array(items)
np.savez(os.path.join(OUT, f"{DATA}_embeddings.npz"), emb=emb, md5=np.array(md5s), meta=meta)

cls = meta[:, 1]
by_md5 = collections.defaultdict(set)
for h, c in zip(md5s, cls):
    by_md5[h].add(c)
dup = collections.Counter(md5s)
print(f"\nexact-duplicate sets: {sum(n > 1 for n in dup.values())}, "
      f"cross-class: {[sorted(v) for h, v in by_md5.items() if len(v) > 1][:20]}")

# Contact sheet of same-class pairs per similarity band
bands = [(0.99, 1.01), (0.98, 0.99), (0.97, 0.98), (0.96, 0.97), (0.95, 0.96), (0.93, 0.95), (0.90, 0.93)]
pairs = []
for c in classes:
    idx = np.where(cls == c)[0]
    s = emb[idx] @ emb[idx].T
    a, b = np.where(np.triu(s >= 0.90, 1))
    pairs += [(idx[i], idx[j], s[i, j]) for i, j in zip(a, b)]
rng = np.random.default_rng(0)
sheet = Image.new("RGB", (12 * 130, len(bands) * 150), "white"); d = ImageDraw.Draw(sheet)
for r, (lo, hi) in enumerate(bands):
    band = [p for p in pairs if lo <= p[2] < hi]
    print(f"similarity [{lo:.2f},{hi:.2f}): {len(band)} same-class pairs")
    for k, q in enumerate(rng.permutation(len(band))[:6] if band else []):
        i, j, s = band[q]
        for t, n in enumerate((i, j)):
            sheet.paste(Image.open(path(items[n])).convert("RGB").resize((125, 125)),
                        ((2 * k + t) * 130, r * 150 + 20))
        name = lambda n: os.path.basename(items[n][2])[:14]
        d.text((2 * k * 130, r * 150 + 4), f"{s:.3f} {name(i)} | {name(j)}", fill="red")
sheet.save(os.path.join(OUT, f"{DATA}_pairs.png"))
print(f"wrote {DATA}_embeddings.npz and {DATA}_pairs.png")
