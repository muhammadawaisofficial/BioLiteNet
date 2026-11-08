"""
Build the duplicate-grouped, class-stratified 5-fold manifest.

The Kaggle BMW images contain exact copies and augmented near-copies (colour
shift, flips, rotations) of the same photograph. Copies are linked into one
group (exact MD5 match, or cosine similarity of flip/rotation/colour-invariant
embeddings >= threshold) and a group never spans two folds.

    python build_folds.py emb.npz --calibrate pairs.png   # inspect pairs per similarity band
    python build_folds.py dedup_embeddings.npz --threshold 0.985 --out manifest.csv
    python build_folds.py mw_embeddings.npz --threshold 0.995 --stem --out manifest_mw.csv
    python build_folds.py imw_embeddings.npz --threshold 0.95 --merge SoB=SDB --drop-conflicts 0.99 --out manifest_imw.csv

--stem also links files that share a name stem within a class: Roboflow copies
(IMG_1_JPG.rf.<hash>.jpg) and stereo frames (<id>.jpeg, <id>_l.png, <id>_r.png).
"""
import argparse, csv, collections, os, re
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS, SEED = 5, 42


def stem_key(f):
    s = os.path.splitext(f)[0].split(".rf.")[0]
    s = re.sub(r"_(jpe?g|png)$", "", s, flags=re.I)
    return re.sub(r"_[lr]$", "", s).lower()


def union_find(n):
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a, b):
        parent[find(a)] = find(b)
    return find, union


def near_pairs(emb, cls, threshold):
    """(i, j, sim) for same-class pairs with cosine similarity >= threshold."""
    out = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        sim = emb[idx] @ emb[idx].T
        np.fill_diagonal(sim, -1)
        a, b = np.where(np.triu(sim >= threshold, 1))
        out += [(idx[i], idx[j], sim[i, j]) for i, j in zip(a, b)]
    return out


def calibrate(emb, cls, meta, zip_path, out_png):
    import io, zipfile
    from PIL import Image, ImageDraw
    z = zipfile.ZipFile(zip_path)
    rng = np.random.default_rng(0)
    pairs = near_pairs(emb, cls, 0.90)
    bands = [(0.99, 1.01), (0.98, 0.99), (0.97, 0.98), (0.96, 0.97), (0.95, 0.96), (0.93, 0.95), (0.90, 0.93)]
    sheet = Image.new("RGB", (6 * 2 * 130, len(bands) * 150), "white")
    d = ImageDraw.Draw(sheet)
    for r, (lo, hi) in enumerate(bands):
        band = [p for p in pairs if lo <= p[2] < hi]
        print(f"similarity [{lo:.2f},{hi:.2f}): {len(band)} pairs")
        for k, p in enumerate(rng.permutation(len(band))[:6] if band else []):
            i, j, s = band[p]
            for t, q in enumerate((i, j)):
                sp, c, f = meta[q]
                im = Image.open(io.BytesIO(z.read(f"outputs/splits/{sp}/{c}/{f}"))).convert("RGB").resize((125, 125))
                sheet.paste(im, ((2 * k + t) * 130, r * 150 + 20))
            d.text((2 * k * 130, r * 150 + 4), f"{s:.3f}", fill="red")
    sheet.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emb")
    ap.add_argument("--threshold", type=float, default=0.97)
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--calibrate")
    ap.add_argument("--zip")
    ap.add_argument("--stem", action="store_true")
    ap.add_argument("--merge", default="", help="OLD=NEW[,OLD=NEW] class relabelling")
    ap.add_argument("--drop-conflicts", type=float, help="drop images with a different-label copy at sim >= this")
    a = ap.parse_args()

    data = np.load(a.emb)
    emb, md5, meta = data["emb"].astype(np.float32), data["md5"], data["meta"].copy()
    for pair in filter(None, a.merge.split(",")):
        old, new = pair.split("=")
        meta[meta[:, 1] == old, 1] = new
    if a.drop_conflicts:
        cls = meta[:, 1]
        i, j = np.where(np.triu(emb @ emb.T >= a.drop_conflicts, 1))
        bad = np.zeros(len(cls), bool)
        bad[i[cls[i] != cls[j]]] = bad[j[cls[i] != cls[j]]] = True
        print(f"dropped {bad.sum()} images with a different-label copy (sim >= {a.drop_conflicts}): "
              f"{dict(collections.Counter(cls[bad]))}")
        emb, md5, meta = emb[~bad], md5[~bad], meta[~bad]
    split, cls, fn = meta[:, 0], meta[:, 1], meta[:, 2]
    if a.calibrate:
        return calibrate(emb, cls, meta, a.zip, a.calibrate)

    find, union = union_find(len(md5))
    by_md5 = collections.defaultdict(list)
    for i, h in enumerate(md5):
        by_md5[h].append(i)
    cross_class_exact = 0
    for ids in by_md5.values():
        if len({cls[i] for i in ids}) > 1:
            cross_class_exact += 1
        for i in ids[1:]:
            union(ids[0], i)
    if a.stem:
        first = {}
        for i in range(len(fn)):
            first.setdefault((cls[i], stem_key(fn[i])), i)
            union(i, first[(cls[i], stem_key(fn[i]))])
    pairs = near_pairs(emb, cls, a.threshold)
    for i, j, _ in pairs:
        union(i, j)
    groups = np.array([find(i) for i in range(len(md5))])

    sizes = collections.Counter(groups)
    multi = [g for g, s in sizes.items() if s > 1]
    print(f"images {len(md5)} | exact-duplicate sets {sum(len(v) > 1 for v in by_md5.values())} "
          f"(cross-class {cross_class_exact}) | near pairs >= {a.threshold}: {len(pairs)}")
    print(f"groups {len(sizes)} | groups with copies {len(multi)} covering {sum(sizes[g] for g in multi)} images")
    leak = sum(1 for i in range(len(md5)) if split[i] == "test" and any(
        split[j] == "train" for j in np.where(groups == groups[i])[0]))
    print(f"original split: test images with a copy in train = {leak} / {(split == 'test').sum()}")

    y = np.unique(cls, return_inverse=True)[1]
    fold = np.full(len(y), -1)
    for k, (_, te) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(y, y, groups)):
        fold[te] = k
    val = np.zeros((N_FOLDS, len(y)), bool)
    for k in range(N_FOLDS):
        rest = np.where(fold != k)[0]
        _, va = next(StratifiedGroupKFold(6, shuffle=True, random_state=SEED + k).split(rest, y[rest], groups[rest]))
        val[k, rest[va]] = True

    for k in range(N_FOLDS):
        te, va = fold == k, val[k]
        assert not set(groups[te]) & set(groups[~te]), "group leaks across folds"
        assert not set(groups[va]) & set(groups[(fold != k) & ~va]), "group leaks train/val"
        print(f"fold {k}: train {((fold != k) & ~va).sum()} val {va.sum()} test {te.sum()}")

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "file", "group", "fold"] + [f"val_f{k}" for k in range(N_FOLDS)] + ["orig_split"])
        for i in np.lexsort((fn, cls)):
            w.writerow([cls[i], fn[i], int(groups[i]), int(fold[i])] + [int(val[k, i]) for k in range(N_FOLDS)] + [split[i]])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
