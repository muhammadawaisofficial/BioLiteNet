"""
BioLiteNet camera-ready experiments: one script, one protocol, every model.

Runs every (config, fold) job of a 5-fold, duplicate-grouped, class-stratified
cross-validation on the background-removed Kaggle BMW images. All configs share
the same folds, seeds, augmentation, optimiser and schedule, so results are
paired and can be compared with significance tests.

Kaggle usage (GPU T4 x2):
    !python /kaggle/input/<your-code-dataset>/run_bln.py --probe     # dataset check + timing
    !python /kaggle/input/<your-code-dataset>/run_bln.py             # all jobs, one worker per GPU
Finished jobs are skipped, including jobs found in attached earlier outputs,
so a run cut off by the 12 h session limit can simply be resumed.
"""
import argparse, csv, glob, json, os, subprocess, sys, time
import numpy as np

# ── Protocol (identical for every config) ────────────────────────────────────
NUM_EPOCHS   = 30
BATCH_SIZE   = 32
LR           = 1e-3
LR_MIN       = 1e-6
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.4
GAMMA        = 2.0
CLIP_NORM    = 1.0
BASE_SEED    = 42
N_FOLDS      = 5
CBAM_STAGES  = [3, 6, 12]

SHARPS = {"used_syringes", "scalpels", "episiotomy_scissors", "mayo_scissors",
          "stitch_removal_scissors", "tweezers", "forceps", "hemostats"}

# Hazard weights h_c used in the paper (Table III)
HAZARD_PAPER = {
    "used_syringes": 3.0, "scalpels": 3.0, "episiotomy_scissors": 3.0,
    "mayo_scissors": 3.0, "stitch_removal_scissors": 3.0,
    "tweezers": 2.8, "forceps": 2.8, "hemostats": 2.8,
    "human_organs": 2.5, "blood_soaked_bandages": 2.5,
    "disinfectant_bottles": 2.0, "ampoules_full": 2.0, "ampuoles_broken": 2.2,
    "expired_tablets": 1.5, "vaccine_or_medicine_vials": 1.5,
    "syrup_bottles": 1.5, "iv_bottles": 1.5,
    "used_masks": 1.2, "used_medical_gloves": 1.2, "uncategorized_or_overlapping": 1.3,
    "used_medical_paper": 1.0, "general_organic_waste": 1.0, "waterbottles": 1.0,
}
# Sensitivity scheme: same tier order, steeper scale (tier weights 5/4/3/2/1)
TIER = {**{c: 5 for c in SHARPS},
        "human_organs": 4, "blood_soaked_bandages": 4,
        "disinfectant_bottles": 3, "ampoules_full": 3, "ampuoles_broken": 3,
        "expired_tablets": 2, "vaccine_or_medicine_vials": 2, "syrup_bottles": 2, "iv_bottles": 2,
        "used_masks": 1, "used_medical_gloves": 1, "uncategorized_or_overlapping": 1,
        "used_medical_paper": 1, "general_organic_waste": 1, "waterbottles": 1}
HAZARD_STEEP = {c: float(t) for c, t in TIER.items()}

# External dataset (Kaggle sumn2u/medical-waste), selected with BLN_DATA=mw.
# Same WHO tiers and the same weight per tier as above.
DATA = os.environ.get("BLN_DATA", "bmw")
if DATA == "mw":
    _body = ["body_tissue_or_organ", "gauze", "urine_bag", "test_tube"]           # pathological / body fluid
    _ppe = ["glove_pair_latex", "glove_pair_nitrile", "glove_pair_surgery", "glove_single_latex",
            "glove_single_nitrile", "glove_single_surgery", "mask", "medical_cap", "medical_glasses",
            "shoe_cover_pair", "shoe_cover_single", "syringe"]                     # needle-free soft/plastic items
    _general = ["glass_equipment_packaging", "metal_equipment_packaging", "paper_equipment_packaging",
                "plastic_equipment_packaging", "organic_waste"]
    SHARPS = {"syringe_needle", "tweezers"}
    TIER = {"syringe_needle": 5, "tweezers": 5, **{c: 4 for c in _body}, **{c: 1 for c in _ppe + _general}}
    HAZARD_PAPER = {"syringe_needle": 3.0, "tweezers": 2.8, **{c: 2.5 for c in _body},
                    **{c: 1.2 for c in _ppe}, **{c: 1.0 for c in _general}}
    HAZARD_STEEP = {c: float(t) for c, t in TIER.items()}
if DATA == "imw":   # Kaggle engineeringubu/infectious-municipal-waste; labels are the folder codes
    _body = ["PW", "BBP", "CAW", "MW", "CLW", "DW", "SoB", "SDB"]   # tissue, blood, cultures, soiled items
    SHARPS = {"SW"}
    TIER = {"SW": 5, **{c: 4 for c in _body}, "PhW": 2, "PPE": 1, "CBL": 1}
    HAZARD_PAPER = {"SW": 3.0, **{c: 2.5 for c in _body}, "PhW": 1.5, "PPE": 1.2, "CBL": 1.2}
    HAZARD_STEEP = {c: float(t) for c, t in TIER.items()}

# name: (backbone, use_cbam, loss, hazard scheme)
CONFIGS = {
    "mnv3_ce":    ("mnv3", False, "ce",   None),
    "mnv3_wce":   ("mnv3", False, "wce",  HAZARD_PAPER),
    "mnv3_fl":    ("mnv3", False, "fl",   None),
    "mnv3_hafl":  ("mnv3", False, "hafl", HAZARD_PAPER),
    "cbam_fl":    ("mnv3", True,  "fl",   None),
    "cbam_hafl":  ("mnv3", True,  "hafl", HAZARD_PAPER),   # attention variant
    "r50_ce":     ("r50",  False, "ce",   None),
    "mnv3_hafl_steep": ("mnv3", False, "hafl", HAZARD_STEEP),  # optional sensitivity
}
CORE = ["mnv3_ce", "mnv3_wce", "mnv3_fl", "mnv3_hafl", "cbam_fl", "cbam_hafl", "r50_ce"]
if DATA in ("mw", "imw"):
    CORE = ["mnv3_ce", "mnv3_wce", "mnv3_fl", "mnv3_hafl", "cbam_hafl"]
SUFFIX = "" if DATA == "bmw" else "_" + DATA
MANIFEST = f"manifest{SUFFIX}.csv"
OUT_ROOT = ("/kaggle/working/results" if os.path.isdir("/kaggle/working") else "results") + SUFFIX


# ── Data ─────────────────────────────────────────────────────────────────────
def find_file(name):
    for root in ("/kaggle/input", os.path.dirname(os.path.abspath(__file__)), "."):
        hits = glob.glob(os.path.join(root, "**", name), recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


def find_dir(name):
    for root, dirs, _ in os.walk("/kaggle/input" if os.path.isdir("/kaggle/input") else "."):
        if name in dirs:
            return os.path.join(root, name)
    return None


def img_root():
    if DATA in ("mw", "imw"):
        marker = "syringe_needle" if DATA == "mw" else "Sharps Waste (SW)"
        for root, dirs, _ in os.walk("/kaggle/input" if os.path.isdir("/kaggle/input") else "."):
            if marker in dirs:
                return root
        return None
    return find_dir("Baackground_Removed")


def img_path(root, r):
    # imw manifests store the path relative to the dataset root; the others store class/file
    return os.path.join(root, r["file"]) if DATA == "imw" else os.path.join(root, r["class"], r["file"])


def load_manifest():
    with open(find_file(MANIFEST), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    classes = sorted({r["class"] for r in rows})
    return rows, classes


def load_cache(rows, img_root):
    """Decode every image once: 256x256 copy for training, 224x224 for evaluation."""
    from PIL import Image
    n = len(rows)
    big = np.empty((n, 256, 256, 3), np.uint8)
    small = np.empty((n, 224, 224, 3), np.uint8)
    t0 = time.time()
    for i, r in enumerate(rows):
        im = Image.open(img_path(img_root, r))
        if DATA == "imw":            # phone photos up to 4032x3024: decode JPEGs at reduced scale (>= 512 px)
            im.draft("RGB", (512, 512))
        im = im.convert("RGB")
        big[i] = np.asarray(im.resize((256, 256), Image.BILINEAR))
        small[i] = np.asarray(im.resize((224, 224), Image.BILINEAR))
    print(f"[DATA] cached {n} images in {time.time()-t0:.0f}s", flush=True)
    return big, small


def make_loaders(big, small, labels, tr_idx, va_idx, te_idx, seed):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision.transforms import v2 as T

    norm = T.Compose([T.ToDtype(torch.float32, scale=True),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_tf = T.Compose([T.RandomCrop(224), T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.2),
                          T.ColorJitter(0.3, 0.3, 0.2, 0.05), T.RandomRotation(15),
                          T.RandomGrayscale(0.05), norm])

    class DS(Dataset):
        def __init__(self, arr, idx, tf):
            self.arr, self.idx, self.tf = arr, idx, tf
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, k):
            i = self.idx[k]
            x = torch.from_numpy(self.arr[i]).permute(2, 0, 1)
            return self.tf(x), int(labels[i]), i

    g = torch.Generator().manual_seed(seed)
    kw = dict(batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, persistent_workers=True)
    return (DataLoader(DS(big, tr_idx, train_tf), shuffle=True, drop_last=True, generator=g, **kw),
            DataLoader(DS(small, va_idx, norm), shuffle=False, **kw),
            DataLoader(DS(small, te_idx, norm), shuffle=False, **kw))


# ── Model ────────────────────────────────────────────────────────────────────
def build_model(backbone, use_cbam, num_classes):
    import torch.nn as nn
    import torchvision.models as tvm
    if backbone == "r50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(2048, num_classes))
        return m
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [here, os.path.dirname(here)]   # Kaggle upload (flat) or repo layout
    from model import BioLiteNet
    return BioLiteNet(num_classes=num_classes, dropout_rate=DROPOUT, use_cbam=use_cbam,
                      cbam_stages=CBAM_STAGES, pretrained=True)


def make_loss(kind, classes, scheme):
    import torch
    import torch.nn.functional as F
    h = torch.tensor([scheme[c] for c in classes] if scheme else [1.0] * len(classes))

    def loss_fn(logits, y):
        logp = F.log_softmax(logits.float(), dim=1)
        logp_t = logp.gather(1, y[:, None]).squeeze(1)
        w = h.to(y.device)[y] if kind in ("wce", "hafl") else 1.0
        mod = (1 - logp_t.exp()) ** GAMMA if kind in ("fl", "hafl") else 1.0
        return (-w * mod * logp_t).mean()
    return loss_fn


# ── One job ──────────────────────────────────────────────────────────────────
def run_job(cfg_name, fold, rows, classes, big, small, device, epochs=NUM_EPOCHS):
    import torch
    from sklearn.metrics import f1_score
    backbone, use_cbam, loss_kind, scheme = CONFIGS[cfg_name]
    seed = BASE_SEED + fold
    torch.manual_seed(seed); np.random.seed(seed)

    labels = np.array([classes.index(r["class"]) for r in rows])
    fold_col = np.array([int(r["fold"]) for r in rows])
    val_col = np.array([r[f"val_f{fold}"] == "1" for r in rows])
    te_idx = np.where(fold_col == fold)[0]
    va_idx = np.where((fold_col != fold) & val_col)[0]
    tr_idx = np.where((fold_col != fold) & ~val_col)[0]
    tr_dl, va_dl, te_dl = make_loaders(big, small, labels, tr_idx, va_idx, te_idx, seed)

    model = build_model(backbone, use_cbam, len(classes)).to(device)
    criterion = make_loss(loss_kind, classes, scheme)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=LR_MIN)
    scaler = torch.amp.GradScaler("cuda")
    sharp_idx = [classes.index(c) for c in SHARPS]

    def predict(dl):
        model.eval(); P, Y, I = [], [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for x, y, i in dl:
                P.append(torch.softmax(model(x.to(device, non_blocking=True)).float(), 1).cpu())
                Y.append(y); I.append(i)
        return torch.cat(P).numpy(), torch.cat(Y).numpy(), torch.cat(I).numpy()

    out = os.path.join(OUT_ROOT, f"{cfg_name}_f{fold}")
    os.makedirs(out, exist_ok=True)
    log, best_f1, best_ep, best_state = [], -1.0, 0, None
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); tl, n = 0.0, 0
        for x, y, _ in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(opt); scaler.update()
            tl += loss.item() * len(y); n += len(y)
        sched.step()
        p, y, _ = predict(va_dl); yhat = p.argmax(1)
        vf1 = f1_score(y, yhat, average="macro")
        log.append([ep, tl / n, (yhat == y).mean(), vf1, time.time() - t0])
        print(f"[{cfg_name} f{fold}] ep {ep:02d} loss {tl/n:.4f} val_acc {(yhat==y).mean():.4f} "
              f"val_f1 {vf1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vf1 > best_f1:
            best_f1, best_ep = vf1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    p, y, idx = predict(te_dl); yhat = p.argmax(1)
    f1c = f1_score(y, yhat, average=None, labels=range(len(classes)))
    metrics = {"data": DATA, "config": cfg_name, "fold": fold, "seed": seed, "best_epoch": best_ep,
               "val_macro_f1": best_f1, "test_acc": float((yhat == y).mean()),
               "test_macro_f1": float(f1c.mean()), "test_sharps_f1": float(f1c[sharp_idx].mean()),
               "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx),
               "params": sum(q.numel() for q in model.parameters()),
               "sec_per_epoch": float(np.mean([r[-1] for r in log]))}
    np.savez_compressed(os.path.join(out, "test_preds.npz"), idx=idx, y=y, prob=p.astype(np.float16),
                        classes=np.array(classes))

    # Domain shift: same test-fold photos, but the Raw version with original backgrounds
    raw_dir = find_dir("Raw") if DATA == "bmw" else None
    if raw_dir:
        from PIL import Image
        raw_arr = np.empty((len(te_idx), 224, 224, 3), np.uint8)
        for k, i in enumerate(te_idx):
            r = rows[i]
            hits = glob.glob(os.path.join(raw_dir, r["class"], os.path.splitext(r["file"])[0] + ".*"))
            raw_arr[k] = np.asarray(Image.open(hits[0]).convert("RGB").resize((224, 224), Image.BILINEAR))
        lookup = np.full(len(rows), -1); lookup[te_idx] = np.arange(len(te_idx))
        raw_labels = labels.copy()
        from torch.utils.data import DataLoader, Dataset
        from torchvision.transforms import v2 as T
        norm = T.Compose([T.ToDtype(torch.float32, scale=True),
                          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        class RawDS(Dataset):
            def __len__(self): return len(te_idx)
            def __getitem__(self, k):
                return norm(torch.from_numpy(raw_arr[k]).permute(2, 0, 1)), int(raw_labels[te_idx[k]]), int(te_idx[k])
        rp, ry, ridx = predict(DataLoader(RawDS(), batch_size=BATCH_SIZE, num_workers=2))
        np.savez_compressed(os.path.join(out, "raw_preds.npz"), idx=ridx, y=ry, prob=rp.astype(np.float16))
        rf1 = f1_score(ry, rp.argmax(1), average="macro")
        print(f"[RAW] {cfg_name} f{fold}: acc {(rp.argmax(1)==ry).mean():.4f} macroF1 {rf1:.4f}", flush=True)
    torch.save(best_state, os.path.join(out, "best.pt"))
    with open(os.path.join(out, "log.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["epoch", "train_loss", "val_acc", "val_macro_f1", "sec"]] + log)
    with open(os.path.join(out, "metrics.json"), "w") as f:  # written last = job complete
        json.dump(metrics, f, indent=2)
    print(f"[DONE] {cfg_name} f{fold}: acc {metrics['test_acc']:.4f} "
          f"macroF1 {metrics['test_macro_f1']:.4f} sharpsF1 {metrics['test_sharps_f1']:.4f}", flush=True)


# ── Orchestration ────────────────────────────────────────────────────────────
def done_jobs():
    found = set()
    for p in glob.glob("/kaggle/input/**/metrics.json", recursive=True) + \
             glob.glob(os.path.join(OUT_ROOT, "*", "metrics.json")):
        m = json.load(open(p))
        if m.get("data", "bmw") == DATA:
            found.add((m["config"], m["fold"]))
    return found


def worker(gpu, shard, nshards, configs, folds):
    import torch
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    rows, classes = load_manifest()
    big, small = load_cache(rows, img_root())
    jobs = [(c, f) for f in folds for c in configs]                  # fold-major: early folds finish first
    mine = [j for k, j in enumerate(jobs) if k % nshards == shard]
    for c, f in mine:
        if (c, f) in done_jobs():
            print(f"[SKIP] {c} f{f}", flush=True); continue
        run_job(c, f, rows, classes, big, small, device)


def probe():
    """Check data, manifest, Raw-folder overlap, and time two epochs of BioLiteNet."""
    import torch
    rows, classes = load_manifest()
    br, raw = img_root(), (find_dir("Raw") if DATA == "bmw" else None)
    print("[PROBE] background-removed:", br, "| raw:", raw, "| GPUs:", torch.cuda.device_count())
    missing = [r for r in rows if not os.path.exists(img_path(br, r))]
    print(f"[PROBE] manifest rows {len(rows)}, classes {len(classes)}, missing files {len(missing)}")
    if missing:
        print("  e.g.", missing[:5]); return
    if raw:
        stem = lambda f: os.path.splitext(f)[0]
        for c in classes[:4]:
            rf = os.listdir(os.path.join(raw, c)) if os.path.isdir(os.path.join(raw, c)) else []
            bf = {stem(r["file"]) for r in rows if r["class"] == c}
            print(f"  raw/{c}: {len(rf)} files, stems shared with BR: {len({stem(x) for x in rf} & bf)}"
                  f"  sample: {rf[:3]}")
    big, small = load_cache(rows, br)
    t0 = time.time()
    run_job("cbam_hafl", 0, rows, classes, big, small, torch.device("cuda:0"), epochs=2)
    print(f"[PROBE] 2-epoch BioLiteNet job took {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--configs", default=",".join(CORE))
    ap.add_argument("--folds", default=",".join(map(str, range(N_FOLDS))))
    a = ap.parse_args()
    configs = a.configs.split(",")
    folds = [int(f) for f in a.folds.split(",")]
    if a.probe:
        global OUT_ROOT
        OUT_ROOT = OUT_ROOT + "_probe"
        return probe()
    if a.worker:
        return worker(a.gpu, a.shard, a.nshards, configs, folds)

    import torch
    n = max(torch.cuda.device_count(), 1)
    os.makedirs(OUT_ROOT, exist_ok=True)
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", "--gpu", str(g),
                               "--shard", str(g), "--nshards", str(n), "--configs", a.configs,
                               "--folds", a.folds],
                              stdout=open(os.path.join(OUT_ROOT, f"worker{g}.log"), "w"),
                              stderr=subprocess.STDOUT) for g in range(n)]
    print(f"[MAIN] {n} workers started; logs in {OUT_ROOT}/worker*.log", flush=True)
    for p in procs:
        p.wait()
    print("[MAIN] finished:", sorted(done_jobs()))


if __name__ == "__main__":
    main()
