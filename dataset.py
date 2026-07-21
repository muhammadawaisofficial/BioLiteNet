import os, zipfile, shutil, random, json
from config import (DATASET_ZIP, DATASET_DIR, OUTPUT_DIR,
                    USE_BACKGROUND_REMOVED, TRAIN_RATIO, VAL_RATIO,
                    SEED, HAZARD_WEIGHTS)

random.seed(SEED)


def extract_dataset():
    if os.path.exists(DATASET_DIR) and os.listdir(DATASET_DIR):
        print("[INFO] Dataset already extracted at:", DATASET_DIR)
        return
    print("[INFO] Extracting dataset...")
    os.makedirs(DATASET_DIR, exist_ok=True)
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zf:
        zf.extractall(DATASET_DIR)
    print("[INFO] Done.")


def get_class_root():
    subfolder = "Baackground_Removed" if USE_BACKGROUND_REMOVED else "Raw"
    root = os.path.join(DATASET_DIR, "Dataset", subfolder)
    if not os.path.isdir(root):
        parent = os.path.join(DATASET_DIR, "Dataset")
        for d in os.listdir(parent):
            if d.lower().replace(" ", "") == subfolder.lower().replace(" ", ""):
                root = os.path.join(parent, d)
                break
    return root


def build_splits():
    root    = get_class_root()
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    print(f"[INFO] Found {len(classes)} classes")

    split_root = os.path.join(OUTPUT_DIR, "splits")
    if os.path.exists(split_root) and os.listdir(split_root):
        print("[INFO] Splits already exist at:", split_root)
        return split_root, classes

    stats = {}
    for cls in classes:
        cls_dir = os.path.join(root, cls)
        images  = [f for f in os.listdir(cls_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        random.shuffle(images)

        n       = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        splits  = {
            "train": images[:n_train],
            "val":   images[n_train: n_train + n_val],
            "test":  images[n_train + n_val:]
        }
        stats[cls] = {s: len(imgs) for s, imgs in splits.items()}

        for split, imgs in splits.items():
            dst_dir = os.path.join(split_root, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for img in imgs:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(dst_dir, img)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "split_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'Class':<40} {'Train':>6} {'Val':>6} {'Test':>6} {'h_c':>5}")
    print("-" * 65)
    for cls in classes:
        hc = HAZARD_WEIGHTS.get(cls, 1.0)
        s  = stats[cls]
        print(f"{cls:<40} {s['train']:>6} {s['val']:>6} {s['test']:>6} {hc:>5.1f}")

    return split_root, classes


if __name__ == "__main__":
    extract_dataset()
    split_root, classes = build_splits()
    print("\nDataset ready at:", split_root)
    print("Total classes:", len(classes))
