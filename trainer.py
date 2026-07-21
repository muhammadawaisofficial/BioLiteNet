import os, time, json, csv, random, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from config import (
    OUTPUT_DIR, IMG_SIZE, BATCH_SIZE, NUM_EPOCHS, LR, LR_MIN,
    WEIGHT_DECAY, DROPOUT_RATE, NUM_WORKERS, SEED, LOG_INTERVAL,
    FOCAL_GAMMA, HAZARD_WEIGHTS, CBAM_STAGES, CBAM_REDUCTION,
    USE_CBAM, USE_HAFL,
)
from model import BioLiteNet
from loss  import build_hafl, FocalLoss

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# classes considered sharps - used to compute sharps-F1 separately
SHARP_CLASSES = {
    "used_syringes", "scalpels", "episiotomy_scissors",
    "mayo_scissors", "stitch_removal_scissors",
    "tweezers", "forceps", "hemostats"
}


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=15),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def get_dataloaders(split_root):
    train_tf, val_tf = get_transforms()
    train_ds = datasets.ImageFolder(os.path.join(split_root, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(split_root, "val"),   transform=val_tf)
    test_ds  = datasets.ImageFolder(os.path.join(split_root, "test"),  transform=val_tf)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_dl, val_dl, test_dl, train_ds.classes


def build_criterion(class_names, use_hafl, num_classes):
    if use_hafl:
        return build_hafl(class_names, HAZARD_WEIGHTS, gamma=FOCAL_GAMMA)
    print("[INFO] Using standard focal loss (no HAFL)")
    return FocalLoss(num_classes=num_classes, gamma=FOCAL_GAMMA)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            print(f"  Epoch {epoch:02d} | Batch {batch_idx+1:>4}/{len(loader)} "
                  f"| Loss: {loss.item():.4f} | Acc: {correct/total*100:.1f}%")

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            preds  = logits.argmax(dim=1)

            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss   = total_loss / total
    accuracy   = correct / total
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_cls_f1 = f1_score(all_labels, all_preds, average=None,    zero_division=0)
    f1_dict    = {cls: round(float(per_cls_f1[i]), 4) for i, cls in enumerate(class_names)}

    return avg_loss, accuracy, macro_f1, f1_dict, all_preds, all_labels


def sharps_f1(f1_dict):
    vals = [v for k, v in f1_dict.items() if k in SHARP_CLASSES]
    return round(float(np.mean(vals)), 4) if vals else 0.0


def train(split_root, variant_name="BioLiteNet",
          use_cbam=USE_CBAM, use_hafl=USE_HAFL, resume=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Variant: {variant_name} | Device: {device}")
    print(f"  CBAM: {use_cbam} | HAFL: {use_hafl}")
    print(f"{'='*55}\n")

    train_dl, val_dl, test_dl, class_names = get_dataloaders(split_root)
    num_classes = len(class_names)
    print(f"[INFO] {num_classes} classes | Train: {len(train_dl)} batches "
          f"| Val: {len(val_dl)} | Test: {len(test_dl)}")

    model = BioLiteNet(
        num_classes=num_classes, dropout_rate=DROPOUT_RATE,
        use_cbam=use_cbam, cbam_stages=CBAM_STAGES,
        cbam_reduction=CBAM_REDUCTION, pretrained=True,
    ).to(device)
    print("[MODEL]", model.count_parameters())

    criterion = build_criterion(class_names, use_hafl, num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR_MIN)

    variant_dir = os.path.join(OUTPUT_DIR, variant_name)
    ckpt_dir    = os.path.join(variant_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_ckpt   = os.path.join(ckpt_dir, "best_model.pth")
    last_ckpt   = os.path.join(ckpt_dir, "last_epoch.pth")
    start_epoch = 1
    best_val_f1 = 0.0
    best_epoch  = 0

    # resume from last saved checkpoint if available
    resume_file = last_ckpt if os.path.exists(last_ckpt) else (
                  best_ckpt if os.path.exists(best_ckpt) else None)

    if resume and resume_file:
        print(f"[RESUME] Loading: {resume_file}")
        ckpt = torch.load(resume_file, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            print("[WARN] Optimizer state mismatch, starting optimizer fresh")
        start_epoch = ckpt["epoch"] + 1
        if os.path.exists(best_ckpt):
            best_ckpt_data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            best_val_f1    = best_ckpt_data["val_f1"]
            best_epoch     = best_ckpt_data["epoch"]
        else:
            best_val_f1 = ckpt["val_f1"]
            best_epoch  = ckpt["epoch"]
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"[RESUME] Continuing from epoch {start_epoch} | Best F1 so far: {best_val_f1:.4f}")
    else:
        print("[INFO] Starting fresh training")

    if start_epoch > NUM_EPOCHS:
        print(f"[DONE] All {NUM_EPOCHS} epochs already complete.")
    else:
        csv_path   = os.path.join(variant_dir, "training_log.csv")
        write_mode = "a" if (resume and os.path.exists(csv_path) and
                             os.path.getsize(csv_path) > 0) else "w"
        csv_file   = open(csv_path, write_mode, newline="")
        csv_writer = csv.writer(csv_file)
        if write_mode == "w":
            csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss",
                                  "val_acc", "val_macro_f1", "sharps_f1", "lr"])

        try:
            for epoch in range(start_epoch, NUM_EPOCHS + 1):
                t0 = time.time()
                tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device, epoch)
                vl_loss, vl_acc, vl_f1, f1_dict, _, _ = evaluate(model, val_dl, criterion, device, class_names)
                sf1 = sharps_f1(f1_dict)
                lr  = optimizer.param_groups[0]["lr"]
                scheduler.step()

                print(f"\nEpoch {epoch:02d}/{NUM_EPOCHS} [{time.time()-t0:.0f}s] "
                      f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc*100:.1f}% | "
                      f"ValAcc={vl_acc*100:.1f}% MacroF1={vl_f1:.4f} SharpsF1={sf1:.4f}")

                csv_writer.writerow([epoch, round(tr_loss, 6), round(tr_acc, 6),
                                     round(vl_loss, 6), round(vl_acc, 6),
                                     round(vl_f1, 6), round(sf1, 6), round(lr, 8)])
                csv_file.flush()

                # save checkpoint every epoch for resume support
                torch.save({
                    "epoch": epoch, "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_f1": vl_f1, "sharps_f1": sf1,
                    "class_names": class_names,
                    "config": {"use_cbam": use_cbam, "use_hafl": use_hafl,
                               "num_classes": num_classes},
                }, last_ckpt)

                if vl_f1 > best_val_f1:
                    best_val_f1 = vl_f1
                    best_epoch  = epoch
                    shutil.copy2(last_ckpt, best_ckpt)
                    print(f"  [BEST] Saved (MacroF1={vl_f1:.4f})")

        except KeyboardInterrupt:
            print("\n[STOPPED] Training interrupted. Progress saved.")
        finally:
            csv_file.close()

    if not os.path.exists(best_ckpt):
        print("[WARN] No best checkpoint found.")
        return None

    print(f"\n{'='*55}")
    print(f"  Test Evaluation (best epoch {best_epoch})")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    te_loss, te_acc, te_f1, te_f1_dict, te_preds, te_labels = evaluate(
        model, test_dl, criterion, device, class_names)
    te_sf1 = sharps_f1(te_f1_dict)

    print(f"  Accuracy : {te_acc*100:.2f}%")
    print(f"  Macro-F1 : {te_f1:.4f}")
    print(f"  Sharps-F1: {te_sf1:.4f}")

    print("\n[Per-Class F1]")
    for cls, f1 in sorted(te_f1_dict.items(), key=lambda x: -x[1]):
        tag = " <- SHARP" if cls in SHARP_CLASSES else ""
        print(f"  {cls:<42}: {f1:.4f}{tag}")

    report = classification_report(te_labels, te_preds, target_names=class_names, zero_division=0)
    with open(os.path.join(variant_dir, "test_report.txt"), "w") as f:
        f.write(f"Variant: {variant_name}\n")
        f.write(f"Accuracy : {te_acc*100:.2f}%\n")
        f.write(f"Macro-F1 : {te_f1:.4f}\n")
        f.write(f"SharpsF1 : {te_sf1:.4f}\n\n")
        f.write(report)

    with open(os.path.join(variant_dir, "per_class_f1.json"), "w") as f:
        json.dump({"variant": variant_name, "macro_f1": te_f1,
                   "accuracy": te_acc, "sharps_f1": te_sf1,
                   "per_class": te_f1_dict}, f, indent=2)

    np.save(os.path.join(variant_dir, "confusion_matrix.npy"),
            confusion_matrix(te_labels, te_preds))

    print(f"\n[DONE] Outputs saved to: {variant_dir}")
    print(f"{'='*55}\n")

    return {"variant": variant_name, "accuracy": te_acc,
            "macro_f1": te_f1, "sharps_f1": te_sf1, "per_class_f1": te_f1_dict}
