"""
PCB Anomaly Classification with MobileViT-S + Grad-CAM Heatmap
===============================================================
UPDATED v4.1 — Resume Fix:
  - Replaced OneCycleLR with CosineAnnealingLR (resumes perfectly)
  - Resume from epoch 50, train epochs 51→100
  - All other settings preserved from v4

Dataset: C:/Users/Sastra/OneDrive/Desktop/PCB_mini/stage2_classification/augumented_data
Classes: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper
"""

import os
import warnings
import random
import math
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import timm

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class CFG:
    seed          = 42
    device        = "cuda" if torch.cuda.is_available() else "cpu"
    img_size      = 256
    num_classes   = 6
    batch_size    = 32
    epochs        = 1000     # total target epochs (resumes from 197 → 300)
    lr            = 5e-4
    weight_decay  = 1e-4
    label_smooth  = 0.10
    mixup_alpha   = 0.4
    amp           = torch.cuda.is_available()
    tta_iters     = 5
    val_split     = 0.15
    test_split    = 0.10
    model_name    = "mobilevit_s"
    pretrained    = True
    num_workers   = 4

    # ── PATHS ─────────────────────────────────────────────────────────────────
    data_root = Path(
        r"C:/Users/Sastra/OneDrive/Desktop/PCB_mini"
        r"/stage2_classification/augumented_data"
    )
    output_dir = Path(
        r"C:/Users/Sastra/OneDrive/Desktop/PCB_mini/outputs"
    )

    classes = [
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper",
    ]
    class2idx = {c: i for i, c in enumerate(classes)}

    # ─────────────────────────────────────────────────────────────────────────
    # ★ CONTROLS
    # ─────────────────────────────────────────────────────────────────────────
    SKIP_TRAINING = False

    # Resume from epoch 50 checkpoint
    RESUME            = True
    RESUME_EPOCH      = 895
    RESUME_CHECKPOINT = output_dir / "best_pcb_vit_s.pth"

    SAVE_EVERY_N_EPOCHS = 5


# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed=CFG.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()
CFG.output_dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
def get_transforms(mode="train"):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(CFG.img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.3, hue=0.15),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomAffine(degrees=0, shear=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2),
        ])
    return transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class PCBDataset(Dataset):
    def __init__(self, samples, mode="train"):
        self.samples   = samples
        self.transform = get_transforms(mode)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def load_all_samples(root: Path):
    samples = []
    for cls in CFG.classes:
        cls_dir = root / cls
        if not cls_dir.exists():
            print(f"  WARNING: folder not found → {cls_dir}")
            continue
        found = (list(cls_dir.glob("*.jpg")) +
                 list(cls_dir.glob("*.png")) +
                 list(cls_dir.glob("*.bmp")))
        print(f"  {cls:<20} : {len(found):>4} images")
        for fp in found:
            samples.append((fp, CFG.class2idx[cls]))
    return samples


def split_samples(samples):
    random.shuffle(samples)
    n      = len(samples)
    n_test = int(n * CFG.test_split)
    n_val  = int(n * CFG.val_split)
    return (samples[n_test + n_val:],
            samples[n_test: n_test + n_val],
            samples[:n_test])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
class PCBViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG.model_name, pretrained=CFG.pretrained, num_classes=0)
        dim = self.backbone.num_features
        print(f"  Backbone feature dim : {dim}")
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, CFG.num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────
def get_gradcam_target_layer(model):
    named_candidates = [
        "backbone.stages",
        "backbone.layer4",
        "backbone.features",
    ]
    for attr_path in named_candidates:
        try:
            obj = model
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            last = list(obj.children())[-1] if list(obj.children()) else obj
            for m in reversed(list(last.modules())):
                if isinstance(m, nn.Conv2d):
                    print(f"  ✔ Grad-CAM target: {attr_path}[-1] (Conv2d)")
                    return m
        except (AttributeError, IndexError):
            continue
    last_conv = None
    for _, module in model.backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        print(f"  ✔ Grad-CAM target: last Conv2d in backbone (fallback)")
        return last_conv
    print("  ⚠ No suitable Grad-CAM target layer found.")
    return None


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.grads = self.acts = None
        self._hooks = [
            target_layer.register_forward_hook(
                lambda m, i, o: setattr(
                    self, "acts", o[0] if isinstance(o, tuple) else o)),
            target_layer.register_full_backward_hook(
                lambda m, i, o: setattr(
                    self, "grads", o[0] if isinstance(o, tuple) else o)),
        ]

    def generate(self, x, class_idx=None):
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        self.model.zero_grad()
        oh = torch.zeros_like(logits)
        oh[0, class_idx] = 1.0
        logits.backward(gradient=oh, retain_graph=True)

        if self.grads is None or self.acts is None:
            print("  ⚠ Grad-CAM: no gradients captured.")
            return np.zeros((CFG.img_size, CFG.img_size)), class_idx

        g = self.grads.detach()
        a = self.acts.detach()

        if g.dim() == 4:
            cam = (g.mean((2, 3), keepdim=True) * a).sum(1)[0].cpu().numpy()
        elif g.dim() == 3:
            g, a = g[:, 1:, :], a[:, 1:, :]
            cam  = (g.mean(-1, keepdim=True) * a).sum(-1)[0].cpu().numpy()
            h    = int(math.sqrt(cam.shape[0]))
            cam  = cam.reshape(h, h)
        else:
            return np.zeros((CFG.img_size, CFG.img_size)), class_idx

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (CFG.img_size, CFG.img_size))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    def remove(self):
        for h in self._hooks:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────
class LabelSmoothCE(nn.Module):
    def __init__(self, eps=CFG.label_smooth):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        n  = logits.size(-1)
        lp = F.log_softmax(logits, -1)
        s  = torch.full_like(lp, self.eps / (n - 1))
        s.scatter_(-1, targets.unsqueeze(1), 1.0 - self.eps)
        return -(s * lp).sum(-1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# MIXUP
# ─────────────────────────────────────────────────────────────────────────────
def mixup(x, y, alpha=CFG.mixup_alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, path):
    torch.save({
        "epoch":     epoch,
        "best_acc":  float(best_acc),
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    print(f"  ✔ Checkpoint saved → {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=CFG.device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if optimizer is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                print("  ⚠ Could not load optimizer state — using fresh optimizer")
        # NOTE: We intentionally do NOT restore scheduler state —
        #       the old OneCycleLR state is incompatible with CosineAnnealingLR
        epoch    = ckpt.get("epoch", 0)
        best_acc = float(ckpt.get("best_acc", 0.0))
        print(f"  ✔ Resumed from epoch {epoch}  (best val acc: {best_acc:.4f})")
        return epoch, best_acc
    else:
        model.load_state_dict(ckpt)
        print(f"  ✔ Loaded weights-only checkpoint")
        return CFG.RESUME_EPOCH, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, opt, criterion, scaler, sched):
    model.train()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(CFG.device), y.to(CFG.device)
        xm, ya, yb, lam = mixup(x, y)
        with torch.autocast(CFG.device, enabled=CFG.amp):
            logits = model(xm)
            loss   = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        # CosineAnnealingLR steps per EPOCH not per batch — called outside
        loss_sum += loss.item() * x.size(0)
        p = logits.argmax(1)
        correct += (lam * (p == ya).float() +
                    (1 - lam) * (p == yb).float()).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    preds, labels, probs_all, loss_sum = [], [], [], 0
    for x, y in loader:
        x, y = x.to(CFG.device), y.to(CFG.device)
        logits   = model(x)
        loss_sum += criterion(logits, y).item() * x.size(0)
        pr = F.softmax(logits, -1)
        preds.extend(pr.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
        probs_all.extend(pr.cpu().numpy())
    preds     = np.array(preds)
    labels    = np.array(labels)
    probs_all = np.array(probs_all)
    return (loss_sum / len(labels),
            (preds == labels).mean(),
            labels, preds, probs_all)


def train(model, tr_loader, vl_loader):
    criterion = LabelSmoothCE()
    opt       = torch.optim.AdamW(model.parameters(),
                                  lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler    = torch.cuda.amp.GradScaler(enabled=CFG.amp)

    start_epoch = 1
    best        = 0.0

    if CFG.RESUME and CFG.RESUME_CHECKPOINT.exists():
        print(f"\nResuming from: {CFG.RESUME_CHECKPOINT}")
        # Load model weights + optimizer (NOT scheduler — incompatible)
        start_epoch, best = load_checkpoint(CFG.RESUME_CHECKPOINT, model, opt)
        start_epoch += 1

        remaining = CFG.epochs - start_epoch + 1
        # Fresh CosineAnnealingLR for remaining epochs only
        # T_max = remaining epochs, starts at lr=5e-4, decays to eta_min
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max   = remaining,
            eta_min = 1e-6,
        )
        print(f"  Scheduler : CosineAnnealingLR  T_max={remaining}  eta_min=1e-6")

    elif CFG.RESUME and not CFG.RESUME_CHECKPOINT.exists():
        print(f"\n⚠  Resume checkpoint not found: {CFG.RESUME_CHECKPOINT}")
        print("   Starting from scratch.\n")
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=CFG.epochs, eta_min=1e-6)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=CFG.epochs, eta_min=1e-6)

    remaining = CFG.epochs - start_epoch + 1
    if remaining <= 0:
        print(f"\n✔ Already completed {CFG.epochs} epochs. Nothing to train.")
        return None

    print(f"\nTraining epochs {start_epoch} → {CFG.epochs}  ({remaining} remaining)")
    print(f"\n{'Ep':>4} | {'TrLoss':>7} | {'TrAcc':>6} | {'VlLoss':>7} | {'VlAcc':>6} | {'LR':>10}")
    print("-" * 58)

    hist = {k: [] for k in ("tl", "vl", "ta", "va")}

    for ep in range(start_epoch, CFG.epochs + 1):
        tl, ta     = train_one_epoch(model, tr_loader, opt, criterion, scaler, sched)
        vl, va, *_ = evaluate(model, vl_loader, criterion)

        # CosineAnnealingLR steps per epoch
        sched.step()

        hist["tl"].append(tl)
        hist["ta"].append(ta)
        hist["vl"].append(vl)
        hist["va"].append(va)

        current_lr = opt.param_groups[0]["lr"]
        marker     = " ← best" if va > best else ""
        print(f"{ep:>4} | {tl:>7.4f} | {ta:>6.4f} | "
              f"{vl:>7.4f} | {va:>6.4f} | {current_lr:>10.2e}{marker}")

        if va > best:
            best = va
            save_checkpoint(model, opt, sched, ep, best,
                            CFG.output_dir / "best_pcb_vit_s.pth")

        if ep % CFG.SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(model, opt, sched, ep, best,
                            CFG.output_dir / f"ckpt_s_epoch{ep}.pth")

    print(f"\n✔ Best Val Acc : {best:.4f}")
    print(f"  Saved to     : {CFG.output_dir / 'best_pcb_vit_s.pth'}")
    return hist


# ─────────────────────────────────────────────────────────────────────────────
# TTA INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
_TTA = [
    transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.RandomVerticalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    transforms.Compose([
        transforms.Resize((int(CFG.img_size * 1.1), int(CFG.img_size * 1.1))),
        transforms.CenterCrop(CFG.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
]


@torch.no_grad()
def predict_tta(model, img: Image.Image):
    model.eval()
    avg = torch.stack([
        F.softmax(model(t(img).unsqueeze(0).to(CFG.device)), -1).cpu()
        for t in _TTA[:CFG.tta_iters]
    ]).mean(0)
    idx = avg.argmax(1).item()
    return CFG.classes[idx], avg[0].numpy()


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def overlay(img_np, cam, alpha=0.45):
    h = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    return np.clip(alpha * h + (1 - alpha) * img_np, 0, 255).astype(np.uint8)


def visualize(model, img_path, save_path=None):
    img    = Image.open(img_path).convert("RGB")
    img_np = np.array(img.resize((CFG.img_size, CFG.img_size)))

    label, probs = predict_tta(model, img)
    idx = CFG.class2idx[label]

    tl  = get_gradcam_target_layer(model)
    cam = np.zeros((CFG.img_size, CFG.img_size))
    if tl is not None:
        gc = GradCAM(model, tl)
        x  = (get_transforms("val")(img)
              .unsqueeze(0).to(CFG.device).requires_grad_(True))
        try:
            cam, _ = gc.generate(x, idx)
        except Exception as e:
            print(f"  ⚠ Grad-CAM failed: {e}")
        gc.remove()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"PCB Defect: {label.replace('_', ' ').upper()}  "
        f"|  Confidence: {probs[idx]:.1%}",
        fontsize=14, fontweight="bold", color="#c0392b")

    axes[0].imshow(img_np);               axes[0].set_title("Original Image"); axes[0].axis("off")
    axes[1].imshow(overlay(img_np, cam)); axes[1].set_title("Grad-CAM");       axes[1].axis("off")

    colors = ["#e74c3c" if i == idx else "#3498db" for i in range(CFG.num_classes)]
    axes[2].barh([c.replace("_", "\n") for c in CFG.classes], probs, color=colors)
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("Probability")
    axes[2].set_title("Class Confidence")
    for i, v in enumerate(probs):
        axes[2].text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✔ Heatmap saved → {save_path}")
    plt.show()
    return label, probs


def plot_history(h, save=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    ep = range(1, len(h["tl"]) + 1)

    axes[0].plot(ep, h["tl"], label="Train", color="#e74c3c")
    axes[0].plot(ep, h["vl"], label="Val",   color="#3498db")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, h["ta"], label="Train", color="#e74c3c")
    axes[1].plot(ep, h["va"], label="Val",   color="#3498db")
    axes[1].set_title("Accuracy"); axes[1].legend()
    axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 1)

    best_ep  = h["va"].index(max(h["va"])) + 1
    best_acc = max(h["va"])
    axes[2].plot(ep, h["va"], color="#2ecc71", linewidth=2)
    axes[2].axvline(best_ep, color="#e74c3c", linestyle="--", alpha=0.7)
    axes[2].scatter([best_ep], [best_acc], color="#e74c3c", s=100, zorder=5)
    axes[2].annotate(f"Best: {best_acc:.4f}\nEpoch {best_ep}",
                     xy=(best_ep, best_acc),
                     xytext=(best_ep + 1, best_acc - 0.05),
                     fontsize=9, color="#e74c3c")
    axes[2].set_title("Val Accuracy — Best Point")
    axes[2].grid(alpha=0.3); axes[2].set_ylim(0, 1)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
        print(f"  ✔ Training history saved → {save}")
    plt.show()


def plot_cm(y_true, y_pred, save=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(CFG.num_classes)); ax.set_yticks(range(CFG.num_classes))
    ax.set_xticklabels(CFG.classes, rotation=45, ha="right")
    ax.set_yticklabels(CFG.classes)
    for i in range(CFG.num_classes):
        for j in range(CFG.num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — PCB Defect Classification (MobileViT-S)")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
        print(f"  ✔ Confusion matrix saved → {save}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ONNX EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def export_onnx(model):
    path = str(CFG.output_dir / "pcb_vit_s.onnx")
    model.eval()
    dummy = torch.randn(1, 3, CFG.img_size, CFG.img_size).to(CFG.device)
    torch.onnx.export(
        model, dummy, path,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17)
    print(f"  ✔ ONNX exported → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  PCB Anomaly Classifier — MobileViT-S v4.1 (Resume Fix)")
    print(f"  Device        : {CFG.device}")
    print(f"  Model         : {CFG.model_name}  (~5.7M params)")
    print(f"  Image Size    : {CFG.img_size}x{CFG.img_size}")
    print(f"  Batch Size    : {CFG.batch_size}")
    print(f"  LR            : {CFG.lr}  (CosineAnnealingLR — resumable)")
    print(f"  Epochs        : {CFG.epochs}  (resuming from {CFG.RESUME_EPOCH})")
    print(f"  Mixup Alpha   : {CFG.mixup_alpha}")
    print(f"  Skip Training : {CFG.SKIP_TRAINING}")
    print("=" * 65)

    if not CFG.data_root.exists():
        print(f"\n❌  Data path not found:\n    {CFG.data_root}"); return

    print("\nScanning dataset ...")
    all_samples = load_all_samples(CFG.data_root)
    if not all_samples:
        print("❌  No images found."); return

    print(f"\nTotal : {len(all_samples)} images")
    train_s, val_s, test_s = split_samples(all_samples)
    print(f"Split : Train={len(train_s)} | Val={len(val_s)} | Test={len(test_s)}")

    train_dl = DataLoader(PCBDataset(train_s, "train"),
                          CFG.batch_size, shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=True)
    val_dl   = DataLoader(PCBDataset(val_s, "val"),
                          CFG.batch_size, shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True)
    test_dl  = DataLoader(PCBDataset(test_s, "val"),
                          CFG.batch_size, shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True)

    model  = PCBViT().to(CFG.device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel : {CFG.model_name}  ({params:.1f}M params)")

    if CFG.SKIP_TRAINING:
        best_ckpt = CFG.output_dir / "best_pcb_vit_s.pth"
        if not best_ckpt.exists():
            print(f"\n❌  No checkpoint found at {best_ckpt}"); return
        print(f"\nSkipping training. Loading: {best_ckpt}")
        load_checkpoint(best_ckpt, model)
        history = None
    else:
        history = train(model, train_dl, val_dl)
        if history:
            plot_history(history, str(CFG.output_dir / "training_history_s.png"))

    print("\nLoading best checkpoint for final test evaluation ...")
    load_checkpoint(CFG.output_dir / "best_pcb_vit_s.pth", model)

    _, acc, y_true, y_pred, y_probs = evaluate(model, test_dl, LabelSmoothCE())
    print(f"\nTest Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CFG.classes))
    try:
        auc = roc_auc_score(y_true, y_probs, multi_class="ovr")
        print(f"ROC-AUC (OvR) : {auc:.4f}")
    except ValueError:
        pass

    plot_cm(y_true, y_pred, str(CFG.output_dir / "confusion_matrix_s.png"))
    export_onnx(model)

    sample = test_s[0][0]
    print(f"\nGenerating demo heatmap on: {sample.name}")
    visualize(model, str(sample), save_path=str(CFG.output_dir / "demo_heatmap_s.png"))


if __name__ == "__main__":
    main()