"""
PCB Anomaly Classification with Vision Transformer (ViT) + Grad-CAM Heatmap
============================================================================
Features:
  - Vision Transformer (ViT-B/16) fine-tuned for PCB defect classification
  - Multi-class anomaly types: short_circuit, open_circuit, solder_bridge,
    missing_component, misalignment, normal
  - Grad-CAM & Attention Rollout heatmaps for localization
  - Mixed-precision training, LR warm-up + cosine annealing
  - Test-Time Augmentation (TTA) for robust inference
  - Metrics: Accuracy, per-class F1, Confusion Matrix, ROC-AUC
  - ONNX export for production deployment
"""

import os
import warnings
import random
import math
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
import timm  # pip install timm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
class CFG:
    seed          = 42
    device        = "cuda" if torch.cuda.is_available() else "cpu"
    img_size      = 224
    num_classes   = 6
    batch_size    = 32
    epochs        = 30
    lr            = 3e-4
    weight_decay  = 1e-4
    warmup_epochs = 5
    label_smooth  = 0.1
    mixup_alpha   = 0.2
    amp           = True           # mixed-precision
    tta_iters     = 5              # test-time augmentation rounds
    model_name    = "vit_base_patch16_224"
    pretrained    = True
    num_workers   = 4

    classes = [
        "normal",
        "short_circuit",
        "open_circuit",
        "solder_bridge",
        "missing_component",
        "misalignment",
    ]
    class2idx = {c: i for i, c in enumerate(classes)}

    data_root  = Path("data/pcb")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = CFG.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
def get_transforms(mode: str = "train"):
    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(CFG.img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CFG.img_size, CFG.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


class PCBDataset(Dataset):
    """
    Expects folder structure:
        data/pcb/train/<class_name>/*.jpg
        data/pcb/val/<class_name>/*.jpg
        data/pcb/test/<class_name>/*.jpg
    """
    def __init__(self, root: Path, split: str = "train"):
        self.transform = get_transforms(split)
        self.samples   = []
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        for cls in CFG.classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            for fp in cls_dir.glob("*.[jp][pn]g"):
                self.samples.append((fp, CFG.class2idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = CFG.mixup_alpha):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class PCBViT(nn.Module):
    """
    Vision Transformer backbone + custom classification head.
    Exposes attention weights for Attention Rollout heatmaps.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG.model_name,
            pretrained=CFG.pretrained,
            num_classes=0,           # remove default head
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, CFG.num_classes),
        )

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        return self.head(feats)

    def forward_features(self, x: torch.Tensor):
        """Return (cls_token, patch_tokens, attn_weights_per_block)."""
        return self.backbone.forward_features(x)

# ─────────────────────────────────────────────
# GRAD-CAM (for CNN-style last conv layer or ViT patch tokens)
# ─────────────────────────────────────────────
class GradCAM:
    """
    Generic Grad-CAM that works on any layer returning spatial feature maps.
    For ViT, hook onto the last transformer block's output (patch dimension).
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model  = model
        self.grads  : Optional[torch.Tensor] = None
        self.acts   : Optional[torch.Tensor] = None
        self._hooks  = []
        self._hooks.append(
            target_layer.register_forward_hook(self._save_acts))
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_grads))

    def _save_acts(self, _, __, output):
        self.acts = output[0] if isinstance(output, tuple) else output

    def _save_grads(self, _, __, grad_output):
        self.grads = grad_output[0] if isinstance(grad_output, tuple) else grad_output

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None):
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        grads = self.grads.detach()   # [B, seq, dim] for ViT
        acts  = self.acts.detach()

        # ViT: acts shape [B, N, D] (N = 1 cls + HW patches)
        if grads.dim() == 3:
            # Remove cls token
            grads = grads[:, 1:, :]
            acts  = acts[:, 1:, :]
            weights = grads.mean(dim=-1, keepdim=True)  # [B, N, 1]
            cam = (weights * acts).sum(dim=-1)           # [B, N]
            h = w = int(math.sqrt(cam.shape[-1]))
            cam = cam.reshape(1, h, w).cpu().numpy()[0]
        else:
            # CNN: acts [B, C, H, W]
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1)[0].cpu().numpy()

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (CFG.img_size, CFG.img_size))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


# ─────────────────────────────────────────────
# ATTENTION ROLLOUT
# ─────────────────────────────────────────────
class AttentionRollout:
    """
    Computes Attention Rollout (Abnar & Zuidema, 2020) across all ViT blocks.
    More faithful than Grad-CAM for pure transformer models.
    """
    def __init__(self, model: PCBViT, discard_ratio: float = 0.9):
        self.model         = model
        self.discard_ratio = discard_ratio
        self.attentions    = []
        self._hooks        = []
        for block in model.backbone.blocks:
            self._hooks.append(
                block.attn.register_forward_hook(self._save_attn))

    def _save_attn(self, _, __, output):
        # timm ViT attention returns (x,) or (x, attn_weights)
        if isinstance(output, tuple) and len(output) == 2:
            self.attentions.append(output[1].detach().cpu())

    @torch.no_grad()
    def generate(self, x: torch.Tensor):
        self.attentions = []
        self.model.eval()
        self.model(x)

        if not self.attentions:
            return None   # attention weights not available

        result = torch.eye(self.attentions[0].shape[-1])
        for attn in self.attentions:
            attn_heads_fused = attn.mean(dim=1)[0]  # mean over heads
            flat    = attn_heads_fused.flatten()
            thresh  = flat.kthvalue(int(flat.numel() * self.discard_ratio)).values
            attn_heads_fused[attn_heads_fused < thresh] = 0
            I       = torch.eye(attn_heads_fused.shape[-1])
            a       = (attn_heads_fused + I) / 2.0
            a       = a / a.sum(dim=-1, keepdim=True)
            result  = torch.matmul(a, result)

        mask = result[0, 1:]   # cls -> all patches
        h = w = int(math.sqrt(mask.shape[0]))
        mask = mask.reshape(h, w).numpy()
        mask = cv2.resize(mask, (CFG.img_size, CFG.img_size))
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
class LabelSmoothCE(nn.Module):
    def __init__(self, eps: float = CFG.label_smooth):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        n = logits.size(-1)
        log_p = F.log_softmax(logits, dim=-1)
        # smooth
        smooth = torch.full_like(log_p, self.eps / (n - 1))
        smooth.scatter_(-1, targets.unsqueeze(1), 1.0 - self.eps)
        return -(smooth * log_p).sum(dim=-1).mean()


def build_scheduler(optimizer, steps_per_epoch: int):
    warmup_steps = CFG.warmup_epochs * steps_per_epoch
    total_steps  = CFG.epochs       * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(CFG.device), y.to(CFG.device)
        # MixUp
        x_mix, ya, yb, lam = mixup(x, y)
        with torch.autocast(CFG.device, enabled=CFG.amp):
            logits = model(x_mix)
            loss   = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        total_loss += loss.item() * x.size(0)
        preds  = logits.argmax(dim=1)
        correct += ((preds == ya).float() * lam +
                    (preds == yb).float() * (1 - lam)).sum().item()
        total  += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(CFG.device), y.to(CFG.device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        probs  = F.softmax(logits, dim=-1)
        preds  = probs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    acc  = (all_preds == all_labels).mean()
    loss = total_loss / len(all_labels)
    return loss, acc, all_labels, all_preds, all_probs


def train(model, train_loader, val_loader):
    criterion = LabelSmoothCE()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CFG.lr,
                                  weight_decay=CFG.weight_decay)
    scaler    = torch.cuda.amp.GradScaler(enabled=CFG.amp)
    scheduler = build_scheduler(optimizer, len(train_loader))

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>9} | {'Val Acc':>8} | {'LR':>8}")
    print("-" * 60)

    for epoch in range(1, CFG.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, scheduler)
        vl_loss, vl_acc, _, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5} | {tr_loss:>10.4f} | {tr_acc:>9.4f} | "
              f"{vl_loss:>9.4f} | {vl_acc:>8.4f} | {lr_now:>8.6f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(),
                       CFG.output_dir / "best_model.pth")

    print(f"\n✔ Best Val Acc: {best_val_acc:.4f}")
    return history

# ─────────────────────────────────────────────
# INFERENCE WITH TTA
# ─────────────────────────────────────────────
_tta_transforms = [
    transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((int(CFG.img_size * 1.1), int(CFG.img_size * 1.1))),
        transforms.CenterCrop(CFG.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
]


@torch.no_grad()
def predict_with_tta(model: nn.Module, img: Image.Image):
    model.eval()
    probs_list = []
    for tf in _tta_transforms[:CFG.tta_iters]:
        x = tf(img).unsqueeze(0).to(CFG.device)
        probs_list.append(F.softmax(model(x), dim=-1).cpu())
    avg_probs = torch.stack(probs_list).mean(0)
    pred_idx  = avg_probs.argmax(dim=1).item()
    return CFG.classes[pred_idx], avg_probs[0].numpy()

# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
def overlay_heatmap(img_np: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    heatmap = cv2.applyColorMap(
        (cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    out = (alpha * heatmap.astype(np.float32) +
           (1 - alpha) * img_np.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


def visualize_prediction(model: nn.Module,
                         img_path: str,
                         save_path: Optional[str] = None):
    """
    Full visualization panel:
      Original | Grad-CAM overlay | Attention Rollout overlay | Confidence Bar
    """
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img.resize((CFG.img_size, CFG.img_size)))

    # ─ TTA prediction ─
    pred_label, probs = predict_with_tta(model, img)
    pred_idx = CFG.class2idx[pred_label]

    # ─ Grad-CAM ─
    target_layer = model.backbone.blocks[-1].norm1
    gc = GradCAM(model, target_layer)
    x  = get_transforms("val")(img).unsqueeze(0).to(CFG.device)
    x.requires_grad_(True)
    cam, _ = gc.generate(x, class_idx=pred_idx)
    gc.remove_hooks()
    grad_overlay = overlay_heatmap(img_np, cam)

    # ─ Attention Rollout ─
    ar = AttentionRollout(model)
    with torch.no_grad():
        attn_map = ar.generate(x)
    ar.remove_hooks()

    # ─ Plot ─
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"PCB Anomaly Detection  |  Prediction: {pred_label.replace('_',' ').title()}"
        f"  (conf: {probs[pred_idx]:.2%})",
        fontsize=14, fontweight="bold")

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(grad_overlay)
    axes[1].set_title("Grad-CAM Localization")
    axes[1].axis("off")

    if attn_map is not None:
        attn_overlay = overlay_heatmap(img_np, attn_map)
        axes[2].imshow(attn_overlay)
        axes[2].set_title("Attention Rollout")
    else:
        axes[2].imshow(img_np)
        axes[2].set_title("Attention Rollout\n(unavailable)")
    axes[2].axis("off")

    # Confidence bar chart
    colors = ["#2ecc71" if i == pred_idx else "#3498db"
              for i in range(CFG.num_classes)]
    labels = [c.replace("_", "\n") for c in CFG.classes]
    axes[3].barh(labels, probs, color=colors)
    axes[3].set_xlim(0, 1)
    axes[3].set_xlabel("Probability")
    axes[3].set_title("Class Confidence")
    for i, v in enumerate(probs):
        axes[3].text(v + 0.01, i, f"{v:.2%}", va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return pred_label, probs


def plot_training_history(history: dict, save_path: Optional[str] = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"],   label="Val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path: Optional[str] = None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(CFG.num_classes))
    ax.set_yticks(range(CFG.num_classes))
    ax.set_xticklabels(CFG.classes, rotation=45, ha="right")
    ax.set_yticklabels(CFG.classes)
    for i in range(CFG.num_classes):
        for j in range(CFG.num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
def export_onnx(model: nn.Module, path: str = "outputs/pcb_vit.onnx"):
    model.eval()
    dummy = torch.randn(1, 3, CFG.img_size, CFG.img_size).to(CFG.device)
    torch.onnx.export(
        model, dummy, path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX model saved: {path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── Build model ────────────────────────────
    model = PCBViT().to(CFG.device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {CFG.model_name}  |  Params: {total_params:.1f}M  "
          f"|  Device: {CFG.device}")

    # ── Load data ──────────────────────────────
    if CFG.data_root.exists():
        train_ds = PCBDataset(CFG.data_root, "train")
        val_ds   = PCBDataset(CFG.data_root, "val")
        test_ds  = PCBDataset(CFG.data_root, "test")
        print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

        train_loader = DataLoader(train_ds, CFG.batch_size, shuffle=True,
                                  num_workers=CFG.num_workers, pin_memory=True)
        val_loader   = DataLoader(val_ds, CFG.batch_size, shuffle=False,
                                  num_workers=CFG.num_workers, pin_memory=True)
        test_loader  = DataLoader(test_ds, CFG.batch_size, shuffle=False,
                                  num_workers=CFG.num_workers, pin_memory=True)

        # ── Train ──────────────────────────────
        history = train(model, train_loader, val_loader)
        plot_training_history(history,
                              str(CFG.output_dir / "training_history.png"))

        # ── Test ───────────────────────────────
        model.load_state_dict(
            torch.load(CFG.output_dir / "best_model.pth",
                       map_location=CFG.device))
        criterion = LabelSmoothCE()
        _, _, y_true, y_pred, y_probs = evaluate(model, test_loader, criterion)

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=CFG.classes))
        try:
            auc = roc_auc_score(y_true, y_probs, multi_class="ovr")
            print(f"ROC-AUC (OvR): {auc:.4f}")
        except ValueError:
            pass

        plot_confusion_matrix(y_true, y_pred,
                              str(CFG.output_dir / "confusion_matrix.png"))
        export_onnx(model)
    else:
        print(f"⚠  Data directory not found: {CFG.data_root}")
        print("   Skipping training — model structure is ready.")
        print("\n   Expected layout:")
        for split in ("train", "val", "test"):
            for cls in CFG.classes:
                print(f"     data/pcb/{split}/{cls}/*.jpg")

    # ── Demo single image (if sample exists) ──
    sample_img = next(
        CFG.data_root.rglob("*.jpg"), None) if CFG.data_root.exists() else None
    if sample_img:
        print(f"\nRunning demo on: {sample_img}")
        visualize_prediction(
            model, str(sample_img),
            save_path=str(CFG.output_dir / "demo_prediction.png"))


if __name__ == "__main__":
    main()
