"""
PCB Anomaly Test Script — MobileViT-S
======================================
Tests the trained MobileViT-S model on a single image or a folder.
Self-contained — no imports from transformer.py needed.

Usage:
    1. Set TEST_IMAGE path in CFG
    2. Run: python test_anomolytype.py
"""

import sys
import math
import warnings
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

warnings.filterwarnings("ignore")
import argparse
from segmentation_model import UNet

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — update TEST_IMAGE or TEST_FOLDER path before running
# ─────────────────────────────────────────────────────────────────────────────
class CFG:
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    img_size    = 256            # must match training (mobilevit_s uses 256)
    num_classes = 6
    model_name  = "mobilevit_s"  # must match training
    tta_iters   = 5
    ambiguity_threshold = 0.02  # 2% threshold for ambiguous predictions

    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper",
    ]
    class2idx = {c: i for i, c in enumerate(classes)}

    # ── Which checkpoint to load ──────────────────────────────────────────────
    # Use best_pcb_vit_s.pth (best val accuracy) OR ckpt_s_epoch{N}.pth
    CHECKPOINT = Path(os.getenv("PHASE2_CHECKPOINT", str(output_dir / "best_pcb_vit_s.pth")))

    # ── Image(s) to test ─────────────────────────────────────────────────────
    # Single image test — set path here:
    TEST_IMAGE = os.getenv("TEST_IMAGE", "")
    # Example: TEST_FOLDER = r"C:\Users\Sastra\OneDrive\Desktop\PCB_mini\stage2_classification\test\open_circuit"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — must match architecture in transformer_updated.py
# ─────────────────────────────────────────────────────────────────────────────
class PCBViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG.model_name, pretrained=False, num_classes=0)
        dim = self.backbone.num_features
        # 3-layer head — must match exactly what was trained
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
# CHECKPOINT LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(path, model):
    ckpt = torch.load(path, map_location=CFG.device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        epoch    = ckpt.get("epoch", "?")
        best_acc = float(ckpt.get("best_acc", 0.0))
        print(f"  ✔ Loaded checkpoint — epoch {epoch}, "
              f"best val acc: {best_acc:.4f} ({best_acc*100:.1f}%)")
    else:
        model.load_state_dict(ckpt)
        print(f"  ✔ Loaded weights-only checkpoint")
    model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS & TTA
# ─────────────────────────────────────────────────────────────────────────────
_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_TTA_TRANSFORMS = [
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
def predict_tta(model, img: Image.Image, temperature=1.2):
    model.eval()
    preds = []

    for t in _TTA_TRANSFORMS[:CFG.tta_iters]:
        x = t(img).unsqueeze(0).to(CFG.device)
        logits = model(x) / temperature   # 🔥 calibration
        preds.append(F.softmax(logits, dim=-1).cpu())

    avg = torch.stack(preds).mean(0)
    
    # Get top 2 probabilities
    top2_probs, top2_indices = torch.topk(avg[0], 2)
    top1_prob = top2_probs[0].item()
    top2_prob = top2_probs[1].item()
    diff_pct = (top1_prob - top2_prob) * 100  # Difference in percentage points
    
    # Check if difference is below threshold (1-2%)
    if diff_pct < (CFG.ambiguity_threshold * 100):
        # Ambiguous case: return both classes
        class1 = CFG.classes[top2_indices[0].item()]
        class2 = CFG.classes[top2_indices[1].item()]
        result_class = f"{class1}/{class2}"
        # Return probabilities for both ambiguous classes
        probs_with_highlights = avg[0].numpy()
        return result_class, probs_with_highlights, True, (class1, class2, top1_prob, top2_prob)
    else:
        # Clear prediction
        idx = avg.argmax(1).item()
        result_class = CFG.classes[idx]
        return result_class, avg[0].numpy(), False, None


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
    # Fallback
    last_conv = None
    for _, module in model.backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        print(f"  ✔ Grad-CAM target: last Conv2d (fallback)")
        return last_conv
    return None


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.grads = None
        self.acts = None

        def forward_hook(m, i, o):
            self.acts = o

        def backward_hook(m, grad_in, grad_out):
            self.grads = grad_out[0]

        self.fwd = target_layer.register_forward_hook(forward_hook)
        self.bwd = target_layer.register_backward_hook(backward_hook)

    def generate(self, x, class_idx=None):
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(1).item()

        self.model.zero_grad()

        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot)

        if self.grads is None or self.acts is None:
            return np.zeros((CFG.img_size, CFG.img_size)), class_idx

        g = self.grads.detach()
        a = self.acts.detach()

        cam = (g.mean((2, 3), keepdim=True) * a).sum(1)[0].cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (CFG.img_size, CFG.img_size))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam, class_idx

    def remove(self):
        self.fwd.remove()
        self.bwd.remove()


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def overlay(img_np, cam, alpha=0.45):
    h = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    return np.clip(alpha * h + (1 - alpha) * img_np, 0, 255).astype(np.uint8)


def visualize(model, img_path, save_path=None, seg_model=None):
    img    = Image.open(img_path).convert("RGB")
    img_np = np.array(img.resize((CFG.img_size, CFG.img_size)))

    result = predict_tta(model, img)
    
    if len(result) == 4:
        label, probs, is_ambiguous, ambiguous_info = result
    else:
        label, probs, is_ambiguous, ambiguous_info = result + (None,)
    
    if is_ambiguous and ambiguous_info:
        class1, class2, prob1, prob2 = ambiguous_info
        idx1 = CFG.class2idx[class1]
        idx2 = CFG.class2idx[class2]
        conf = max(prob1, prob2)  # Use higher confidence for display
        # For Grad-CAM, use the first class
        idx = idx1
    else:
        idx = CFG.class2idx[label]
        conf = float(probs[idx])

    # Grad-CAM
    tl  = get_gradcam_target_layer(model)
    cam = np.zeros((CFG.img_size, CFG.img_size))
    if tl is not None:
        gc = GradCAM(model, tl)
        x  = (_VAL_TRANSFORM(img)
              .unsqueeze(0).to(CFG.device).requires_grad_(True))
        try:
            cam, _ = gc.generate(x, idx)
        except Exception as e:
            print(f"  ⚠ Grad-CAM failed: {e}")
        gc.remove()

    # ── Bounding box from heatmap ──────────────────────────────────────────
    # 🔥 Segmentation first, fallback to Grad-CAM
    if seg_model is not None:
        x = _VAL_TRANSFORM(img).unsqueeze(0).to(CFG.device)
        with torch.no_grad():
            mask = seg_model(x)[0,0].cpu().numpy()

        binary = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x1, y1, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            bbox = (x1, y1, x1+w, y1+h)
        else:
            bbox = None
    else:
        bbox = get_bounding_box(cam)
    
    # Draw bounding box with appropriate label
    display_label = label if not is_ambiguous else f"{class1}/{class2}"
    img_box = draw_bbox(img_np, bbox, display_label, conf, color=(0, 255, 80) if not is_ambiguous else (255, 165, 0))
    if bbox:
        x1, y1, x2, y2 = bbox
        print(f"  ✔ Bounding box : ({x1},{y1}) → ({x2},{y2})")
    else:
        print(f"  ⚠ No bounding box found")

    # ── Plot: 4 panels ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    # Highlight ambiguous predictions in title
    if is_ambiguous:
        title = f"⚠ AMBIGUOUS: {class1.replace('_',' ').upper()} / {class2.replace('_',' ').upper()}  |  Diff: {(prob1-prob2)*100:.1f}%"
        title_color = "#e67e22"
    else:
        title = f"PCB Defect: {label.replace('_',' ').upper()}  |  Confidence: {conf:.1%}"
        title_color = "#c0392b"
    
    fig.suptitle(title, fontsize=14, fontweight="bold", color=title_color)

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(overlay(img_np, cam,alpha=0.6))
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(img_box)
    axes[2].set_title("Bounding Box")
    axes[2].axis("off")

    # Highlight ambiguous classes in bar chart
    colors = []
    for i in range(CFG.num_classes):
        if is_ambiguous and i in [idx1, idx2]:
            colors.append("#e67e22")  # Orange for ambiguous classes
        elif i == idx:
            colors.append("#e74c3c")  # Red for predicted class
        else:
            colors.append("#3498db")  # Blue for others
    
    bars = axes[3].barh([c.replace("_", "\n") for c in CFG.classes],
                 probs, color=colors)
    axes[3].set_xlim(0, 1)
    axes[3].set_xlabel("Probability")
    axes[3].set_title("Class Confidence")
    for i, v in enumerate(probs):
        if is_ambiguous and i in [idx1, idx2]:
            axes[3].text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9, fontweight="bold", color="#e67e22")
        else:
            axes[3].text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✔ Saved → {save_path}")
    plt.show()
    return label, probs, bbox


# ─────────────────────────────────────────────────────────────────────────────
# FOLDER TEST
# ─────────────────────────────────────────────────────────────────────────────
def test_folder(model, folder_path):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import pandas as pd
    import matplotlib.pyplot as plt

    results = []

    folder = Path(folder_path)
    images = (list(folder.glob("*.jpg")) +
              list(folder.glob("*.png")) +
              list(folder.glob("*.bmp")))

    if not images:
        print(f"⚠ No images found in {folder}")
        return

    correct = 0
    labeled_count = 0
    ambiguous_count = 0

    print(f"\n{'Image':<45} {'Predicted':<25} {'Conf':>8}  {'✔/✗':>4}")
    print("-" * 85)

    for img_path in sorted(images):
        img = Image.open(img_path).convert("RGB")

        result = predict_tta(model, img)
        
        if len(result) == 4:
            label, probs, is_ambiguous, ambiguous_info = result
            if is_ambiguous and ambiguous_info:
                class1, class2, prob1, prob2 = ambiguous_info
                display_label = f"{class1}/{class2}"
                conf = max(prob1, prob2)
                ambiguous_count += 1
        else:
            label, probs, is_ambiguous, ambiguous_info = result + (None,)
            display_label = label
            conf = float(probs[CFG.class2idx[label]])

        # 🔍 Extract true label from filename
        true_label = None
        for cls in CFG.classes:
            if cls in img_path.name.lower():
                true_label = cls
                break

        # For ambiguous predictions, consider it correct if either class matches true label
        match = " "
        if true_label is not None:
            labeled_count += 1
            if is_ambiguous and ambiguous_info:
                if true_label in [class1, class2]:
                    correct += 1
                    match = "✔"
                else:
                    match = "✗"
            elif not is_ambiguous and true_label == label:
                correct += 1
                match = "✔"
            elif not is_ambiguous:
                match = "✗"

        # Add indicator for ambiguous predictions
        amb_marker = " ⚠" if is_ambiguous else ""
        
        print(f"{img_path.name:<45} {display_label:<25} {conf:>7.1%}  {match}{amb_marker}")

        # 📊 Store results
        results.append({
            "image": img_path.name,
            "predicted": display_label,
            "is_ambiguous": is_ambiguous,
            "confidence": conf,
            "true_label": true_label,
            "ambiguous_classes": f"{class1}/{class2}" if is_ambiguous and ambiguous_info else ""
        })

    # 📈 Accuracy
    if labeled_count > 0:
        acc = correct / labeled_count
        print(f"\n📊 Folder Statistics:")
        print(f"   Total images: {len(images)}")
        print(f"   Labeled images: {labeled_count}")
        print(f"   Ambiguous predictions: {ambiguous_count}")
        print(f"   Correct predictions: {correct}")
        print(f"   Accuracy: {correct}/{labeled_count} = {acc:.2%}")
    else:
        print("\n⚠ No true labels found in filenames")

    # 💾 Save CSV
    df = pd.DataFrame(results)
    csv_path = Path("results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✔ Saved results → {csv_path}")

    # 📉 Confusion Matrix (skip ambiguous predictions or handle them)
    y_true = [r["true_label"] for r in results if r["true_label"] is not None and not r["is_ambiguous"]]
    y_pred = [r["predicted"] for r in results if r["true_label"] is not None and not r["is_ambiguous"]]

    if y_true:
        cm = confusion_matrix(y_true, y_pred, labels=CFG.classes)

        disp = ConfusionMatrixDisplay(cm, display_labels=CFG.classes)
        disp.plot(xticks_rotation=45)
        plt.title("Confusion Matrix (excluding ambiguous predictions)")
        plt.tight_layout()
        plt.show()
    else:
        print("⚠ Skipping confusion matrix (no clear predictions)")


def get_bounding_box(cam):
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    # 🔥 Lower threshold (more sensitive)
    thresh = np.percentile(cam, 70)

    binary = (cam > thresh).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # 🔥 Fallback: take highest activation region
        y, x = np.unravel_index(np.argmax(cam), cam.shape)

        x1 = max(0, x - 25)
        y1 = max(0, y - 25)
        x2 = min(CFG.img_size, x + 25)
        y2 = min(CFG.img_size, y + 25)

        return (x1, y1, x2, y2)

    x_min, y_min, x_max, y_max = 9999, 9999, 0, 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    return (x_min, y_min, x_max, y_max)


def draw_bbox(img_np, bbox, label, conf, color=(0, 255, 0)):
    """Draw bounding box + label on image."""
    if bbox is None:
        return img_np.copy()
    img = img_np.copy()
    x1, y1, x2, y2 = bbox

    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Draw label background
    text    = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

    # Draw label text
    cv2.putText(img, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=CFG.TEST_IMAGE)
    parser.add_argument("--folder", type=str, default=None)
    args = parser.parse_args()

    CFG.TEST_IMAGE = args.img
    CFG.TEST_FOLDER = args.folder
    print("=" * 60)
    print("  PCB Anomaly Test — MobileViT-S")
    print(f"  Device     : {CFG.device}")
    print(f"  Model      : {CFG.model_name}")
    print(f"  Image Size : {CFG.img_size}x{CFG.img_size}")
    print(f"  Checkpoint : {CFG.CHECKPOINT.name}")
    print(f"  Ambiguity Threshold: {CFG.ambiguity_threshold*100}%")
    print("=" * 60)
    

    # ── Check checkpoint exists ───────────────────────────────────────────────
    if not CFG.CHECKPOINT.exists():
        print(f"\n❌ Checkpoint not found: {CFG.CHECKPOINT}")
        print("   Is training still running? Wait for it to finish at least")
        print("   epoch 1 and save best_pcb_vit_s.pth, then run this again.")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    model = PCBViT().to(CFG.device)
    load_checkpoint(CFG.CHECKPOINT, model)
    print("✔ Model loaded\n")
    # Optional segmentation model
    seg_model = UNet().to(CFG.device)

    seg_path = CFG.output_dir / "unet.pth"
    if seg_path.exists():
        seg_model.load_state_dict(torch.load(seg_path, map_location=CFG.device))
        seg_model.eval()
        print("✔ Segmentation model loaded")
    else:
        seg_model = None
        print("⚠ No segmentation model found → using Grad-CAM")

    # ── Single image test ─────────────────────────────────────────────────────
    if CFG.TEST_IMAGE:
        img_path = Path(CFG.TEST_IMAGE)
        if not img_path.exists():
            print(f"❌ Image not found: {img_path}")
        else:
            print(f"Testing: {img_path.name}")
            img          = Image.open(img_path).convert("RGB")
            result = predict_tta(model, img)
            
            if len(result) == 4:
                label, probs, is_ambiguous, ambiguous_info = result
                if is_ambiguous and ambiguous_info:
                    class1, class2, prob1, prob2 = ambiguous_info
                    print(f"\n  ⚠ AMBIGUOUS PREDICTION")
                    print(f"  Class 1: {class1} ({prob1:.1%})")
                    print(f"  Class 2: {class2} ({prob2:.1%})")
                    print(f"  Difference: {(prob1-prob2)*100:.1f}%")
                    print(f"  Result: {class1}/{class2}")
                    display_label = f"{class1}/{class2}"
                else:
                    display_label = label
                    print(f"\n  Predicted Class : {label}")
                    print(f"  Confidence      : {probs[CFG.class2idx[label]]:.1%}")
            else:
                label, probs, is_ambiguous, ambiguous_info = result + (None,)
                print(f"\n  Predicted Class : {label}")
                print(f"  Confidence      : {probs[CFG.class2idx[label]]:.1%}")
                display_label = label

            print("\n  All class probabilities:")
            for cls, prob in zip(CFG.classes, probs):
                bar    = "█" * int(prob * 30)
                # Highlight ambiguous classes
                if is_ambiguous and ambiguous_info and cls in [class1, class2]:
                    marker = " ← AMBIGUOUS" if cls == class1 and cls == class2 else " ← ambiguous"
                elif cls == display_label:
                    marker = " ← predicted"
                else:
                    marker = ""
                print(f"    {cls:<22} {prob:>5.1%}  {bar}{marker}")

            save_path = str(CFG.output_dir /
                            f"heatmap_s_{img_path.stem}.png")
            print(f"\nGenerating Grad-CAM heatmap ...")
            visualize(model, str(img_path), save_path=save_path, seg_model=seg_model)

    # ── Folder test ───────────────────────────────────────────────────────────
    # Add a quick sanity check in your test script
    import torch

    # Use the full path like you have in CFG:
    ckpt = torch.load(CFG.CHECKPOINT, map_location='cpu')
    if isinstance(ckpt, dict) and 'best_acc' in ckpt:
        print(f"Best validation accuracy: {ckpt['best_acc']:.2%}")

    if CFG.TEST_FOLDER:
        print(f"\nFolder test: {CFG.TEST_FOLDER}")
        test_folder(model, CFG.TEST_FOLDER)

    print("\n✔ Done.")