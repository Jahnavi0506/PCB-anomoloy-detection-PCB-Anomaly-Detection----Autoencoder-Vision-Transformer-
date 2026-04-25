import streamlit as st
from PIL import Image
import torch
import os
import csv
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np

# 🔥 Import visualize also
from test import PCBViT, predict_tta, CFG, load_checkpoint, visualize
from segmentation_model import UNet
from phase1_autoencoder import load_phase1_model, phase1_predict, Phase1Config
import matplotlib.pyplot as plt

st.title("🔍 PCB Defect Detection System")
PHASE2_UNCERTAIN_THRESHOLD = 0.45
PREDICTION_LOG_PATH = Path("outputs/predictions_log.csv")
PHASE1_DOWNLOAD_URL = os.getenv("PHASE1_MODEL_URL", "")
PHASE2_DOWNLOAD_URL = os.getenv("PHASE2_MODEL_URL", "")


def maybe_download_model(destination: Path, url: str):
    if destination.exists() or not url:
        return destination.exists()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(url, destination)
        return True
    except Exception:
        return False


def prepare_model_files():
    # Phase 1 candidates (checked by phase1_autoencoder loader).
    maybe_download_model(Path("outputs/ae_model.pth"), PHASE1_DOWNLOAD_URL)
    # Phase 2 main checkpoint.
    maybe_download_model(Path(CFG.CHECKPOINT), PHASE2_DOWNLOAD_URL)

# ─────────────────────────────────────────────
# Load classification model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = PCBViT().to(CFG.device)
    load_checkpoint(CFG.CHECKPOINT, model)
    return model

# ─────────────────────────────────────────────
# Load segmentation model (optional)
# ─────────────────────────────────────────────
@st.cache_resource
def load_seg_model():
    seg_model = UNet().to(CFG.device)
    seg_path = CFG.output_dir / "unet.pth"

    if seg_path.exists():
        seg_model.load_state_dict(torch.load(seg_path, map_location=CFG.device))
        seg_model.eval()
        return seg_model
    return None


@st.cache_resource
def load_phase1():
    return load_phase1_model()

def append_prediction_log(
    image_name,
    phase1_result,
    phase1_score,
    phase2_result,
    phase2_confidence,
):
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = PREDICTION_LOG_PATH.exists()
    with open(PREDICTION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "image_name",
                "phase1_result",
                "phase1_score",
                "phase2_result",
                "phase2_confidence",
            ])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            image_name,
            phase1_result,
            f"{phase1_score:.6f}" if phase1_score is not None else "",
            phase2_result,
            f"{phase2_confidence:.6f}" if phase2_confidence is not None else "",
        ])


prepare_model_files()
try:
    model = load_model()
except Exception as e:
    st.error(
        "Phase 2 checkpoint could not be loaded. "
        "Place model at `outputs/best_pcb_vit_s.pth` or set `PHASE2_MODEL_URL`."
    )
    st.exception(e)
    st.stop()
seg_model = load_seg_model()
phase1_model = load_phase1()

# ─────────────────────────────────────────────
# File upload
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg", "png"])

if uploaded_file:
    image_name = uploaded_file.name
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Save temporary image
    temp_path = "temp.jpg"
    img.save(temp_path)

    st.subheader("Phase 1: Autoencoder Binary Detection")
    if phase1_model is None:
        searched = "\n".join(str(p) for p in Phase1Config.checkpoint_candidates)
        st.warning(f"Phase 1 checkpoint not found. Searched:\n{searched}")
        st.info("Upload or train autoencoder checkpoint to enable two-phase flow.")
        append_prediction_log(
            image_name=image_name,
            phase1_result="checkpoint_missing",
            phase1_score=None,
            phase2_result="not_run",
            phase2_confidence=None,
        )
    else:
        phase1_label, phase1_score = phase1_predict(phase1_model, temp_path)
        st.write(f"Phase 1 Result: **{phase1_label.upper()}**")
        st.write(f"Reconstruction Error: **{phase1_score:.6f}**")
        st.write(f"Threshold: **{Phase1Config.anomaly_threshold:.6f}**")

        if phase1_label == "anomaly":
            st.subheader("Phase 2: Transformer (6-class anomaly type)")
            label, probs, _ = visualize(
                model,
                temp_path,
                save_path=None,
                seg_model=seg_model
            )
            probs = np.asarray(probs, dtype=float)
            conf = float(probs.max())
            if conf < PHASE2_UNCERTAIN_THRESHOLD:
                final_label = "uncertain_anomaly"
                st.warning(
                    f"Anomaly Type: {final_label} "
                    f"(best class confidence {conf:.2%} < {PHASE2_UNCERTAIN_THRESHOLD:.0%})"
                )
            else:
                final_label = label
                st.success(f"Anomaly Type: {final_label}")
            st.write(f"Confidence: {conf:.2%}")
            st.pyplot(plt.gcf())

            st.subheader("📊 Anomaly Type Probabilities")
            for cls, p in zip(CFG.classes, probs):
                st.write(f"{cls}: {p:.2%}")
            append_prediction_log(
                image_name=image_name,
                phase1_result=phase1_label,
                phase1_score=phase1_score,
                phase2_result=final_label,
                phase2_confidence=conf,
            )
        else:
            st.success("Board looks normal. Phase 2 skipped.")
            append_prediction_log(
                image_name=image_name,
                phase1_result=phase1_label,
                phase1_score=phase1_score,
                phase2_result="not_run_normal",
                phase2_confidence=None,
            )

    # Cleanup temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)