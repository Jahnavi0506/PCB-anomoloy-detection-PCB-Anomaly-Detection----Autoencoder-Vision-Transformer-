# PCB Defect Detection System (Two-Phase Pipeline)

This project detects PCB defects using a two-stage inference flow:

- Phase 1: Autoencoder binary detection (`normal` vs `anomaly`)
- Phase 2: Transformer-based anomaly classification (6 classes), only when Phase 1 predicts anomaly

## Features

- Streamlit UI for image upload and visualization
- Two-phase decision pipeline for efficient inference
- Grad-CAM-based visual explanation from Phase 2 model
- Optional segmentation model loading (`unet.pth`) for localization fallback
- Prediction logging to CSV (`outputs/predictions_log.csv`)
- Uncertain anomaly handling when Phase 2 confidence is low

## Project Structure

- `app.py` - Streamlit application (two-phase pipeline)
- `phase1_autoencoder.py` - Phase 1 model loading and binary inference
- `test.py` - Phase 2 testing and visualization utilities
- `transformer_updated.py` - Phase 2 training pipeline
- `PCB_dataset_processing1.py` - Autoencoder training/evaluation script
- `segmentation_model.py` - U-Net model definition
- `outputs/` - Model checkpoints, heatmaps, and logs

## Phase 2 Classes

- `missing_hole`
- `mouse_bite`
- `open_circuit`
- `short`
- `spur`
- `spurious_copper`

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training From Scratch

Before training, make sure dataset paths inside scripts match your local folders.

### Phase 1: Autoencoder (binary normal vs anomaly)

Train with:

```bash
python PCB_dataset_processing1.py
```

Expected checkpoint output:

- `ae_model.pth` (project root by default in current script)

Recommended move/copy after training:

```bash
mkdir outputs
copy ae_model.pth outputs\ae_model.pth
```

### Phase 2: Transformer (6 anomaly classes)

Train with:

```bash
python transformer_updated.py
```

Expected checkpoint output:

- `outputs/best_pcb_vit_s.pth`
- intermediate checkpoints like `outputs/ckpt_s_epoch*.pth`

### Optional: Segmentation model

If you train a segmentation model separately, place:

- `outputs/unet.pth`

The app auto-loads this file if available.

## Model Checkpoints

Place checkpoints in these locations:

- Phase 1 autoencoder:
  - `ae_model.pth` in project root, or
  - `outputs/ae_model.pth`
- Phase 2 transformer:
  - `outputs/best_pcb_vit_s.pth`
- Optional segmentation:
  - `outputs/unet.pth`

## Run Streamlit App

From project root:

```bash
streamlit run app.py
```

Open the URL shown in terminal (usually `http://localhost:8501`).

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create a new app and select this repo.
3. Set main file path to `app.py`.
4. Add optional environment variables in app settings:
   - `PHASE1_MODEL_URL` - direct download URL for `ae_model.pth`
   - `PHASE2_MODEL_URL` - direct download URL for `best_pcb_vit_s.pth`
   - `PHASE1_CHECKPOINT` - custom local path for phase-1 checkpoint (optional)
   - `PHASE2_CHECKPOINT` - custom local path for phase-2 checkpoint (optional)
5. Deploy.

If checkpoint files are not present in the container, the app will try to download them from `PHASE1_MODEL_URL` and `PHASE2_MODEL_URL` at startup.

## Inference Logic

1. Upload PCB image
2. Phase 1 computes reconstruction error
3. If Phase 1 is `normal`, stop and skip Phase 2
4. If Phase 1 is `anomaly`, run Phase 2 for 6-class anomaly type
5. If Phase 2 confidence is below threshold, output `uncertain_anomaly`

## Output Artifacts

- `outputs/predictions_log.csv` - inference history
- `outputs/heatmap_s_*.png` - Grad-CAM visualizations from test/app flow

## Notes

- If app reports Phase 1 checkpoint missing, verify the `ae_model.pth` path.
- If architecture mismatch occurs while loading Phase 1, loader auto-fallback builds a compatible model from checkpoint shapes.
- Use CPU by default when CUDA is unavailable.
- For cloud deployment, avoid hardcoded local absolute paths; use environment variables and relative `outputs/` paths.

## Quick Verification

After launching Streamlit:

- Upload a known normal image:
  - Expect Phase 1 = `NORMAL`
  - Phase 2 skipped
- Upload a known anomaly image:
  - Expect Phase 1 = `ANOMALY`
  - Phase 2 returns one of 6 classes (or `uncertain_anomaly`)

