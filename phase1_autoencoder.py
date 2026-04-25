from pathlib import Path
import os

import cv2
import numpy as np
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AutoEncoderFromCheckpoint(nn.Module):
    def __init__(self, enc_c1: int, enc_c2: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, enc_c1, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(enc_c1, enc_c2, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Matches checkpoints that used transposed-convolution decoder.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(enc_c2, enc_c1, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(enc_c1, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class Phase1Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 128
    # Keep this close to your training script default; tune with validation data.
    anomaly_threshold = 0.015
    checkpoint_candidates = [
        Path(os.getenv("PHASE1_CHECKPOINT", "ae_model.pth")),
        Path("outputs/ae_model.pth"),
    ]
    resolved_checkpoint = None


def _preprocess_for_autoencoder(img_bgr: np.ndarray) -> torch.Tensor:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (Phase1Config.image_size, Phase1Config.image_size))
    gray = gray.astype(np.float32) / 255.0
    x = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
    return x.to(Phase1Config.device)


@torch.no_grad()
def phase1_predict(model: AutoEncoder, image_path: str):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    x = _preprocess_for_autoencoder(img_bgr)
    recon = model(x)
    score = torch.mean((x - recon) ** 2).item()
    is_anomaly = score > Phase1Config.anomaly_threshold

    result = "anomaly" if is_anomaly else "normal"
    return result, score


def load_phase1_model():
    ckpt_path = None
    for candidate in Phase1Config.checkpoint_candidates:
        if candidate.exists():
            ckpt_path = candidate
            break

    if ckpt_path is None:
        Phase1Config.resolved_checkpoint = None
        return None

    Phase1Config.resolved_checkpoint = ckpt_path
    state = torch.load(ckpt_path, map_location=Phase1Config.device)
    # Try the default architecture first.
    model = AutoEncoder().to(Phase1Config.device)
    try:
        model.load_state_dict(state)
        model.eval()
        return model
    except RuntimeError:
        pass

    # Fallback: build architecture to match checkpoint channels/layout.
    enc_c1 = int(state["encoder.0.weight"].shape[0])
    enc_c2 = int(state["encoder.2.weight"].shape[0])
    model = AutoEncoderFromCheckpoint(enc_c1=enc_c1, enc_c2=enc_c2).to(Phase1Config.device)
    model.load_state_dict(state)
    model.eval()
    return model
