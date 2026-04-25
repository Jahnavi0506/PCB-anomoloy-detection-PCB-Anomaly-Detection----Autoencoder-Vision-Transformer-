import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

############################################
# DEVICE
############################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

############################################
# DATASET (MEMORY SAFE)
############################################
class PCBDataset(Dataset):
    def __init__(self, folder):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return img

############################################
# AUTOENCODER (MEMORY OPTIMIZED)
############################################
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

############################################
# TRAIN
############################################
train_dir = r"E:\PCB_MiniProject\stage1_anomaly\train\normal"

dataset = PCBDataset(train_dir)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

model = AutoEncoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("\nStarting training...\n")

for epoch in range(15):
    total_loss = 0

    for imgs in loader:
        imgs = imgs.to(device)

        recon = model(imgs)
        loss = criterion(recon, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        del recon
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.6f}")

torch.save(model.state_dict(), "ae_model.pth")
print("\nModel saved ✅")

############################################
# ANOMALY SCORE
############################################
def anomaly_score(img, recon):
    return torch.mean((img - recon) ** 2).item()

############################################
# EVALUATION (MEMORY SAFE)
############################################
test_normal = r"E:\PCB_MiniProject\stage1_anomaly\test\normal"
test_anomaly = r"E:\PCB_MiniProject\stage1_anomaly\test\anomaly"

scores = []
labels = []

model.eval()
torch.set_grad_enabled(False)

def eval_folder(folder, label, limit=None):
    for i, f in enumerate(os.listdir(folder)):
        if limit and i >= limit:
            break

        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128,128))
        img = img / 255.0

        img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)

        recon = model(img)
        score = anomaly_score(img, recon)

        scores.append(score)
        labels.append(label)

        del img, recon
        if device == "cuda":
            torch.cuda.empty_cache()

print("\nEvaluating model...\n")

eval_folder(test_normal, 0)
eval_folder(test_anomaly, 1)

auc = roc_auc_score(labels, scores)
print("🔥 AUROC:", round(auc, 4))

############################################
# HEATMAP WITH CLASSIFICATION
############################################
print("\n" + "="*60)
print("GENERATING HEATMAP WITH CLASSIFICATION")
print("="*60 + "\n")

# Set threshold (adjust based on your data)
THRESHOLD = 0.015

# Output folder for heatmap
OUTPUT_FOLDER = r"E:\PCB_MiniProject"

# CORRECTED IMAGE PATH - Use forward slashes or raw string
sample_img = r"E:\PCB_MiniProject\stage1_anomaly\train\normal\00041055_temp.jpg"

# Check if file exists
if not os.path.exists(sample_img):
    print(f"❌ ERROR: File does not exist!")
    print(f"   Looking for: {sample_img}")
    print(f"\n💡 Available files in anomaly folder:")
    anomaly_folder = r"E:\PCB_MiniProject\stage1_anomaly\test\anomaly"
    if os.path.exists(anomaly_folder):
        files = [f for f in os.listdir(anomaly_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for f in files[:10]:  # Show first 10 files
            print(f"   - {f}")
        if len(files) > 10:
            print(f"   ... and {len(files)-10} more files")
    else:
        print(f"   Folder not found: {anomaly_folder}")
else:
    img = cv2.imread(sample_img, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"❌ ERROR: Image file exists but cannot be read (corrupted?)")
        print(f"   Path: {sample_img}")
    else:
        img = cv2.resize(img, (128, 128))
        img_normalized = img / 255.0

        img_tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            recon = model(img_tensor)

        # Calculate anomaly score
        anomaly_score_value = torch.mean((img_tensor - recon) ** 2).item()
        
        # Classify
        if anomaly_score_value > THRESHOLD:
            classification = "ANOMALY"
            color = "red"
        else:
            classification = "NORMAL"
            color = "green"
        
        print(f"{'='*50}")
        print(f"🎯 CLASSIFICATION: {classification}")
        print(f"📊 Anomaly Score: {anomaly_score_value:.6f}")
        print(f"📏 Threshold: {THRESHOLD:.6f}")
        print(f"{'='*50}\n")

        # Generate heatmap
        heatmap = torch.mean((img_tensor - recon) ** 2, dim=1).squeeze().cpu().numpy()
        recon_img = recon.squeeze().cpu().numpy()

        # Create visualization with 3 panels
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original Image
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Original Image", fontsize=14, weight='bold')
        axes[0].axis("off")
        
        # Reconstructed Image
        axes[1].imshow(recon_img, cmap="gray")
        axes[1].set_title("Reconstructed", fontsize=14, weight='bold')
        axes[1].axis("off")
        
        # Heatmap with Classification
        im = axes[2].imshow(heatmap, cmap="jet")
        axes[2].set_title(f"Anomaly Heatmap\n{classification}", 
                          fontsize=14, color=color, weight='bold')
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Add score text below heatmap
        axes[2].text(0.5, -0.1, f"Score: {anomaly_score_value:.6f}", 
                    transform=axes[2].transAxes, ha='center', fontsize=12)
        
        plt.tight_layout()
        
        # Get original filename without extension
        original_filename = os.path.basename(sample_img)
        filename_without_ext = os.path.splitext(original_filename)[0]
        
        # Create output path in specified folder
        output_path = os.path.join(OUTPUT_FOLDER, f"{filename_without_ext}_heatmap.png")
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Heatmap saved: {output_path}\n")
        print("="*60)
        print("PROCESS COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)

        del img_tensor, recon
        if device == "cuda":
            torch.cuda.empty_cache()