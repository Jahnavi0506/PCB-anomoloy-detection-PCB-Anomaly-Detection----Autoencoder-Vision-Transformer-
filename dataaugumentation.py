import os
import cv2
import numpy as np
from itertools import combinations

# INPUT DATASET
input_root = r"C:\Users\Sastra\OneDrive\Desktop\PCB_mini\stage2_classification\train"

# OUTPUT DATASET
output_root = r"C:\Users\Sastra\OneDrive\Desktop\PCB_mini\stage2_classification\augumented_data"

os.makedirs(output_root, exist_ok=True)

# ─────────────────────────────────────────────
#  Augmentation Functions
# ─────────────────────────────────────────────

def add_noise(img, std=25):
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def brighten(img, factor=1.3):
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

def darken(img, factor=0.7):
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

def adjust_contrast(img, alpha=1.5):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def zoom(img, factor=1.2):
    h, w = img.shape[:2]
    new_h, new_w = int(h / factor), int(w / factor)
    y1, x1 = (h - new_h) // 2, (w - new_w) // 2
    return cv2.resize(img[y1:y1 + new_h, x1:x1 + new_w], (w, h))

def translate(img, tx, ty):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))

def shear(img, shear_factor=0.2):
    h, w = img.shape[:2]
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h))

def blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def gamma_correction(img, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)

def equalize_hist(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def resize_variants(img, scale):
    h, w = img.shape[:2]
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    return cv2.resize(resized, (w, h))  # resize back to original

def elastic_distortion(img, alpha=30, sigma=4):
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1) * alpha, (0, 0), sigma)
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1) * alpha, (0, 0), sigma)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

def channel_shuffle(img):
    channels = list(cv2.split(img))
    np.random.shuffle(channels)
    return cv2.merge(channels)

def cutout(img, num_patches=3, patch_size=20):
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(num_patches):
        x = np.random.randint(0, w - patch_size)
        y = np.random.randint(0, h - patch_size)
        out[y:y + patch_size, x:x + patch_size] = 0
    return out

# ─────────────────────────────────────────────
#  Build Augmentation Dictionary
# ─────────────────────────────────────────────

def get_augmentations(img):
    return {
        # Basic transforms
        "_orig":            img,
        "_flipH":           cv2.flip(img, 1),
        "_flipV":           cv2.flip(img, 0),
        "_flipBoth":        cv2.flip(img, -1),

        # Rotations
        "_rot15":           rotate(img, 15),
        "_rot-15":          rotate(img, -15),
        "_rot30":           rotate(img, 30),
        "_rot-30":          rotate(img, -30),
        "_rot45":           rotate(img, 45),
        "_rot-45":          rotate(img, -45),
        "_rot60":           rotate(img, 60),
        "_rot90":           rotate(img, 90),
        "_rot180":          rotate(img, 180),
        "_rot270":          rotate(img, 270),

        # Brightness / Contrast
        "_bright1.3":       brighten(img, 1.3),
        "_bright1.6":       brighten(img, 1.6),
        "_dark0.7":         darken(img, 0.7),
        "_dark0.5":         darken(img, 0.5),
        "_contrast1.5":     adjust_contrast(img, 1.5),
        "_contrast2.0":     adjust_contrast(img, 2.0),
        "_gamma0.5":        gamma_correction(img, 0.5),
        "_gamma1.5":        gamma_correction(img, 1.5),
        "_gamma2.0":        gamma_correction(img, 2.0),
        "_equalhist":       equalize_hist(img),

        # Noise
        "_noise15":         add_noise(img, 15),
        "_noise25":         add_noise(img, 25),
        "_noise40":         add_noise(img, 40),

        # Blur / Sharpen
        "_blur3":           blur(img, 3),
        "_blur5":           blur(img, 5),
        "_blur7":           blur(img, 7),
        "_sharpen":         sharpen(img),

        # Zoom
        "_zoom1.2":         zoom(img, 1.2),
        "_zoom1.5":         zoom(img, 1.5),
        "_zoom1.8":         zoom(img, 1.8),

        # Resize variants
        "_resize0.8":       resize_variants(img, 0.8),
        "_resize0.5":       resize_variants(img, 0.5),

        # Translate
        "_trans+20+20":     translate(img, 20, 20),
        "_trans-20-20":     translate(img, -20, -20),
        "_trans+40+0":      translate(img, 40, 0),
        "_trans0+40":       translate(img, 0, 40),

        # Shear
        "_shear0.2":        shear(img, 0.2),
        "_shear-0.2":       shear(img, -0.2),

        # Advanced
        "_elastic":         elastic_distortion(img),
        "_cutout":          cutout(img),
        "_chshuffle":       channel_shuffle(img),

        # ── Combinations ──────────────────────────
        "_flipH_bright":    brighten(cv2.flip(img, 1), 1.3),
        "_flipH_dark":      darken(cv2.flip(img, 1), 0.7),
        "_flipV_noise":     add_noise(cv2.flip(img, 0), 25),
        "_rot45_bright":    brighten(rotate(img, 45), 1.3),
        "_rot90_noise":     add_noise(rotate(img, 90), 25),
        "_rot180_dark":     darken(rotate(img, 180), 0.7),
        "_zoom_sharpen":    sharpen(zoom(img, 1.2)),
        "_zoom_bright":     brighten(zoom(img, 1.2), 1.3),
        "_noise_blur":      blur(add_noise(img, 25), 3),
        "_contrast_gamma":  gamma_correction(adjust_contrast(img, 1.5), 1.5),
        "_flipH_rot45":     rotate(cv2.flip(img, 1), 45),
        "_flipV_zoom":      zoom(cv2.flip(img, 0), 1.2),
        "_dark_sharpen":    sharpen(darken(img, 0.7)),
        "_bright_blur":     blur(brighten(img, 1.3), 3),
        "_elastic_bright":  brighten(elastic_distortion(img), 1.3),
        "_cutout_noise":    add_noise(cutout(img), 20),
        "_trans_rot15":     rotate(translate(img, 20, 20), 15),
        "_shear_bright":    brighten(shear(img, 0.2), 1.3),
        "_rot45_zoom":      zoom(rotate(img, 45), 1.2),
        "_equalhist_noise": add_noise(equalize_hist(img), 15),
    }

# ─────────────────────────────────────────────
#  Main Loop
# ─────────────────────────────────────────────

total_saved = 0

for class_name in os.listdir(input_root):

    class_input_path = os.path.join(input_root, class_name)
    if not os.path.isdir(class_input_path):
        continue

    class_output_path = os.path.join(output_root, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    print(f"\nProcessing class: {class_name}")

    class_count = 0

    for filename in os.listdir(class_input_path):
        img_path = os.path.join(class_input_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"  ⚠️  Skipping: {filename}")
            continue

        base_name = os.path.splitext(filename)[0]
        augmentations = get_augmentations(img)

        for suffix, aug_img in augmentations.items():
            out_path = os.path.join(class_output_path, base_name + suffix + ".jpg")
            cv2.imwrite(out_path, aug_img)
            class_count += 1

    total_saved += class_count
    print(f"  ✅ Saved {class_count} images for class: {class_name}")

print(f"\n✅ Augmentation complete! Total images saved: {total_saved}")