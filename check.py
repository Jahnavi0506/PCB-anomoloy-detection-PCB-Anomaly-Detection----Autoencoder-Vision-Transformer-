from pathlib import Path
from collections import Counter

data_root = Path(r"C:\Users\Sastra\OneDrive\Desktop\PCB_mini\stage2_classification\augumented_data")
classes = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

print("Class distribution:")
for cls in classes:
    imgs = list((data_root / cls).glob("*.jpg")) + list((data_root / cls).glob("*.png"))
    print(f"  {cls:<20} : {len(imgs):>4} images")