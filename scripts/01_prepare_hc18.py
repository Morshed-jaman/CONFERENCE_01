from pathlib import Path
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Project root = FETAL_SR_PROJECT
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "HC18"
TRAIN_IMG_DIR = RAW_DIR / "training_set"

OUT_DIR = ROOT / "data" / "processed" / "HC18_SR"
HR_DIR = OUT_DIR / "HR"
MASK_DIR = OUT_DIR / "masks"

HR_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

# Find all png files
all_pngs = sorted(list(TRAIN_IMG_DIR.rglob("*.png")))

# Remove annotation/mask files from HR image list
image_paths = [
    p for p in all_pngs
    if "annotation" not in p.name.lower()
]

print(f"Found {len(image_paths)} ultrasound images")

records = []

for i, img_path in enumerate(tqdm(image_paths)):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Could not read image: {img_path}")
        continue

    # Resize image
    img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    out_name = f"hc18_{i:04d}.png"
    out_path = HR_DIR / out_name
    cv2.imwrite(str(out_path), img_resized)

    # Try to find matching annotation/mask
    possible_masks = [
        img_path.with_name(img_path.stem + "_Annotation.png"),
        img_path.with_name(img_path.stem + "_annotation.png"),
        img_path.with_name(img_path.stem.replace("_HC", "_Annotation") + ".png"),
        img_path.with_name(img_path.stem.replace("_2HC", "_Annotation") + ".png"),
        img_path.with_name(img_path.stem.replace("_3HC", "_Annotation") + ".png"),
    ]

    mask_path_found = None

    for mp in possible_masks:
        if mp.exists():
            mask_path_found = mp
            break

    saved_mask_path = ""

    if mask_path_found is not None:
        mask = cv2.imread(str(mask_path_found), cv2.IMREAD_GRAYSCALE)

        if mask is not None:
            mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            saved_mask_path = MASK_DIR / out_name
            cv2.imwrite(str(saved_mask_path), mask_resized)

    records.append({
        "image_id": f"hc18_{i:04d}",
        "original_name": img_path.name,
        "hr_path": str(out_path),
        "mask_path": str(saved_mask_path),
        "source_path": str(img_path),
        "has_mask": 1 if saved_mask_path != "" else 0,
        "dataset": "HC18"
    })

df = pd.DataFrame(records)

# 70 / 15 / 15 split
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

final_df.to_csv(OUT_DIR / "hc18_manifest.csv", index=False)
train_df.to_csv(OUT_DIR / "train.csv", index=False)
val_df.to_csv(OUT_DIR / "val.csv", index=False)
test_df.to_csv(OUT_DIR / "test.csv", index=False)

print("Done preparing HC18.")
print(final_df["split"].value_counts())

print("Mask availability:")
print(final_df["has_mask"].value_counts())

print(f"Saved to: {OUT_DIR}")