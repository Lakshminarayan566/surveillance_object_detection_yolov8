"""
train.py
--------
Fine-tunes YOLOv8 on VisDrone / xView aerial surveillance datasets.
Includes mosaic augmentation and multi-scale training.

Dataset setup:
- VisDrone: https://github.com/VisDrone/VisDrone-Dataset
- xView:    https://xviewdataset.org/

Folder structure expected:
datasets/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
"""

from ultralytics import YOLO
import yaml
import os

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_YAML   = "dataset.yaml"       # path to your dataset config
MODEL_BASE     = "yolov8m.pt"         # yolov8n / yolov8s / yolov8m / yolov8l
PROJECT_NAME   = "surveillance_yolov8"
RUN_NAME       = "defence_v1"
EPOCHS         = 10
IMG_SIZE       = 640
BATCH_SIZE     = 16
DEVICE         = "cpu"                # change to 0 for GPU, "mps" for Mac M1/M2

# ── Create dataset.yaml if not exists ────────────────────────────────────────
def create_dataset_yaml():
    config = {
        "path": "./datasets",
        "train": "images/val",
        "val":   "images/val",
        "nc":    2,                   # number of classes
        "names": ["vehicle", "human"] # class names
    }
    with open(DATASET_YAML, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ dataset.yaml created → {DATASET_YAML}")

# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    if not os.path.exists(DATASET_YAML):
        create_dataset_yaml()

    print(f"Loading base model: {MODEL_BASE}")
    model = YOLO(MODEL_BASE)

    print("Starting training...")
    results = model.train(
        data       = DATASET_YAML,
        epochs     = EPOCHS,
        imgsz      = IMG_SIZE,
        batch      = BATCH_SIZE,
        device     = DEVICE,
        project    = PROJECT_NAME,
        name       = RUN_NAME,

        # ── Augmentation (mosaic + multi-scale) ──────────────────────────────
        mosaic     = 1.0,        # mosaic augmentation probability
        mixup      = 0.1,        # mixup augmentation
        scale      = 0.5,        # multi-scale training (±50% image size)
        fliplr     = 0.5,        # horizontal flip
        flipud     = 0.0,        # vertical flip (off for aerial)
        degrees    = 10.0,       # rotation augmentation
        translate  = 0.1,        # translation augmentation
        hsv_h      = 0.015,      # HSV hue augmentation
        hsv_s      = 0.7,        # HSV saturation
        hsv_v      = 0.4,        # HSV value (brightness — helps low-light)

        # ── Training settings ─────────────────────────────────────────────────
        optimizer  = "AdamW",
        lr0        = 0.001,
        lrf        = 0.01,
        warmup_epochs = 3,
        patience   = 20,          # early stopping
        save       = True,
        plots      = True,
        verbose    = True,
    )

    print("\n✅ Training complete!")
    print(f"Best model saved → {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    return results

if __name__ == "__main__":
    train()