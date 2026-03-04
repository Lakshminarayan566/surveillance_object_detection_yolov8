"""
evaluate.py
-----------
Evaluates the trained YOLOv8 model on the validation set.
Reports:
  - mAP@0.5
  - mAP@0.5:0.95
  - Precision, Recall, F1
  - Per-class breakdown (vehicle vs human)
  - Confusion matrix
  - Performance under challenging conditions (occlusion, low-light)

Usage:
  python evaluate.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "surveillance_yolov8/defence_v1/weights/best.pt"
DATASET_YAML = "dataset.yaml"
IMG_SIZE     = 640
CONF_THRESH  = 0.45
IOU_THRESH   = 0.45
OUTPUT_DIR   = "outputs/evaluation"
CLASS_NAMES  = ["vehicle", "human"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_evaluation(model):
    """Run full mAP evaluation on validation set."""
    print("── Running Validation Evaluation ────────────────────────────")
    metrics = model.val(
        data    = DATASET_YAML,
        imgsz   = IMG_SIZE,
        conf    = CONF_THRESH,
        iou     = IOU_THRESH,
        verbose = True,
        plots   = True,
        save_json = True,
    )

    print("\n── Evaluation Results ───────────────────────────────────────")
    print(f"  mAP@0.5        : {metrics.box.map50:.4f}  ({metrics.box.map50*100:.2f}%)")
    print(f"  mAP@0.5:0.95   : {metrics.box.map:.4f}   ({metrics.box.map*100:.2f}%)")
    print(f"  Precision      : {metrics.box.mp:.4f}")
    print(f"  Recall         : {metrics.box.mr:.4f}")

    print("\n── Per-Class Results ────────────────────────────────────────")
    for i, name in enumerate(CLASS_NAMES):
        if i < len(metrics.box.ap50):
            print(f"  {name:<10} AP@0.5: {metrics.box.ap50[i]*100:.2f}%")

    return metrics


def benchmark_fps(model, num_frames=100):
    """Benchmark inference speed on random frames."""
    print("\n── FPS Benchmark ────────────────────────────────────────────")
    import time

    times = []
    dummy_frame = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        model.predict(dummy_frame, conf=CONF_THRESH, verbose=False)

    # Benchmark
    for _ in range(num_frames):
        t0 = time.time()
        model.predict(dummy_frame, conf=CONF_THRESH, imgsz=IMG_SIZE, verbose=False)
        times.append(time.time() - t0)

    avg_ms  = (sum(times) / len(times)) * 1000
    avg_fps = 1000 / avg_ms
    min_fps = 1 / max(times)
    max_fps = 1 / min(times)

    print(f"  Avg inference  : {avg_ms:.1f} ms")
    print(f"  Avg FPS        : {avg_fps:.1f}")
    print(f"  Min FPS        : {min_fps:.1f}")
    print(f"  Max FPS        : {max_fps:.1f}")
    print(f"  Resolution     : {IMG_SIZE}×{IMG_SIZE}")

    return avg_fps


def evaluate_challenging_conditions(model, val_images_dir):
    """
    Evaluate on subsets of challenging images.
    Assumes images are named with tags:
      - *_dark* or *_night*  → low-light
      - *_occ*               → occlusion
      - *_shift*             → viewpoint shift
    """
    print("\n── Challenging Conditions Evaluation ────────────────────────")

    conditions = {
        "low_light":       ["dark", "night", "lowlight"],
        "occlusion":       ["occ", "occluded"],
        "viewpoint_shift": ["shift", "aerial", "tilt"],
    }

    val_path = Path(val_images_dir)
    if not val_path.exists():
        print(f"  ⚠️  Val images dir not found: {val_images_dir}")
        print("     Skipping challenging conditions evaluation.")
        return

    for condition, tags in conditions.items():
        matched = []
        for tag in tags:
            matched.extend(list(val_path.glob(f"*{tag}*")))

        if not matched:
            print(f"  {condition:<20}: No tagged images found (tag images with _{tags[0]}_)")
            continue

        correct = 0
        total   = len(matched)
        for img_path in matched:
            frame   = cv2.imread(str(img_path))
            results = model.predict(frame, conf=CONF_THRESH,
                                    imgsz=IMG_SIZE, verbose=False)
            if results and len(results[0].boxes) > 0:
                correct += 1

        detection_rate = correct / total * 100
        print(f"  {condition:<20}: {detection_rate:.1f}% detection rate ({correct}/{total} images)")


def plot_confidence_distribution(model, val_images_dir):
    """Plot confidence score distribution of detections."""
    print("\n── Plotting Confidence Distribution ─────────────────────────")

    val_path = Path(val_images_dir)
    if not val_path.exists():
        print(f"  ⚠️  Val images dir not found. Skipping plot.")
        return

    images = list(val_path.glob("*.jpg"))[:200]  # sample 200 images
    all_confs = []

    for img_path in images:
        frame   = cv2.imread(str(img_path))
        results = model.predict(frame, conf=0.1,  # low threshold to capture all
                                imgsz=IMG_SIZE, verbose=False)
        for r in results:
            all_confs.extend(r.boxes.conf.cpu().numpy().tolist())

    if not all_confs:
        print("  No detections found for confidence plot.")
        return

    plt.figure(figsize=(8, 4))
    plt.hist(all_confs, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
    plt.axvline(CONF_THRESH, color="red", linestyle="--",
                label=f"Threshold = {CONF_THRESH}")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.title("Detection Confidence Distribution")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "confidence_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  ✅ Confidence distribution saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 1. Full mAP evaluation
    metrics = run_evaluation(model)

    # 2. FPS benchmark
    avg_fps = benchmark_fps(model)

    # 3. Challenging conditions (update path to your val images)
    val_images = "datasets/images/val"
    evaluate_challenging_conditions(model, val_images)

    # 4. Confidence distribution plot
    plot_confidence_distribution(model, val_images)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("FINAL EVALUATION SUMMARY")
    print("="*55)
    print(f"  mAP@0.5        : {metrics.box.map50*100:.2f}%")
    print(f"  mAP@0.5:0.95   : {metrics.box.map*100:.2f}%")
    print(f"  Precision      : {metrics.box.mp*100:.2f}%")
    print(f"  Recall         : {metrics.box.mr*100:.2f}%")
    print(f"  Avg FPS        : {avg_fps:.1f} @ {IMG_SIZE}×{IMG_SIZE}")
    print("="*55)
    print(f"\n📊 All plots saved → {OUTPUT_DIR}/")