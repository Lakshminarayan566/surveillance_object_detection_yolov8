import os
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
WEIGHTS_PATH   = "weights/best.pt"
IMAGE_FOLDER   = "images/"
OUTPUT_FOLDER  = "results/sample_predictions/"
CONF_THRESHOLD = 0.3
DEVICE         = "cuda:0"  # change to "cpu" if no GPU

# SAHI Slicing Settings
SLICE_HEIGHT   = 512
SLICE_WIDTH    = 512
OVERLAP_H      = 0.2
OVERLAP_W      = 0.2

# VisDrone Class Names
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights not found at {WEIGHTS_PATH}")
    print("Loading YOLOv8l model...")
    model = AutoDetectionModel.from_pretrained(
        model_type           = "ultralytics",
        model_path           = WEIGHTS_PATH,
        confidence_threshold = CONF_THRESHOLD,
        device               = DEVICE,
    )
    print("✅ Model loaded successfully\n")
    return model

# ─────────────────────────────────────────────
# RUN INFERENCE
# ─────────────────────────────────────────────
def run_inference(model, img_path):
    result = get_sliced_prediction(
        image                       = str(img_path),
        detection_model             = model,
        slice_height                = SLICE_HEIGHT,
        slice_width                 = SLICE_WIDTH,
        overlap_height_ratio        = OVERLAP_H,
        overlap_width_ratio         = OVERLAP_W,
        perform_standard_pred       = True,
        postprocess_type            = "NMM",
        postprocess_match_threshold = 0.5,
        verbose                     = 0,
    )
    return result

# ─────────────────────────────────────────────
# PRINT DETECTION SUMMARY
# ─────────────────────────────────────────────
def print_summary(img_name, detections):
    print(f"📷 {img_name} → {len(detections)} object(s) detected")
    class_counts = {}
    for det in detections:
        label = det.category.name
        class_counts[label] = class_counts.get(label, 0) + 1
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"   {cls:<20} : {cnt}")
    print()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Setup
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Check image folder
    if not os.path.exists(IMAGE_FOLDER):
        raise FileNotFoundError(f"Image folder not found: {IMAGE_FOLDER}")

    # Get images
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in Path(IMAGE_FOLDER).iterdir() if p.suffix.lower() in supported]

    if not images:
        print(f"❌ No images found in {IMAGE_FOLDER}")
        return

    print(f"Found {len(images)} image(s). Starting SAHI inference...\n")

    # Load model once
    model = load_model()

    # Process each image
    for img_path in images:
        result = run_inference(model, img_path)
        print_summary(img_path.name, result.object_prediction_list)

        # Save annotated image
        result.export_visuals(
            export_dir = OUTPUT_FOLDER,
            file_name  = img_path.stem,
        )
        print(f"   ✅ Saved → {OUTPUT_FOLDER}{img_path.stem}.png\n")

    print("─" * 40)
    print(f"✅ Done! Results saved to '{OUTPUT_FOLDER}'")

if __name__ == "__main__":
    main()
