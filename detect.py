"""
detect.py
---------
Real-time YOLOv8 inference supporting:
  - Video file
  - Webcam / live camera
  - Images folder

Usage:
  python detect.py --source video    --input path/to/video.mp4
  python detect.py --source webcam
  python detect.py --source images   --input path/to/images/
"""

import cv2
import argparse
import os
import time
from pathlib import Path
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH        = "surveillance_yolov8/defence_v1/weights/best.pt"
CONFIDENCE_THRESH = 0.45      # suppress false alarms below this threshold
IOU_THRESH        = 0.45      # NMS IoU threshold
IMG_SIZE          = 640
OUTPUT_DIR        = "outputs/detections"
CLASS_NAMES       = ["vehicle", "human"]
CLASS_COLORS      = {
    "vehicle": (0, 255, 100),   # green
    "human":   (0, 100, 255),   # orange
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    return model


def draw_detections(frame, results):
    """Draw bounding boxes and labels on frame."""
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
            color = CLASS_COLORS.get(label, (255, 255, 255))

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return frame


def run_on_video(model, input_path):
    """Run detection on a video file."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {input_path}")
        return

    # Output video writer
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(OUTPUT_DIR, "output_video.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))

    frame_count = 0
    total_time  = 0

    print(f"Running detection on video: {input_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        results = model.predict(
            frame,
            conf    = CONFIDENCE_THRESH,
            iou     = IOU_THRESH,
            imgsz   = IMG_SIZE,
            verbose = False
        )
        t1 = time.time()

        elapsed     = t1 - t0
        total_time += elapsed
        frame_count += 1
        fps = 1 / elapsed if elapsed > 0 else 0

        frame = draw_detections(frame, results)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        writer.write(frame)
        cv2.imshow("Surveillance Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\n✅ Done! Processed {frame_count} frames")
    print(f"   Average FPS : {avg_fps:.1f}")
    print(f"   Output saved: {out_path}")


def run_on_webcam(model):
    """Run detection on live webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("Running live detection — press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        results = model.predict(
            frame,
            conf    = CONFIDENCE_THRESH,
            iou     = IOU_THRESH,
            imgsz   = IMG_SIZE,
            verbose = False
        )
        fps = 1 / (time.time() - t0)

        frame = draw_detections(frame, results)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Live Surveillance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_on_images(model, folder_path):
    """Run detection on all images in a folder."""
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    images = [f for f in Path(folder_path).iterdir()
              if f.suffix.lower() in extensions]

    if not images:
        print(f"❌ No images found in: {folder_path}")
        return

    print(f"Found {len(images)} images — running detection...")
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        results = model.predict(
            frame,
            conf    = CONFIDENCE_THRESH,
            iou     = IOU_THRESH,
            imgsz   = IMG_SIZE,
            verbose = False
        )

        frame = draw_detections(frame, results)
        out_path = os.path.join(OUTPUT_DIR, img_path.name)
        cv2.imwrite(out_path, frame)
        print(f"  Saved: {out_path}")

    print(f"\n✅ All detections saved → {OUTPUT_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Surveillance Detection")
    parser.add_argument("--source", type=str, required=True,
                        choices=["video", "webcam", "images"],
                        help="Input source type")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to video file or images folder")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESH,
                        help="Confidence threshold")
    args = parser.parse_args()

    CONFIDENCE_THRESH = args.conf
    model = load_model()

    if args.source == "video":
        if not args.input:
            print("❌ Provide --input path/to/video.mp4")
        else:
            run_on_video(model, args.input)

    elif args.source == "webcam":
        run_on_webcam(model)

    elif args.source == "images":
        if not args.input:
            print("❌ Provide --input path/to/images/")
        else:
            run_on_images(model, args.input)