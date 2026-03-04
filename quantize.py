"""
quantize.py
-----------
Quantizes the trained YOLOv8 model for edge deployment (Jetson Nano).
Exports to:
  - ONNX  (cross-platform, fast CPU inference)
  - TensorRT INT8 (Jetson Nano GPU — uncomment when running on device)

Usage:
  python quantize.py

Run on laptop/PC: exports ONNX model
Run on Jetson Nano: uncomment TensorRT section for INT8 quantization
"""

import os
import time
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "surveillance_yolov8/defence_v1/weights/best.pt"
OUTPUT_DIR  = "model_exports"
IMG_SIZE    = 640

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_model_size_mb(path):
    """Returns file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)


def export_onnx(model):
    """Export to ONNX format — works on laptop and Jetson Nano CPU."""
    print("\n── Exporting to ONNX ────────────────────────────────────────")
    onnx_path = model.export(
        format   = "onnx",
        imgsz    = IMG_SIZE,
        dynamic  = False,
        simplify = True,
        opset    = 12,
    )
    size = get_model_size_mb(onnx_path)
    print(f"✅ ONNX model saved: {onnx_path}")
    print(f"   Model size: {size:.1f} MB")
    return onnx_path


def export_torchscript(model):
    """Export to TorchScript — good for CPU inference benchmarking."""
    print("\n── Exporting to TorchScript ─────────────────────────────────")
    ts_path = model.export(
        format = "torchscript",
        imgsz  = IMG_SIZE,
    )
    size = get_model_size_mb(ts_path)
    print(f"✅ TorchScript model saved: {ts_path}")
    print(f"   Model size: {size:.1f} MB")
    return ts_path


def benchmark_onnx(onnx_path, num_runs=50):
    """Benchmark ONNX inference speed."""
    print("\n── Benchmarking ONNX Inference ──────────────────────────────")
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(onnx_path,
                                    providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        # Warmup
        dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        for _ in range(5):
            sess.run(None, {input_name: dummy})

        # Benchmark
        times = []
        for _ in range(num_runs):
            t0 = time.time()
            sess.run(None, {input_name: dummy})
            times.append(time.time() - t0)

        avg_ms  = (sum(times) / len(times)) * 1000
        avg_fps = 1000 / avg_ms
        print(f"   Avg inference time : {avg_ms:.1f} ms")
        print(f"   Avg FPS            : {avg_fps:.1f}")
        print(f"   Model size         : {get_model_size_mb(onnx_path):.1f} MB")

    except ImportError:
        print("   onnxruntime not installed. Run: pip install onnxruntime")


# ── TensorRT INT8 — Uncomment when running on Jetson Nano ─────────────────────
# def export_tensorrt(model):
#     """
#     Export to TensorRT INT8 — ONLY run this on Jetson Nano.
#     Requires: TensorRT installed on Jetson Nano (comes pre-installed with JetPack)
#     """
#     print("\n── Exporting to TensorRT INT8 ───────────────────────────────")
#     trt_path = model.export(
#         format  = "engine",
#         imgsz   = IMG_SIZE,
#         int8    = True,       # INT8 quantization
#         device  = 0,          # GPU
#         batch   = 1,
#         simplify= True,
#     )
#     size = get_model_size_mb(trt_path)
#     print(f"✅ TensorRT engine saved: {trt_path}")
#     print(f"   Model size: {size:.1f} MB  (target: ~6.2 MB)")
#     return trt_path


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 1. Export ONNX (works on laptop)
    onnx_path = export_onnx(model)

    # 2. Export TorchScript
    export_torchscript(model)

    # 3. Benchmark ONNX
    benchmark_onnx(onnx_path)

    # 4. On Jetson Nano — uncomment TensorRT export above

    print("\n📋 Summary:")
    print(f"   ONNX model  → ready for CPU / Jetson Nano inference")
    print(f"   TorchScript → ready for PyTorch CPU inference")
    print(f"   TensorRT    → uncomment export_tensorrt() on Jetson Nano for INT8")