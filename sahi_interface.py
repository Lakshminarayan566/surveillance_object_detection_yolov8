from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os


def run_sahi_inference(image_path, model_path="weights/best.pt"):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.3,
        device="cpu",  # change to "cuda:0" if GPU available
    )

    result = get_sliced_prediction(
        image=image_path,
        detection_model=detection_model,
        slice_height=320,
        slice_width=320,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    os.makedirs("results/sample_predictions", exist_ok=True)
    result.export_visuals(export_dir="results/sample_predictions")

    print("✅ SAHI inference completed. Results saved to results/sample_predictions")


if __name__ == "__main__":
    test_image = "images/demo1.jpg"  # replace with your test image
    run_sahi_inference(test_image)