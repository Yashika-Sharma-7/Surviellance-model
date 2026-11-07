from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
SEGMENTATION_CLASSES = (
    "Dead Vegetation",
    "Dry Area",
    "Flower bed",
    "Lawn",
    "Pathway",
    "Vegetation",
    "Water Body",
    "Weeds",
    "litter",
)

COVERAGE_CLASS_IDS = {
    "Lawn Coverage": 3,
    "Tree Canopy Cover (Vegetation)": 5,
    "Bare Patches (Dry Area)": 1,
    "Weed Infestation": 7,
    "Litter Detection": 8,
    "Dead Vegetation": 0,
    "Flower Bed": 2,
    "Water Body": 6,
    "Pathway": 4,
}


def load_segmentation_model():
    """Load the existing park segmentation weights."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Segmentation model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    model_names = model.names
    names = tuple(
        model_names[index]
        for index in range(len(model_names))
    )
    if names != SEGMENTATION_CLASSES:
        raise ValueError(
            "Segmentation model classes do not match the expected dataset "
            f"order: {names}"
        )
    return model


def analyze_image(img, model):
    """Return a Markdown report, annotated image, and numeric coverage metrics."""
    empty_metrics = {name: 0.0 for name in COVERAGE_CLASS_IDS}
    if img is None:
        return "No image uploaded.", None, empty_metrics

    if img.ndim < 2 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("The image must have non-zero width and height.")

    results = model(img)[0]
    annotated_img = results.plot()

    height, width = img.shape[:2]
    total_pixels = height * width
    best_confidence = np.full((height, width), -np.inf, dtype=np.float32)
    pixel_classes = np.full((height, width), -1, dtype=np.int16)

    if results.masks is not None and results.boxes is not None:
        for polygon, class_tensor, confidence_tensor in zip(
            results.masks.xy,
            results.boxes.cls,
            results.boxes.conf,
        ):
            class_id = int(class_tensor.item())
            if class_id not in range(len(SEGMENTATION_CLASSES)):
                raise ValueError(
                    f"Segmentation model returned unexpected class ID {class_id}."
                )

            mask_canvas = np.zeros((height, width), dtype=np.uint8)
            polygon = np.asarray(polygon, dtype=np.int32)
            if polygon.size == 0:
                continue
            cv2.fillPoly(mask_canvas, [polygon], 1)

            confidence = float(confidence_tensor.item())
            replace_pixels = (mask_canvas != 0) & (
                confidence > best_confidence
            )
            best_confidence[replace_pixels] = confidence
            pixel_classes[replace_pixels] = class_id

    class_pixel_counts = np.bincount(
        pixel_classes[pixel_classes >= 0],
        minlength=len(SEGMENTATION_CLASSES),
    )
    coverage_metrics = {
        metric_name: float(
            class_pixel_counts[class_id] / total_pixels * 100
        )
        for metric_name, class_id in COVERAGE_CLASS_IDS.items()
    }

    report_markdown = "### 📊 Analysis Report\n\n"
    report_markdown += "| Target metric | Coverage (%) | Detected (YES/NO) |\n"
    report_markdown += "| :--- | :--- | :--- |\n"

    for metric_name, percentage in coverage_metrics.items():
        detected = "✅ **YES**" if percentage > 0 else "❌ NO"
        report_markdown += (
            f"| {metric_name} | {percentage:.2f}% | {detected} |\n"
        )

    report_markdown += "\n### 🔍 Presence of all classes:\n"
    for class_id, class_name in enumerate(SEGMENTATION_CLASSES):
        present = (
            "✅ YES" if class_pixel_counts[class_id] > 0 else "❌ NO"
        )
        report_markdown += f"- **{class_name}**: {present}\n"

    return report_markdown, annotated_img, coverage_metrics
