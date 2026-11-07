
from ultralytics import YOLO
from pathlib import Path


# --------------------------------------------------
# MODEL INITIALIZATION
# --------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "models" / "Count.pt"


def load_count_model():
    """Load the trained object counting model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Counting model not found: {MODEL_PATH}")

    return YOLO(str(MODEL_PATH))


# --------------------------------------------------
# OBJECT COUNTING
# --------------------------------------------------

def analyze_count(img, model, confidence=0.25):

    if img is None:
        return "No image uploaded.", None, {}

    # Run object detection
    results = model.predict(
        source=img,
        conf=confidence,
        verbose=False
    )[0]

    # Generate annotated image
    annotated_img = results.plot()

    # Get model class names
    class_names = results.names

    # Initialize counts for every trained class
    object_counts = {
        name: 0 for name in class_names.values()
    }

    # Count detected objects
    if results.boxes is not None and len(results.boxes) > 0:

        for class_tensor in results.boxes.cls:

            class_id = int(class_tensor.item())
            class_name = class_names[class_id]

            object_counts[class_name] += 1

    # --------------------------------------------------
    # REPORT GENERATION
    # --------------------------------------------------

    report_markdown = "### Object Detection & Counting Report\n\n"

    report_markdown += (
        "| Object Category | Count | Detected |\n"
        "| :--- | :---: | :---: |\n"
    )

    # Standardized display names
    display_names = {
        "Bench": "Benches / Chairs",
        "Dustbin": "Dustbins",
        "Tree": "Trees",
        "Light_pole": "Light Poles"
    }

    for class_name, count in object_counts.items():

        display_name = display_names.get(
            class_name, class_name
        )

        detected = "YES" if count > 0 else "NO"

        report_markdown += (
            f"| {display_name} | {count} | {detected} |\n"
        )

    total_objects = sum(object_counts.values())

    report_markdown += (
        f"\n**Total Objects Detected: {total_objects}**\n"
    )

    return report_markdown, annotated_img, object_counts
