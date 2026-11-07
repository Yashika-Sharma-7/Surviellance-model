from pathlib import Path

from ultralytics import YOLO

project_root = Path(__file__).resolve().parent

model = YOLO("yolo11s-seg.pt")

results = model.train(
    data=str(project_root / "DATASET.yolov11" / "data.yaml"),
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    project=str(project_root / "runs" / "segment"),
    name="yolo11s-seg-dataset",
    plots=True
)