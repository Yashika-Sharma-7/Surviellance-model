import os
import shutil
from pathlib import Path

import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise EnvironmentError(
        "Set the ROBOFLOW_API_KEY environment variable before running TRAIN_2.py."
    )

project_root = Path(__file__).resolve().parent
rf = Roboflow(api_key=api_key)

# Retrieve your default workspace
ws = rf.workspace()

print("Workspace name:", ws.name)
print("Workspace ID:", ws.url)

print("\nAccessible projects:")
ws.list_projects()

# Project 1: Obstacle_detection
project1 = ws.project("obstacle_detection-8cn5s-7hvqd")
version1 = project1.version(1)
dataset1 = version1.download("yolov11")

# Project 2: OOD
project2 = ws.project("ood-pbnro-jdfq3-zcp5j")
version2 = project2.version(1)
dataset2 = version2.download("yolov11")

print("Both datasets downloaded successfully!")
print("Dataset 1:", dataset1.location)
print("Dataset 2:", dataset2.location)

sources = {
    "obstacle": Path(dataset1.location),
    "ood": Path(dataset2.location)
}

output = project_root / "combined_park_dataset"

splits = ["train", "valid", "test"]

class_names = ["Bench", "Dustbin", "Light_pole", "Tree"]

# Create destination folders
for split in splits:
    (output / split / "images").mkdir(parents=True, exist_ok=True)
    (output / split / "labels").mkdir(parents=True, exist_ok=True)

# Copy images and corresponding labels
for source_name, source_path in sources.items():

    for split in splits:

        image_dir = source_path / split / "images"
        label_dir = source_path / split / "labels"

        for image_path in image_dir.iterdir():

            if not image_path.is_file():
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                print("Missing label:", image_path.name)
                continue

            # Prefix filenames to avoid collisions
            new_name = f"{source_name}_{image_path.name}"
            new_label = f"{source_name}_{image_path.stem}.txt"

            shutil.copy2(
                image_path,
                output / split / "images" / new_name
            )

            shutil.copy2(
                label_path,
                output / split / "labels" / new_label
            )

# Create YOLO dataset configuration
data_config = {
    "path": str(output),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 4,
    "names": class_names
}

with open(output / "data.yaml", "w") as f:
    yaml.safe_dump(data_config, f, sort_keys=False)

print("\nMERGE COMPLETED!")

# Verify merged dataset
for split in splits:

    images = list((output / split / "images").iterdir())
    labels = list((output / split / "labels").glob("*.txt"))

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Labels:", len(labels))

print("\nYAML saved at:", output / "data.yaml")

device = 0 if torch.cuda.is_available() else "cpu"
if device == 0:
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU unavailable; training on CPU.")

# Load pretrained YOLO11 Small model
model = YOLO("yolo11s.pt")

# Train
results = model.train(
    data=str(output / "data.yaml"),
    epochs=80,
    imgsz=640,
    batch=-1,           # Automatically selects a suitable batch size
    patience=15,        # Early stopping
    device=device,
    workers=2,
    pretrained=True,
    project=str(project_root / "runs" / "detect"),
    name="park_inventory_yolo11s",
    save=True,
    plots=True,
    close_mosaic=10
)