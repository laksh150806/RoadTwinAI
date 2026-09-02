"""
RoadTwin AI — Vision Agent YOLO Fine-Tuning Script
===================================================
Trains a YOLOv8 object detection model on the road damage dataset in data/
(2,009 annotated images of potholes, cracks, and manholes).

Usage
-----
    python ml/train_vision_model.py --epochs 10 --imgsz 640

Saves weights to:
    runs/detect/roadtwin_yolo/weights/best.pt
"""

import os
import sys
import argparse
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_YAML = os.path.join(PROJECT_ROOT, "data", "roadtwin_dataset.yaml")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Road Twin AI dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights")
    args = parser.parse_args()

    logger.info("Initializing YOLOv8 base model: %s", args.model)
    model = YOLO(args.model)

    logger.info("Starting training on dataset config: %s", DATA_YAML)
    results = model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=RUNS_DIR,
        name="roadtwin_yolo",
        exist_ok=True,
        workers=2,
        device="cpu",      # uses CPU / CUDA automatically
        verbose=True,
    )

    best_weights = os.path.join(RUNS_DIR, "roadtwin_yolo", "weights", "best.pt")
    logger.info("✓ Training finished. Best weights saved to: %s", best_weights)
    return best_weights


if __name__ == "__main__":
    main()
