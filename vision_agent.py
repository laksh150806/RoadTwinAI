"""
RoadTwin AI — Vision Agent (YOLO + OpenCV)
===========================================
Detects road defects (potholes, cracks, manholes) from dashcam video/image streams.
Converts detections into structured defect events and posts them directly to
the FastAPI backend (/defects).

Architecture
------------
    Dashcam Image/Frame
           │
           ▼
    VisionAgent.detect(image_path_or_array)
           │
           ▼
    YOLO Model Inference  →  [ {defect_type, severity, confidence, bbox} ]
           │
           ▼
    VisionAgent.ingest_to_backend(road_id, detections)  →  POST /defects

Usage
-----
    from vision_agent import VisionAgent

    agent = VisionAgent()
    detections = agent.detect("data/images/20250216_164325.jpg")
    results = agent.ingest_to_backend(road_id=1, detections=detections)
"""

import os
import logging
from typing import List, Dict, Union, Optional
import urllib.request
import json

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Class mapping matching dataset README
CLASS_MAP = {
    0: "pothole",
    1: "crack",
    2: "manhole",
}

# Base API URL
DEFAULT_BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8001")


class VisionAgent:
    """YOLOv8-powered Vision Agent for road defect detection."""

    def __init__(self, model_weights: Optional[str] = None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        trained_path = os.path.join(project_root, "runs", "roadtwin_yolo", "weights", "best.pt")

        if model_weights and os.path.exists(model_weights):
            weights = model_weights
        elif os.path.exists(trained_path):
            weights = trained_path
        else:
            weights = "yolov8n.pt"  # base model fallback

        logger.info("VisionAgent loading weights: %s", weights)
        self.model = YOLO(weights)

    def detect(
        self,
        source: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
    ) -> List[Dict]:
        """
        Run defect detection on an image path or numpy frame.

        Returns
        -------
        List of dicts:
            [
              {
                "defect_type": "pothole" | "crack" | "manhole",
                "severity": float (0–1 area ratio relative to image),
                "confidence": float (0–1),
                "bbox": [x_min, y_min, x_max, y_max]
              }
            ]
        """
        results = self.model.predict(source=source, conf=conf_threshold, verbose=False)
        detections = []

        if not results:
            return detections

        res = results[0]
        orig_shape = res.orig_shape  # (h, w)
        img_area = max(1.0, orig_shape[0] * orig_shape[1])

        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            box_w = max(0.0, xyxy[2] - xyxy[0])
            box_h = max(0.0, xyxy[3] - xyxy[1])
            box_area = box_w * box_h

            # Severity = normalized area ratio (larger defect = higher severity, capped at 1.0)
            severity = min(1.0, (box_area / img_area) * 15.0)  # scaled to 0-1 range

            defect_name = CLASS_MAP.get(cls_id, "crack" if cls_id == 1 else "pothole")

            detections.append({
                "defect_type": defect_name,
                "severity": round(severity, 3),
                "confidence": round(conf, 3),
                "bbox": [round(c, 1) for c in xyxy],
            })

        return detections

    def ingest_to_backend(
        self,
        road_id: int,
        detections: List[Dict],
        backend_url: str = DEFAULT_BACKEND_URL,
    ) -> List[Dict]:
        """
        Post detected defects to the FastAPI backend /defects endpoint.
        """
        ingested = []
        for det in detections:
            payload = {
                "road_id": road_id,
                "defect_type": det["defect_type"],
                "severity": det["severity"],
                "confidence": det["confidence"],
                "source": "vision_agent_yolo",
            }
            try:
                url = f"{backend_url.rstrip('/')}/defects"
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as res:
                    ingested.append(json.loads(res.read().decode("utf-8")))
            except Exception as exc:
                logger.error("Failed to ingest defect %s to backend: %s", det["defect_type"], exc)

        return ingested
