import cv2
import numpy as np
from ultralytics import YOLO
import random
import time
import os
import json
from .models import BoundingBox, Detection, FrameResult

# List of defects we want to track
DEFECT_CLASSES = [
    "pothole",
    "crack",
    "waterlogging",
    "damaged_barrier",
    "faded_marking",
    "broken_streetlight"
]

class VisionAgent:
    def __init__(self, model_path="yolov8n.pt", mock_mode=True):
        """
        Initializes the Vision Agent.
        If mock_mode is True, we use a generic YOLO model (like yolov8n.pt) 
        and map random detections to our target classes for demo purposes.
        """
        self.mock_mode = mock_mode
        print(f"Initializing YOLO model: {model_path} (Mock Mode: {mock_mode})")
        self.model = YOLO(model_path)
        
    def _calculate_severity(self, bbox_area, frame_area):
        """
        A heuristic to determine severity based on the relative size of the defect.
        Returns (numeric_severity 0-1, label) — numeric feeds the backend's
        condition-score formula directly, label is for human-readable display.
        """
        ratio = bbox_area / frame_area if frame_area > 0 else 0
        if ratio < 0.05:
            return round(min(0.4, 0.15 + ratio), 3), "low"
        elif ratio < 0.15:
            return round(min(0.7, 0.4 + ratio), 3), "medium"
        else:
            return round(min(1.0, 0.7 + ratio), 3), "high"

    def process_frame(self, frame, frame_id="frame_0", timestamp=None, lat=None, lng=None):
        """
        Processes a single cv2 image frame and returns structured detections.
        lat/lng is an interim pass-through until Member 2's real GPS+IMU sensor
        fusion agent supplies it — see NOTE in process_video.
        """
        h, w = frame.shape[:2]
        frame_area = h * w

        # Run YOLO detection
        results = self.model(frame, verbose=False)[0]

        detections_list = []

        for box in results.boxes:
            # Extract box data
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            bbox_area = (x_max - x_min) * (y_max - y_min)
            severity, severity_label = self._calculate_severity(bbox_area, frame_area)

            bbox_obj = BoundingBox(
                x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max
            )

            if self.mock_mode:
                # NOTE: no trained road-defect weights exist yet (needs Member 1's
                # own dataset + training run). Until then this randomly labels
                # whatever the generic COCO model detects as one of our target
                # classes, purely so the rest of the pipeline has real traffic
                # to work with. Swap mock_mode=False once real weights land.
                defect_type = random.choice(DEFECT_CLASSES)
            else:
                # In real mode, use the class name from the model
                class_id = int(box.cls[0])
                defect_type = self.model.names[class_id]

            detection_obj = Detection(
                defect_type=defect_type,
                severity=severity,
                severity_label=severity_label,
                bbox=bbox_obj,
                confidence=conf
            )
            detections_list.append(detection_obj)

        if self.mock_mode and not detections_list:
            # The generic COCO-pretrained model finds nothing on footage with no
            # real COCO objects in frame (e.g. the synthetic demo video) — with
            # no trained road-defect weights yet, that would otherwise mean the
            # whole pipeline sees empty frames forever. Synthesize one low-
            # confidence detection so downstream stages (condition scoring,
            # backend, dashboard) have real structured data to run on. Remove
            # this once real weights make actual detection reliable.
            bbox_obj = BoundingBox(
                x_min=w * 0.35, y_min=h * 0.55, x_max=w * 0.55, y_max=h * 0.75
            )
            bbox_area = (bbox_obj.x_max - bbox_obj.x_min) * (bbox_obj.y_max - bbox_obj.y_min)
            severity, severity_label = self._calculate_severity(bbox_area, frame_area)
            detections_list.append(Detection(
                defect_type=random.choice(DEFECT_CLASSES),
                severity=severity,
                severity_label=severity_label,
                bbox=bbox_obj,
                confidence=0.35
            ))

        frame_result = FrameResult(
            frame_id=frame_id,
            timestamp=timestamp or time.time(),
            lat=lat, lng=lng,
            detections=detections_list
        )

        return frame_result

    def process_video(self, video_path, output_json_path, output_video_path=None,
                       process_every_n_frames=5, gps_track=None):
        """
        Processes a dashcam video and outputs a JSON lines file of structured events.
        Simulates an edge device sending structured data instead of raw video.

        gps_track: optional list of (lat, lng) covering the drive start-to-end.
        Until Member 2's real GPS+IMU sensor fusion agent is ready, this linearly
        interpolates position across the video by frame index as a stand-in —
        replace with real fused coordinates per frame when that agent lands.
        """
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        out = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_count = 0
        processed_count = 0

        with open(output_json_path, 'w') as json_file:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % process_every_n_frames == 0:
                    timestamp = frame_count / fps if fps > 0 else time.time()
                    frame_id = f"frame_{frame_count}"

                    lat = lng = None
                    if gps_track and len(gps_track) >= 2:
                        t = min(1.0, frame_count / total_frames)
                        (lat1, lng1), (lat2, lng2) = gps_track[0], gps_track[-1]
                        lat = lat1 + (lat2 - lat1) * t
                        lng = lng1 + (lng2 - lng1) * t

                    # Process the frame
                    result = self.process_frame(frame, frame_id=frame_id, timestamp=timestamp, lat=lat, lng=lng)

                    # Write structured event as JSON line
                    json_file.write(result.model_dump_json() + '\n')
                    processed_count += 1
                    
                    # Optional visualization
                    if out:
                        for det in result.detections:
                            box = det.bbox
                            cv2.rectangle(
                                frame, 
                                (int(box.x_min), int(box.y_min)), 
                                (int(box.x_max), int(box.y_max)), 
                                (0, 255, 0), 2
                            )
                            label = f"{det.defect_type} ({det.severity_label})"
                            cv2.putText(
                                frame, label, 
                                (int(box.x_min), int(box.y_min) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                            )
                
                if out:
                    out.write(frame)
                    
                frame_count += 1
                
        cap.release()
        if out:
            out.release()
            
        print(f"Video processing complete. Processed {processed_count} frames.")
        print(f"Structured events saved to: {output_json_path}")
        if output_video_path:
            print(f"Visualization saved to: {output_video_path}")
