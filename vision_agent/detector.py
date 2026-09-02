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
        """
        ratio = bbox_area / frame_area if frame_area > 0 else 0
        if ratio < 0.05:
            return "low"
        elif ratio < 0.15:
            return "medium"
        else:
            return "high"

    def process_frame(self, frame, frame_id="frame_0", timestamp=None):
        """
        Processes a single cv2 image frame and returns structured detections.
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
            severity = self._calculate_severity(bbox_area, frame_area)
            
            bbox_obj = BoundingBox(
                x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max
            )
            
            if self.mock_mode:
                # In mock mode, we randomly assign one of our target defect classes
                defect_type = random.choice(DEFECT_CLASSES)
            else:
                # In real mode, use the class name from the model
                class_id = int(box.cls[0])
                defect_type = self.model.names[class_id]
                
            detection_obj = Detection(
                defect_type=defect_type,
                severity=severity,
                bbox=bbox_obj,
                confidence=conf
            )
            detections_list.append(detection_obj)
            
        frame_result = FrameResult(
            frame_id=frame_id,
            timestamp=timestamp or time.time(),
            detections=detections_list
        )
        
        return frame_result
        
    def process_video(self, video_path, output_json_path, output_video_path=None, process_every_n_frames=5):
        """
        Processes a dashcam video and outputs a JSON lines file of structured events.
        Simulates an edge device sending structured data instead of raw video.
        """
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
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
                    
                    # Process the frame
                    result = self.process_frame(frame, frame_id=frame_id, timestamp=timestamp)
                    
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
                            label = f"{det.defect_type} ({det.severity})"
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
