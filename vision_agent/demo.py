import cv2
import json
import numpy as np
import os
from .detector import VisionAgent

def create_dummy_video(filename="dummy_dashcam.mp4", duration_sec=3, fps=10):
    """Creates a simple dummy video simulating a dashcam feed."""
    print(f"Generating dummy dashcam video: {filename}...")
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for i in range(duration_sec * fps):
        # Create a moving "road" background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw some moving "defects" to trigger YOLO detections
        # We'll just draw some rectangles that move downwards to simulate driving forward
        offset = (i * 15) % height
        cv2.rectangle(img, (200, offset), (250, offset + 50), (200, 200, 200), -1)
        cv2.rectangle(img, (400, (offset + 200) % height), (460, (offset + 250) % height), (150, 150, 150), -1)
        
        cv2.putText(img, f'Frame {i}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(img)
        
    out.release()
    print("Dummy video generation complete.")
    return filename

def main():
    print("Starting Vision Agent Dashcam Demo...")
    
    # Initialize the agent
    agent = VisionAgent(model_path="yolov8n.pt", mock_mode=True)
    
    # Create a dummy video since we likely don't have a real dashcam feed here
    input_video = "dummy_dashcam.mp4"
    if not os.path.exists(input_video):
        create_dummy_video(input_video)
        
    output_json = "structured_events.jsonl"
    output_video = "demo_dashcam_output.mp4"
    
    # Process the video stream
    # Process every 2nd frame to simulate a 5 fps edge processing rate on a 10 fps video
    print("\n--- Simulating Edge Device Processing ---")
    agent.process_video(
        video_path=input_video, 
        output_json_path=output_json, 
        output_video_path=output_video,
        process_every_n_frames=2 
    )
    
    print("\n--- Preview of Edge Events (First 2 Lines) ---")
    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            for i, line in enumerate(f):
                if i < 2:
                    parsed = json.loads(line)
                    print(json.dumps(parsed, indent=2))
                else:
                    break

if __name__ == "__main__":
    main()
