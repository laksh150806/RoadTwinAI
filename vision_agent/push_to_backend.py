"""
Pushes structured_events.jsonl (from detector.VisionAgent.process_video) into
the Decision Engine backend's POST /defects endpoint.

This is the missing wire between Member 1's Vision Agent and Member 3's backend —
until this exists, detections never reach the condition score / forecast / repair
priority pipeline no matter how good the detector gets.

Usage:
    python -m vision_agent.push_to_backend structured_events.jsonl --road-id 1
"""

import argparse
import json
import requests


def push_events(jsonl_path, backend_url, road_id, min_confidence=0.25, source="vision_agent"):
    sent = 0
    skipped = 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            for det in frame["detections"]:
                if det["confidence"] < min_confidence:
                    skipped += 1
                    continue
                payload = {
                    "road_id": road_id,
                    "defect_type": det["defect_type"],
                    "severity": det["severity"],
                    "confidence": det["confidence"],
                    "lat": frame.get("lat"),
                    "lng": frame.get("lng"),
                    "source": source,
                }
                resp = requests.post(f"{backend_url}/defects", json=payload, timeout=5)
                resp.raise_for_status()
                sent += 1
    print(f"Pushed {sent} defects to {backend_url}/defects ({skipped} skipped below confidence threshold)")
    return sent, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push Vision Agent structured events into the RoadTwin backend")
    parser.add_argument("jsonl_path", help="Path to structured_events.jsonl produced by detector.process_video")
    parser.add_argument("--road-id", type=int, required=True, help="Road ID in the backend to attach detections to")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--min-confidence", type=float, default=0.25)
    args = parser.parse_args()
    push_events(args.jsonl_path, args.backend_url, args.road_id, args.min_confidence)
