from pydantic import BaseModel
from typing import List, Optional

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class Detection(BaseModel):
    defect_type: str
    severity: float        # 0.0-1.0, matches the backend's Defect.severity contract
    severity_label: str    # 'low' | 'medium' | 'high', for display only
    bbox: BoundingBox
    confidence: float

class FrameResult(BaseModel):
    frame_id: str
    timestamp: Optional[float] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    detections: List[Detection]
