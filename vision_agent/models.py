from pydantic import BaseModel
from typing import List, Optional

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class Detection(BaseModel):
    defect_type: str
    severity: str # e.g., 'low', 'medium', 'high'
    bbox: BoundingBox
    confidence: float

class FrameResult(BaseModel):
    frame_id: str
    timestamp: Optional[float] = None
    detections: List[Detection]
