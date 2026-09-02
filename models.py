from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


class Road(Base):
    __tablename__ = "roads"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)          # e.g. "R-1042"
    start_lat = Column(Float, nullable=True)
    start_lng = Column(Float, nullable=True)
    end_lat = Column(Float, nullable=True)
    end_lng = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    defects = relationship("Defect", back_populates="road")
    condition_scores = relationship("ConditionScore", back_populates="road")
    forecasts = relationship("DeteriorationForecast", back_populates="road")
    priorities = relationship("RepairPriority", back_populates="road")
    sensor_readings = relationship("SensorReading", back_populates="road")


class Defect(Base):
    __tablename__ = "defects"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"))
    defect_type = Column(String, nullable=False)     # pothole, crack, waterlogging, etc.
    severity = Column(Float, nullable=False)          # normalised 0–1
    confidence = Column(Float, nullable=True)         # detection confidence (0–1)
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    detected_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String, nullable=True)            # e.g. "dashcam_upload_23"

    road = relationship("Road", back_populates="defects")


class ConditionScore(Base):
    __tablename__ = "condition_scores"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"))
    score = Column(Float, nullable=False)             # 0–100
    computed_at = Column(DateTime, default=datetime.utcnow)
    breakdown = Column(JSON, nullable=True)            # defect counts/weights used

    road = relationship("Road", back_populates="condition_scores")


class DeteriorationForecast(Base):
    __tablename__ = "deterioration_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"))
    forecast_7d = Column(Float, nullable=True)
    forecast_30d = Column(Float, nullable=True)
    forecast_60d = Column(Float, nullable=True)
    method = Column(String, nullable=True)             # "ml" or "linear_fallback"
    created_at = Column(DateTime, default=datetime.utcnow)

    road = relationship("Road", back_populates="forecasts")


class TrafficRisk(Base):
    __tablename__ = "traffic_risk"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"))
    traffic_volume = Column(Float, nullable=True)          # 0–100 normalised
    pedestrian_density = Column(Float, nullable=True)      # 0–100 normalised
    near_school_hospital = Column(Integer, default=0)      # 0/1 flag
    weather_factor = Column(Float, nullable=True)          # 0–1
    time_of_day_factor = Column(Float, nullable=True)      # 0–1
    avg_speed_kmph = Column(Float, nullable=True)          # km/h (from data contract)
    accident_history_score = Column(Float, nullable=True)  # 0–1 (from data contract)
    risk_score = Column(Float, nullable=True)              # computed combined risk (0–100)
    computed_at = Column(DateTime, default=datetime.utcnow)


class RepairPriority(Base):
    __tablename__ = "repair_priorities"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"))
    priority_rank = Column(Integer, nullable=True)
    priority_score = Column(Float, nullable=True)
    computed_at = Column(DateTime, default=datetime.utcnow)

    road = relationship("Road", back_populates="priorities")


class TrafficSimulation(Base):
    __tablename__ = "traffic_simulations"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"))
    scenario = Column(String, nullable=True)              # "road_closure"
    affected_roads = Column(JSON, nullable=True)           # {"Road B": "+34%", ...}
    recommendation = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class SensorReading(Base):
    """Persists a raw GPS+IMU batch and its fused output."""
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"), nullable=True)
    gps_samples = Column(JSON, nullable=True)              # raw GPS list
    imu_samples = Column(JSON, nullable=True)              # raw IMU list
    fused_points = Column(JSON, nullable=True)             # output of SensorFusionAgent
    defect_event_count = Column(Integer, default=0)        # IMU spikes detected
    best_defect_lat = Column(Float, nullable=True)         # highest-confidence event
    best_defect_lng = Column(Float, nullable=True)
    fusion_confidence = Column(Float, nullable=True)       # 0–1
    imu_threshold_g = Column(Float, nullable=True)
    recorded_at = Column(DateTime, default=datetime.utcnow)

    road = relationship("Road", back_populates="sensor_readings")
