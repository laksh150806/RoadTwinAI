"""
RoadTwin AI — Decision Engine (FastAPI backend)
===============================================
5-layer architecture wired together:
    Data Collection  →  AI Agents  →  Data Fusion  →  Digital Twin  →  Decision Engine

Endpoints
---------
  Roads
    POST /roads                           create a road segment
    GET  /roads                           list all roads
    POST /seed                            seed baseline roads and defects for demo
    DELETE /roads/{road_id}/reset         reset a road back to baseline state

  Defect Ingestion (Vision Agent output)
    POST /defects                         ingest a detected defect
    GET  /roads/{road_id}/defects         list defects for a road

  Sensor Fusion Agent  ← NEW
    POST /sensor-fusion                   fuse a GPS+IMU batch, optionally ingest
                                          the best defect location directly

  Condition Agent
    GET  /roads/{road_id}/condition-score compute/return 0–100 road condition score

  Deterioration Predictor  ← UPGRADED (ML)
    GET  /roads/{road_id}/forecast        predict +7/+30/+60 day scores via RF model

  Traffic Risk Agent
    POST /roads/{road_id}/traffic-risk    set contextual risk factors

  Decision Engine
    GET  /repair-priorities               rank all roads by priority score
    POST /simulate                        what-if traffic simulation

  Full Analysis Pipeline  ← NEW
    GET  /roads/{road_id}/full-analysis   chains condition → forecast → priority in one call

  Summary (for dashboard)
    GET  /roads/{road_id}/summary         full snapshot for a road
    GET  /ml/status                       shows whether ML model is loaded
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

from database import engine, get_db, Base
from models import (
    Road, Defect, ConditionScore, DeteriorationForecast,
    TrafficRisk, RepairPriority, TrafficSimulation, SensorReading,
)
from sensor_fusion import SensorFusionAgent
from ml.deterioration_model import DeteriorationPredictor

# ---------- App bootstrap ----------

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="RoadTwin AI — Decision Engine",
    description="Living digital twin for road infrastructure. Observe → Predict → Prioritize → Repair → Repeat.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons — loaded once at startup
_fusion_agent = SensorFusionAgent()
_predictor = DeteriorationPredictor()


# ===========================================================================
# ROADS
# ===========================================================================

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "RoadTwin AI backend running",
        "ml_model_loaded": _predictor.is_ml_ready(),
        "version": "2.0.0",
        "dashboard_url": "/dashboard",
    }


@app.get("/dashboard", tags=["Dashboard UI"])
@app.get("/app", tags=["Dashboard UI"])
def get_dashboard():
    import os
    html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    return FileResponse(html_path)


@app.post("/roads", tags=["Roads"])
def create_road(
    name: str,
    start_lat: float = None, start_lng: float = None,
    end_lat: float = None,   end_lng: float = None,
    db: Session = Depends(get_db),
):
    road = Road(name=name, start_lat=start_lat, start_lng=start_lng,
                end_lat=end_lat, end_lng=end_lng)
    db.add(road)
    db.commit()
    db.refresh(road)
    return road


@app.get("/roads", tags=["Roads"])
def list_roads(db: Session = Depends(get_db)):
    return db.query(Road).all()


# ---------- SEED / RESET (frontend bootstrap for the demo) ----------

SEED_ROADS = [
    {"name": "R-1042", "start_lat": 12.8245, "start_lng": 80.0430, "end_lat": 12.8215, "end_lng": 80.0460},
    {"name": "R-1015", "start_lat": 12.8280, "start_lng": 80.0400, "end_lat": 12.8265, "end_lng": 80.0430},
    {"name": "R-1033", "start_lat": 12.8210, "start_lng": 80.0470, "end_lat": 12.8190, "end_lng": 80.0500},
    {"name": "R-1050", "start_lat": 12.8195, "start_lng": 80.0390, "end_lat": 12.8175, "end_lng": 80.0420},
]


SEED_BASELINE_DEFECTS = {
    "R-1042": [("crack", 0.6), ("faded_marking", 1.0)],
}


@app.post("/seed", tags=["Demo Setup"])
def seed(db: Session = Depends(get_db)):
    created = []
    for r in SEED_ROADS:
        existing = db.query(Road).filter(Road.name == r["name"]).first()
        if existing:
            continue
        road = Road(**r)
        db.add(road)
        db.commit()
        db.refresh(road)
        for defect_type, severity in SEED_BASELINE_DEFECTS.get(r["name"], []):
            db.add(Defect(road_id=road.id, defect_type=defect_type, severity=severity, source="baseline"))
        db.commit()
        compute_condition_score(road.id, db)
        created.append(road.name)
    return {"created": created, "roads": db.query(Road).all()}


@app.delete("/roads/{road_id}/reset", tags=["Demo Setup"])
def reset_road(road_id: int, db: Session = Depends(get_db)):
    road = db.query(Road).filter(Road.id == road_id).first()
    if not road:
        raise HTTPException(status_code=404, detail="Road not found")
    db.query(Defect).filter(Defect.road_id == road_id).delete()
    db.query(ConditionScore).filter(ConditionScore.road_id == road_id).delete()
    db.query(DeteriorationForecast).filter(DeteriorationForecast.road_id == road_id).delete()
    db.query(TrafficRisk).filter(TrafficRisk.road_id == road_id).delete()
    db.query(RepairPriority).filter(RepairPriority.road_id == road_id).delete()
    db.commit()
    compute_condition_score(road_id, db)  # back to baseline 100
    return {"road_id": road_id, "status": "reset"}


# ===========================================================================
# DEFECT INGESTION  (Vision Agent → Decision Engine)
# ===========================================================================

class DefectIn(BaseModel):
    road_id: int
    defect_type: str        # pothole | crack | waterlogging | damaged_barrier | faded_marking | broken_streetlight
    severity: float         # normalised 0–1  (Vision Agent must normalise before posting)
    confidence: Optional[float] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    source: Optional[str] = None


@app.post("/defects", tags=["Defects"])
def add_defect(defect: DefectIn, db: Session = Depends(get_db)):
    road = db.query(Road).filter(Road.id == defect.road_id).first()
    if not road:
        raise HTTPException(status_code=404, detail="Road not found")

    # Normalise severity to 0–1 if Vision Agent sent 0–100
    severity = defect.severity if defect.severity <= 1.0 else defect.severity / 100.0
    record = Defect(
        road_id=defect.road_id,
        defect_type=defect.defect_type,
        severity=round(severity, 4),
        confidence=defect.confidence,
        lat=defect.lat,
        lng=defect.lng,
        source=defect.source,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    # Auto-recompute condition score on every new defect
    compute_condition_score(defect.road_id, db)
    db.refresh(record)
    return record


@app.get("/roads/{road_id}/defects", tags=["Defects"])
def get_defects(road_id: int, db: Session = Depends(get_db)):
    return db.query(Defect).filter(Defect.road_id == road_id).all()


# ===========================================================================
# SENSOR FUSION AGENT  (GPS + IMU → fused defect location)
# ===========================================================================

class IMUSample(BaseModel):
    ax: float           # m/s² lateral
    ay: float           # m/s² longitudinal
    az: float = 9.81    # m/s² vertical (gravity ~9.81 on flat road)
    timestamp_ms: int


class GPSSample(BaseModel):
    lat: float
    lng: float
    timestamp_ms: int


class SensorFusionIn(BaseModel):
    road_id: Optional[int] = None           # if provided, result is persisted
    gps_samples: List[GPSSample]
    imu_samples: List[IMUSample]
    threshold_g: float = 0.4                # g-force magnitude to flag defect event
    auto_ingest_defect: bool = False        # if True, post best defect event as a Defect record


@app.post("/sensor-fusion", tags=["Sensor Fusion"])
def fuse_sensors(payload: SensorFusionIn, db: Session = Depends(get_db)):
    """
    Fuse GPS + IMU samples via Kalman Filter.
    Returns fused trajectory points and flags IMU spikes as defect events.
    If road_id is provided, persists the reading to SensorReading table.
    If auto_ingest_defect=True, the best defect event is also saved as a Defect.
    """
    gps_dicts = [s.model_dump() for s in payload.gps_samples]
    imu_dicts = [s.model_dump() for s in payload.imu_samples]

    fused = _fusion_agent.fuse_batch(gps_dicts, imu_dicts, threshold_g=payload.threshold_g)
    best = _fusion_agent.best_defect_location(fused)
    defect_events = [p for p in fused if p["is_defect_event"]]

    # Persist if road_id given
    db_record_id = None
    if payload.road_id is not None:
        road = db.query(Road).filter(Road.id == payload.road_id).first()
        if not road:
            raise HTTPException(status_code=404, detail="Road not found")

        reading = SensorReading(
            road_id=payload.road_id,
            gps_samples=gps_dicts,
            imu_samples=imu_dicts,
            fused_points=fused,
            defect_event_count=len(defect_events),
            best_defect_lat=best["fused_lat"] if best else None,
            best_defect_lng=best["fused_lng"] if best else None,
            fusion_confidence=best["confidence"] if best else None,
            imu_threshold_g=payload.threshold_g,
        )
        db.add(reading)
        db.commit()
        db.refresh(reading)
        db_record_id = reading.id

        # Optionally auto-ingest the defect event
        if payload.auto_ingest_defect and best:
            defect = Defect(
                road_id=payload.road_id,
                defect_type="pothole",    # IMU spikes most commonly = potholes; Vision Agent refines
                severity=min(1.0, best["imu_magnitude_g"] / (payload.threshold_g * 3)),
                confidence=best["confidence"],
                lat=best["fused_lat"],
                lng=best["fused_lng"],
                source="sensor_fusion",
            )
            db.add(defect)
            db.commit()
            compute_condition_score(payload.road_id, db)

    return {
        "fused_points": fused,
        "defect_events": defect_events,
        "best_defect_location": best,
        "defect_event_count": len(defect_events),
        "sensor_reading_id": db_record_id,
    }


# ===========================================================================
# CONDITION AGENT  (rule-weighted 0–100 score)
# ===========================================================================

# Defect type base weights (impact per unit severity)
DEFECT_WEIGHTS = {
    "pothole":           22.0,   # highest structural risk
    "crack":             10.0,
    "waterlogging":      16.0,   # accelerates all other damage
    "damaged_barrier":   12.0,
    "faded_marking":      4.0,
    "broken_streetlight": 6.0,
}
DEFAULT_WEIGHT = 5.0


def compute_condition_score(road_id: int, db: Session):
    """
    Condition Agent: convert raw defects into a 0–100 road condition score.
    Score = 100 - Σ(weight_type × severity_normalised) clamped to [0, 100].
    severity is stored normalised 0–1; each defect contributes weight×severity points.
    """
    defects = db.query(Defect).filter(Defect.road_id == road_id).all()
    score = 100.0
    breakdown: dict = {}

    for d in defects:
        weight = DEFECT_WEIGHTS.get(d.defect_type, DEFAULT_WEIGHT)
        # severity already normalised 0–1 in DB
        severity_norm = d.severity if d.severity <= 1.0 else d.severity / 100.0
        penalty = weight * severity_norm
        score -= penalty
        breakdown[d.defect_type] = breakdown.get(d.defect_type, 0) + 1

    score = max(0.0, min(100.0, round(score, 1)))

    record = ConditionScore(road_id=road_id, score=score, breakdown=breakdown)
    db.add(record)
    db.commit()
    return score, breakdown


@app.get("/roads/{road_id}/condition-score", tags=["Condition Agent"])
def get_condition_score(road_id: int, db: Session = Depends(get_db)):
    road = db.query(Road).filter(Road.id == road_id).first()
    if not road:
        raise HTTPException(status_code=404, detail="Road not found")
    score, breakdown = compute_condition_score(road_id, db)
    return {"road_id": road_id, "score": score, "breakdown": breakdown}


# ===========================================================================
# DETERIORATION PREDICTOR  (ML Random Forest — +7/+30/+60 days)
# ===========================================================================

def _build_predictor_input(road_id: int, current_score: float, db: Session) -> dict:
    """Collect defect counts and latest traffic risk to build predictor features."""
    defects = db.query(Defect).filter(Defect.road_id == road_id).all()
    counts: dict = {}
    for d in defects:
        counts[d.defect_type] = counts.get(d.defect_type, 0) + 1

    latest_risk = (
        db.query(TrafficRisk)
        .filter(TrafficRisk.road_id == road_id)
        .order_by(TrafficRisk.computed_at.desc())
        .first()
    )

    return {
        "current_score":          current_score,
        "pothole_count":          counts.get("pothole", 0),
        "crack_count":            counts.get("crack", 0),
        "waterlogging_present":   int("waterlogging" in counts),
        "damaged_barrier_count":  counts.get("damaged_barrier", 0),
        "faded_marking_count":    counts.get("faded_marking", 0),
        "broken_streetlight_count": counts.get("broken_streetlight", 0),
        "traffic_volume":         latest_risk.traffic_volume if latest_risk else 50.0,
        "near_school_hospital":   latest_risk.near_school_hospital if latest_risk else 0,
        "weather_factor":         latest_risk.weather_factor if latest_risk else 0.0,
        "avg_speed_kmph":         latest_risk.avg_speed_kmph if latest_risk else 40.0,
        "accident_history_score": latest_risk.accident_history_score if latest_risk else 0.0,
    }


@app.get("/roads/{road_id}/forecast", tags=["Deterioration Predictor"])
def get_forecast(road_id: int, db: Session = Depends(get_db)):
    """
    Return +7/+30/+60 day deterioration forecast using the Random Forest model.
    Falls back to linear decay if the model hasn't been trained yet.
    Run `python ml/train.py` to train the ML model.
    """
    latest = (
        db.query(ConditionScore)
        .filter(ConditionScore.road_id == road_id)
        .order_by(ConditionScore.computed_at.desc())
        .first()
    )
    if not latest:
        raise HTTPException(status_code=404, detail="No condition score yet — POST defects first.")

    features = _build_predictor_input(road_id, latest.score, db)
    forecast = _predictor.predict(**features)

    record = DeteriorationForecast(
        road_id=road_id,
        forecast_7d=forecast["forecast_7d"],
        forecast_30d=forecast["forecast_30d"],
        forecast_60d=forecast["forecast_60d"],
        method=forecast["method"],
    )
    db.add(record)
    db.commit()

    return {
        "road_id":       road_id,
        "current_score": latest.score,
        "forecast_7d":   forecast["forecast_7d"],
        "forecast_30d":  forecast["forecast_30d"],
        "forecast_60d":  forecast["forecast_60d"],
        "method":        forecast["method"],
    }


# ===========================================================================
# TRAFFIC RISK AGENT  (aligned to data-contract.json)
# ===========================================================================

class TrafficRiskIn(BaseModel):
    road_id: Optional[int] = None
    traffic_volume: Optional[float] = 50.0          # 0–100 normalised
    pedestrian_density: Optional[float] = 0.0       # 0–100 normalised
    near_school_hospital: Optional[int] = 0         # 0/1
    weather_factor: Optional[float] = 0.0           # 0–1
    time_of_day_factor: Optional[float] = 0.0       # 0–1 (0=night, 1=peak hour)
    avg_speed_kmph: Optional[float] = 40.0          # from data contract
    accident_history_score: Optional[float] = 0.0   # 0–1 from data contract


@app.post("/roads/{road_id}/traffic-risk", tags=["Traffic Risk"])
def set_traffic_risk(road_id: int, risk: TrafficRiskIn, db: Session = Depends(get_db)):
    """
    Traffic Risk Agent: compute a composite risk score (0–100).
    Risk = Damage × Exposure × Context (Risk = f(traffic, pedestrians,
    school/hospital proximity, weather, time-of-day, accident history).
    """
    road = db.query(Road).filter(Road.id == road_id).first()
    if not road:
        raise HTTPException(status_code=404, detail="Road not found")

    # Weighted risk formula — mirrors data-contract.json traffic_risk_context
    risk_score = (
        risk.traffic_volume          * 0.30
        + risk.pedestrian_density    * 0.20
        + risk.near_school_hospital  * 20.0   # binary flag → large fixed contribution
        + risk.weather_factor        * 10.0
        + risk.time_of_day_factor    * 8.0
        + risk.accident_history_score * 12.0
        + (risk.avg_speed_kmph / 80) * 10.0  # normalised; 80 km/h = max expected
    )
    risk_score = round(min(100.0, risk_score), 2)

    record = TrafficRisk(
        road_id=road_id,
        traffic_volume=risk.traffic_volume,
        pedestrian_density=risk.pedestrian_density,
        near_school_hospital=risk.near_school_hospital,
        weather_factor=risk.weather_factor,
        time_of_day_factor=risk.time_of_day_factor,
        avg_speed_kmph=risk.avg_speed_kmph,
        accident_history_score=risk.accident_history_score,
        risk_score=risk_score,
    )
    db.add(record)
    db.commit()

    return {"road_id": road_id, "risk_score": risk_score}


# ===========================================================================
# REPAIR PRIORITY RANKING  (Decision Engine core)
# ===========================================================================

@app.get("/repair-priorities", tags=["Decision Engine"])
def get_repair_priorities(db: Session = Depends(get_db)):
    """
    Rank all roads by priority score.
    priority_score = (100 - condition_score) × (1 + risk_score / 100)
    Higher score = more urgent repair needed.
    """
    roads = db.query(Road).all()
    results = []

    for road in roads:
        latest_score = (
            db.query(ConditionScore)
            .filter(ConditionScore.road_id == road.id)
            .order_by(ConditionScore.computed_at.desc())
            .first()
        )
        latest_risk = (
            db.query(TrafficRisk)
            .filter(TrafficRisk.road_id == road.id)
            .order_by(TrafficRisk.computed_at.desc())
            .first()
        )

        if not latest_score:
            continue

        condition_score = latest_score.score
        risk_score = latest_risk.risk_score if latest_risk else 1.0

        # Composite priority: bad condition + high risk = urgent
        priority_score = round((100 - condition_score) * (1 + risk_score / 100), 2)

        results.append({
            "road_id":         road.id,
            "road_name":       road.name,
            "condition_score": condition_score,
            "risk_score":      risk_score,
            "priority_score":  priority_score,
        })

    results.sort(key=lambda x: x["priority_score"], reverse=True)
    for i, r in enumerate(results):
        r["priority_rank"] = i + 1
        record = RepairPriority(
            road_id=r["road_id"],
            priority_rank=r["priority_rank"],
            priority_score=r["priority_score"],
        )
        db.add(record)
    db.commit()

    return results


# ===========================================================================
# WHAT-IF TRAFFIC SIMULATION
# ===========================================================================

class SimulationIn(BaseModel):
    road_id: int
    scenario: str = "road_closure"


@app.post("/simulate", tags=["Decision Engine"])
def simulate_closure(sim: SimulationIn, db: Session = Depends(get_db)):
    road = db.query(Road).filter(Road.id == sim.road_id).first()
    if not road:
        raise HTTPException(status_code=404, detail="Road not found")

    # Realistic rule-based diversion impact (placeholder for routing model)
    # A production version would call OSRM / Valhalla here
    affected_roads = {
        "Road B (R-1033)": "+34% congestion",
        "Road C (R-1050)": "+21% congestion",
        "Road D (R-1015)": "+8% congestion",
    }
    recommendation = "Repair window 23:00–05:00 · divert via Route C (R-1050)"

    record = TrafficSimulation(
        road_id=sim.road_id,
        scenario=sim.scenario,
        affected_roads=affected_roads,
        recommendation=recommendation,
    )
    db.add(record)
    db.commit()

    return {
        "road_id":       sim.road_id,
        "scenario":      sim.scenario,
        "affected_roads": affected_roads,
        "recommendation": recommendation,
    }


# ===========================================================================
# FULL ANALYSIS PIPELINE  ← NEW
# ===========================================================================

@app.get("/roads/{road_id}/full-analysis", tags=["Full Pipeline"])
def full_analysis(road_id: int, db: Session = Depends(get_db)):
    """
    Convenience endpoint that chains the entire intelligence pipeline:
        Condition Agent → Deterioration Predictor → Priority Ranking

    Returns a single response matching the decision_engine_output shape
    from data-contract.json.
    """
    road = db.query(Road).filter(Road.id == road_id).first()
    if not road:
        raise HTTPException(status_code=404, detail="Road not found")

    # 1. Condition score
    score, breakdown = compute_condition_score(road_id, db)

    # 2. Deterioration forecast
    features = _build_predictor_input(road_id, score, db)
    forecast = _predictor.predict(**features)

    # Persist forecast
    db.add(DeteriorationForecast(
        road_id=road_id,
        forecast_7d=forecast["forecast_7d"],
        forecast_30d=forecast["forecast_30d"],
        forecast_60d=forecast["forecast_60d"],
        method=forecast["method"],
    ))

    # 3. Repair priority across all roads
    all_roads = db.query(Road).all()
    priority_list = []
    for r in all_roads:
        ls = (db.query(ConditionScore).filter(ConditionScore.road_id == r.id)
              .order_by(ConditionScore.computed_at.desc()).first())
        lr = (db.query(TrafficRisk).filter(TrafficRisk.road_id == r.id)
              .order_by(TrafficRisk.computed_at.desc()).first())
        if ls:
            ps = round((100 - ls.score) * (1 + (lr.risk_score if lr else 1) / 100), 2)
            priority_list.append((r.id, ps))
    priority_list.sort(key=lambda x: x[1], reverse=True)
    rank_map = {rid: i + 1 for i, (rid, _) in enumerate(priority_list)}
    priority_rank = rank_map.get(road_id, None)

    # 4. Latest risk for response
    latest_risk = (db.query(TrafficRisk).filter(TrafficRisk.road_id == road_id)
                   .order_by(TrafficRisk.computed_at.desc()).first())

    db.commit()

    # Determine risk level label
    risk_val = latest_risk.risk_score if latest_risk else 0
    risk_level = "low" if risk_val < 35 else ("medium" if risk_val < 65 else "high")

    return {
        "segment_id":    road.name,
        "road_id":       road_id,
        "condition_score":   score,
        "breakdown":         breakdown,
        "deterioration_forecast": {
            "d7":    forecast["forecast_7d"],
            "d30":   forecast["forecast_30d"],
            "d60":   forecast["forecast_60d"],
            "method": forecast["method"],
        },
        "risk_score":    risk_val,
        "risk_level":    risk_level,
        "priority_rank": priority_rank,
        "recommended_action":  "repair" if score < 50 else ("monitor" if score < 75 else "ok"),
        "recommended_window":  {"start": "23:00", "end": "05:00"},
        "ml_model_active": _predictor.is_ml_ready(),
    }


# ===========================================================================
# SUMMARY  (dashboard / demo moment)
# ===========================================================================

@app.get("/roads/{road_id}/summary", tags=["Dashboard"])
def get_summary(road_id: int, db: Session = Depends(get_db)):
    road = db.query(Road).filter(Road.id == road_id).first()
    if not road:
        raise HTTPException(status_code=404, detail="Road not found")

    latest_score = (db.query(ConditionScore).filter(ConditionScore.road_id == road_id)
                    .order_by(ConditionScore.computed_at.desc()).first())
    latest_forecast = (db.query(DeteriorationForecast).filter(DeteriorationForecast.road_id == road_id)
                       .order_by(DeteriorationForecast.created_at.desc()).first())
    latest_priority = (db.query(RepairPriority).filter(RepairPriority.road_id == road_id)
                       .order_by(RepairPriority.computed_at.desc()).first())

    return {
        "road":            {"id": road.id, "name": road.name},
        "condition_score": latest_score.score if latest_score else None,
        "forecast": {
            "7d":    latest_forecast.forecast_7d,
            "30d":   latest_forecast.forecast_30d,
            "60d":   latest_forecast.forecast_60d,
            "method": latest_forecast.method,
        } if latest_forecast else None,
        "priority_rank": latest_priority.priority_rank if latest_priority else None,
    }


# ===========================================================================
# ML STATUS
# ===========================================================================

@app.get("/ml/status", tags=["ML"])
def ml_status():
    """Check whether the Random Forest deterioration model is loaded."""
    return {
        "ml_model_loaded": _predictor.is_ml_ready(),
        "message": (
            "ML model active — using Random Forest forecasting."
            if _predictor.is_ml_ready()
            else "No trained model found. Run `python ml/train.py` to enable ML forecasting."
        ),
    }
