from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

from database import engine, get_db, Base
from models import (
    Road, Defect, ConditionScore, DeteriorationForecast,
    TrafficRisk, RepairPriority, TrafficSimulation
)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="RoadTwin AI - Decision Engine")


@app.get("/")
def root():
    return {"status": "RoadTwin backend running"}


# ---------- ROADS ----------

@app.post("/roads")
def create_road(name: str, start_lat: float = None, start_lng: float = None,
                 end_lat: float = None, end_lng: float = None, db: Session = Depends(get_db)):
    road = Road(name=name, start_lat=start_lat, start_lng=start_lng,
                end_lat=end_lat, end_lng=end_lng)
    db.add(road)
    db.commit()
    db.refresh(road)
    return road


@app.get("/roads")
def list_roads(db: Session = Depends(get_db)):
    return db.query(Road).all()


# ---------- DEFECT INGESTION (from Vision Agent / Member 1) ----------

class DefectIn(BaseModel):
    road_id: int
    defect_type: str
    severity: float
    confidence: Optional[float] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    source: Optional[str] = None


@app.post("/defects")
def add_defect(defect: DefectIn, db: Session = Depends(get_db)):
    road = db.query(Road).filter(Road.id == defect.road_id).first()
    if not road:
        raise HTTPException(status_code=404, detail="Road not found")

    record = Defect(**defect.dict())
    db.add(record)
    db.commit()
    db.refresh(record)

    # auto-recompute condition score whenever a new defect comes in
    compute_condition_score(defect.road_id, db)

    return record


@app.get("/roads/{road_id}/defects")
def get_defects(road_id: int, db: Session = Depends(get_db)):
    return db.query(Defect).filter(Defect.road_id == road_id).all()


# ---------- CONDITION SCORE (Decision Engine core) ----------

DEFECT_WEIGHTS = {
    "pothole": 10,
    "crack": 5,
    "waterlogging": 8,
    "damaged_barrier": 6,
    "faded_marking": 2,
    "broken_streetlight": 3,
}


def compute_condition_score(road_id: int, db: Session):
    defects = db.query(Defect).filter(Defect.road_id == road_id).all()
    score = 100.0
    breakdown = {}

    for d in defects:
        weight = DEFECT_WEIGHTS.get(d.defect_type, 3)
        penalty = weight * (d.severity if d.severity <= 1 else d.severity / 100)
        score -= penalty
        breakdown[d.defect_type] = breakdown.get(d.defect_type, 0) + 1

    score = max(0, min(100, round(score, 1)))

    record = ConditionScore(road_id=road_id, score=score, breakdown=breakdown)
    db.add(record)
    db.commit()
    return score, breakdown


@app.get("/roads/{road_id}/condition-score")
def get_condition_score(road_id: int, db: Session = Depends(get_db)):
    road = db.query(Road).filter(Road.id == road_id).first()
    if not road:
        raise HTTPException(status_code=404, detail="Road not found")
    score, breakdown = compute_condition_score(road_id, db)
    return {"road_id": road_id, "score": score, "breakdown": breakdown}


# ---------- DETERIORATION FORECAST (placeholder until Member 2's model is ready) ----------

@app.get("/roads/{road_id}/forecast")
def get_forecast(road_id: int, db: Session = Depends(get_db)):
    latest = db.query(ConditionScore).filter(ConditionScore.road_id == road_id)\
        .order_by(ConditionScore.computed_at.desc()).first()
    if not latest:
        raise HTTPException(status_code=404, detail="No condition score yet for this road")

    current = latest.score
    # simple linear decay placeholder — swap with Member 2's real model later
    forecast_7d = max(0, round(current - 5, 1))
    forecast_30d = max(0, round(current - 18, 1))
    forecast_60d = max(0, round(current - 33, 1))

    record = DeteriorationForecast(
        road_id=road_id, forecast_7d=forecast_7d,
        forecast_30d=forecast_30d, forecast_60d=forecast_60d
    )
    db.add(record)
    db.commit()

    return {
        "road_id": road_id,
        "current_score": current,
        "forecast_7d": forecast_7d,
        "forecast_30d": forecast_30d,
        "forecast_60d": forecast_60d
    }


# ---------- TRAFFIC RISK (data contract with Member 2) ----------

class TrafficRiskIn(BaseModel):
    road_id: int
    traffic_volume: Optional[float] = 0
    pedestrian_density: Optional[float] = 0
    near_school_hospital: Optional[int] = 0
    weather_factor: Optional[float] = 0
    time_of_day_factor: Optional[float] = 0


@app.post("/roads/{road_id}/traffic-risk")
def set_traffic_risk(road_id: int, risk: TrafficRiskIn, db: Session = Depends(get_db)):
    # simple weighted combination — Member 2 can override by posting their own risk_score directly if preferred
    risk_score = (
        risk.traffic_volume * 0.35 +
        risk.pedestrian_density * 0.25 +
        risk.near_school_hospital * 20 +
        risk.weather_factor * 0.1 +
        risk.time_of_day_factor * 0.1
    )
    record = TrafficRisk(
        road_id=road_id,
        traffic_volume=risk.traffic_volume,
        pedestrian_density=risk.pedestrian_density,
        near_school_hospital=risk.near_school_hospital,
        weather_factor=risk.weather_factor,
        time_of_day_factor=risk.time_of_day_factor,
        risk_score=round(risk_score, 2)
    )
    db.add(record)
    db.commit()
    return {"road_id": road_id, "risk_score": round(risk_score, 2)}


# ---------- REPAIR PRIORITY RANKING (Decision Engine core) ----------

@app.get("/repair-priorities")
def get_repair_priorities(db: Session = Depends(get_db)):
    roads = db.query(Road).all()
    results = []

    for road in roads:
        latest_score = db.query(ConditionScore).filter(ConditionScore.road_id == road.id)\
            .order_by(ConditionScore.computed_at.desc()).first()
        latest_risk = db.query(TrafficRisk).filter(TrafficRisk.road_id == road.id)\
            .order_by(TrafficRisk.computed_at.desc()).first()

        if not latest_score:
            continue

        condition_score = latest_score.score
        risk_score = latest_risk.risk_score if latest_risk else 1

        # priority = worse condition + higher risk = higher priority
        priority_score = round((100 - condition_score) * (1 + risk_score / 100), 2)

        results.append({
            "road_id": road.id,
            "road_name": road.name,
            "condition_score": condition_score,
            "risk_score": risk_score,
            "priority_score": priority_score
        })

    results.sort(key=lambda x: x["priority_score"], reverse=True)
    for i, r in enumerate(results):
        r["priority_rank"] = i + 1
        record = RepairPriority(
            road_id=r["road_id"], priority_rank=r["priority_rank"],
            priority_score=r["priority_score"]
        )
        db.add(record)
    db.commit()

    return results


# ---------- WHAT-IF TRAFFIC SIMULATION ----------

class SimulationIn(BaseModel):
    road_id: int
    scenario: str = "road_closure"


@app.post("/simulate")
def simulate_closure(sim: SimulationIn, db: Session = Depends(get_db)):
    road = db.query(Road).filter(Road.id == sim.road_id).first()
    if not road:
        raise HTTPException(status_code=404, detail="Road not found")

    # placeholder simulation logic — replace with real routing model if time allows
    affected_roads = {
        "Road B": "+34%",
        "Road C": "+21%",
        "Road D": "+8%"
    }
    recommendation = "Repair 11 PM–5 AM · divert via Route C"

    record = TrafficSimulation(
        road_id=sim.road_id,
        scenario=sim.scenario,
        affected_roads=affected_roads,
        recommendation=recommendation
    )
    db.add(record)
    db.commit()

    return {
        "road_id": sim.road_id,
        "scenario": sim.scenario,
        "affected_roads": affected_roads,
        "recommendation": recommendation
    }


# ---------- SUMMARY (for Member 4's dashboard / the demo moment) ----------

@app.get("/roads/{road_id}/summary")
def get_summary(road_id: int, db: Session = Depends(get_db)):
    road = db.query(Road).filter(Road.id == road_id).first()
    if not road:
        raise HTTPException(status_code=404, detail="Road not found")

    latest_score = db.query(ConditionScore).filter(ConditionScore.road_id == road_id)\
        .order_by(ConditionScore.computed_at.desc()).first()
    latest_forecast = db.query(DeteriorationForecast).filter(DeteriorationForecast.road_id == road_id)\
        .order_by(DeteriorationForecast.created_at.desc()).first()
    latest_priority = db.query(RepairPriority).filter(RepairPriority.road_id == road_id)\
        .order_by(RepairPriority.computed_at.desc()).first()

    return {
        "road": {"id": road.id, "name": road.name},
        "condition_score": latest_score.score if latest_score else None,
        "forecast": {
            "7d": latest_forecast.forecast_7d,
            "30d": latest_forecast.forecast_30d,
            "60d": latest_forecast.forecast_60d
        } if latest_forecast else None,
        "priority_rank": latest_priority.priority_rank if latest_priority else None
  }
