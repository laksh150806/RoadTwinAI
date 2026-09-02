"""
RoadTwin AI — Deterioration Predictor
======================================
Wraps a trained scikit-learn Random Forest that forecasts road condition
scores at +7 / +30 / +60 days given the current snapshot of defects,
traffic exposure, and environmental factors.

Usage
-----
    from ml.deterioration_model import DeteriorationPredictor

    predictor = DeteriorationPredictor()          # auto-loads model.pkl
    result = predictor.predict(
        current_score=64,
        pothole_count=3,
        crack_count=4,
        waterlogging_present=1,
        damaged_barrier_count=0,
        faded_marking_count=0,
        broken_streetlight_count=0,
        traffic_volume=80,          # 0-100 normalised
        near_school_hospital=1,     # 0/1 flag
        weather_factor=0.3,         # 0-1 (0=clear, 1=heavy rain)
        avg_speed_kmph=42,
        accident_history_score=0.3, # 0-1
    )
    # result → {"forecast_7d": 58.2, "forecast_30d": 47.6, "forecast_60d": 31.1}

Falls back to linear decay if no trained model exists yet.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Path to persisted model bundle (relative to this file)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Feature order must match training (see train.py)
FEATURE_NAMES = [
    "current_score",
    "pothole_count",
    "crack_count",
    "waterlogging_present",
    "damaged_barrier_count",
    "faded_marking_count",
    "broken_streetlight_count",
    "traffic_volume",
    "near_school_hospital",
    "weather_factor",
    "avg_speed_kmph",
    "accident_history_score",
]


class DeteriorationPredictor:
    """Load-once, predict-many interface to the Random Forest model."""

    def __init__(self):
        self._model_7d = None
        self._model_30d = None
        self._model_60d = None
        self._loaded = False
        self._try_load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        current_score: float,
        pothole_count: int = 0,
        crack_count: int = 0,
        waterlogging_present: int = 0,
        damaged_barrier_count: int = 0,
        faded_marking_count: int = 0,
        broken_streetlight_count: int = 0,
        traffic_volume: float = 50.0,
        near_school_hospital: int = 0,
        weather_factor: float = 0.0,
        avg_speed_kmph: float = 40.0,
        accident_history_score: float = 0.0,
    ) -> dict:
        """
        Return deterioration forecasts as a dict:
            {"forecast_7d": float, "forecast_30d": float, "forecast_60d": float,
             "method": "ml" | "linear_fallback"}
        All returned scores are clamped to [0, 100].
        """
        if self._loaded:
            return self._ml_predict(
                current_score, pothole_count, crack_count, waterlogging_present,
                damaged_barrier_count, faded_marking_count, broken_streetlight_count,
                traffic_volume, near_school_hospital, weather_factor,
                avg_speed_kmph, accident_history_score,
            )
        else:
            return self._linear_fallback(current_score)

    def is_ml_ready(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_load(self):
        if not os.path.exists(_MODEL_PATH):
            logger.warning(
                "No trained model found at %s — using linear decay fallback. "
                "Run `python ml/train.py` to train the model.",
                _MODEL_PATH,
            )
            return
        try:
            import joblib
            bundle = joblib.load(_MODEL_PATH)
            self._model_7d = bundle["model_7d"]
            self._model_30d = bundle["model_30d"]
            self._model_60d = bundle["model_60d"]
            self._loaded = True
            logger.info("DeteriorationPredictor: loaded model from %s", _MODEL_PATH)
        except Exception as exc:
            logger.error("Failed to load model: %s — using linear fallback.", exc)

    def _build_feature_vector(self, **kw) -> np.ndarray:
        return np.array([[kw[f] for f in FEATURE_NAMES]])

    def _ml_predict(self, current_score, pothole_count, crack_count,
                    waterlogging_present, damaged_barrier_count, faded_marking_count,
                    broken_streetlight_count, traffic_volume, near_school_hospital,
                    weather_factor, avg_speed_kmph, accident_history_score) -> dict:
        X = self._build_feature_vector(
            current_score=current_score,
            pothole_count=pothole_count,
            crack_count=crack_count,
            waterlogging_present=waterlogging_present,
            damaged_barrier_count=damaged_barrier_count,
            faded_marking_count=faded_marking_count,
            broken_streetlight_count=broken_streetlight_count,
            traffic_volume=traffic_volume,
            near_school_hospital=near_school_hospital,
            weather_factor=weather_factor,
            avg_speed_kmph=avg_speed_kmph,
            accident_history_score=accident_history_score,
        )
        f7 = float(np.clip(self._model_7d.predict(X)[0], 0, 100))
        f30 = float(np.clip(self._model_30d.predict(X)[0], 0, 100))
        f60 = float(np.clip(self._model_60d.predict(X)[0], 0, 100))
        return {
            "forecast_7d": round(f7, 1),
            "forecast_30d": round(f30, 1),
            "forecast_60d": round(f60, 1),
            "method": "ml",
        }

    @staticmethod
    def _linear_fallback(current_score: float) -> dict:
        """Simple linear decay used before the RF model is trained."""
        return {
            "forecast_7d": max(0, round(current_score - 5, 1)),
            "forecast_30d": max(0, round(current_score - 18, 1)),
            "forecast_60d": max(0, round(current_score - 33, 1)),
            "method": "linear_fallback",
        }
