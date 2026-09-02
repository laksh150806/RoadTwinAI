"""
RoadTwin AI — Deterioration Model Training Script
==================================================
Generates synthetic but realistic road deterioration trajectories and trains
three Random Forest regressors (one per horizon: +7d, +30d, +60d).

Run from the project root:
    python ml/train.py

Output
------
    ml/model.pkl  — joblib bundle {"model_7d", "model_30d", "model_60d"}

After training, the FastAPI server automatically uses the ML model instead
of the linear-decay fallback.

Synthetic Data Generation
--------------------------
Each training sample represents a road segment observation:
  • condition score (0–100)
  • defect counts by type and severity
  • traffic/environmental context

Future scores are computed using a realistic decay model:
    base_decay = f(defect_severity, weather, traffic)
    score_future = max(0, current_score - base_decay * days)
    + Gaussian noise to simulate real-world variance

This produces ~10 000 samples covering the full range of road conditions.
"""

import os
import sys
import logging
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# Ensure project root is on path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.deterioration_model import FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
RANDOM_STATE = 42
N_SAMPLES = 12_000


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_synthetic_data(n: int = N_SAMPLES, seed: int = RANDOM_STATE) -> tuple:
    """
    Generate synthetic road condition observations and their future scores.

    Returns
    -------
    X : np.ndarray  shape (n, len(FEATURE_NAMES))
    y_7d, y_30d, y_60d : np.ndarray  shape (n,)
    """
    rng = np.random.default_rng(seed)

    # --- Features ---
    current_score         = rng.uniform(10, 100, n)
    pothole_count         = rng.integers(0, 15, n).astype(float)
    crack_count           = rng.integers(0, 20, n).astype(float)
    waterlogging_present  = rng.integers(0, 2, n).astype(float)
    damaged_barrier_count = rng.integers(0, 8, n).astype(float)
    faded_marking_count   = rng.integers(0, 10, n).astype(float)
    broken_streetlight    = rng.integers(0, 5, n).astype(float)
    traffic_volume        = rng.uniform(0, 100, n)        # 0–100 normalised
    near_school_hospital  = rng.integers(0, 2, n).astype(float)
    weather_factor        = rng.uniform(0, 1, n)          # 0=clear,1=heavy rain
    avg_speed_kmph        = rng.uniform(10, 80, n)
    accident_history      = rng.uniform(0, 1, n)

    X = np.column_stack([
        current_score, pothole_count, crack_count, waterlogging_present,
        damaged_barrier_count, faded_marking_count, broken_streetlight,
        traffic_volume, near_school_hospital, weather_factor,
        avg_speed_kmph, accident_history,
    ])

    # --- Deterioration rate (score points lost per day) ---
    #
    # Physics-inspired model:
    #   • Each pothole drains ~0.18 pts/day, each crack ~0.08 pts/day
    #   • Waterlogging accelerates by 30 %
    #   • Traffic volume adds up to 0.15 pts/day
    #   • Rain adds up to 0.10 pts/day
    #   • Near schools/hospitals: additional 0.05 pts/day (high-stakes)
    #   • High-speed roads deteriorate faster (vibration, impact)
    #   • Accident history raises urgency but not mechanical decay
    #   • Random Gaussian noise to model real-world variance

    base_decay = (
        pothole_count         * 0.18
        + crack_count         * 0.08
        + waterlogging_present * 0.25 * (1 + 0.30)
        + damaged_barrier_count * 0.10
        + faded_marking_count  * 0.02
        + broken_streetlight   * 0.03
        + traffic_volume / 100 * 0.15
        + weather_factor       * 0.10
        + near_school_hospital * 0.05
        + avg_speed_kmph / 100 * 0.08
        + accident_history     * 0.04
    )

    noise_std = rng.uniform(0.2, 1.2, n)

    def score_at(days: int) -> np.ndarray:
        decay = base_decay * days + rng.normal(0, noise_std * np.sqrt(days), n)
        # Roads below a certain score deteriorate faster (non-linear)
        accelerator = np.where(current_score < 40, 1.4, 1.0)
        return np.clip(current_score - decay * accelerator, 0, 100)

    y_7d  = score_at(7)
    y_30d = score_at(30)
    y_60d = score_at(60)

    return X, y_7d, y_30d, y_60d


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(X: np.ndarray, y: np.ndarray, label: str) -> RandomForestRegressor:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    logger.info("  [%s] MAE=%.2f  R²=%.4f  (n_test=%d)", label, mae, r2, len(y_test))

    return model


def print_feature_importance(model: RandomForestRegressor, label: str):
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    logger.info("  [%s] Top-5 features:", label)
    for name, imp in importances[:5]:
        logger.info("    %-30s %.4f", name, imp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("Generating %d synthetic deterioration samples…", N_SAMPLES)
    X, y_7d, y_30d, y_60d = generate_synthetic_data(N_SAMPLES)

    logger.info("Training Random Forest — +7 day horizon…")
    m7 = train_model(X, y_7d, "+7d")
    print_feature_importance(m7, "+7d")

    logger.info("Training Random Forest — +30 day horizon…")
    m30 = train_model(X, y_30d, "+30d")
    print_feature_importance(m30, "+30d")

    logger.info("Training Random Forest — +60 day horizon…")
    m60 = train_model(X, y_60d, "+60d")
    print_feature_importance(m60, "+60d")

    bundle = {"model_7d": m7, "model_30d": m30, "model_60d": m60}
    joblib.dump(bundle, MODEL_PATH, compress=3)
    logger.info("Model saved → %s", MODEL_PATH)
    logger.info("✓ Training complete. Restart the FastAPI server to activate ML forecasting.")


if __name__ == "__main__":
    main()
