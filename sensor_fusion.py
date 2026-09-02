"""
RoadTwin AI — GPS + IMU Sensor Fusion Agent
============================================
Implements a lightweight 1-D Kalman Filter per axis to fuse:
  • GPS coordinates   — low-frequency (~1 Hz), accurate long-term
  • IMU accelerometer — high-frequency (~50 Hz), accurate short-term

The fused output is a high-confidence defect location that accounts for
vehicle motion between GPS fixes — critical for pinpointing potholes
that occur between two GPS samples.

Architecture
------------
    raw GPS + IMU batch
          │
          ▼
    KalmanFusion1D (lat axis)  +  KalmanFusion1D (lng axis)
          │
          ▼
    fused_lat, fused_lng, confidence_score (0–1)
          │
          ▼
    SensorFusionAgent.fuse_batch()  →  list[FusedPoint]

Usage (programmatic)
--------------------
    from sensor_fusion import SensorFusionAgent

    agent = SensorFusionAgent()
    results = agent.fuse_batch(
        gps_samples=[
            {"lat": 12.8230, "lng": 80.0450, "timestamp_ms": 0},
            {"lat": 12.8231, "lng": 80.0452, "timestamp_ms": 1000},
        ],
        imu_samples=[
            {"ax": 0.12, "ay": -0.05, "az": 9.78, "timestamp_ms": 0},
            {"ax": 0.18, "ay": -0.02, "az": 9.81, "timestamp_ms": 20},
            # ... more at ~50 Hz
        ],
        threshold_g=0.4,    # g-force threshold to flag a defect event
    )
    # results → list of FusedPoint dicts

FastAPI endpoint: POST /sensor-fusion (defined in main.py)
"""

import math
from typing import List, Optional
from dataclasses import dataclass, asdict

import numpy as np


# ---------------------------------------------------------------------------
# Kalman Filter (1-D, constant-velocity model)
# ---------------------------------------------------------------------------

class KalmanFusion1D:
    """
    Fuses a 1-D position signal (GPS) with velocity updates derived from
    accelerometer integration using a Kalman Filter.

    State vector: [position, velocity]
    Measurement:  [position]  (from GPS)
    Control input: [acceleration] (from IMU)
    """

    def __init__(self, process_noise_q: float = 1e-4, measurement_noise_r: float = 1e-3):
        # State: [pos, vel]
        self.x = np.zeros((2, 1))
        # State covariance
        self.P = np.eye(2) * 1.0
        # Process noise covariance
        self.Q = np.array([[process_noise_q, 0], [0, process_noise_q * 10]])
        # Measurement noise covariance (GPS accuracy ~3 m ≈ 2.7e-5 degrees)
        self.R = np.array([[measurement_noise_r]])
        # Measurement matrix
        self.H = np.array([[1.0, 0.0]])

    def predict(self, dt: float, acceleration: float = 0.0):
        """Propagate state forward by dt seconds with optional acceleration."""
        # State transition
        F = np.array([[1.0, dt], [0.0, 1.0]])
        # Control matrix
        B = np.array([[0.5 * dt ** 2], [dt]])
        self.x = F @ self.x + B * acceleration
        self.P = F @ self.P @ F.T + self.Q

    def update(self, gps_measurement: float):
        """Correct state with a GPS measurement."""
        z = np.array([[gps_measurement]])
        y = z - self.H @ self.x                          # innovation
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

    @property
    def position(self) -> float:
        return float(self.x[0, 0])

    @property
    def velocity(self) -> float:
        return float(self.x[1, 0])


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FusedPoint:
    timestamp_ms: int
    raw_lat: Optional[float]
    raw_lng: Optional[float]
    fused_lat: float
    fused_lng: float
    horizontal_error_m: float        # estimated GPS uncertainty at this point
    imu_magnitude_g: float           # combined acceleration magnitude
    is_defect_event: bool            # True if IMU spike exceeds threshold
    confidence: float                # 0–1 fusion confidence score

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _haversine_m(lat1, lng1, lat2, lng2) -> float:
    """Distance in metres between two WGS-84 coordinates."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _degrees_per_meter_lat() -> float:
    return 1.0 / 111_320.0


def _degrees_per_meter_lng(lat: float) -> float:
    return 1.0 / (111_320.0 * math.cos(math.radians(lat)))


# ---------------------------------------------------------------------------
# Sensor Fusion Agent
# ---------------------------------------------------------------------------

class SensorFusionAgent:
    """
    Main entry point for fusing GPS + IMU sensor data.

    Parameters
    ----------
    gps_noise_r : float
        Kalman measurement noise for GPS (lower = trust GPS more).
    process_noise_q : float
        Kalman process noise (lower = smoother trajectory).
    """

    def __init__(self, gps_noise_r: float = 1e-3, process_noise_q: float = 1e-4):
        self.gps_noise_r = gps_noise_r
        self.process_noise_q = process_noise_q

    def fuse_batch(
        self,
        gps_samples: List[dict],
        imu_samples: List[dict],
        threshold_g: float = 0.4,
    ) -> List[dict]:
        """
        Fuse a batch of GPS and IMU samples.

        Parameters
        ----------
        gps_samples : list of dicts
            Each dict: {"lat": float, "lng": float, "timestamp_ms": int}
        imu_samples : list of dicts
            Each dict: {"ax": float, "ay": float, "az": float, "timestamp_ms": int}
            Accelerations in m/s² (or g — the filter handles both via threshold_g)
        threshold_g : float
            IMU spike magnitude (in same units as ax/ay/az) that flags a
            potential defect event (pothole, bump, etc.).

        Returns
        -------
        List of FusedPoint dicts, one per GPS sample, enriched with fused
        coordinates and IMU-derived defect flags.
        """
        if not gps_samples:
            return []

        # Sort by time
        gps_sorted = sorted(gps_samples, key=lambda s: s["timestamp_ms"])
        imu_sorted = sorted(imu_samples, key=lambda s: s["timestamp_ms"])

        # Initialise Kalman filters at first GPS fix
        kf_lat = KalmanFusion1D(self.process_noise_q, self.gps_noise_r)
        kf_lng = KalmanFusion1D(self.process_noise_q, self.gps_noise_r)
        kf_lat.x[0, 0] = gps_sorted[0]["lat"]
        kf_lng.x[0, 0] = gps_sorted[0]["lng"]

        # Build IMU index for fast lookup by time
        imu_by_time = {s["timestamp_ms"]: s for s in imu_sorted}
        imu_times = np.array([s["timestamp_ms"] for s in imu_sorted])

        results: List[FusedPoint] = []
        prev_t_ms = gps_sorted[0]["timestamp_ms"]

        for gps in gps_sorted:
            t_ms = gps["timestamp_ms"]
            dt_s = max((t_ms - prev_t_ms) / 1000.0, 1e-6)

            # --- Find IMU samples in the [prev_t, t] window ---
            window_mask = (imu_times > prev_t_ms) & (imu_times <= t_ms)
            window_imu = [imu_sorted[i] for i in np.where(window_mask)[0]]

            # Compute mean acceleration in lat/lng directions from IMU window
            accel_lat = 0.0
            accel_lng = 0.0
            magnitudes = []

            for s in window_imu:
                ax, ay = s.get("ax", 0.0), s.get("ay", 0.0)
                az = s.get("az", 9.81)
                # Horizontal magnitude (remove gravity from z)
                horiz_mag = math.sqrt(ax**2 + ay**2)
                magnitudes.append(horiz_mag)
                # Naive projection: ax→lat direction, ay→lng direction
                accel_lat += ax * _degrees_per_meter_lat() * dt_s**2 * 0.5
                accel_lng += ay * _degrees_per_meter_lng(gps["lat"]) * dt_s**2 * 0.5

            mean_mag = float(np.mean(magnitudes)) if magnitudes else 0.0

            # --- Kalman predict (IMU motion update) ---
            kf_lat.predict(dt_s, accel_lat / max(dt_s, 1e-6))
            kf_lng.predict(dt_s, accel_lng / max(dt_s, 1e-6))

            # --- Kalman update (GPS correction) ---
            kf_lat.update(gps["lat"])
            kf_lng.update(gps["lng"])

            # --- Confidence: based on GPS–fused discrepancy ---
            raw_lat, raw_lng = gps["lat"], gps["lng"]
            err_m = _haversine_m(raw_lat, raw_lng, kf_lat.position, kf_lng.position)
            # Confidence: 1 when err=0, ~0.5 when err=15m (typical GPS drift)
            confidence = float(np.clip(1.0 / (1.0 + err_m / 10.0), 0.0, 1.0))

            results.append(FusedPoint(
                timestamp_ms=t_ms,
                raw_lat=raw_lat,
                raw_lng=raw_lng,
                fused_lat=round(kf_lat.position, 7),
                fused_lng=round(kf_lng.position, 7),
                horizontal_error_m=round(err_m, 2),
                imu_magnitude_g=round(mean_mag, 4),
                is_defect_event=(mean_mag >= threshold_g),
                confidence=round(confidence, 3),
            ))

            prev_t_ms = t_ms

        return [p.to_dict() for p in results]

    def best_defect_location(self, fused_points: List[dict]) -> Optional[dict]:
        """
        From a list of fused points, return the highest-confidence defect
        event location (for storing a single representative lat/lng per defect).
        Returns None if no defect events are present.
        """
        events = [p for p in fused_points if p["is_defect_event"]]
        if not events:
            return None
        return max(events, key=lambda p: p["confidence"])
