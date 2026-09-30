from dataclasses import dataclass

from app.db.schemas import Thresholds


@dataclass(frozen=True)
class AggregateFeatures:
    gaze_on_task_ratio: float
    posture_score_avg: float
    movement_anomaly_ratio: float
    presence_stability_ratio: float
    presence_count: int


def calculate_scai(features: AggregateFeatures, thresholds: Thresholds) -> float:
    """Calculate a collective score; no per-person value leaves the pipeline."""
    score = (
        thresholds.gaze_weight * features.gaze_on_task_ratio
        + thresholds.posture_weight * features.posture_score_avg
        + thresholds.movement_weight * (1 - features.movement_anomaly_ratio)
        + thresholds.presence_weight * features.presence_stability_ratio
    )
    return round(max(0.0, min(100.0, score * 100)), 2)


def weights_dict(thresholds: Thresholds) -> dict[str, float]:
    return {
        "gaze": thresholds.gaze_weight,
        "posture": thresholds.posture_weight,
        "movement": thresholds.movement_weight,
        "presence": thresholds.presence_weight,
    }
