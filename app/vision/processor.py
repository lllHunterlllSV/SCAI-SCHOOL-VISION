from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.scai.engine import AggregateFeatures
from app.vision.detector import Detection, YoloDetector
from app.vision.landmarks import MediaPipeEstimator


@dataclass(frozen=True)
class FrameAnalysis:
    features: AggregateFeatures
    phone_detected: bool
    annotations: tuple[PersonAnnotation, ...] = ()


@dataclass(frozen=True)
class PersonAnnotation:
    box: tuple[int, int, int, int]
    confidence: float
    gaze_on_task: float
    posture_score: float


class ClassroomFrameProcessor:
    def __init__(self, detector: YoloDetector, estimator: MediaPipeEstimator) -> None:
        self.detector = detector
        self.estimator = estimator
        self._previous_centroids: list[tuple[float, float]] = []
        self._previous_presence: int | None = None

    def process(self, frame: Any) -> FrameAnalysis:
        detections = self.detector.detect(frame)
        people = [item for item in detections if item.class_name == "person"]
        phone_detected = any(item.class_name == "cell phone" for item in detections)
        gaze_scores: list[float] = []
        posture_scores: list[float] = []
        annotations: list[PersonAnnotation] = []
        height, width = frame.shape[:2]
        for person in people:
            x1, y1 = max(0, person.x1), max(0, person.y1)
            x2, y2 = min(width, person.x2), min(height, person.y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            crop = frame[y1:y2, x1:x2]
            landmark_features = self.estimator.estimate(crop)
            gaze_scores.append(landmark_features.gaze_on_task)
            posture_scores.append(landmark_features.posture_score)
            annotations.append(
                PersonAnnotation(
                    box=(x1, y1, x2, y2),
                    confidence=person.confidence,
                    gaze_on_task=landmark_features.gaze_on_task,
                    posture_score=landmark_features.posture_score,
                )
            )

        movement = self._movement_ratio(people, width, height)
        presence = len(people)
        if self._previous_presence is None or max(presence, self._previous_presence) == 0:
            stability = 1.0 if presence else 0.0
        else:
            stability = 1 - abs(presence - self._previous_presence) / max(
                presence, self._previous_presence
            )
        self._previous_presence = presence
        self._previous_centroids = [person.centroid for person in people]
        features = AggregateFeatures(
            gaze_on_task_ratio=sum(gaze_scores) / len(gaze_scores) if gaze_scores else 0.5,
            posture_score_avg=sum(posture_scores) / len(posture_scores) if posture_scores else 0.5,
            movement_anomaly_ratio=movement,
            presence_stability_ratio=max(0.0, min(1.0, stability)),
            presence_count=presence,
        )
        # No crop, landmark, detection, or frame is retained on the instance.
        return FrameAnalysis(
            features=features,
            phone_detected=phone_detected,
            annotations=tuple(annotations),
        )

    def _movement_ratio(self, people: list[Detection], width: int, height: int) -> float:
        if not people or not self._previous_centroids:
            return 0.0
        diagonal = max(math.hypot(width, height), 1)
        distances = []
        for person in people:
            cx, cy = person.centroid
            nearest = min(
                math.hypot(cx - old_x, cy - old_y) for old_x, old_y in self._previous_centroids
            )
            distances.append(min(1.0, nearest / (diagonal * 0.12)))
        return sum(distances) / len(distances)

    def close(self) -> None:
        self.estimator.close()
