from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.vision.source import VisionDependencyError, require_cv2


@dataclass(frozen=True)
class LandmarkFeatures:
    gaze_on_task: float
    posture_score: float


class MediaPipeEstimator:
    """Head-orientation and posture heuristics from transient MediaPipe landmarks."""

    def __init__(self) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise VisionDependencyError(
                "MediaPipe is not installed. Use Python 3.11-3.13 and install '.[vision]'"
            ) from exc
        self._face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.35,
        )
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.35,
        )

    def estimate(self, person_crop: Any) -> LandmarkFeatures:
        cv2 = require_cv2()
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        face_result = self._face.process(rgb)
        pose_result = self._pose.process(rgb)
        gaze = self._gaze_score(face_result)
        posture = self._posture_score(pose_result)
        return LandmarkFeatures(gaze_on_task=gaze, posture_score=posture)

    @staticmethod
    def _gaze_score(result: Any) -> float:
        if not result.multi_face_landmarks:
            return 0.5
        points = result.multi_face_landmarks[0].landmark
        nose_x = points[1].x
        left_cheek_x = points[234].x
        right_cheek_x = points[454].x
        face_center = (left_cheek_x + right_cheek_x) / 2
        half_width = max(abs(right_cheek_x - left_cheek_x) / 2, 0.01)
        normalized_yaw = abs(nose_x - face_center) / half_width
        nose_y = points[1].y
        eye_y = (points[33].y + points[263].y) / 2
        chin_y = points[152].y
        normalized_pitch = abs((nose_y - eye_y) / max(chin_y - eye_y, 0.01) - 0.45)
        return max(0.0, min(1.0, 1 - 0.7 * normalized_yaw - normalized_pitch))

    @staticmethod
    def _posture_score(result: Any) -> float:
        if not result.pose_landmarks:
            return 0.5
        points = result.pose_landmarks.landmark
        left_shoulder, right_shoulder = points[11], points[12]
        left_hip, right_hip = points[23], points[24]
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        torso_height = max(
            (left_hip.y + right_hip.y) / 2 - (left_shoulder.y + right_shoulder.y) / 2, 0.05
        )
        lean = abs(shoulder_center_x - hip_center_x) / torso_height
        shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
        visibility = min(
            left_shoulder.visibility,
            right_shoulder.visibility,
            left_hip.visibility,
            right_hip.visibility,
        )
        return max(0.0, min(1.0, (1 - lean - shoulder_tilt * 2) * visibility))

    def close(self) -> None:
        self._face.close()
        self._pose.close()


class NeutralLandmarkEstimator:
    """YOLO-only fallback for runtimes where MediaPipe has no compatible wheel."""

    def estimate(self, person_crop: Any) -> LandmarkFeatures:
        return LandmarkFeatures(gaze_on_task=0.5, posture_score=0.5)

    def close(self) -> None:
        pass
