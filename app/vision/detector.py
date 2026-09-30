from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.vision.source import VisionDependencyError


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class YoloDetector:
    """Ultralytics YOLO adapter; only numeric boxes/classes leave this component."""

    def __init__(self, model_path: str, device: str = "cpu", confidence: float = 0.35) -> None:
        config_dir = Path(".runtime/ultralytics").resolve()
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise VisionDependencyError(
                "Ultralytics is not installed. Install: pip install -e '.[vision]'"
            ) from exc
        self._model = YOLO(model_path)
        self._device = device
        self._confidence = confidence

    def detect(self, frame: Any) -> list[Detection]:
        result = self._model.predict(
            source=frame,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )[0]
        names = result.names
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = str(names[class_id])
            if class_name not in {"person", "cell phone"}:
                continue
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            detections.append(Detection(x1, y1, x2, y2, float(box.conf[0].item()), class_name))
        return detections
