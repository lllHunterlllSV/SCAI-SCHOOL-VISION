from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from app.core.config import settings


class VisionDependencyError(RuntimeError):
    pass


def require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise VisionDependencyError(
            "OpenCV is not installed. Install the project with: pip install -e '.[vision]'"
        ) from exc
    return cv2


@dataclass
class OpenCVVideoSource:
    """Capture wrapper. Frames remain in process memory and are never encoded or written."""

    source: str | int
    capture: Any = None

    def open(self) -> None:
        cv2 = require_cv2()
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(f"Could not open video source: {self.source!r}")

    def read(self) -> tuple[bool, Any | None]:
        if self.capture is None:
            raise RuntimeError("Video source is not open")
        return self.capture.read()

    @property
    def native_fps(self) -> float:
        if self.capture is None:
            return 0.0
        cv2 = require_cv2()
        return float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def source_from_config(source_type: str, config: dict) -> OpenCVVideoSource:
    if source_type == "live":
        source = config.get("camera_url", config.get("device_index", 0))
    else:
        source = config.get("file_path")
        if source == "test":
            source = settings.test_video_path
        if not source:
            raise ValueError("file_demo requires source_config.file_path or synthetic=true")
    return OpenCVVideoSource(source=source)
