import numpy as np

from app.vision.detector import Detection
from app.vision.landmarks import LandmarkFeatures
from app.vision.processor import ClassroomFrameProcessor


class FakeDetector:
    def __init__(self, batches):
        self.batches = iter(batches)

    def detect(self, _frame):
        return next(self.batches)


class FakeEstimator:
    def estimate(self, _crop):
        return LandmarkFeatures(gaze_on_task=0.8, posture_score=0.7)

    def close(self):
        pass


def test_frame_processor_returns_collective_aggregates_only():
    person_first = Detection(10, 10, 60, 90, 0.9, "person")
    person_moved = Detection(15, 10, 65, 90, 0.9, "person")
    phone = Detection(20, 40, 30, 55, 0.8, "cell phone")
    processor = ClassroomFrameProcessor(
        FakeDetector([[person_first], [person_moved, phone]]), FakeEstimator()
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    first = processor.process(frame)
    second = processor.process(frame)

    assert first.features.presence_count == 1
    assert first.features.gaze_on_task_ratio == 0.8
    assert first.features.posture_score_avg == 0.7
    assert second.phone_detected is True
    assert 0 < second.features.movement_anomaly_ratio < 1
    assert not hasattr(processor, "frame")
    assert not hasattr(processor, "crops")


def test_empty_classroom_is_handled_without_division_by_zero():
    processor = ClassroomFrameProcessor(FakeDetector([[]]), FakeEstimator())
    result = processor.process(np.zeros((50, 50, 3), dtype=np.uint8))
    assert result.features.presence_count == 0
    assert result.features.presence_stability_ratio == 0
