from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class SessionCreate(BaseModel):
    class_id: str = Field(min_length=1, max_length=100)
    mode: Literal["lecture", "exam"]
    video_source_type: Literal["live", "file_demo"]
    source_config: dict = Field(default_factory=dict)
    consent_confirmed: bool

    @model_validator(mode="after")
    def validate_source(self) -> "SessionCreate":
        allowed = (
            {
                "device_index",
                "camera_url",
                "model_path",
                "device",
                "confidence",
                "sampling_fps",
            }
            if self.video_source_type == "live"
            else {
                "file_path",
                "uploaded_file_id",
                "synthetic",
                "model_path",
                "device",
                "confidence",
                "sampling_fps",
                "max_duration_seconds",
            }
        )
        if self.source_config and not set(self.source_config).issubset(allowed):
            raise ValueError(f"source_config only accepts: {sorted(allowed)}")
        return self


class SessionRead(ORMModel):
    id: str
    class_id: str
    mode: str
    video_source_type: str
    consent_confirmed: bool
    status: str
    started_at: datetime
    ended_at: datetime | None
    avg_scai_score: float | None
    avg_processing_fps: float | None


class SnapshotRead(ORMModel):
    id: str
    session_id: str
    timestamp: datetime
    window_seconds: int
    scai_score: float
    avg_gaze_on_task_pct: float
    avg_posture_score: float
    presence_count: int
    movement_index: float
    raw_component_weights: dict


class EventRead(ORMModel):
    id: str
    session_id: str
    timestamp: datetime
    event_type: str
    confidence: float
    zone: str | None


class FeedbackCreate(BaseModel):
    perceived_attention_score: int = Field(ge=1, le=5)
    comments: str | None = Field(default=None, max_length=2000)


class FeedbackRead(ORMModel):
    id: str
    session_id: str
    perceived_attention_score: int
    comments: str | None
    submitted_at: datetime


class RecommendationRead(ORMModel):
    id: str
    session_id: str
    generated_at: datetime
    summary_text: str
    suggested_actions: list[str]
    model_used: str


class Thresholds(BaseModel):
    gaze_weight: float = Field(default=0.4, ge=0, le=1)
    posture_weight: float = Field(default=0.25, ge=0, le=1)
    movement_weight: float = Field(default=0.2, ge=0, le=1)
    presence_weight: float = Field(default=0.15, ge=0, le=1)
    low_attention_threshold: float = Field(default=50, ge=0, le=100)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "Thresholds":
        total = self.gaze_weight + self.posture_weight + self.movement_weight + self.presence_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError("SCAI weights must sum to 1")
        return self
