import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class SessionStatus(str, enum.Enum):
    active = "active"
    completed = "completed"
    error = "error"


class ClassroomSession(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    class_id: Mapped[str] = mapped_column(String(100), index=True)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    mode: Mapped[str] = mapped_column(String(20))
    video_source_type: Mapped[str] = mapped_column(String(20))
    source_config: Mapped[dict] = mapped_column(JSON, default=dict)
    consent_confirmed: Mapped[bool]
    status: Mapped[str] = mapped_column(String(20), default=SessionStatus.active.value)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    avg_scai_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_processing_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    snapshots: Mapped[list["ScaiSnapshot"]] = relationship(cascade="all, delete-orphan")
    events: Mapped[list["BehaviorEvent"]] = relationship(cascade="all, delete-orphan")
    recommendations: Mapped[list["Recommendation"]] = relationship(cascade="all, delete-orphan")
    feedback: Mapped[list["TeacherFeedback"]] = relationship(cascade="all, delete-orphan")


class ScaiSnapshot(Base):
    __tablename__ = "scai_snapshots"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    window_seconds: Mapped[int] = mapped_column(Integer, default=30)
    scai_score: Mapped[float]
    avg_gaze_on_task_pct: Mapped[float]
    avg_posture_score: Mapped[float]
    presence_count: Mapped[int]
    movement_index: Mapped[float]
    raw_component_weights: Mapped[dict] = mapped_column(JSON)


class BehaviorEvent(Base):
    __tablename__ = "behavior_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    event_type: Mapped[str] = mapped_column(String(50))
    confidence: Mapped[float]
    zone: Mapped[str | None] = mapped_column(String(100), nullable=True)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


class Recommendation(Base):
    __tablename__ = "recommendations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    summary_text: Mapped[str] = mapped_column(Text)
    suggested_actions: Mapped[list] = mapped_column(JSON)
    model_used: Mapped[str] = mapped_column(String(100))


class TeacherFeedback(Base):
    __tablename__ = "teacher_feedback"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    teacher_id: Mapped[str] = mapped_column(String(100))
    perceived_attention_score: Mapped[int]
    comments: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
