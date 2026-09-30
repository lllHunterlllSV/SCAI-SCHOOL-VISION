import asyncio
import math
import random
import time
from datetime import UTC, datetime

from sqlalchemy import select

from app.core.config import settings
from app.db.base import SessionLocal
from app.db.models import BehaviorEvent, ClassroomSession, ScaiSnapshot
from app.db.schemas import Thresholds
from app.realtime.connection_manager import manager
from app.scai.engine import AggregateFeatures, calculate_scai, weights_dict
from app.services.thresholds import threshold_store
from app.vision.detector import YoloDetector
from app.vision.landmarks import MediaPipeEstimator, NeutralLandmarkEstimator
from app.vision.preview import preview_hub
from app.vision.processor import ClassroomFrameProcessor, FrameAnalysis
from app.vision.source import VisionDependencyError, require_cv2, source_from_config


class PipelineRegistry:
    """Runs synthetic or real vision workers while persisting aggregate data only."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}

    def start(self, session_id: str) -> None:
        if session_id not in self._tasks:
            self._tasks[session_id] = asyncio.create_task(self._dispatch(session_id))

    async def stop(self, session_id: str) -> None:
        task = self._tasks.pop(session_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def shutdown(self) -> None:
        await asyncio.gather(*(self.stop(key) for key in list(self._tasks)))

    async def _dispatch(self, session_id: str) -> None:
        async with SessionLocal() as db:
            session = await db.get(ClassroomSession, session_id)
            if not session:
                return
            synthetic = bool(session.source_config.get("synthetic"))
        if synthetic:
            await self._run_synthetic(session_id)
        else:
            await self._run_vision(session_id)

    async def _run_synthetic(self, session_id: str) -> None:
        rng = random.Random(session_id)
        tick = 0
        started = time.perf_counter()
        try:
            while not settings.demo_max_ticks or tick < settings.demo_max_ticks:
                async with SessionLocal() as db:
                    session = await db.scalar(
                        select(ClassroomSession).where(ClassroomSession.id == session_id)
                    )
                    if not session or session.status != "active":
                        return
                    wave = math.sin(tick / 4) * 0.08
                    features = AggregateFeatures(
                        gaze_on_task_ratio=max(
                            0.35, min(0.95, 0.72 + wave + rng.uniform(-0.06, 0.06))
                        ),
                        posture_score_avg=max(0.4, min(0.95, 0.75 + rng.uniform(-0.08, 0.08))),
                        movement_anomaly_ratio=max(
                            0.02, min(0.5, 0.14 - wave + rng.uniform(-0.04, 0.04))
                        ),
                        presence_stability_ratio=max(0.65, min(1, 0.92 + rng.uniform(-0.04, 0.04))),
                        presence_count=rng.randint(18, 25),
                    )
                    session.avg_processing_fps = round(
                        (tick + 1) / max(time.perf_counter() - started, 0.001), 2
                    )
                    await self._persist_analysis(db, session, FrameAnalysis(features, False))
                tick += 1
                await asyncio.sleep(settings.demo_tick_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            async with SessionLocal() as db:
                session = await db.get(ClassroomSession, session_id)
                if session:
                    session.status = "error"
                    await db.commit()
            await manager.broadcast(
                session_id,
                {"type": "session_status", "status": "error", "detail": type(exc).__name__},
            )
        finally:
            self._tasks.pop(session_id, None)

    async def _run_vision(self, session_id: str) -> None:
        source = None
        processor = None
        try:
            async with SessionLocal() as db:
                session = await db.get(ClassroomSession, session_id)
                if not session:
                    return
                config = session.source_config
                source = source_from_config(session.video_source_type, config)
                detector = YoloDetector(
                    model_path=str(config.get("model_path", settings.yolo_model_path)),
                    device=str(config.get("device", settings.yolo_device)),
                    confidence=float(config.get("confidence", settings.yolo_confidence)),
                )
                try:
                    estimator = MediaPipeEstimator()
                except VisionDependencyError:
                    estimator = NeutralLandmarkEstimator()
                processor = ClassroomFrameProcessor(detector, estimator)
            await asyncio.to_thread(source.open)
            native_fps = source.native_fps
            target_fps = float(config.get("sampling_fps", settings.default_sampling_fps))
            max_duration_seconds = float(config.get("max_duration_seconds", 0))
            frame_interval = max(1, round(native_fps / target_fps)) if native_fps > 0 else 1
            frame_number = processed = 0
            started = time.perf_counter()
            while True:
                ok, frame = await asyncio.to_thread(source.read)
                if not ok:
                    break
                frame_number += 1
                if (
                    max_duration_seconds > 0
                    and native_fps > 0
                    and frame_number > native_fps * max_duration_seconds
                ):
                    break
                if frame_number % frame_interval:
                    del frame
                    continue
                analysis = await asyncio.to_thread(processor.process, frame)
                await asyncio.to_thread(self._publish_preview, session_id, frame, analysis)
                del frame  # Explicit privacy boundary: discard raw pixels immediately.
                processed += 1
                async with SessionLocal() as db:
                    session = await db.get(ClassroomSession, session_id)
                    if not session or session.status != "active":
                        return
                    session.avg_processing_fps = round(
                        processed / max(time.perf_counter() - started, 0.001), 2
                    )
                    await self._persist_analysis(db, session, analysis)
                if native_fps <= 0:
                    await asyncio.sleep(max(0, 1 / target_fps))
            await self._complete_file_session(session_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._mark_error(session_id, exc)
        finally:
            if source is not None:
                await asyncio.to_thread(source.close)
            if processor is not None:
                await asyncio.to_thread(processor.close)
            preview_hub.remove(session_id)
            self._tasks.pop(session_id, None)

    @staticmethod
    def _publish_preview(session_id: str, frame, analysis: FrameAnalysis) -> None:
        cv2 = require_cv2()
        for person in analysis.annotations:
            gaze = round(person.gaze_on_task * 100)
            posture = round(person.posture_score * 100)
            color = (
                (75, 224, 138) if gaze >= 70 else (89, 189, 255) if gaze >= 50 else (119, 101, 255)
            )
            x1, y1, x2, y2 = person.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"mirada {gaze}%  postura {posture}%"
            cv2.rectangle(
                frame, (x1, max(0, y1 - 25)), (min(frame.shape[1], x1 + 245), y1), color, -1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 5, max(16, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (5, 12, 20),
                1,
                cv2.LINE_AA,
            )
        score_text = f"SCAI {calculate_scai(analysis.features, Thresholds())}  personas {analysis.features.presence_count}"
        cv2.rectangle(frame, (8, 8), (310, 43), (8, 17, 31), -1)
        cv2.putText(
            frame,
            score_text,
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (55, 215, 255),
            2,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if ok:
            preview_hub.publish(session_id, encoded.tobytes())

    async def _persist_analysis(
        self, db, session: ClassroomSession, analysis: FrameAnalysis
    ) -> None:
        features = analysis.features
        thresholds = await threshold_store.get()
        score = calculate_scai(features, thresholds)
        now = datetime.now(UTC)
        snapshot = ScaiSnapshot(
            session_id=session.id,
            timestamp=now,
            window_seconds=30,
            scai_score=score,
            avg_gaze_on_task_pct=round(features.gaze_on_task_ratio, 3),
            avg_posture_score=round(features.posture_score_avg, 3),
            presence_count=features.presence_count,
            movement_index=round(features.movement_anomaly_ratio, 3),
            raw_component_weights=weights_dict(thresholds),
        )
        db.add(snapshot)
        if score < thresholds.low_attention_threshold:
            db.add(
                BehaviorEvent(
                    session_id=session.id,
                    event_type="low_group_attention",
                    confidence=round((100 - score) / 100, 2),
                    metadata_json={"collective_only": True},
                )
            )
        if session.mode == "exam" and analysis.phone_detected:
            db.add(
                BehaviorEvent(
                    session_id=session.id,
                    event_type="possible_exam_irregularity",
                    confidence=0.7,
                    metadata_json={"signal": "cell_phone", "human_review_required": True},
                )
            )
        await db.commit()
        await manager.broadcast(
            session.id,
            {
                "type": "scai_update",
                "timestamp": now.isoformat(),
                "scai_score": score,
                "presence_count": features.presence_count,
                "avg_gaze_on_task_pct": snapshot.avg_gaze_on_task_pct,
                "movement_index": snapshot.movement_index,
            },
        )

    async def _mark_error(self, session_id: str, exc: Exception) -> None:
        async with SessionLocal() as db:
            session = await db.get(ClassroomSession, session_id)
            if session:
                session.status = "error"
                await db.commit()
        await manager.broadcast(
            session_id,
            {"type": "session_status", "status": "error", "detail": str(exc)},
        )

    async def _complete_file_session(self, session_id: str) -> None:
        async with SessionLocal() as db:
            session = await db.get(ClassroomSession, session_id)
            if (
                not session
                or session.video_source_type != "file_demo"
                or session.status != "active"
            ):
                return
            scores = (
                await db.scalars(
                    select(ScaiSnapshot.scai_score).where(ScaiSnapshot.session_id == session_id)
                )
            ).all()
            session.status = "completed"
            session.ended_at = datetime.now(UTC)
            session.avg_scai_score = round(sum(scores) / len(scores), 2) if scores else None
            await db.commit()
        await manager.broadcast(session_id, {"type": "session_status", "status": "completed"})


pipeline_registry = PipelineRegistry()
