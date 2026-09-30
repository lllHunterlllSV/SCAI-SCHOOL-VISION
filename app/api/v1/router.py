from collections import Counter
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from app.core.security import create_access_token, current_user, validate_ws_token
from app.db.base import SessionLocal, get_db
from app.db.models import (
    BehaviorEvent,
    ClassroomSession,
    Recommendation,
    ScaiSnapshot,
    TeacherFeedback,
)
from app.db.schemas import (
    EventRead,
    FeedbackCreate,
    FeedbackRead,
    RecommendationRead,
    SessionCreate,
    SessionRead,
    SnapshotRead,
    Thresholds,
)
from app.realtime.connection_manager import manager
from app.services.recommendations import generate_recommendation
from app.services.thresholds import threshold_store
from app.vision.pipeline import pipeline_registry
from app.vision.preview import preview_hub

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/auth/login")
async def login(payload: LoginRequest) -> dict[str, str | int]:
    # MVP identity provider seam. Replace with institutional OIDC before production.
    if not payload.username or not payload.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "access_token": create_access_token(payload.username),
        "token_type": "bearer",
        "expires_in": 3600,
    }


async def owned_session(
    session_id: str, db: AsyncSession, user: dict[str, str]
) -> ClassroomSession:
    session = await db.get(ClassroomSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if user["role"] != "admin" and session.owner_id != user["sub"]:
        raise HTTPException(status_code=403, detail="Not allowed")
    return session


@router.post("/sessions", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionCreate,
    db: AsyncSession = Depends(get_db),
    user: dict[str, str] = Depends(current_user),
) -> ClassroomSession:
    if not payload.consent_confirmed:
        raise HTTPException(status_code=422, detail="Explicit consent confirmation is required")
    session = ClassroomSession(**payload.model_dump(), owner_id=user["sub"])
    db.add(session)
    await db.commit()
    await db.refresh(session)
    pipeline_registry.start(session.id)
    return session


@router.get("/sessions", response_model=list[SessionRead])
async def list_sessions(
    class_id: str | None = None,
    db: AsyncSession = Depends(get_db),
    user: dict[str, str] = Depends(current_user),
) -> list[ClassroomSession]:
    query = select(ClassroomSession).order_by(ClassroomSession.started_at.desc())
    if user["role"] != "admin":
        query = query.where(ClassroomSession.owner_id == user["sub"])
    if class_id:
        query = query.where(ClassroomSession.class_id == class_id)
    return list((await db.scalars(query)).all())


@router.get("/sessions/{session_id}", response_model=SessionRead)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict[str, str] = Depends(current_user),
) -> ClassroomSession:
    return await owned_session(session_id, db, user)


@router.patch("/sessions/{session_id}/stop", response_model=SessionRead)
async def stop_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict[str, str] = Depends(current_user),
) -> ClassroomSession:
    session = await owned_session(session_id, db, user)
    if session.status != "active":
        return session
    await pipeline_registry.stop(session_id)
    scores = (
        await db.scalars(
            select(ScaiSnapshot.scai_score).where(ScaiSnapshot.session_id == session_id)
        )
    ).all()
    session.status = "completed"
    session.ended_at = datetime.now(UTC)
    session.avg_scai_score = round(sum(scores) / len(scores), 2) if scores else None
    await db.commit()
    await db.refresh(session)
    await generate_recommendation(db, session)
    await manager.broadcast(session_id, {"type": "session_status", "status": "completed"})
    return session


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict[str, str] = Depends(current_user),
) -> None:
    session = await owned_session(session_id, db, user)
    await pipeline_registry.stop(session_id)
    await db.delete(session)
    await db.commit()


@router.get("/sessions/{session_id}/scai", response_model=list[SnapshotRead])
async def session_scai(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict[str, str] = Depends(current_user),
) -> list[ScaiSnapshot]:
    await owned_session(session_id, db, user)
    return list(
        (
            await db.scalars(
                select(ScaiSnapshot)
                .where(ScaiSnapshot.session_id == session_id)
                .order_by(ScaiSnapshot.timestamp)
            )
        ).all()
    )


@router.get("/sessions/{session_id}/events", response_model=list[EventRead])
async def session_events(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict[str, str] = Depends(current_user),
) -> list[BehaviorEvent]:
    await owned_session(session_id, db, user)
    return list(
        (
            await db.scalars(
                select(BehaviorEvent)
                .where(BehaviorEvent.session_id == session_id)
                .order_by(BehaviorEvent.timestamp)
            )
        ).all()
    )


@router.get("/sessions/{session_id}/report")
async def session_report(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict[str, str] = Depends(current_user),
) -> dict:
    session = await owned_session(session_id, db, user)
    snapshots = list(
        (await db.scalars(select(ScaiSnapshot).where(ScaiSnapshot.session_id == session_id))).all()
    )
    events = list(
        (
            await db.scalars(select(BehaviorEvent).where(BehaviorEvent.session_id == session_id))
        ).all()
    )
    values = [item.scai_score for item in snapshots]
    return {
        "session": SessionRead.model_validate(session),
        "summary": {
            "samples": len(values),
            "average": round(sum(values) / len(values), 2) if values else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "event_counts": Counter(item.event_type for item in events),
        },
        "disclaimer": "Las alertas son señales probabilísticas para revisión humana, no evidencia concluyente ni base para sanciones automáticas.",
    }


@router.post("/sessions/{session_id}/feedback", response_model=FeedbackRead, status_code=201)
async def add_feedback(
    session_id: str,
    payload: FeedbackCreate,
    db: AsyncSession = Depends(get_db),
    user: dict[str, str] = Depends(current_user),
) -> TeacherFeedback:
    await owned_session(session_id, db, user)
    feedback = TeacherFeedback(
        session_id=session_id, teacher_id=user["sub"], **payload.model_dump()
    )
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)
    return feedback


@router.post("/sessions/{session_id}/recommendations/generate", response_model=RecommendationRead)
async def create_recommendation(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict[str, str] = Depends(current_user),
) -> Recommendation:
    session = await owned_session(session_id, db, user)
    return await generate_recommendation(db, session)


@router.get("/sessions/{session_id}/recommendations", response_model=list[RecommendationRead])
async def get_recommendations(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict[str, str] = Depends(current_user),
) -> list[Recommendation]:
    await owned_session(session_id, db, user)
    return list(
        (
            await db.scalars(
                select(Recommendation)
                .where(Recommendation.session_id == session_id)
                .order_by(Recommendation.generated_at.desc())
            )
        ).all()
    )


@router.get("/classes/{class_id}/trends")
async def class_trends(
    class_id: str, db: AsyncSession = Depends(get_db), user: dict[str, str] = Depends(current_user)
) -> dict:
    query = (
        select(ClassroomSession)
        .where(ClassroomSession.class_id == class_id, ClassroomSession.status == "completed")
        .order_by(ClassroomSession.started_at)
    )
    if user["role"] != "admin":
        query = query.where(ClassroomSession.owner_id == user["sub"])
    sessions = list((await db.scalars(query)).all())
    return {
        "class_id": class_id,
        "sessions": [
            {
                "session_id": item.id,
                "started_at": item.started_at,
                "avg_scai_score": item.avg_scai_score,
            }
            for item in sessions
        ],
    }


@router.get("/config/thresholds", response_model=Thresholds)
async def get_thresholds(_: dict[str, str] = Depends(current_user)) -> Thresholds:
    return await threshold_store.get()


@router.put("/config/thresholds", response_model=Thresholds)
async def put_thresholds(
    payload: Thresholds, _: dict[str, str] = Depends(current_user)
) -> Thresholds:
    return await threshold_store.set(payload)


@router.websocket("/ws/sessions/{session_id}")
async def session_websocket(
    websocket: WebSocket, session_id: str, token: str | None = Query(default=None)
) -> None:
    if not validate_ws_token(token):
        await websocket.close(code=4401)
        return
    async with SessionLocal() as db:
        if not await db.get(ClassroomSession, session_id):
            await websocket.close(code=4404)
            return
    await manager.connect(session_id, websocket)
    try:
        while True:
            message = await websocket.receive_json()
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.get("type") == "request_snapshot":
                async with SessionLocal() as db:
                    latest = await db.scalar(
                        select(ScaiSnapshot)
                        .where(ScaiSnapshot.session_id == session_id)
                        .order_by(ScaiSnapshot.timestamp.desc())
                    )
                    await websocket.send_json(
                        {
                            "type": "snapshot",
                            "data": SnapshotRead.model_validate(latest).model_dump(mode="json")
                            if latest
                            else None,
                        }
                    )
    except WebSocketDisconnect:
        await manager.disconnect(session_id, websocket)


@router.get("/sessions/{session_id}/preview", include_in_schema=False)
async def session_preview(
    session_id: str,
    token: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    if not validate_ws_token(token):
        raise HTTPException(status_code=401, detail="Invalid preview token")
    if not await db.get(ClassroomSession, session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return StreamingResponse(
        preview_hub.stream(session_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )
