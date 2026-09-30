from collections import Counter

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import BehaviorEvent, ClassroomSession, Recommendation, ScaiSnapshot


async def generate_recommendation(db: AsyncSession, session: ClassroomSession) -> Recommendation:
    cached = await db.scalar(
        select(Recommendation)
        .where(Recommendation.session_id == session.id)
        .order_by(Recommendation.generated_at.desc())
    )
    if cached:
        return cached
    snapshots = list(
        (await db.scalars(select(ScaiSnapshot).where(ScaiSnapshot.session_id == session.id))).all()
    )
    events = list(
        (
            await db.scalars(select(BehaviorEvent).where(BehaviorEvent.session_id == session.id))
        ).all()
    )
    average = (
        round(sum(item.scai_score for item in snapshots) / len(snapshots), 1) if snapshots else 0.0
    )
    counts = Counter(item.event_type for item in events)
    if average < 50:
        summary = f"La atención colectiva fue baja (SCAI promedio {average})."
        actions = [
            "Introducir una actividad breve de participación.",
            "Cambiar el ritmo o formato cada 10–15 minutos.",
            "Comprobar comprensión con una pregunta anónima.",
        ]
    elif average < 75:
        summary = f"La atención colectiva fue moderada (SCAI promedio {average})."
        actions = [
            "Reforzar los segmentos con caída de atención.",
            "Alternar explicación con práctica guiada.",
        ]
    else:
        summary = f"La atención colectiva se mantuvo alta (SCAI promedio {average})."
        actions = [
            "Mantener la dinámica usada.",
            "Revisar los eventos aislados sin atribuirlos a personas.",
        ]
    if counts.get("low_group_attention"):
        summary += (
            f" Se observaron {counts['low_group_attention']} alertas colectivas de baja atención."
        )
    # Deterministic fallback deliberately keeps the MVP useful without sending data externally.
    recommendation = Recommendation(
        session_id=session.id,
        summary_text=summary,
        suggested_actions=actions,
        model_used="rules-v1"
        if not settings.openai_api_key
        else "rules-v1 (OpenAI adapter pending)",
    )
    db.add(recommendation)
    await db.commit()
    await db.refresh(recommendation)
    return recommendation
