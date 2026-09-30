from sqlalchemy import inspect

from app.db.models import BehaviorEvent, ClassroomSession, ScaiSnapshot
from app.vision.preview import InMemoryPreviewHub


def test_persistent_schema_has_no_image_or_identity_fields():
    forbidden = {
        "frame",
        "image",
        "video",
        "face_embedding",
        "student_id",
        "name",
        "anonymous_track_id",
    }
    for model in (ClassroomSession, ScaiSnapshot, BehaviorEvent):
        columns = {column.name for column in inspect(model).columns}
        assert not columns.intersection(forbidden)


def test_preview_hub_keeps_only_latest_frame_in_memory():
    hub = InMemoryPreviewHub()
    hub.publish("session", b"first")
    hub.publish("session", b"second")
    assert hub._latest["session"][1] == b"second"
    hub.remove("session")
    assert "session" not in hub._latest
