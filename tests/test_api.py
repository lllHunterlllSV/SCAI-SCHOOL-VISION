import asyncio


async def test_full_session_lifecycle(client):
    rejected = await client.post(
        "/api/v1/sessions",
        json={
            "class_id": "math",
            "mode": "lecture",
            "video_source_type": "file_demo",
            "source_config": {"synthetic": True},
            "consent_confirmed": False,
        },
    )
    assert rejected.status_code == 422

    created = await client.post(
        "/api/v1/sessions",
        json={
            "class_id": "math",
            "mode": "lecture",
            "video_source_type": "file_demo",
            "source_config": {"synthetic": True},
            "consent_confirmed": True,
        },
    )
    assert created.status_code == 201
    session_id = created.json()["id"]
    await asyncio.sleep(0.08)

    stopped = await client.patch(f"/api/v1/sessions/{session_id}/stop")
    assert stopped.status_code == 200
    assert stopped.json()["status"] == "completed"

    metrics = await client.get(f"/api/v1/sessions/{session_id}/scai")
    assert metrics.status_code == 200
    assert metrics.json()
    assert all(0 <= row["scai_score"] <= 100 for row in metrics.json())

    report = await client.get(f"/api/v1/sessions/{session_id}/report")
    assert report.status_code == 200
    assert report.json()["summary"]["samples"] >= 1

    recommendations = await client.get(f"/api/v1/sessions/{session_id}/recommendations")
    assert recommendations.status_code == 200
    assert len(recommendations.json()) == 1

    removed = await client.delete(f"/api/v1/sessions/{session_id}")
    assert removed.status_code == 204


async def test_weights_must_sum_to_one(client):
    response = await client.put(
        "/api/v1/config/thresholds",
        json={
            "gaze_weight": 0.8,
            "posture_weight": 0.25,
            "movement_weight": 0.2,
            "presence_weight": 0.15,
            "low_attention_threshold": 50,
        },
    )
    assert response.status_code == 422


async def test_monitor_page_is_available(client):
    response = await client.get("/monitor")
    assert response.status_code == 200
    assert "Monitor de atención" in response.text
