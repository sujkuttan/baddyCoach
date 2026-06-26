from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_get_nonexistent_job_returns_404():
    response = client.get("/api/jobs/nonexistent")
    assert response.status_code == 404


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_shuttle_coach_endpoint():
    """Test shuttle-coach analysis endpoint."""
    response = client.get("/api/shuttle-coach/analyze/test_job")
    assert response.status_code == 404


FAKE_ANALYZE_RESULT = {
    "player_ids": ["P1", "P2"],
    "capabilities": ["attack", "defense"],
    "metrics": [],
    "findings": [],
    "report_md": "# Report",
    "report_json": {"summary": "test"},
}


def test_shuttle_coach_analyze_happy_path():
    upload = client.post(
        "/api/upload",
        files={"file": ("match.mp4", b"fake-video-content", "video/mp4")},
    )
    assert upload.status_code == 200
    job_id = upload.json()["job_id"]

    with patch("app.shuttle_coach.engine.analyze", return_value=FAKE_ANALYZE_RESULT):
        response = client.get(f"/api/shuttle-coach/analyze/{job_id}")

    assert response.status_code == 200
    data = response.json()
    for key in FAKE_ANALYZE_RESULT:
        assert key in data, f"Missing key: {key}"
    assert data["player_ids"] == ["P1", "P2"]
    assert data["capabilities"] == ["attack", "defense"]
