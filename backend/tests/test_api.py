from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_upload_endpoint():
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
