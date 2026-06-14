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
