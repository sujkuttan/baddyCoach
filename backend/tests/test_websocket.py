from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_websocket_connect_and_ping():
    with client.websocket_connect("/api/jobs/test123/progress") as ws:
        ws.send_text("ping")
        data = ws.receive_text()
        assert data == '{"type": "pong"}'
