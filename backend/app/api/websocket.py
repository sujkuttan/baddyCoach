from fastapi import WebSocket, WebSocketDisconnect
from typing import Any


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, job_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

    def disconnect(self, job_id: str, websocket: WebSocket) -> None:
        if job_id in self.active_connections:
            self.active_connections[job_id] = [
                ws for ws in self.active_connections[job_id] if ws != websocket
            ]

    async def broadcast(self, job_id: str, message: dict[str, Any]) -> None:
        if job_id in self.active_connections:
            import json
            for ws in self.active_connections[job_id]:
                await ws.send_text(json.dumps(message))


ws_manager = ConnectionManager()
