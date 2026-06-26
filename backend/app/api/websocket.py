import asyncio
import json
from fastapi import WebSocket, WebSocketDisconnect
from typing import Any


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}
        self._main_loop: asyncio.AbstractEventLoop | None = None

    def set_main_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._main_loop = loop

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

    async def _send(self, job_id: str, message: str) -> None:
        if job_id in self.active_connections:
            dead = []
            for ws in self.active_connections[job_id]:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.active_connections[job_id].remove(ws)

    def broadcast_sync(self, job_id: str, message: dict[str, Any]) -> None:
        """Thread-safe broadcast from background tasks."""
        if not self._main_loop or self._main_loop.is_closed():
            return
        msg = json.dumps(message)
        asyncio.run_coroutine_threadsafe(self._send(job_id, msg), self._main_loop)

    async def broadcast(self, job_id: str, message: dict[str, Any]) -> None:
        if job_id in self.active_connections:
            msg = json.dumps(message)
            await self._send(job_id, msg)


ws_manager = ConnectionManager()
