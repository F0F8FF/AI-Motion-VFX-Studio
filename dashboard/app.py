"""
FastAPI 실시간 대시보드.
WebSocket으로 AI 파이프라인 상태를 브라우저에 스트리밍한다.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from main import MotionVFXPipeline


def create_app(pipeline: MotionVFXPipeline) -> FastAPI:
    app = FastAPI(title="AI Motion VFX Studio")

    static_dir = Path(__file__).parent.parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = static_dir / "index.html"
        return html_path.read_text(encoding="utf-8")

    @app.get("/api/status")
    async def status():
        return pipeline.latest_state

    @app.get("/api/config")
    async def get_config():
        cfg = pipeline.config
        return {
            "osc": {"host": cfg.osc.host, "port": cfg.osc.port, "enabled": cfg.osc.enabled},
            "ai": {
                "pose": cfg.ai.pose_enabled,
                "hand": cfg.ai.hand_enabled,
                "face": cfg.ai.face_enabled,
                "gesture": cfg.ai.gesture_enabled,
                "emotion": cfg.ai.emotion_enabled,
            },
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                state = pipeline.latest_state
                await websocket.send_text(json.dumps(state))
                await asyncio.sleep(0.05)  # 20 FPS로 상태 전송
        except WebSocketDisconnect:
            pass

    return app
