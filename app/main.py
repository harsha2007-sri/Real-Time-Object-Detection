from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketState

from app.database import get_analytics_data, get_dashboard_stats, get_recent_events, get_setting, init_db, save_settings
from app.detector import DEFAULT_BROWSER_CONFIG, DEFAULT_PROMPTS, create_manager

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="Real Object Detection System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
app.state.detector = create_manager()

BASE_DIR = __import__("pathlib").Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_app() -> FileResponse:
    return FileResponse(TEMPLATE_DIR / "index.html")


@app.get("/api/system")
def get_system_status() -> dict[str, Any]:
    return app.state.detector.get_system_status()


@app.get("/api/settings")
def read_settings() -> dict[str, Any]:
    return {
        "confidence": float(get_setting("confidence", str(DEFAULT_BROWSER_CONFIG["confidence"]))),
        "iou": float(get_setting("iou", str(DEFAULT_BROWSER_CONFIG["iou"]))),
        "line_y_ratio": float(get_setting("line_y_ratio", str(DEFAULT_BROWSER_CONFIG["line_y_ratio"]))),
        "object_prompts": json.loads(get_setting("object_prompts", json.dumps(DEFAULT_PROMPTS))),
    }


def _validated_setting_number(payload: dict[str, Any], key: str, minimum: float, maximum: float) -> float:
    value = payload.get(key, DEFAULT_BROWSER_CONFIG[key])
    if isinstance(value, bool):
        raise HTTPException(status_code=422, detail=f"{key} must be a number between {minimum} and {maximum}.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"{key} must be a number between {minimum} and {maximum}.") from None
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise HTTPException(status_code=422, detail=f"{key} must be a number between {minimum} and {maximum}.")
    return number


@app.post("/api/settings")
def save_configuration(payload: dict[str, Any]) -> dict[str, Any]:
    confidence = _validated_setting_number(payload, "confidence", 0.05, 0.95)
    iou = _validated_setting_number(payload, "iou", 0.1, 0.95)
    line_y_ratio = _validated_setting_number(payload, "line_y_ratio", 0.1, 0.9)
    prompts = payload.get("object_prompts", DEFAULT_PROMPTS)
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
    ):
        raise HTTPException(status_code=422, detail="object_prompts must be a non-empty list of non-empty strings.")
    prompts = list(dict.fromkeys(prompt.strip() for prompt in prompts))

    try:
        save_settings({
            "confidence": confidence,
            "iou": iou,
            "line_y_ratio": line_y_ratio,
            "object_prompts": json.dumps(prompts),
        })
        app.state.detector.apply_settings(confidence, iou, line_y_ratio, prompts)
    except Exception as exc:
        logger.exception("Failed to apply or save detection settings")
        raise HTTPException(status_code=503, detail="Settings could not be applied or saved.") from exc

    return {
        "confidence": confidence,
        "iou": iou,
        "line_y_ratio": line_y_ratio,
        "object_prompts": prompts,
    }


@app.get("/api/dashboard")
def read_dashboard() -> dict[str, Any]:
    try:
        stats = get_dashboard_stats()
        snapshot = app.state.detector.get_dashboard_snapshot()
        stats["visible_objects"] = snapshot["visible_objects"]
        return {
            "stats": stats,
            "system": snapshot["system"],
            "activity": snapshot["activity"],
        }
    except Exception as exc:
        logger.exception("Failed to load dashboard statistics")
        raise HTTPException(status_code=503, detail="Dashboard statistics are temporarily unavailable.") from exc


@app.get("/api/analytics")
def read_analytics() -> dict[str, Any]:
    try:
        return get_analytics_data()
    except Exception as exc:
        logger.exception("Failed to load analytics data")
        raise HTTPException(status_code=503, detail="Analytics data is temporarily unavailable.") from exc


@app.get("/api/history")
def read_history() -> list[dict[str, Any]]:
    return get_recent_events(limit=100)


@app.websocket("/ws")
async def detection_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(uuid.uuid4())
    detector = app.state.detector
    detector.activate_session(session_id)
    received_frames = 0
    logger.info("Detection WebSocket connected: session=%s", session_id)

    try:
        while True:
            payload = await websocket.receive_text()
            try:
                message = json.loads(payload)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON message."})
                continue
            if not isinstance(message, dict):
                await websocket.send_json({"type": "error", "message": "WebSocket message must be a JSON object."})
                continue
            kind = message.get("type")

            if kind == "init":
                config = message.get("config", DEFAULT_BROWSER_CONFIG)
                detector.update_settings_from_store()
                await websocket.send_json({"type": "ready", "session_id": session_id, "config": config})
                continue

            if kind == "frame":
                image_data = message.get("image")
                if not isinstance(image_data, str):
                    await websocket.send_json({"type": "error", "message": "Camera frame is missing image data."})
                    continue
                config = message.get("config", DEFAULT_BROWSER_CONFIG)
                if image_data.startswith("data:image"):
                    parts = image_data.split(",", 1)
                    if len(parts) != 2:
                        await websocket.send_json({"type": "error", "message": "Camera frame has an invalid data URL."})
                        continue
                    image_data = parts[1]
                try:
                    frame_bytes = base64.b64decode(image_data, validate=True)
                except (binascii.Error, ValueError):
                    await websocket.send_json({"type": "error", "message": "Camera frame is not valid base64 image data."})
                    continue
                if not frame_bytes:
                    await websocket.send_json({"type": "error", "message": "Camera frame is empty."})
                    continue

                received_frames += 1
                frame_id = message.get("frame_id", received_frames)
                started_at = time.perf_counter()
                try:
                    result = await run_in_threadpool(detector.process_frame, session_id, frame_bytes, config)
                except Exception:
                    logger.exception(
                        "Frame processing failed: session=%s frame=%s bytes=%s",
                        session_id,
                        frame_id,
                        len(frame_bytes),
                    )
                    await websocket.send_json({
                        "type": "error",
                        "frame_id": frame_id,
                        "message": "The backend failed while processing this camera frame. Check the server log for details.",
                    })
                    continue

                result["frame_id"] = frame_id
                result["frame_size_bytes"] = len(frame_bytes)
                result["inference_ms"] = round((time.perf_counter() - started_at) * 1000, 1)
                if received_frames == 1 or received_frames % 25 == 0:
                    logger.info(
                        "Received camera frame: session=%s frame=%s bytes=%s inference_ms=%s result=%s",
                        session_id,
                        frame_id,
                        len(frame_bytes),
                        result["inference_ms"],
                        result.get("type"),
                    )
                await websocket.send_json(result)
                continue

            if kind == "ping":
                await websocket.send_json({"type": "pong", "session_id": session_id})
    except WebSocketDisconnect:
        logger.info("Detection WebSocket disconnected: session=%s frames=%s", session_id, received_frames)
    except Exception:
        logger.exception("Detection WebSocket failed: session=%s frames=%s", session_id, received_frames)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason="Internal server error")
    finally:
        detector.deactivate_session(session_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
