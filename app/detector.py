from __future__ import annotations

import base64
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLOWorld

from app.database import get_dashboard_counts, get_setting, save_event

logger = logging.getLogger(__name__)
_bbox_diagnostic_lock = Lock()
_last_bbox_diagnostic_at = 0.0

DEFAULT_PROMPTS = [
    "person",
    "bottle",
    "mobile phone",
    "book",
    "backpack",
    "handbag",
    "eyeglasses",
    "pen",
    "ballpoint pen",
    "pencil",
    "charger",
    "adapter",
    "usb cable",
    "computer mouse",
    "keyboard",
    "laptop",
    "monitor",
    "chair",
    "cup",
    "scissors",
    "screwdriver",
    "hammer",
    "electronic component",
]


@dataclass
class SessionStats:
    total_in: int = 0
    total_out: int = 0
    visible_objects: int = 0
    total_detections: int = 0
    line_y_ratio: float = 0.72
    line_y: int = 0
    track_state: dict[int, dict[str, float]] = field(default_factory=dict)
    recent_events: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    recent_activity: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=50))


class DetectionManager:
    def __init__(self) -> None:
        self.sessions: dict[str, SessionStats] = {}
        self.active_session_id: str | None = None
        self.model: YOLOWorld | None = None
        self.model_ready = False
        self.last_error: str | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "yolov8s-worldv2.pt"
        self.default_confidence = 0.25
        self.default_iou = 0.5
        self.default_line_y_ratio = DEFAULT_BROWSER_CONFIG["line_y_ratio"]
        self.default_prompt = DEFAULT_PROMPTS
        self._load_model()

    def _load_model(self) -> None:
        try:
            self.model = YOLOWorld(self.model_name)
            self.model.set_classes(self.default_prompt)
            self.model.to(self.device)
            self.model_ready = True
            self.last_error = None
        except Exception as exc:  # pragma: no cover - runtime dependent
            self.last_error = str(exc)
            self.model_ready = False
            self.model = None

    def get_system_status(self) -> dict[str, Any]:
        gpu_name = "CPU fallback"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
        return {
            "device": self.device,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": gpu_name,
            "model_name": self.model_name,
            "model_ready": self.model_ready,
            "status": "ready" if self.model_ready else "error",
            "message": "YOLO-World is active" if self.model_ready else (self.last_error or "Model unavailable"),
        }

    def get_session(self, session_id: str) -> SessionStats:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionStats()
        return self.sessions[session_id]

    def activate_session(self, session_id: str) -> None:
        self.active_session_id = session_id
        self.get_session(session_id)

    def deactivate_session(self, session_id: str) -> None:
        if self.active_session_id == session_id:
            self.sessions[session_id].visible_objects = 0
            self.active_session_id = None

    def update_settings_from_store(self) -> None:
        confidence = float(get_setting("confidence", str(DEFAULT_BROWSER_CONFIG["confidence"])))
        iou = float(get_setting("iou", str(DEFAULT_BROWSER_CONFIG["iou"])))
        line_y_ratio = float(get_setting("line_y_ratio", str(DEFAULT_BROWSER_CONFIG["line_y_ratio"])))
        prompt_value = get_setting("object_prompts", json.dumps(DEFAULT_PROMPTS))
        try:
            prompt_list = json.loads(prompt_value)
            if isinstance(prompt_list, list) and prompt_list:
                prompts = [str(item).strip() for item in prompt_list if str(item).strip()]
            else:
                prompts = DEFAULT_PROMPTS
        except json.JSONDecodeError:
            prompts = DEFAULT_PROMPTS

        self.apply_settings(confidence, iou, line_y_ratio, prompts)

    def apply_settings(
        self,
        confidence: float,
        iou: float,
        line_y_ratio: float,
        prompts: list[str],
    ) -> None:
        if self.model is not None:
            try:
                self.model.set_classes(prompts)
            except Exception:
                logger.exception("YOLO-World could not apply object prompts")
                raise
        self.default_confidence = confidence
        self.default_iou = iou
        self.default_line_y_ratio = line_y_ratio
        self.default_prompt = prompts

    def _encode_frame(self, frame: np.ndarray) -> str:
        encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    def process_frame(self, session_id: str, image_bytes: bytes, config: dict[str, Any]) -> dict[str, Any]:
        session_stats = self.get_session(session_id)
        if self.model is None or not self.model_ready:
            session_stats.visible_objects = 0
            return {
                "type": "error",
                "message": self.last_error or "YOLO-World model is unavailable.",
                "stats": {
                    **get_dashboard_counts(),
                    "visible_objects": session_stats.visible_objects,
                },
            }

        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"type": "error", "message": "Unable to decode the camera frame."}

        height, width = frame.shape[:2]
        line_y = max(20, min(height - 20, int(height * float(config.get("line_y_ratio", self.default_line_y_ratio)))))
        session_stats.line_y = line_y

        conf_threshold = float(config.get("confidence", self.default_confidence))
        iou_threshold = float(config.get("iou", self.default_iou))

        results = self.model.track(
            frame,
            persist=True,
            conf=conf_threshold,
            iou=iou_threshold,
            tracker="bytetrack.yaml",
            imgsz=640,
            verbose=False,
        )

        global _last_bbox_diagnostic_at
        with _bbox_diagnostic_lock:
            diagnostic_now = time.monotonic()
            log_bbox_diagnostics = diagnostic_now - _last_bbox_diagnostic_at >= 3.0
            if log_bbox_diagnostics:
                _last_bbox_diagnostic_at = diagnostic_now

        diagnostic_result = results[0] if results and len(results) > 0 else None
        if log_bbox_diagnostics:
            result_shape = getattr(diagnostic_result, "orig_shape", None)
            logger.info(
                "Bounding-box diagnostic: original_frame_dimensions=(height=%s, width=%s) result.orig_shape=%s",
                height,
                width,
                result_shape,
            )
            box_data = getattr(diagnostic_result, "boxes", None)
            if box_data is not None:
                for idx, box in enumerate(box_data.xyxy):
                    cls_idx = int(box_data.cls[idx].item())
                    label = self.model.names.get(cls_idx, "Object")
                    confidence = float(box_data.conf[idx].item())
                    x1, y1, x2, y2 = [float(value) for value in box.tolist()]
                    box_width = x2 - x1
                    box_height = y2 - y1
                    area_percentage = (box_width * box_height) / (height * width) * 100
                    tracking_id = idx
                    if hasattr(box_data, "id") and box_data.id is not None and idx < len(box_data.id):
                        tracking_id = int(box_data.id[idx].item())
                    logger.info(
                        "Bounding-box diagnostic object: class=%s confidence=%.4f tracking_id=%s "
                        "coordinates=(x1=%.2f, y1=%.2f, x2=%.2f, y2=%.2f) "
                        "box_width=%.2f box_height=%.2f frame_area_percent=%.2f",
                        label,
                        confidence,
                        tracking_id,
                        x1,
                        y1,
                        x2,
                        y2,
                        box_width,
                        box_height,
                        area_percentage,
                    )

        detections: list[dict[str, Any]] = []
        if results and len(results) > 0:
            result = results[0]
            box_data = getattr(result, "boxes", None)
            if box_data is not None:
                for idx, box in enumerate(box_data.xyxy):
                    cls_idx = int(box_data.cls[idx].item())
                    label = self.model.names.get(cls_idx, "Object")
                    confidence = float(box_data.conf[idx].item())
                    if confidence < conf_threshold:
                        continue

                    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                    tracking_id = idx
                    if hasattr(box_data, "id") and box_data.id is not None and idx < len(box_data.id):
                        tracking_id = int(box_data.id[idx].item())
                    center_y = (y1 + y2) / 2.0
                    state = session_stats.track_state.get(tracking_id)
                    previous_y = state["last_y"] if state else None
                    if previous_y is not None:
                        if previous_y < line_y and center_y >= line_y:
                            session_stats.total_in += 1
                            session_stats.recent_events.append("in")
                            save_event(session_id, label, tracking_id, confidence, (x1, y1, x2, y2), "in")
                        elif previous_y > line_y and center_y <= line_y:
                            session_stats.total_out += 1
                            session_stats.recent_events.append("out")
                            save_event(session_id, label, tracking_id, confidence, (x1, y1, x2, y2), "out")
                    session_stats.track_state[tracking_id] = {"last_y": center_y, "last_seen": time.time()}

                    detections.append({
                        "tracking_id": tracking_id,
                        "label": label,
                        "confidence": round(confidence, 3),
                        "box": [x1, y1, x2, y2],
                        "center_y": center_y,
                    })
                    save_event(session_id, label, tracking_id, confidence, (x1, y1, x2, y2), "detection")

        now = time.time()
        stale_ids = [track_id for track_id, info in session_stats.track_state.items() if now - float(info.get("last_seen", now)) > 4.0]
        for stale_id in stale_ids:
            session_stats.track_state.pop(stale_id, None)

        session_stats.visible_objects = len(detections)
        session_stats.total_detections = len(detections) if session_stats.total_detections < len(detections) else session_stats.total_detections

        annotated = frame.copy()
        cv2.line(annotated, (0, line_y), (width, line_y), (0, 0, 255), 2)
        cv2.putText(annotated, "IN", (18, line_y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(annotated, "OUT", (width - 72, line_y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if log_bbox_diagnostics:
            logger.info(
                "Bounding-box diagnostic: frame_dimensions_immediately_before_drawing=(height=%s, width=%s)",
                annotated.shape[0],
                annotated.shape[1],
            )
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 176, 255), 2)
            label_text = f"{detection['label']} ID:{detection['tracking_id']} {detection['confidence'] * 100:.0f}%"
            cv2.putText(annotated, label_text, (int(x1), max(20, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        session_stats.recent_activity.append({
            "timestamp": time.strftime("%H:%M:%S", time.localtime()),
            "in": session_stats.total_in,
            "out": session_stats.total_out,
            "visible": session_stats.visible_objects,
        })

        return {
            "type": "result",
            "image": self._encode_frame(annotated),
            "stats": {
                **get_dashboard_counts(),
                "visible_objects": session_stats.visible_objects,
            },
            "detections": detections,
            "events": list(session_stats.recent_events),
            "system": self.get_system_status(),
        }

    def get_dashboard_snapshot(self) -> dict[str, Any]:
        activity: list[dict[str, Any]] = []
        for session in self.sessions.values():
            if session.recent_activity:
                activity.extend(list(session.recent_activity))
        active_session = self.sessions.get(self.active_session_id) if self.active_session_id else None
        return {
            "sessions": len(self.sessions),
            "system": self.get_system_status(),
            "activity": activity[-25:],
            "visible_objects": active_session.visible_objects if active_session else 0,
        }


def create_manager() -> DetectionManager:
    manager = DetectionManager()
    manager.update_settings_from_store()
    return manager


DEFAULT_BROWSER_CONFIG = {
    "confidence": 0.25,
    "iou": 0.5,
    "line_y_ratio": 0.72,
}
