from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.database as database
from app.database import Base, save_event
from app.detector import DetectionManager
from app.main import app


@pytest.fixture
def isolated_database(monkeypatch: pytest.MonkeyPatch) -> None:
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    test_session = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(database, "SessionLocal", test_session)
    yield
    test_engine.dispose()


def add_event(event_type: str, object_name: str = "person") -> None:
    save_event("test-session", object_name, 7, 0.9, (1, 2, 30, 40), event_type)


def test_dashboard_api_uses_persisted_event_type_counts(
    isolated_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_event("detection")
    add_event("detection", "bottle")
    add_event("in")
    add_event("out")
    add_event("out")

    class DashboardDetector:
        def get_dashboard_snapshot(self) -> dict[str, object]:
            return {
                "visible_objects": 3,
                "system": {"model_ready": True},
                "activity": [],
            }

    monkeypatch.setattr(app.state, "detector", DashboardDetector(), raising=False)

    with TestClient(app) as client:
        response = client.get("/api/dashboard")

    assert response.status_code == 200
    assert response.json()["stats"] == {
        "total_detections": 2,
        "total_in": 1,
        "total_out": 2,
        "object_distribution": {"person": 4, "bottle": 1},
        "visible_objects": 3,
    }


def test_live_frame_stats_use_persisted_counts_and_latest_frame_visibility(
    isolated_database: None,
) -> None:
    add_event("detection")
    add_event("in")
    add_event("out")

    class FakeBoxes:
        def __init__(self, y1: int | None) -> None:
            boxes = [[10, y1, 40, y1 + 20]] if y1 is not None else []
            self.xyxy = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            self.cls = torch.tensor([0] if y1 is not None else [])
            self.conf = torch.tensor([0.9] if y1 is not None else [])
            self.id = torch.tensor([7] if y1 is not None else [])

    class FakeModel:
        names = {0: "person"}

        def __init__(self) -> None:
            self.y1: int | None = 10

        def track(self, *args: object, **kwargs: object) -> list[SimpleNamespace]:
            return [SimpleNamespace(boxes=FakeBoxes(self.y1))]

    manager = DetectionManager.__new__(DetectionManager)
    manager.sessions = {}
    manager.active_session_id = "live-session"
    fake_model = FakeModel()
    manager.model = fake_model
    manager.model_ready = True
    manager.last_error = None
    manager.device = "cpu"
    manager.model_name = "test-model"
    manager.default_confidence = 0.25
    manager.default_iou = 0.5
    manager.default_line_y_ratio = 0.72
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()
    config = {"confidence": 0.25, "iou": 0.5, "line_y_ratio": 0.5}

    first = manager.process_frame("live-session", frame_bytes, config)
    assert first["stats"] == {
        "total_detections": 2,
        "total_in": 1,
        "total_out": 1,
        "visible_objects": 1,
    }

    fake_model.y1 = 60
    second = manager.process_frame("live-session", frame_bytes, config)
    assert second["stats"] == {
        "total_detections": 3,
        "total_in": 2,
        "total_out": 1,
        "visible_objects": 1,
    }

    fake_model.y1 = None
    third = manager.process_frame("live-session", frame_bytes, config)
    assert third["stats"] == {
        "total_detections": 3,
        "total_in": 2,
        "total_out": 1,
        "visible_objects": 0,
    }


def test_visible_count_is_cleared_when_active_session_ends() -> None:
    manager = DetectionManager.__new__(DetectionManager)
    manager.sessions = {}
    manager.active_session_id = None
    manager.get_system_status = lambda: {}

    manager.activate_session("live-session")
    manager.sessions["live-session"].visible_objects = 4
    assert manager.get_dashboard_snapshot()["visible_objects"] == 4

    manager.deactivate_session("live-session")
    assert manager.get_dashboard_snapshot()["visible_objects"] == 0


def test_dashboard_api_reports_database_failure(
    isolated_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DashboardDetector:
        def get_dashboard_snapshot(self) -> dict[str, object]:
            return {"visible_objects": 0, "system": {}, "activity": []}

    monkeypatch.setattr(app.state, "detector", DashboardDetector(), raising=False)
    monkeypatch.setattr(database, "SessionLocal", lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")))

    with TestClient(app) as client:
        response = client.get("/api/dashboard")

    assert response.status_code == 503
    assert response.json()["detail"] == "Dashboard statistics are temporarily unavailable."
