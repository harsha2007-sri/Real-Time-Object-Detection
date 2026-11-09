from __future__ import annotations

import base64
import json
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
import app.main as main
from app.database import Base, DetectionEvent, SystemSetting
from app.detector import DEFAULT_PROMPTS, DetectionManager, create_manager


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


class FakeBoxes:
    def __init__(self, y1: int) -> None:
        self.xyxy = torch.tensor([[10, y1, 40, y1 + 20]], dtype=torch.float32)
        self.cls = torch.tensor([0])
        self.conf = torch.tensor([0.9])
        self.id = torch.tensor([7])


class FakeModel:
    names = {0: "person"}

    def __init__(self) -> None:
        self.classes: list[str] = []
        self.track_calls: list[dict[str, object]] = []
        self.box_y1 = 20

    def set_classes(self, prompts: list[str]) -> None:
        self.classes = list(prompts)

    def track(self, frame: np.ndarray, **kwargs: object) -> list[SimpleNamespace]:
        self.track_calls.append(kwargs)
        return [SimpleNamespace(boxes=FakeBoxes(self.box_y1))]


def new_manager(model: FakeModel) -> DetectionManager:
    manager = DetectionManager.__new__(DetectionManager)
    manager.sessions = {}
    manager.active_session_id = None
    manager.model = model
    manager.model_ready = True
    manager.last_error = None
    manager.device = "cpu"
    manager.model_name = "test-model"
    manager.default_confidence = 0.25
    manager.default_iou = 0.5
    manager.default_line_y_ratio = 0.72
    manager.default_prompt = DEFAULT_PROMPTS
    return manager


def test_settings_api_persists_all_values_and_reloads_after_restart(
    isolated_database: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel()
    manager = new_manager(model)
    monkeypatch.setattr(main.app.state, "detector", manager, raising=False)
    settings = {
        "confidence": 0.61,
        "iou": 0.37,
        "line_y_ratio": 0.4,
        "object_prompts": [" person ", "bicycle", "bicycle"],
    }

    with TestClient(main.app) as client:
        response = client.post("/api/settings", json=settings)
        assert response.status_code == 200
        assert response.json() == {
            "confidence": 0.61,
            "iou": 0.37,
            "line_y_ratio": 0.4,
            "object_prompts": ["person", "bicycle"],
        }
        assert client.get("/api/settings").json() == response.json()

    assert manager.default_confidence == 0.61
    assert manager.default_iou == 0.37
    assert manager.default_line_y_ratio == 0.4
    assert model.classes == ["person", "bicycle"]

    with database.SessionLocal() as session:
        saved = {row.key: row.value for row in session.query(SystemSetting).all()}
    assert saved == {
        "confidence": "0.61",
        "iou": "0.37",
        "line_y_ratio": "0.4",
        "object_prompts": '["person", "bicycle"]',
    }

    restarted_model = FakeModel()
    restarted_manager = new_manager(restarted_model)
    restarted_manager.update_settings_from_store()
    assert restarted_manager.default_confidence == 0.61
    assert restarted_manager.default_iou == 0.37
    assert restarted_manager.default_line_y_ratio == 0.4
    assert restarted_model.classes == ["person", "bicycle"]


def test_persisted_settings_configure_yolo_thresholds_and_counting_line(
    isolated_database: None,
) -> None:
    model = FakeModel()
    manager = new_manager(model)
    database.save_settings({
        "confidence": 0.61,
        "iou": 0.37,
        "line_y_ratio": 0.4,
        "object_prompts": json.dumps(["person", "bicycle"]),
    })
    manager.update_settings_from_store()
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()

    first = manager.process_frame("settings-test", frame_bytes, {})
    assert first["type"] == "result"
    assert manager.sessions["settings-test"].line_y == 48
    assert model.track_calls[-1]["conf"] == 0.61
    assert model.track_calls[-1]["iou"] == 0.37
    assert model.classes == ["person", "bicycle"]

    model.box_y1 = 50
    second = manager.process_frame("settings-test", frame_bytes, {})
    assert second["stats"]["total_in"] == 1

    annotated = cv2.imdecode(
        np.frombuffer(
            base64.b64decode(second["image"]),
            dtype=np.uint8,
        ),
        cv2.IMREAD_COLOR,
    )
    assert annotated[48, 0, 2] > 200
    assert annotated[48, 0, 0] < 30
    assert annotated[48, 0, 1] < 30


def test_backend_startup_loads_persisted_settings_into_model(
    isolated_database: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database.save_settings({
        "confidence": 0.63,
        "iou": 0.41,
        "line_y_ratio": 0.35,
        "object_prompts": json.dumps(["traffic cone", "person"]),
    })
    model = FakeModel()

    class StartupManager(DetectionManager):
        def __init__(self) -> None:
            super().__init__()

        def _load_model(self) -> None:
            self.model = model
            self.model_ready = True

    monkeypatch.setattr("app.detector.DetectionManager", StartupManager)

    manager = create_manager()

    assert manager.default_confidence == 0.63
    assert manager.default_iou == 0.41
    assert manager.default_line_y_ratio == 0.35
    assert manager.default_prompt == ["traffic cone", "person"]
    assert model.classes == ["traffic cone", "person"]


@pytest.mark.parametrize(
    "override",
    [
        {"confidence": 1.1},
        {"iou": -0.1},
        {"line_y_ratio": "NaN"},
        {"object_prompts": []},
        {"object_prompts": ["person", " "]},
        {"object_prompts": "person, bicycle"},
    ],
)
def test_settings_api_rejects_invalid_values_without_persisting(
    isolated_database: None,
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, object],
) -> None:
    model = FakeModel()
    manager = new_manager(model)
    monkeypatch.setattr(main.app.state, "detector", manager, raising=False)
    settings: dict[str, object] = {
        "confidence": 0.5,
        "iou": 0.5,
        "line_y_ratio": 0.5,
        "object_prompts": ["person"],
    }
    settings.update(override)

    with TestClient(main.app) as client:
        response = client.post("/api/settings", json=settings)

    assert response.status_code == 422
    assert model.classes == []
    with database.SessionLocal() as session:
        assert session.query(SystemSetting).count() == 0


def test_settings_api_reports_model_prompt_application_failure(
    isolated_database: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedModel(FakeModel):
        def set_classes(self, prompts: list[str]) -> None:
            raise RuntimeError("model rejected prompts")

    manager = new_manager(FailedModel())
    monkeypatch.setattr(main.app.state, "detector", manager, raising=False)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/settings",
            json={
                "confidence": 0.5,
                "iou": 0.5,
                "line_y_ratio": 0.5,
                "object_prompts": ["person"],
            },
        )

    assert response.status_code == 503
    with database.SessionLocal() as session:
        assert session.query(SystemSetting).count() == 4
        assert session.get(SystemSetting, "object_prompts").value == '["person"]'


def test_settings_api_uses_detector_prompt_defaults(
    isolated_database: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main.app.state, "detector", new_manager(FakeModel()), raising=False)

    with TestClient(main.app) as client:
        settings = client.get("/api/settings").json()

    assert settings["object_prompts"] == DEFAULT_PROMPTS
