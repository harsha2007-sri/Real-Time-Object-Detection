from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.database as database
import app.main as main
from app.database import Base, DetectionEvent


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


def add_event(
    event_type: str,
    object_name: str,
    timestamp: datetime,
    tracking_id: int | None = 1,
    confidence: float = 0.8,
) -> None:
    with database.SessionLocal() as session:
        session.add(
            DetectionEvent(
                timestamp=timestamp,
                session_id="analytics-test",
                object_name=object_name,
                tracking_id=tracking_id,
                confidence=confidence,
                x1=1,
                y1=2,
                x2=30,
                y2=40,
                event_type=event_type,
            )
        )
        session.commit()


def test_analytics_aggregates_real_detection_and_crossing_rows(
    isolated_database: None,
) -> None:
    add_event("detection", "person", datetime(2025, 1, 2, 9, 1))
    add_event("detection", "person", datetime(2025, 1, 2, 9, 59))
    add_event("detection", "bottle", datetime(2025, 1, 2, 10, 0))
    add_event("in", "person", datetime(2025, 1, 2, 9, 30))
    add_event("out", "bottle", datetime(2025, 1, 2, 11, 1))

    data = database.get_analytics_data()

    assert data["entry_exit"] == {"total_in": 1, "total_out": 1}
    assert data["detection_activity"] == [
        {"timestamp": "2025-01-02T09:00:00Z", "count": 2},
        {"timestamp": "2025-01-02T10:00:00Z", "count": 1},
    ]
    assert data["traffic_flow"] == [
        {"timestamp": "2025-01-02T09:00:00Z", "total_in": 1, "total_out": 0},
        {"timestamp": "2025-01-02T11:00:00Z", "total_in": 0, "total_out": 1},
    ]
    assert data["object_distribution"] == [
        {"object_name": "bottle", "count": 1},
        {"object_name": "person", "count": 2},
    ]


def test_recent_events_are_newest_first_and_preserve_optional_values(
    isolated_database: None,
) -> None:
    add_event("detection", "older", datetime(2025, 1, 2, 9, 0), tracking_id=4)
    add_event("out", "newer", datetime(2025, 1, 2, 10, 0), tracking_id=None, confidence=0.6)

    events = database.get_analytics_data()["recent_events"]

    assert events == [
        {
            "timestamp": "2025-01-02T10:00:00.000000Z",
            "object_name": "newer",
            "event_type": "out",
            "confidence": 0.6,
            "tracking_id": None,
        },
        {
            "timestamp": "2025-01-02T09:00:00.000000Z",
            "object_name": "older",
            "event_type": "detection",
            "confidence": 0.8,
            "tracking_id": 4,
        },
    ]


def test_analytics_handles_empty_database(isolated_database: None) -> None:
    assert database.get_analytics_data() == {
        "entry_exit": {"total_in": 0, "total_out": 0},
        "detection_activity": [],
        "traffic_flow": [],
        "object_distribution": [],
        "recent_events": [],
    }

    with TestClient(main.app) as client:
        response = client.get("/api/analytics")

    assert response.status_code == 200
    assert response.json()["detection_activity"] == []
    assert response.json()["traffic_flow"] == []
    assert response.json()["object_distribution"] == []
    assert response.json()["recent_events"] == []


def test_analytics_api_returns_json_aggregates(
    isolated_database: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_event("detection", "person", datetime(2025, 1, 2, 9, 1))
    add_event("in", "person", datetime(2025, 1, 2, 9, 30))

    class DashboardDetector:
        def get_dashboard_snapshot(self) -> dict[str, object]:
            return {"visible_objects": 0, "system": {}, "activity": []}

    monkeypatch.setattr(main.app.state, "detector", DashboardDetector(), raising=False)

    with TestClient(main.app) as client:
        response = client.get("/api/analytics")
        dashboard_response = client.get("/api/dashboard")

    assert response.status_code == 200
    assert dashboard_response.status_code == 200
    payload = response.json()
    assert payload["entry_exit"] == {"total_in": 1, "total_out": 0}
    assert dashboard_response.json()["stats"]["total_in"] == payload["entry_exit"]["total_in"]
    assert dashboard_response.json()["stats"]["total_out"] == payload["entry_exit"]["total_out"]
    assert payload["detection_activity"] == [
        {"timestamp": "2025-01-02T09:00:00Z", "count": 1}
    ]
    assert payload["object_distribution"] == [{"object_name": "person", "count": 1}]


def test_analytics_api_reports_database_errors(
    isolated_database: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_load() -> dict[str, object]:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(main, "get_analytics_data", fail_to_load)

    with TestClient(main.app) as client:
        response = client.get("/api/analytics")

    assert response.status_code == 503
    assert response.json() == {"detail": "Analytics data is temporarily unavailable."}
