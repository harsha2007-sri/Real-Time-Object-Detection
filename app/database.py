from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Float, Integer, String, case, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DATABASE_PATH = "sqlite:///./detection.db"


class Base(DeclarativeBase):
    pass


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    object_name: Mapped[str] = mapped_column(String, nullable=False)
    tracking_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    x1: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    y1: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    x2: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    y2: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    event_type: Mapped[str] = mapped_column(String, nullable=False, default="detection")


class SystemSetting(Base):
    __tablename__ = "system_settings"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String, nullable=False)


engine = create_engine(DATABASE_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def save_event(session_id: str, object_name: str, tracking_id: int | None, confidence: float, box: tuple[float, float, float, float], event_type: str = "detection") -> None:
    with SessionLocal() as session:
        event = DetectionEvent(
            session_id=session_id,
            object_name=object_name,
            tracking_id=tracking_id,
            confidence=float(confidence),
            x1=float(box[0]),
            y1=float(box[1]),
            x2=float(box[2]),
            y2=float(box[3]),
            event_type=event_type,
        )
        session.add(event)
        session.commit()


def save_setting(key: str, value: Any) -> None:
    with SessionLocal() as session:
        existing = session.get(SystemSetting, key)
        if existing is None:
            session.add(SystemSetting(key=key, value=str(value)))
        else:
            existing.value = str(value)
        session.commit()


def save_settings(settings: dict[str, Any]) -> None:
    with SessionLocal() as session:
        for key, value in settings.items():
            existing = session.get(SystemSetting, key)
            if existing is None:
                session.add(SystemSetting(key=key, value=str(value)))
            else:
                existing.value = str(value)
        session.commit()


def get_setting(key: str, default: str = "") -> str:
    try:
        with SessionLocal() as session:
            record = session.get(SystemSetting, key)
            if record is None:
                return default
            return record.value
    except Exception:
        return default


def get_recent_events(limit: int = 20) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        rows = (
            session.query(DetectionEvent)
            .order_by(DetectionEvent.id.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "timestamp": row.timestamp.isoformat(timespec="seconds"),
                "session_id": row.session_id,
                "object_name": row.object_name,
                "tracking_id": row.tracking_id,
                "confidence": row.confidence,
                "event_type": row.event_type,
            }
            for row in rows
        ][::-1]


def _dashboard_counts(session: Session) -> dict[str, int]:
    row = session.query(
        func.count(case((DetectionEvent.event_type == "detection", 1))).label("total_detections"),
        func.count(case((DetectionEvent.event_type == "in", 1))).label("total_in"),
        func.count(case((DetectionEvent.event_type == "out", 1))).label("total_out"),
    ).one()
    return {
        "total_detections": row.total_detections,
        "total_in": row.total_in,
        "total_out": row.total_out,
    }


def get_dashboard_counts() -> dict[str, int]:
    with SessionLocal() as session:
        return _dashboard_counts(session)


def get_dashboard_stats() -> dict[str, Any]:
    with SessionLocal() as session:
        counts = _dashboard_counts(session)
        object_breakdown = {}
        for row in session.query(DetectionEvent.object_name, DetectionEvent.id).all():
            object_breakdown[row[0]] = object_breakdown.get(row[0], 0) + 1
    return {
        **counts,
        "object_distribution": object_breakdown,
    }


def get_analytics_data(recent_limit: int = 20) -> dict[str, Any]:
    with SessionLocal() as session:
        counts = _dashboard_counts(session)

        hour_bucket = func.strftime("%Y-%m-%dT%H:00:00Z", DetectionEvent.timestamp)
        hourly_rows = (
            session.query(
                hour_bucket.label("timestamp"),
                func.sum(case((DetectionEvent.event_type == "detection", 1), else_=0)).label("detections"),
                func.sum(case((DetectionEvent.event_type == "in", 1), else_=0)).label("total_in"),
                func.sum(case((DetectionEvent.event_type == "out", 1), else_=0)).label("total_out"),
            )
            .filter(DetectionEvent.event_type.in_(("detection", "in", "out")))
            .group_by(hour_bucket)
            .order_by(hour_bucket)
            .all()
        )

        distribution_rows = (
            session.query(
                DetectionEvent.object_name,
                func.count(DetectionEvent.id).label("count"),
            )
            .filter(DetectionEvent.event_type == "detection")
            .group_by(DetectionEvent.object_name)
            .order_by(DetectionEvent.object_name)
            .all()
        )

        recent_rows = (
            session.query(DetectionEvent)
            .order_by(DetectionEvent.timestamp.desc(), DetectionEvent.id.desc())
            .limit(recent_limit)
            .all()
        )

        return {
            "entry_exit": {
                "total_in": counts["total_in"],
                "total_out": counts["total_out"],
            },
            "detection_activity": [
                {"timestamp": row.timestamp, "count": row.detections}
                for row in hourly_rows
                if row.detections
            ],
            "traffic_flow": [
                {
                    "timestamp": row.timestamp,
                    "total_in": row.total_in,
                    "total_out": row.total_out,
                }
                for row in hourly_rows
                if row.total_in or row.total_out
            ],
            "object_distribution": [
                {"object_name": row.object_name, "count": row.count}
                for row in distribution_rows
            ],
            "recent_events": [
                {
                    "timestamp": f"{row.timestamp.isoformat(timespec='microseconds')}Z" if row.timestamp else None,
                    "object_name": row.object_name,
                    "event_type": row.event_type,
                    "confidence": row.confidence,
                    "tracking_id": row.tracking_id,
                }
                for row in recent_rows
            ],
        }
