"""Lakebase (Postgres) connection management with OAuth credential rotation."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import threading
import time
from datetime import datetime
from typing import Optional, Generator, Any

from databricks.sdk import WorkspaceClient
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine


logger = logging.getLogger(__name__)

# Refresh this many seconds before the credential actually expires.
_REFRESH_MARGIN_S = 300
# Conservative assumed lifetime if the SDK gives an expiry we cannot parse.
_ASSUMED_TTL_S = 2_700


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"{name} is not set. In Databricks Apps this comes from the attached "
            f"'postgres' resource; locally, export it (see docs/LAKEBASE_SETUP.md)."
        )
    return val


class _CredentialCache:
    """Caches one database credential and refreshes it before expiry.

    Thread-safe: Streamlit serves reruns on worker threads, and the pool may open
    connections concurrently.
    """

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._client = WorkspaceClient()
        self._lock = threading.Lock()
        self._token: Optional[str] = None
        self._expires_at: float = 0.0

    def token(self) -> str:
        with self._lock:
            if self._token is None or time.time() >= self._expires_at - _REFRESH_MARGIN_S:
                logger.info("Minting Lakebase database credential for %s", self._endpoint)
                cred = self._client.postgres.generate_database_credential(
                    endpoint=self._endpoint
                )
                self._token = cred.token
                self._expires_at = self._parse_expiry(cred.expire_time)
            return self._token

    @staticmethod
    def _parse_expiry(expire_time: object) -> float:
        try:
            if isinstance(expire_time, datetime):
                return expire_time.timestamp()
            if isinstance(expire_time, str):
                return datetime.fromisoformat(
                    expire_time.replace("Z", "+00:00")
                ).timestamp()
            seconds = getattr(expire_time, "seconds", None)
            if seconds:
                return float(seconds)
        except Exception:  # noqa: BLE001
            logger.warning("Could not parse credential expiry; assuming %ss", _ASSUMED_TTL_S)
        return time.time() + _ASSUMED_TTL_S


_engine: Engine | None = None
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine. Safe to call repeatedly."""
    global _engine
    if _engine is not None:
        return _engine

    with _engine_lock:
        if _engine is not None:
            return _engine

        # Single escape hatch for local testing against explicit connection URL
        lakebase_url = os.environ.get("LAKEBASE_URL")
        if lakebase_url:
            logger.info("Connecting to Lakebase using LAKEBASE_URL environment override")
            _engine = create_engine(
                lakebase_url,
                pool_size=5,
                max_overflow=5,
                pool_recycle=1_800,
                pool_pre_ping=True,
            )
            return _engine

        host = _require("PGHOST")
        user = _require("PGUSER")
        database = os.environ.get("PGDATABASE", "databricks_postgres")
        port = os.environ.get("PGPORT", "5432")
        sslmode = os.environ.get("PGSSLMODE", "require")
        schema = os.environ.get("PGSCHEMA", "capstone")

        engine = create_engine(
            f"postgresql+psycopg://{user}@{host}:{port}/{database}?sslmode={sslmode}",
            pool_size=5,
            max_overflow=5,
            pool_recycle=1_800,
            pool_pre_ping=True,
            connect_args={
                "application_name": os.environ.get("PGAPPNAME", "capstone-app"),
                "connect_timeout": 15,
                "options": f"-c search_path={schema},public",
            },
        )

        credentials = _CredentialCache(_require("PGENDPOINT"))

        @event.listens_for(engine, "do_connect")
        def _inject_credential(dialect, conn_rec, cargs, cparams):  # noqa: ANN001
            cparams["password"] = credentials.token()

        _engine = engine
        return _engine


def is_postgres_available() -> bool:
    """Interim compatibility check using pooled engine. Deleted in T05."""
    try:
        engine = get_engine()
        with engine.connect():
            return True
    except Exception:
        return False


@contextmanager
def get_db_connection() -> Generator[Any, None, None]:
    """Interim compatibility context manager using pooled engine. Deleted in T05."""
    engine = get_engine()
    with engine.connect() as conn:
        yield conn


