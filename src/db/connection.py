"""
Database connection management for Lakebase (PostgreSQL + pgvector).
Supports Unity Catalog Secret Scope lookup, base64 connection URL decoding,
and local in-memory fallback for testing environments.
"""

import base64
from contextlib import contextmanager
import logging
import os
from typing import Any, Generator, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


def _decode_if_base64(val: str) -> str:
    """
    Decodes base64 string if encoded, otherwise returns raw string.
    """
    if not val:
        return ""
    val_str = val.strip()
    if val_str.startswith("postgresql://") or val_str.startswith("postgres://"):
        return val_str
    try:
        decoded = base64.b64decode(val_str).decode("utf-8").strip()
        if decoded.startswith("postgresql://") or decoded.startswith("postgres://"):
            return decoded
    except Exception:
        pass
    return val_str


def get_lakebase_url() -> Optional[str]:
    """
    Retrieves the Lakebase / PostgreSQL connection URL.
    Order of preference:
    1. Environment variables: LAKEBASE_URL, DATABASE_URL, POSTGRES_URL.
    2. Databricks Secret Scope (dbutils or databricks-sdk WorkspaceClient), decoding base64 if needed.
    """
    url = os.getenv("LAKEBASE_URL") or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
    if url:
        return _decode_if_base64(url)

    scope = os.getenv("SECRET_SCOPE", "capstone_secrets")
    key = os.getenv("SECRET_KEY", "lakebase_url")

    # Try dbutils inside Databricks Runtime / Apps
    try:
        import dbutils  # type: ignore
        val = dbutils.secrets.get(scope=scope, key=key)
        if val:
            return _decode_if_base64(val)
    except Exception:
        pass

    # Try Databricks SDK WorkspaceClient
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        resp = w.secrets.get_secret(scope=scope, key=key)
        if resp and resp.value:
            return _decode_if_base64(resp.value)
    except Exception:
        pass

    return None


def is_postgres_available() -> bool:
    lakebase_url = get_lakebase_url()
    if not lakebase_url:
        return False
    try:
        conn = psycopg2.connect(lakebase_url, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


@contextmanager
def get_db_connection() -> Generator[Any, None, None]:
    """
    Context manager yielding a psycopg2 connection with RealDictCursor.
    """
    lakebase_url = get_lakebase_url()
    if not lakebase_url:
        raise ValueError("Lakebase connection URL is not configured.")
    conn = psycopg2.connect(lakebase_url, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()
