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

import urllib.parse

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


def get_oauth_lakebase_url() -> Optional[str]:
    """
    Generates a PostgreSQL connection URL authenticated via Service Principal OAuth 2.0 token.
    Programmatically resolves host, user, and database from Databricks Apps bound environment variables.
    """
    use_oauth = (os.getenv("LAKEBASE_USE_OAUTH", "").lower() == "true")
    host = os.getenv("LAKEBASE_HOST") or os.getenv("PGHOST") or os.getenv("POSTGRES_HOST") or os.getenv("DATABRICKS_POSTGRES_HOST")
    user = os.getenv("LAKEBASE_USER") or os.getenv("PGUSER") or os.getenv("POSTGRES_USER") or os.getenv("DATABRICKS_CLIENT_ID")
    db = os.getenv("LAKEBASE_DB") or os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB") or "capstone_lakebase"
    port = os.getenv("LAKEBASE_PORT") or os.getenv("PGPORT") or "5432"

    if not (use_oauth or (host and user)):
        return None

    if not host or not user:
        logger.warning("Service Principal OAuth enabled but LAKEBASE_HOST or LAKEBASE_USER missing.")
        return None

    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        token = None

        # Try endpoint-scoped postgres credential first if endpoint path provided
        endpoint_path = os.getenv("LAKEBASE_ENDPOINT_PATH")
        if endpoint_path and hasattr(w, "postgres") and hasattr(w.postgres, "generate_database_credential"):
            try:
                cred = w.postgres.generate_database_credential(endpoint=endpoint_path)
                if cred and cred.token:
                    token = cred.token
            except Exception as e:
                logger.debug(f"Failed to generate endpoint database credential: {e}")

        # Fallback to workspace client M2M OAuth access token
        if not token:
            token_info = w.config.get_token()
            if token_info and token_info.access_token:
                token = token_info.access_token

        if not token:
            logger.warning("Could not acquire OAuth access token via Databricks SDK.")
            return None

        user_enc = urllib.parse.quote(user, safe="")
        pass_enc = urllib.parse.quote(token, safe="")
        url = f"postgresql://{user_enc}:{pass_enc}@{host}:{port}/{db}?sslmode=require"
        logger.info(f"Successfully constructed Service Principal OAuth PostgreSQL URL for host '{host}'.")
        return url
    except Exception as e:
        logger.warning(f"Error fetching Service Principal OAuth token for Lakebase: {e}")
        return None


def get_lakebase_url() -> Optional[str]:
    """
    Retrieves the Lakebase / PostgreSQL connection URL.
    Order of preference:
    1. Service Principal OAuth token authentication (LAKEBASE_USE_OAUTH=true or host/user configured).
    2. Explicit environment variables: LAKEBASE_URL, DATABASE_URL, POSTGRES_URL.
    3. Databricks Secret Scope (dbutils or databricks-sdk WorkspaceClient), decoding base64 if needed.
    """
    oauth_url = get_oauth_lakebase_url()
    if oauth_url:
        return oauth_url

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
