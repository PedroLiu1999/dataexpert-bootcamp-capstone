"""Pytest configuration and Testcontainers PostgreSQL + pgvector fixture."""

from __future__ import annotations

import os
import pytest
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres_container():
    """Spins up a Postgres container with pgvector for database integration tests."""
    # If explicit LAKEBASE_URL or PGHOST is set, use existing database
    if os.environ.get("LAKEBASE_URL") or os.environ.get("PGHOST"):
        import src.db.connection as conn_mod

        conn_mod._engine = None
        from src.db.repository import init_db

        init_db()
        yield None
        return

    # Use pgvector container for integration tests
    container = PostgresContainer("pgvector/pgvector:pg17")
    container.start()

    db_url = container.get_connection_url()
    if "+psycopg2" in db_url:
        db_url = db_url.replace("+psycopg2", "+psycopg")
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

    os.environ["LAKEBASE_URL"] = db_url

    # Reset connection engine cache and initialize schema
    import src.db.connection as conn_mod

    conn_mod._engine = None
    from src.db.repository import init_db

    init_db()
    yield container
    container.stop()


@pytest.fixture(autouse=True)
def _auto_db_integration(request):
    """Triggers postgres_container fixture only for tests decorated with @pytest.mark.integration."""
    if "integration" in request.node.keywords:
        request.getfixturevalue("postgres_container")
