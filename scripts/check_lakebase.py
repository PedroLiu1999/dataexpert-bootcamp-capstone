"""Smoke test script for Lakebase PostgreSQL + pgvector connection."""

import logging
import sys
from sqlalchemy import text
from src.db.connection import get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("check_lakebase")


def main() -> None:
    logger.info("Connecting to Lakebase via SQLAlchemy pooled engine...")
    engine = get_engine()
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version();")).scalar()
        user = conn.execute(text("SELECT current_user;")).scalar()
        schema = conn.execute(text("SELECT current_schema();")).scalar()

        vector_ver = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        ).scalar()

        logger.info("PostgreSQL Version: %s", version)
        logger.info("Current User: %s", user)
        logger.info("Current Schema: %s", schema)
        logger.info("pgvector Extension Version: %s", vector_ver or "Not installed yet (run T03 schema bootstrap)")

    logger.info("Lakebase connection check successful!")


if __name__ == "__main__":
    main()
