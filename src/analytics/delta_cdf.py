"""Lakebase Persistent Event Analytics Module.

Logs domain operational events (reading progress updates, notes, agent tool executions)
into Lakebase capstone.events table, providing real-time persistent analytics metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from src.db.repository import get_user_analytics_summary, log_analytics_event

logger = logging.getLogger(__name__)


class DeltaCDFTracker:
    def __init__(self):
        pass

    def log_event(self, event_type: str, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Logs an operational event ('tool_call', 'progress_update', 'note_added', 'paper_added')

        to persistent Lakebase capstone.events storage.
        """
        return log_analytics_event(event_type=event_type, user_id=user_id, payload=payload)

    def get_cdf_analytics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Reads persistent event history from Lakebase capstone.events table and computes metrics."""
        return get_user_analytics_summary(user_id=user_id)


# Global singleton instance
cdf_tracker = DeltaCDFTracker()


def get_delta_cdf_sql_schema() -> str:
    """Returns SQL DDL statement for Databricks Delta table with Change Data Feed enabled."""
    return """
    CREATE TABLE IF NOT EXISTS delta_cdf_events (
        event_id BIGINT NOT NULL,
        event_type STRING NOT NULL,
        user_id STRING NOT NULL,
        payload STRING NOT NULL,
        timestamp TIMESTAMP NOT NULL
    )
    USING DELTA
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true'
    );
    """
