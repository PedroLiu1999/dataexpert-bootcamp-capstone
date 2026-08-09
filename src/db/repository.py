"""Lakebase Repository module handling database CRUD operations across domain tables:

users, learning_goals, papers, authors, paper_authors, collections, collection_papers,
reading_progress, notes, paper_chunks + pgvector cosine similarity search.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy import text
from src.db.connection import get_engine

logger = logging.getLogger(__name__)


def init_db() -> None:
    """Initializes PostgreSQL tables and pgvector schema."""
    engine = get_engine()
    ddl = """
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE SCHEMA IF NOT EXISTS capstone;

    CREATE TABLE IF NOT EXISTS capstone.users (
        user_id VARCHAR(64) PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        full_name VARCHAR(255),
        role VARCHAR(32) DEFAULT 'student',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS capstone.learning_goals (
        goal_id VARCHAR(64) PRIMARY KEY,
        user_id VARCHAR(64) NOT NULL REFERENCES capstone.users(user_id) ON DELETE CASCADE,
        title VARCHAR(255) NOT NULL,
        description TEXT,
        target_level VARCHAR(64) DEFAULT 'Intermediate',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS capstone.papers (
        paper_id VARCHAR(64) PRIMARY KEY,
        doi VARCHAR(255),
        title TEXT NOT NULL,
        abstract TEXT,
        publication_year INT,
        citation_count INT DEFAULT 0,
        open_access_url TEXT,
        topics TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS capstone.authors (
        author_id VARCHAR(64) PRIMARY KEY,
        display_name VARCHAR(255) NOT NULL,
        institution VARCHAR(255),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS capstone.paper_authors (
        paper_id VARCHAR(64) NOT NULL REFERENCES capstone.papers(paper_id) ON DELETE CASCADE,
        author_id VARCHAR(64) NOT NULL REFERENCES capstone.authors(author_id) ON DELETE CASCADE,
        author_position INT DEFAULT 1,
        PRIMARY KEY (paper_id, author_id)
    );

    CREATE TABLE IF NOT EXISTS capstone.collections (
        collection_id VARCHAR(64) PRIMARY KEY,
        user_id VARCHAR(64) NOT NULL REFERENCES capstone.users(user_id) ON DELETE CASCADE,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS capstone.collection_papers (
        collection_id VARCHAR(64) NOT NULL REFERENCES capstone.collections(collection_id) ON DELETE CASCADE,
        paper_id VARCHAR(64) NOT NULL REFERENCES capstone.papers(paper_id) ON DELETE CASCADE,
        added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (collection_id, paper_id)
    );

    CREATE TABLE IF NOT EXISTS capstone.reading_progress (
        progress_id BIGSERIAL PRIMARY KEY,
        user_id VARCHAR(64) NOT NULL REFERENCES capstone.users(user_id) ON DELETE CASCADE,
        paper_id VARCHAR(64) NOT NULL REFERENCES capstone.papers(paper_id) ON DELETE CASCADE,
        status VARCHAR(32) DEFAULT 'unread',
        sequence_order INT DEFAULT 1,
        rating INT,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_user_paper UNIQUE (user_id, paper_id)
    );

    CREATE TABLE IF NOT EXISTS capstone.notes (
        note_id VARCHAR(64) PRIMARY KEY,
        user_id VARCHAR(64) NOT NULL REFERENCES capstone.users(user_id) ON DELETE CASCADE,
        paper_id VARCHAR(64) REFERENCES capstone.papers(paper_id) ON DELETE CASCADE,
        goal_id VARCHAR(64) REFERENCES capstone.learning_goals(goal_id) ON DELETE CASCADE,
        content TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS capstone.paper_chunks (
        chunk_id VARCHAR(64) PRIMARY KEY,
        paper_id VARCHAR(64) NOT NULL REFERENCES capstone.papers(paper_id) ON DELETE CASCADE,
        chunk_index INT NOT NULL,
        chunk_text TEXT NOT NULL,
        embedding vector(384),
        model_name VARCHAR(128) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_paper_chunk UNIQUE (paper_id, chunk_index)
    );

    CREATE TABLE IF NOT EXISTS capstone.events (
        event_id BIGSERIAL PRIMARY KEY,
        event_type VARCHAR(64) NOT NULL,
        user_id VARCHAR(64) NOT NULL REFERENCES capstone.users(user_id) ON DELETE CASCADE,
        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_paper_chunks_embedding
    ON capstone.paper_chunks USING hnsw (embedding vector_cosine_ops);
    """


    with engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("Lakebase database schema initialized successfully.")


# --- Users CRUD ---
def create_user(email: str, full_name: str) -> Dict[str, Any]:
    user_id = str(uuid.uuid4())
    query = text("""
        INSERT INTO capstone.users (user_id, email, full_name)
        VALUES (:user_id, :email, :full_name)
        ON CONFLICT (email) DO UPDATE SET full_name = EXCLUDED.full_name
        RETURNING user_id, email, full_name, role, created_at;
    """)
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(query, {"user_id": user_id, "email": email, "full_name": full_name})
        row = result.fetchone()
        if row:
            d = dict(row._mapping)
            if isinstance(d.get("created_at"), datetime):
                d["created_at"] = d["created_at"].isoformat()
            return d
        return {"user_id": user_id, "email": email, "full_name": full_name}


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    query = text("SELECT * FROM capstone.users WHERE LOWER(email) = LOWER(:email);")
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(query, {"email": email})
        row = result.fetchone()
        if not row:
            return None
        d = dict(row._mapping)
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].isoformat()
        return d


# --- Learning Goals ---
def create_learning_goal(
    user_id: str, title: str, description: str, target_level: str = "Intermediate"
) -> Dict[str, Any]:
    goal_id = str(uuid.uuid4())
    query = text("""
        INSERT INTO capstone.learning_goals (goal_id, user_id, title, description, target_level)
        VALUES (:goal_id, :user_id, :title, :description, :target_level)
        RETURNING goal_id, user_id, title, description, target_level, created_at;
    """)
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            query,
            {
                "goal_id": goal_id,
                "user_id": user_id,
                "title": title,
                "description": description,
                "target_level": target_level,
            },
        )
        row = result.fetchone()
        d = dict(row._mapping) if row else {"goal_id": goal_id, "user_id": user_id, "title": title}
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].isoformat()
        return d


def get_user_learning_goals(user_id: str) -> List[Dict[str, Any]]:
    query = text("SELECT * FROM capstone.learning_goals WHERE user_id = :user_id ORDER BY created_at DESC;")
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(query, {"user_id": user_id})
        res = []
        for r in result.fetchall():
            d = dict(r._mapping)
            if isinstance(d.get("created_at"), datetime):
                d["created_at"] = d["created_at"].isoformat()
            res.append(d)
        return res


# --- Papers & Authors ---
def upsert_paper(
    paper_id: str,
    title: str,
    abstract: Optional[str] = None,
    doi: Optional[str] = None,
    publication_year: Optional[int] = None,
    citation_count: int = 0,
    open_access_url: Optional[str] = None,
    topics: Optional[str] = None,
) -> Dict[str, Any]:
    query = text("""
        INSERT INTO capstone.papers (paper_id, doi, title, abstract, publication_year, citation_count, open_access_url, topics)
        VALUES (:paper_id, :doi, :title, :abstract, :publication_year, :citation_count, :open_access_url, :topics)
        ON CONFLICT (paper_id) DO UPDATE SET
            doi = EXCLUDED.doi,
            title = EXCLUDED.title,
            abstract = EXCLUDED.abstract,
            publication_year = EXCLUDED.publication_year,
            citation_count = EXCLUDED.citation_count,
            open_access_url = EXCLUDED.open_access_url,
            topics = EXCLUDED.topics
        RETURNING *;
    """)
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            query,
            {
                "paper_id": paper_id,
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "publication_year": publication_year,
                "citation_count": citation_count,
                "open_access_url": open_access_url,
                "topics": topics,
            },
        )
        row = result.fetchone()
        d = dict(row._mapping) if row else {"paper_id": paper_id, "title": title}
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].isoformat()
        return d


def get_paper_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
    query = text("SELECT * FROM capstone.papers WHERE paper_id = :paper_id;")
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(query, {"paper_id": paper_id})
        row = result.fetchone()
        if not row:
            return None
        d = dict(row._mapping)
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].isoformat()
        return d


def upsert_author(author_id: str, display_name: str, institution: Optional[str] = None) -> Dict[str, Any]:
    query = text("""
        INSERT INTO capstone.authors (author_id, display_name, institution)
        VALUES (:author_id, :display_name, :institution)
        ON CONFLICT (author_id) DO UPDATE SET
            display_name = EXCLUDED.display_name,
            institution = EXCLUDED.institution
        RETURNING *;
    """)
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(query, {"author_id": author_id, "display_name": display_name, "institution": institution})
        row = result.fetchone()
        d = dict(row._mapping) if row else {"author_id": author_id, "display_name": display_name}
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].isoformat()
        return d


def upsert_paper_author(paper_id: str, author_id: str, author_position: int = 1) -> Dict[str, Any]:
    query = text("""
        INSERT INTO capstone.paper_authors (paper_id, author_id, author_position)
        VALUES (:paper_id, :author_id, :author_position)
        ON CONFLICT (paper_id, author_id) DO UPDATE SET
            author_position = EXCLUDED.author_position
        RETURNING *;
    """)
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(query, {"paper_id": paper_id, "author_id": author_id, "author_position": author_position})
        row = result.fetchone()
        return dict(row._mapping) if row else {"paper_id": paper_id, "author_id": author_id}


# --- Collections ---
def create_collection(user_id: str, name: str, description: Optional[str] = None) -> Dict[str, Any]:
    collection_id = str(uuid.uuid4())
    query = text("""
        INSERT INTO capstone.collections (collection_id, user_id, name, description)
        VALUES (:collection_id, :user_id, :name, :description)
        RETURNING *;
    """)
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            query,
            {"collection_id": collection_id, "user_id": user_id, "name": name, "description": description or ""},
        )
        row = result.fetchone()
        d = dict(row._mapping) if row else {"collection_id": collection_id, "user_id": user_id, "name": name}
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].isoformat()
        return d


def get_user_collections(user_id: str) -> List[Dict[str, Any]]:
    query = text("SELECT * FROM capstone.collections WHERE user_id = :user_id ORDER BY created_at DESC;")
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(query, {"user_id": user_id})
        res = []
        for r in result.fetchall():
            d = dict(r._mapping)
            if isinstance(d.get("created_at"), datetime):
                d["created_at"] = d["created_at"].isoformat()
            res.append(d)
        return res


def add_paper_to_collection(collection_id: str, paper_id: str) -> bool:
    query = text("""
        INSERT INTO capstone.collection_papers (collection_id, paper_id)
        VALUES (:collection_id, :paper_id)
        ON CONFLICT DO NOTHING;
    """)
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(query, {"collection_id": collection_id, "paper_id": paper_id})
    return True


def get_collection_papers(collection_id: str) -> List[Dict[str, Any]]:
    query = text("""
        SELECT p.* FROM capstone.papers p
        JOIN capstone.collection_papers cp ON p.paper_id = cp.paper_id
        WHERE cp.collection_id = :collection_id
        ORDER BY cp.added_at DESC;
    """)
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(query, {"collection_id": collection_id})
        res = []
        for r in result.fetchall():
            d = dict(r._mapping)
            if isinstance(d.get("created_at"), datetime):
                d["created_at"] = d["created_at"].isoformat()
            res.append(d)
        return res


# --- Reading Progress ---
def update_reading_progress(
    user_id: str,
    paper_id: str,
    status: str = "in_progress",
    sequence_order: int = 1,
    rating: Optional[int] = None,
) -> Dict[str, Any]:
    query = text("""
        INSERT INTO capstone.reading_progress (user_id, paper_id, status, sequence_order, rating)
        VALUES (:user_id, :paper_id, :status, :sequence_order, :rating)
        ON CONFLICT (user_id, paper_id) DO UPDATE SET
            status = EXCLUDED.status,
            sequence_order = EXCLUDED.sequence_order,
            rating = EXCLUDED.rating,
            updated_at = NOW()
        RETURNING progress_id, user_id, paper_id, status, sequence_order, rating, updated_at;
    """)
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            query,
            {
                "user_id": user_id,
                "paper_id": paper_id,
                "status": status,
                "sequence_order": sequence_order,
                "rating": rating,
            },
        )
        row = result.fetchone()
        d = dict(row._mapping) if row else {"user_id": user_id, "paper_id": paper_id, "status": status}
        if isinstance(d.get("updated_at"), datetime):
            d["updated_at"] = d["updated_at"].isoformat()
        return d


def get_user_reading_progress(user_id: str) -> List[Dict[str, Any]]:
    query = text("""
        SELECT rp.*, p.title, p.abstract, p.citation_count, p.open_access_url
        FROM capstone.reading_progress rp
        JOIN capstone.papers p ON rp.paper_id = p.paper_id
        WHERE rp.user_id = :user_id
        ORDER BY rp.sequence_order ASC;
    """)
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(query, {"user_id": user_id})
        res = []
        for r in result.fetchall():
            d = dict(r._mapping)
            if isinstance(d.get("updated_at"), datetime):
                d["updated_at"] = d["updated_at"].isoformat()
            res.append(d)
        return res


# --- Notes ---
def add_note(
    user_id: str, content: str, paper_id: Optional[str] = None, goal_id: Optional[str] = None
) -> Dict[str, Any]:
    note_id = str(uuid.uuid4())
    query = text("""
        INSERT INTO capstone.notes (note_id, user_id, paper_id, goal_id, content)
        VALUES (:note_id, :user_id, :paper_id, :goal_id, :content)
        RETURNING *;
    """)
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            query,
            {"note_id": note_id, "user_id": user_id, "paper_id": paper_id, "goal_id": goal_id, "content": content},
        )
        row = result.fetchone()
        d = dict(row._mapping) if row else {"note_id": note_id, "user_id": user_id, "content": content}
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].isoformat()
        return d


def get_user_notes(user_id: str) -> List[Dict[str, Any]]:
    query = text("SELECT * FROM capstone.notes WHERE user_id = :user_id ORDER BY created_at DESC;")
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(query, {"user_id": user_id})
        res = []
        for r in result.fetchall():
            d = dict(r._mapping)
            if isinstance(d.get("created_at"), datetime):
                d["created_at"] = d["created_at"].isoformat()
            res.append(d)
        return res


# --- Paper Chunks & Vector Search ---
def insert_paper_embeddings(embeddings_data: List[Dict[str, Any]]) -> int:
    if not embeddings_data:
        return 0

    engine = get_engine()
    query = text("""
        INSERT INTO capstone.paper_chunks (chunk_id, paper_id, chunk_index, chunk_text, embedding, model_name)
        VALUES (:chunk_id, :paper_id, :chunk_index, :chunk_text, CAST(:embedding AS vector), :model_name)
        ON CONFLICT (paper_id, chunk_index) DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name;

    """)

    # Single transaction for batch vector chunk insertion
    with engine.begin() as conn:
        for e in embeddings_data:
            chunk_id = e.get("chunk_id") or f"{e['paper_id']}_c{e['chunk_index']}"
            vec = e["embedding"]
            vec_str = "[" + ",".join(str(float(v)) for v in vec) + "]"
            model_name = e.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            conn.execute(
                query,
                {
                    "chunk_id": chunk_id,
                    "paper_id": e["paper_id"],
                    "chunk_index": e["chunk_index"],
                    "chunk_text": e["chunk_text"],
                    "embedding": vec_str,
                    "model_name": model_name,
                },
            )
    return len(embeddings_data)


def vector_search_papers(
    query_vector: List[float],
    top_k: int = 5,
    similarity_threshold: Optional[float] = 0.3,
    collection_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    top_k = max(1, min(20, top_k))
    vec_str = "[" + ",".join(str(float(x)) for x in query_vector) + "]"

    sql_str = """
        WITH ranked_chunks AS (
            SELECT DISTINCT ON (p.paper_id)
                   p.paper_id, p.title, p.abstract, p.publication_year, p.citation_count, p.open_access_url,
                   pc.chunk_text, 1 - (pc.embedding <=> CAST(:vec AS vector)) AS similarity
            FROM capstone.paper_chunks pc
            JOIN capstone.papers p ON p.paper_id = pc.paper_id
    """
    if collection_id:
        sql_str += " JOIN capstone.collection_papers cp ON cp.paper_id = p.paper_id WHERE cp.collection_id = :collection_id "

    sql_str += """
            ORDER BY p.paper_id, (pc.embedding <=> CAST(:vec AS vector)) ASC
        )
        SELECT * FROM ranked_chunks
    """

    if similarity_threshold is not None:
        sql_str += " WHERE similarity >= :min_sim "

    sql_str += " ORDER BY similarity DESC LIMIT :top_k;"

    query = text(sql_str)
    params: Dict[str, Any] = {"vec": vec_str, "top_k": top_k}
    if similarity_threshold is not None:
        params["min_sim"] = float(similarity_threshold)
    if collection_id:
        params["collection_id"] = collection_id

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(query, params)
        res = []
        for row in result.fetchall():
            d = dict(row._mapping)
            d["similarity"] = round(float(d["similarity"]), 4) if d.get("similarity") is not None else None
            res.append(d)
        return res


# --- Persistent Events & Analytics ---

def log_analytics_event(event_type: str, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    import json

    query = text("""
        INSERT INTO capstone.events (event_type, user_id, payload)
        VALUES (:event_type, :user_id, CAST(:payload AS jsonb))
        RETURNING event_id, event_type, user_id, payload, created_at;
    """)

    engine = get_engine()
    payload_str = json.dumps(payload)
    with engine.begin() as conn:
        result = conn.execute(query, {"event_type": event_type, "user_id": user_id, "payload": payload_str})
        row = result.fetchone()
        d = dict(row._mapping) if row else {"event_type": event_type, "user_id": user_id, "payload": payload}
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].isoformat()
        return d


def get_user_analytics_summary(user_id: Optional[str] = None) -> Dict[str, Any]:
    import json

    engine = get_engine()
    if user_id:
        sql = text("SELECT event_type, payload FROM capstone.events WHERE user_id = :user_id;")
        params = {"user_id": user_id}
    else:
        sql = text("SELECT event_type, payload FROM capstone.events;")
        params = {}

    with engine.connect() as conn:
        result = conn.execute(sql, params)
        rows = result.fetchall()

    total_events = len(rows)
    plans_generated = 0
    papers_added = 0
    completed_count = 0
    total_progress_events = 0
    tool_counts: Dict[str, int] = {}

    for row in rows:
        d = dict(row._mapping)
        e_type = d.get("event_type")
        payload = d.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}

        if e_type == "tool_call":
            t_name = payload.get("tool_name", "unknown_tool")
            tool_counts[t_name] = tool_counts.get(t_name, 0) + 1
            if t_name == "tool_generate_sequenced_reading_plan":
                plans_generated += 1
            elif t_name == "tool_add_paper_to_collection":
                papers_added += 1

        elif e_type in ("paper_added", "collection_add"):
            papers_added += 1

        elif e_type == "progress_update":
            total_progress_events += 1
            if payload.get("status") == "completed":
                completed_count += 1

    completion_rate = round((completed_count / total_progress_events * 100), 1) if total_progress_events > 0 else 0.0

    return {
        "total_events_logged": total_events,
        "plans_generated": plans_generated,
        "papers_added": papers_added,
        "completed_reading_count": completed_count,
        "completion_rate_pct": completion_rate,
        "tool_call_counts": tool_counts,
        "cdf_enabled": True,
        "storage_backend": "Lakebase PostgreSQL Events Table",
    }


