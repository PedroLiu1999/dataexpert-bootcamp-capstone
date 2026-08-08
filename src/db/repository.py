"""
Lakebase Repository module handling CRUD operations across all 9 domain tables
(users, learning_goals, papers, authors, paper_authors, collections, collection_papers,
reading_progress, notes) + pgvector cosine similarity search.
"""

from datetime import datetime, timezone
import hashlib
import logging
import math
import uuid
from typing import Any, Dict, List, Optional
from psycopg2.extras import execute_values
from src.db.connection import get_db_connection, is_postgres_available
from src.db.models import CREATE_TABLES_SQL

logger = logging.getLogger(__name__)

# In-memory storage mock fallbacks for testing/offline environments
_MOCK_USERS: Dict[str, Dict[str, Any]] = {}
_MOCK_GOALS: Dict[str, Dict[str, Any]] = {}
_MOCK_PAPERS: Dict[str, Dict[str, Any]] = {}
_MOCK_AUTHORS: Dict[str, Dict[str, Any]] = {}
_MOCK_PAPER_AUTHORS: List[Dict[str, Any]] = []
_MOCK_COLLECTIONS: Dict[str, Dict[str, Any]] = {}
_MOCK_COLLECTION_PAPERS: List[Dict[str, Any]] = []
_MOCK_READING_PROGRESS: Dict[str, Dict[str, Any]] = {}
_MOCK_NOTES: Dict[str, Dict[str, Any]] = {}
_MOCK_EMBEDDINGS: List[Dict[str, Any]] = []


def init_db() -> None:
    """
    Initializes PostgreSQL tables and pgvector index if available.
    """
    if not is_postgres_available():
        logger.info("Lakebase PostgreSQL not available; operating in-memory mock mode.")
        return

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLES_SQL)
        conn.commit()


# --- Users CRUD ---
def create_user(email: str, full_name: str) -> Dict[str, Any]:
    user_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    record = {"user_id": user_id, "email": email, "full_name": full_name, "created_at": now_iso}

    if not is_postgres_available():
        _MOCK_USERS[user_id] = record
        return record

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (user_id, email, full_name, created_at) VALUES (%s, %s, %s, %s) ON CONFLICT (email) DO UPDATE SET full_name = EXCLUDED.full_name RETURNING *;",
                (user_id, email, full_name, now_iso),
            )
            row = cur.fetchone()
            conn.commit()
            return dict(row) if row else record


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    if not is_postgres_available():
        for u in _MOCK_USERS.values():
            if u["email"].lower() == email.lower():
                return u
        return None

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(%s);", (email,))
            row = cur.fetchone()
            return dict(row) if row else None


# --- Learning Goals ---
def create_learning_goal(user_id: str, title: str, description: str, target_level: str = "Intermediate") -> Dict[str, Any]:
    goal_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "goal_id": goal_id,
        "user_id": user_id,
        "title": title,
        "description": description,
        "target_level": target_level,
        "created_at": now_iso,
    }

    if not is_postgres_available():
        _MOCK_GOALS[goal_id] = record
        return record

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO learning_goals (goal_id, user_id, title, description, target_level, created_at) VALUES (%s, %s, %s, %s, %s, %s);",
                (goal_id, user_id, title, description, target_level, now_iso),
            )
        conn.commit()
    return record


def get_user_learning_goals(user_id: str) -> List[Dict[str, Any]]:
    if not is_postgres_available():
        return [g for g in _MOCK_GOALS.values() if g["user_id"] == user_id]

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM learning_goals WHERE user_id = %s ORDER BY created_at DESC;", (user_id,))
            return [dict(r) for r in cur.fetchall()]


# --- Papers & Authors ---
def upsert_paper(
    paper_id: str,
    title: str,
    abstract: str,
    doi: Optional[str] = None,
    publication_year: Optional[int] = None,
    citation_count: int = 0,
    open_access_url: Optional[str] = None,
    topics: Optional[str] = None,
) -> Dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "paper_id": paper_id,
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "publication_year": publication_year,
        "citation_count": citation_count,
        "open_access_url": open_access_url,
        "topics": topics,
        "created_at": now_iso,
    }

    if not is_postgres_available():
        _MOCK_PAPERS[paper_id] = record
        return record

    query = """
        INSERT INTO papers (paper_id, doi, title, abstract, publication_year, citation_count, open_access_url, topics, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (paper_id) DO UPDATE SET
            doi = EXCLUDED.doi,
            title = EXCLUDED.title,
            abstract = EXCLUDED.abstract,
            publication_year = EXCLUDED.publication_year,
            citation_count = EXCLUDED.citation_count,
            open_access_url = EXCLUDED.open_access_url,
            topics = EXCLUDED.topics;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (paper_id, doi, title, abstract, publication_year, citation_count, open_access_url, topics, now_iso),
            )
        conn.commit()
    return record


def get_paper_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
    if not is_postgres_available():
        return _MOCK_PAPERS.get(paper_id)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM papers WHERE paper_id = %s;", (paper_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def upsert_author(author_id: str, display_name: str, institution: Optional[str] = None) -> Dict[str, Any]:
    record = {"author_id": author_id, "display_name": display_name, "institution": institution}
    if not is_postgres_available():
        _MOCK_AUTHORS[author_id] = record
        return record

    query = """
        INSERT INTO authors (author_id, display_name, institution)
        VALUES (%s, %s, %s)
        ON CONFLICT (author_id) DO UPDATE SET
            display_name = EXCLUDED.display_name,
            institution = EXCLUDED.institution;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (author_id, display_name, institution))
        conn.commit()
    return record


def upsert_paper_author(paper_id: str, author_id: str, author_position: int = 1) -> Dict[str, Any]:
    record = {"paper_id": paper_id, "author_id": author_id, "author_position": author_position}
    if not is_postgres_available():
        # Upsert into mock list
        for idx, existing in enumerate(_MOCK_PAPER_AUTHORS):
            if existing["paper_id"] == paper_id and existing["author_id"] == author_id:
                _MOCK_PAPER_AUTHORS[idx] = record
                return record
        _MOCK_PAPER_AUTHORS.append(record)
        return record

    query = """
        INSERT INTO paper_authors (paper_id, author_id, author_position)
        VALUES (%s, %s, %s)
        ON CONFLICT (paper_id, author_id) DO UPDATE SET
            author_position = EXCLUDED.author_position;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (paper_id, author_id, author_position))
        conn.commit()
    return record


# --- Collections ---
def create_collection(user_id: str, name: str, description: Optional[str] = None) -> Dict[str, Any]:
    collection_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "collection_id": collection_id,
        "user_id": user_id,
        "name": name,
        "description": description or "",
        "created_at": now_iso,
    }

    if not is_postgres_available():
        _MOCK_COLLECTIONS[collection_id] = record
        return record

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO collections (collection_id, user_id, name, description, created_at) VALUES (%s, %s, %s, %s, %s);",
                (collection_id, user_id, name, description or "", now_iso),
            )
        conn.commit()
    return record


def get_user_collections(user_id: str) -> List[Dict[str, Any]]:
    if not is_postgres_available():
        return [c for c in _MOCK_COLLECTIONS.values() if c["user_id"] == user_id]

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM collections WHERE user_id = %s ORDER BY created_at DESC;", (user_id,))
            return [dict(r) for r in cur.fetchall()]


def add_paper_to_collection(collection_id: str, paper_id: str) -> bool:
    now_iso = datetime.now(timezone.utc).isoformat()
    if not is_postgres_available():
        _MOCK_COLLECTION_PAPERS.append({"collection_id": collection_id, "paper_id": paper_id, "added_at": now_iso})
        return True

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO collection_papers (collection_id, paper_id, added_at) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
                (collection_id, paper_id, now_iso),
            )
        conn.commit()
    return True


def get_collection_papers(collection_id: str) -> List[Dict[str, Any]]:
    if not is_postgres_available():
        paper_ids = [cp["paper_id"] for cp in _MOCK_COLLECTION_PAPERS if cp["collection_id"] == collection_id]
        return [p for p_id, p in _MOCK_PAPERS.items() if p_id in paper_ids]

    query = """
        SELECT p.* FROM papers p
        JOIN collection_papers cp ON p.paper_id = cp.paper_id
        WHERE cp.collection_id = %s
        ORDER BY cp.added_at DESC;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (collection_id,))
            return [dict(r) for r in cur.fetchall()]


# --- Reading Progress ---
def update_reading_progress(
    user_id: str, paper_id: str, status: str = "in_progress", sequence_order: int = 1, rating: Optional[int] = None
) -> Dict[str, Any]:
    progress_id = f"prog-{user_id[:8]}-{paper_id[:16]}"
    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "progress_id": progress_id,
        "user_id": user_id,
        "paper_id": paper_id,
        "status": status,
        "sequence_order": sequence_order,
        "rating": rating,
        "updated_at": now_iso,
    }

    if not is_postgres_available():
        _MOCK_READING_PROGRESS[progress_id] = record
        return record

    query = """
        INSERT INTO reading_progress (progress_id, user_id, paper_id, status, sequence_order, rating, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (progress_id) DO UPDATE SET
            status = EXCLUDED.status,
            sequence_order = EXCLUDED.sequence_order,
            rating = EXCLUDED.rating,
            updated_at = EXCLUDED.updated_at;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (progress_id, user_id, paper_id, status, sequence_order, rating, now_iso))
        conn.commit()
    return record


def get_user_reading_progress(user_id: str) -> List[Dict[str, Any]]:
    if not is_postgres_available():
        res = []
        for rp in _MOCK_READING_PROGRESS.values():
            if rp["user_id"] == user_id:
                p = _MOCK_PAPERS.get(rp["paper_id"], {})
                combined = {**rp, "title": p.get("title", ""), "abstract": p.get("abstract", "")}
                res.append(combined)
        return sorted(res, key=lambda x: x["sequence_order"])

    query = """
        SELECT rp.*, p.title, p.abstract, p.citation_count, p.open_access_url
        FROM reading_progress rp
        JOIN papers p ON rp.paper_id = p.paper_id
        WHERE rp.user_id = %s
        ORDER BY rp.sequence_order ASC;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id,))
            return [dict(r) for r in cur.fetchall()]


# --- Notes ---
def add_note(user_id: str, content: str, paper_id: Optional[str] = None, goal_id: Optional[str] = None) -> Dict[str, Any]:
    note_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "note_id": note_id,
        "user_id": user_id,
        "paper_id": paper_id,
        "goal_id": goal_id,
        "content": content,
        "created_at": now_iso,
    }

    if not is_postgres_available():
        _MOCK_NOTES[note_id] = record
        return record

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO notes (note_id, user_id, paper_id, goal_id, content, created_at) VALUES (%s, %s, %s, %s, %s, %s);",
                (note_id, user_id, paper_id, goal_id, content, now_iso),
            )
        conn.commit()
    return record


def get_user_notes(user_id: str) -> List[Dict[str, Any]]:
    if not is_postgres_available():
        return [n for n in _MOCK_NOTES.values() if n["user_id"] == user_id]

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notes WHERE user_id = %s ORDER BY created_at DESC;", (user_id,))
            return [dict(r) for r in cur.fetchall()]


# --- Paper Embeddings & Vector Search ---
def insert_paper_embeddings(embeddings_data: List[Dict[str, Any]]) -> int:
    if not embeddings_data:
        return 0

    if not is_postgres_available():
        inserted_count = 0
        for item in embeddings_data:
            p_id = item["paper_id"]
            c_idx = item["chunk_index"]
            # Upsert into mock storage
            updated = False
            for idx, existing in enumerate(_MOCK_EMBEDDINGS):
                if existing["paper_id"] == p_id and existing["chunk_index"] == c_idx:
                    _MOCK_EMBEDDINGS[idx] = item
                    updated = True
                    break
            if not updated:
                _MOCK_EMBEDDINGS.append(item)
            inserted_count += 1
        return inserted_count

    insert_sql = """
        INSERT INTO paper_embeddings (paper_id, chunk_index, chunk_text, embedding, model_name, created_at)
        VALUES (%s, %s, %s, %s::vector, %s, %s)
        ON CONFLICT (paper_id, chunk_index) DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name,
            created_at = EXCLUDED.created_at;
    """
    tuples = [
        (
            e["paper_id"],
            e["chunk_index"],
            e["chunk_text"],
            "[" + ",".join(str(float(v)) for v in e["embedding"]) + "]",
            e["model_name"],
            e["created_at"],
        )
        for e in embeddings_data
    ]

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, tuples)
        conn.commit()
    return len(embeddings_data)


def vector_search_papers(query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    top_k = max(1, min(20, top_k))

    if not is_postgres_available():
        q_len = math.sqrt(sum(x * x for x in query_vector))
        if q_len == 0:
            return []

        results = []
        for emb in _MOCK_EMBEDDINGS:
            paper = _MOCK_PAPERS.get(emb["paper_id"])
            if not paper:
                continue
            e_vec = emb["embedding"]
            e_len = math.sqrt(sum(x * x for x in e_vec))
            if e_len == 0:
                continue
            dot = sum(a * b for a, b in zip(query_vector, e_vec))
            sim = dot / (q_len * e_len)
            results.append({
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                "publication_year": paper.get("publication_year"),
                "citation_count": paper.get("citation_count", 0),
                "open_access_url": paper.get("open_access_url"),
                "chunk_text": emb["chunk_text"],
                "similarity": round(float(sim), 4),
            })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    vec_str = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
    query = """
        SELECT p.paper_id, p.title, p.abstract, p.publication_year, p.citation_count, p.open_access_url,
               e.chunk_text, 1 - (e.embedding <=> %s::vector) AS similarity
        FROM paper_embeddings e
        JOIN papers p ON p.paper_id = e.paper_id
        ORDER BY e.embedding <=> %s::vector ASC
        LIMIT %s;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (vec_str, vec_str, top_k))
            results = []
            for r in cur.fetchall():
                r_dict = dict(r)
                r_dict["similarity"] = round(float(r_dict["similarity"]), 4)
                results.append(r_dict)
            return results
