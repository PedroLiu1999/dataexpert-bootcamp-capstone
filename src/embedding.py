"""Pure-Python Text Embedding & Lakebase Ingestion Module (Spark-free).

Handles sentence transformer model caching, vector embedding generation,
text chunking, paper metadata upserts, and sequential Lakebase database ingestion.
Designed for serverless containers (Databricks Apps) without JVM / PySpark dependencies.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict, List

# Ensure HuggingFace & PyTorch use writable /tmp cache on Databricks Serverless worker nodes
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")

from src.db.repository import init_db, insert_paper_embeddings, upsert_paper, upsert_author, upsert_paper_author

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_DIMS = 384

_model_instance = None


def get_embedding_model():
    """Loads or retrieves process-cached SentenceTransformer model."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    os.environ["HF_HOME"] = "/tmp/hf_cache"
    os.environ["TORCH_HOME"] = "/tmp/torch_cache"

    from sentence_transformers import SentenceTransformer

    logger.info("Loading SentenceTransformer model '%s'...", MODEL_NAME)
    _model_instance = SentenceTransformer(MODEL_NAME)
    return _model_instance


def generate_embedding(text: str) -> List[float]:
    """Generates 384-dim normalized vector embedding for given text."""
    model = get_embedding_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Splits text into overlapping chunks for vector embedding index."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        start += step
    return chunks


def upsert_paper_metadata(papers_list: List[Dict[str, Any]]) -> int:
    """Driver-side upsert of paper, author and paper_author rows into Lakebase.

    Returns the number of papers upserted.
    """
    for paper in papers_list:
        upsert_paper(
            paper_id=paper["paper_id"],
            title=paper["title"],
            abstract=paper.get("abstract"),
            doi=paper.get("doi"),
            publication_year=paper.get("publication_year"),
            citation_count=paper.get("citation_count", 0),
            open_access_url=paper.get("open_access_url"),
            topics=paper.get("topics"),
        )
        for author in paper.get("authors", []):
            a_id = author.get("author_id")
            a_name = author.get("display_name")
            if a_id and a_name:
                upsert_author(a_id, a_name, author.get("institution"))
                upsert_paper_author(paper["paper_id"], a_id, author.get("author_position", 1))
    return len(papers_list)


def build_chunk_records(papers_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chunk + embed papers in-process, returning rows shaped for
    insert_paper_embeddings: paper_id, chunk_index, chunk_text, embedding,
    model_name, created_at.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    records = []
    for paper in papers_list:
        p_id = paper["paper_id"]
        text_to_embed = f"Title: {paper['title']}\n\nAbstract: {paper.get('abstract', '')}"
        chunks = chunk_text(text_to_embed)
        for idx, chunk_str in enumerate(chunks):
            vector_vec = generate_embedding(chunk_str)
            records.append({
                "paper_id": p_id,
                "chunk_index": idx,
                "chunk_text": chunk_str,
                "embedding": vector_vec,
                "model_name": MODEL_NAME,
                "created_at": now_iso,
            })
    return records


def process_and_embed_papers(papers_list: List[Dict[str, Any]]) -> int:
    """Sequential in-process paper ingestion and embedding pipeline for Lakebase."""
    init_db()
    if not papers_list:
        return 0
    upsert_paper_metadata(papers_list)
    records = build_chunk_records(papers_list)
    return insert_paper_embeddings(records)
