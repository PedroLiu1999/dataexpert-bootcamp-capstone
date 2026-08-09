"""PySpark Data Ingestion & Vector Embedding Pipeline for Academic Papers.

Cleans raw OpenAlex paper metadata, chunks abstracts/text, computes 384-dim vector embeddings
via distributed Spark partitions, and batch-syncs structured records into Lakebase.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Iterator, Tuple

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


def _embed_partition(iterator: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Partition worker function loading SentenceTransformer once per partition

    and computing embeddings over partition records.
    """
    model = get_embedding_model()
    now_iso = datetime.now(timezone.utc).isoformat()

    for paper_dict in iterator:
        p_id = paper_dict["paper_id"]
        p_title = paper_dict.get("title", "")
        p_abstract = paper_dict.get("abstract", "")
        text_to_embed = f"Title: {p_title}\n\nAbstract: {p_abstract}"
        chunks = chunk_text(text_to_embed)

        for idx, c_str in enumerate(chunks):
            vec = model.encode(c_str, normalize_embeddings=True).tolist()
            yield {
                "paper_id": p_id,
                "chunk_index": idx,
                "chunk_text": c_str,
                "embedding": vec,
                "model_name": MODEL_NAME,
                "created_at": now_iso,
            }


def process_and_embed_papers(papers_list: List[Dict[str, Any]]) -> int:
    """Ingests and embeds paper records into Lakebase.

    Uses active PySpark session for partition-level model loading and driver batch write when Spark is available,
    falling back to Python batch execution if PySpark is not installed.
    """
    init_db()
    if not papers_list:
        return 0

    logger.info("Ingesting %d paper records...", len(papers_list))
    now_iso = datetime.now(timezone.utc).isoformat()

    # 1. Driver upserts paper & author metadata into Lakebase
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

    # 2. PySpark partition execution or fallback
    try:
        from pyspark.sql import SparkSession

        # Attach to active Spark session on serverless / cluster, or create local SparkSession without hardcoded master
        spark = SparkSession.builder.appName("CapstoneSparkNativePipeline").getOrCreate()

        logger.info("PySpark session active. Parallelizing embedding computations across Spark partitions...")

        rdd = spark.sparkContext.parallelize(papers_list)
        embeddings_rdd = rdd.mapPartitions(_embed_partition)
        all_embeddings = embeddings_rdd.collect()

        if all_embeddings:
            embedded_chunks_count = insert_paper_embeddings(all_embeddings)
        else:
            embedded_chunks_count = 0

        logger.info("PySpark pipeline completed: inserted %d vector chunks.", embedded_chunks_count)
        return embedded_chunks_count

    except ImportError:
        logger.warning("PySpark package not installed. Running Python batch embedding pipeline.")
    except Exception as e:
        logger.error("PySpark execution failed: %s", e)
        raise

    # Fallback for environment without PySpark installed
    embeddings_batch = []
    for paper in papers_list:
        p_id = paper["paper_id"]
        text_to_embed = f"Title: {paper['title']}\n\nAbstract: {paper.get('abstract', '')}"
        chunks = chunk_text(text_to_embed)
        for idx, chunk_str in enumerate(chunks):
            vector_vec = generate_embedding(chunk_str)
            embeddings_batch.append({
                "paper_id": p_id,
                "chunk_index": idx,
                "chunk_text": chunk_str,
                "embedding": vector_vec,
                "model_name": MODEL_NAME,
                "created_at": now_iso,
            })

    embedded_chunks_count = insert_paper_embeddings(embeddings_batch)
    logger.info("Python batch pipeline completed: inserted %d vector chunks.", embedded_chunks_count)
    return embedded_chunks_count


if __name__ == "__main__":
    from src.openalex_client import OpenAlexClient

    client = OpenAlexClient()
    papers = client.search_works("Artificial Intelligence in Healthcare", limit=5)
    count = process_and_embed_papers(papers)
    print(f"PySpark batch pipeline complete. Processed {count} chunks.")
