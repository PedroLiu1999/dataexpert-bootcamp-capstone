"""
PySpark Data Ingestion & Vector Embedding Pipeline for Academic Papers.
Cleans raw OpenAlex paper metadata, chunks abstracts/text, computes 384-dim vector embeddings,
and syncs structured records into Lakebase (papers & paper_embeddings).
"""

from datetime import datetime, timezone
import hashlib
import logging
import math
import os
import sys
from typing import Any, Dict, List

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.db.repository import init_db, insert_paper_embeddings, upsert_paper, upsert_author, upsert_paper_author

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_DIMS = 384

_model_instance = None


def get_embedding_model():
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading SentenceTransformer model '{MODEL_NAME}'...")
    _model_instance = SentenceTransformer(MODEL_NAME)
    return _model_instance


def generate_embedding(text: str) -> List[float]:
    model = get_embedding_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
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


def process_and_embed_papers(papers_list: List[Dict[str, Any]]) -> int:
    """
    Ingests and embeds paper records into Lakebase.
    Uses PySpark distributed DataFrame transformations when Spark is active,
    falling back to Python batch processing when Spark is unavailable.
    """
    init_db()
    if not papers_list:
        return 0

    logger.info(f"Ingesting {len(papers_list)} paper records...")
    now_iso = datetime.now(timezone.utc).isoformat()

    # 1. Upsert paper metadata records into Lakebase
    for paper in papers_list:
        upsert_paper(
            paper_id=paper["paper_id"],
            title=paper["title"],
            abstract=paper["abstract"],
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

    # Try PySpark distributed execution
    spark_success = False
    embedded_chunks_count = 0

    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import udf, explode, col
        from pyspark.sql.types import ArrayType, StringType, StructType, StructField, IntegerType

        spark = SparkSession.builder \
            .appName("CapstoneSparkNativePipeline") \
            .master("local[*]") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .getOrCreate()

        logger.info("PySpark session active. Parallelizing chunking and embedding across Spark partitions...")

        # Create Spark RDD / DataFrame of papers
        rdd = spark.sparkContext.parallelize(papers_list)

        def process_paper_partition(paper_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
            p_id = paper_dict["paper_id"]
            p_title = paper_dict["title"]
            p_abstract = paper_dict["abstract"]
            text_to_embed = f"Title: {p_title}\n\nAbstract: {p_abstract}"
            chunks = chunk_text(text_to_embed)

            result = []
            for idx, c_str in enumerate(chunks):
                vec = generate_embedding(c_str)
                result.append({
                    "paper_id": p_id,
                    "chunk_index": idx,
                    "chunk_text": c_str,
                    "embedding": vec,
                    "model_name": MODEL_NAME,
                    "created_at": now_iso
                })
            return result

        # Distributed flatMap across Spark partitions
        embeddings_rdd = rdd.flatMap(process_paper_partition)
        all_embeddings = embeddings_rdd.collect()

        if all_embeddings:
            embedded_chunks_count = insert_paper_embeddings(all_embeddings)
        spark_success = True
        logger.info(f"PySpark distributed pipeline completed: stored {embedded_chunks_count} vector chunks.")
    except Exception as e:
        logger.info(f"PySpark distributed pipeline fallback ({e}); using Python batch processor.")

    if not spark_success:
        embeddings_batch = []
        for paper in papers_list:
            p_id = paper["paper_id"]
            text_to_embed = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
            chunks = chunk_text(text_to_embed)
            for idx, chunk_str in enumerate(chunks):
                vector_vec = generate_embedding(chunk_str)
                embeddings_batch.append({
                    "paper_id": p_id,
                    "chunk_index": idx,
                    "chunk_text": chunk_str,
                    "embedding": vector_vec,
                    "model_name": MODEL_NAME,
                    "created_at": now_iso
                })

        embedded_chunks_count = insert_paper_embeddings(embeddings_batch)
        logger.info(f"Python batch pipeline completed: stored {embedded_chunks_count} vector chunks.")

    return embedded_chunks_count


if __name__ == "__main__":
    from src.openalex_client import OpenAlexClient
    client = OpenAlexClient()
    papers = client.search_works("Artificial Intelligence in Healthcare", limit=5)
    count = process_and_embed_papers(papers)
    print(f"PySpark batch pipeline complete. Processed {count} chunks.")
