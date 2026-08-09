"""PySpark Data Ingestion & Vector Embedding Pipeline for Academic Papers.

Cleans raw OpenAlex paper metadata, chunks abstracts/text, computes 384-dim vector embeddings
via PySpark DataFrame mapInPandas transforms (Serverless compatible), and batch-syncs records into Lakebase.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Iterator

# Ensure workspace root is present in sys.path for standalone PySpark task execution
if "__file__" in globals():
    root_dir = str(Path(__file__).resolve().parents[1])
else:
    root_dir = str(Path.cwd())

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


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


def process_and_embed_papers(papers_list: List[Dict[str, Any]]) -> int:
    """Ingests and embeds paper records into Lakebase.

    Uses PySpark DataFrame mapInPandas API (serverless compatible) for partition-level model loading and driver batch write,
    falling back to Python batch execution if PySpark environment is unavailable.
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

    # 2. PySpark mapInPandas DataFrame execution (Serverless compatible)
    try:
        import pandas as pd
        from pyspark.sql import SparkSession
        from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, StructField, StructType

        # Attach to active Spark session on serverless / cluster
        spark = SparkSession.builder.appName("CapstoneSparkNativePipeline").getOrCreate()

        logger.info("PySpark session active. Parallelizing embeddings via mapInPandas DataFrame transform...")

        schema = StructType([
            StructField("paper_id", StringType(), False),
            StructField("chunk_index", IntegerType(), False),
            StructField("chunk_text", StringType(), False),
            StructField("embedding", ArrayType(FloatType()), False),
            StructField("model_name", StringType(), False),
            StructField("created_at", StringType(), False),
        ])

        # Prepare simple driver records for DataFrame conversion
        input_data = [
            {
                "paper_id": p["paper_id"],
                "title": p.get("title") or "",
                "abstract": p.get("abstract") or "",
            }
            for p in papers_list
        ]
        df_input = spark.createDataFrame(input_data)

        def _embed_pandas_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
            model = get_embedding_model()
            iso_timestamp = datetime.now(timezone.utc).isoformat()

            for pdf in iterator:
                records = []
                for _, row in pdf.iterrows():
                    p_id = str(row["paper_id"])
                    p_title = str(row.get("title", ""))
                    p_abstract = str(row.get("abstract", ""))
                    text_to_embed = f"Title: {p_title}\n\nAbstract: {p_abstract}"
                    chunks = chunk_text(text_to_embed)

                    for idx, c_str in enumerate(chunks):
                        vec = model.encode(c_str, normalize_embeddings=True).tolist()
                        records.append({
                            "paper_id": p_id,
                            "chunk_index": idx,
                            "chunk_text": c_str,
                            "embedding": vec,
                            "model_name": MODEL_NAME,
                            "created_at": iso_timestamp,
                        })
                yield pd.DataFrame(records)

        df_embeddings = df_input.mapInPandas(_embed_pandas_partition, schema=schema)
        pdf_res = df_embeddings.toPandas()
        all_embeddings = pdf_res.to_dict(orient="records") if not pdf_res.empty else []

        if all_embeddings:
            embedded_chunks_count = insert_paper_embeddings(all_embeddings)
        else:
            embedded_chunks_count = 0

        logger.info("PySpark mapInPandas pipeline completed: inserted %d vector chunks.", embedded_chunks_count)
        return embedded_chunks_count

    except ImportError:
        logger.warning("PySpark/Pandas package not installed. Running Python batch embedding pipeline.")
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
