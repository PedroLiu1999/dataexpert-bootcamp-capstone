"""PySpark Data Ingestion & Vector Embedding Pipeline for Academic Papers.

Cleans raw OpenAlex paper metadata, transforms DataFrames, computes 384-dim vector embeddings
via PySpark DataFrame mapInPandas transforms, and batch-syncs records into Lakebase.
Requires an active PySpark session on Databricks compute.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Iterator
import pandas as pd

# Ensure workspace root is present in sys.path for standalone PySpark task execution
if "__file__" in globals():
    root_dir = str(Path(__file__).resolve().parents[1])
else:
    root_dir = str(Path.cwd())

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, StructField, StructType
from pyspark.sql.functions import col, concat, concat_ws, length, lit, trim, when

from src.db.repository import init_db, insert_paper_embeddings
from src.embedding import (
    MODEL_NAME,
    chunk_text,
    get_embedding_model,
    upsert_paper_metadata,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_SCHEMA = StructType([
    StructField("paper_id", StringType(), False),
    StructField("chunk_index", IntegerType(), False),
    StructField("chunk_text", StringType(), False),
    StructField("embedding", ArrayType(FloatType()), False),
    StructField("model_name", StringType(), False),
    StructField("created_at", StringType(), False),
])


def embed_pandas_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Pandas UDF partition handler for distributed text chunking & vector embedding."""
    model = get_embedding_model()
    iso_timestamp = datetime.now(timezone.utc).isoformat()

    for pdf in iterator:
        records = []
        for _, row in pdf.iterrows():
            p_id = str(row["paper_id"])
            embed_text = str(row.get("embed_text") or "")
            if not embed_text and ("title" in row or "abstract" in row):
                p_title = str(row.get("title") or "")
                p_abstract = str(row.get("abstract") or "")
                embed_text = f"Title: {p_title}\n\nAbstract: {p_abstract}"

            chunks = chunk_text(embed_text)

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
        yield pd.DataFrame(records, columns=[f.name for f in EMBEDDING_SCHEMA.fields])


def clean_papers_df(df: DataFrame) -> DataFrame:
    """DataFrame transformation stage for cleaning and enriching paper metadata."""
    current_year = datetime.now(timezone.utc).year
    return (
        df.dropDuplicates(["paper_id"])
        .filter(col("abstract").isNotNull() & (length(trim(col("abstract"))) > 50))
        .withColumn("abstract_length", length(col("abstract")))
        .withColumn(
            "citation_bucket",
            when(col("citation_count") >= 1000, "highly_cited")
            .when(col("citation_count") >= 100, "well_cited")
            .when(col("citation_count") >= 10, "cited")
            .otherwise("emerging"),
        )
        .withColumn("years_since_publication", lit(current_year) - col("publication_year"))
        .withColumn(
            "embed_text",
            concat_ws("\n\n", concat(lit("Title: "), col("title")), concat(lit("Abstract: "), col("abstract"))),
        )
        .repartition(4)
    )


def spark_embed_papers(spark: SparkSession | None, papers_list: List[Dict[str, Any]]) -> int:
    """Distributed PySpark paper metadata transformation and mapInPandas embedding pipeline.

    Requires an active Spark session attached to Databricks compute.
    """
    if spark is None or SparkSession.getActiveSession() is None:
        raise RuntimeError(
            "src.spark_pipeline requires an active Spark session on Databricks compute. "
            "Use src.embedding.process_and_embed_papers for in-process local/app execution."
        )

    init_db()
    if not papers_list:
        return 0

    logger.info("Ingesting %d paper records via PySpark...", len(papers_list))
    upsert_paper_metadata(papers_list)

    input_data = [
        {
            "paper_id": p["paper_id"],
            "title": p.get("title") or "",
            "abstract": p.get("abstract") or "",
            "publication_year": p.get("publication_year") or datetime.now(timezone.utc).year,
            "citation_count": p.get("citation_count", 0),
        }
        for p in papers_list
    ]
    df_input = spark.createDataFrame(input_data)
    df_clean = clean_papers_df(df_input)

    df_embeddings = df_clean.mapInPandas(embed_pandas_partition, schema=EMBEDDING_SCHEMA)
    pdf_res = df_embeddings.toPandas()
    all_embeddings = pdf_res.to_dict(orient="records") if not pdf_res.empty else []

    if all_embeddings:
        embedded_chunks_count = insert_paper_embeddings(all_embeddings)
    else:
        embedded_chunks_count = 0

    logger.info("PySpark mapInPandas pipeline completed: inserted %d vector chunks.", embedded_chunks_count)
    return embedded_chunks_count
