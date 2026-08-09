# Databricks notebook source
# MAGIC %md
# MAGIC # 🚀 Serverless PySpark Academic Paper Data Pipeline & Vector Embedding Index
# MAGIC 
# MAGIC This notebook demonstrates the distributed data processing and vector embedding generation pipeline for academic literature:
# MAGIC 1. **OpenAlex REST API Ingestion**: Driver-side fetching and metadata extraction.
# MAGIC 2. **PySpark DataFrame Transformations**: Automated deduplication, abstract filtering, citation bucketing, publication age computation, and prompt formatting via `src.spark_pipeline.clean_papers_df`.
# MAGIC 3. **Distributed Vector Embedding (`mapInPandas`)**: Partition-level model loading (`sentence-transformers/all-MiniLM-L6-v2`) generating 384-dimensional normalized vector embeddings.
# MAGIC 4. **Lakebase Sync & Vector Search Verification**: Atomic transactional writes into Lakebase PostgreSQL pgvector table (`capstone.paper_chunks`) and read-back verification via vector search.
# MAGIC 
# MAGIC > **Design Note**: The Streamlit web app in `app.py` deliberately uses the pure-Python in-process embedding path in `src/embedding.py`. Databricks Apps run in plain serverless Python containers without a JVM or attached cluster, while this scheduled job notebook leverages full serverless Databricks PySpark compute.

# COMMAND ----------

dbutils.widgets.text("search_query", "Graph Neural Networks for Drug Discovery")
dbutils.widgets.text("limit", "25")
dbutils.widgets.text("user_id", "scheduled-job")

search_query = dbutils.widgets.get("search_query")
limit = int(dbutils.widgets.get("limit"))
user_id = dbutils.widgets.get("user_id")

print(f"Configured Pipeline Search Query : '{search_query}'")
print(f"Paper Fetch Limit                : {limit}")
print(f"Execution User Context           : '{user_id}'")

# COMMAND ----------

import os
import sys

# Bootstrap workspace root into sys.path to enable src.* module imports
try:
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    notebook_dir = os.path.dirname(notebook_path)
    repo_root = "/Workspace" + os.path.dirname(notebook_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    print(f"Resolved repo root: {repo_root}")
    print(f"Repo contents: {os.listdir(repo_root)}")
except Exception as err:
    print(f"Warning: Workspace path resolution fallback triggered ({err})")
    if "." not in sys.path:
        sys.path.insert(0, ".")

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CapstoneSparkIngestionPipelineNotebook").getOrCreate()

print(f"Spark Engine Version : {spark.version}")
print(f"Active App Name      : {spark.conf.get('spark.app.name', 'N/A')}")
assert SparkSession.getActiveSession() is not None, "Error: Active SparkSession is required for execution."

# COMMAND ----------

from src.openalex_client import OpenAlexClient

client = OpenAlexClient()
papers = client.search_works(search_query, limit=limit)
print(f"Successfully fetched {len(papers)} paper records from OpenAlex API.")
assert len(papers) > 0, f"Assertion Failed: OpenAlex search returned 0 records for query '{search_query}'"

# COMMAND ----------

input_data = [
    {
        "paper_id": p["paper_id"],
        "title": p.get("title") or "",
        "abstract": p.get("abstract") or "",
        "publication_year": p.get("publication_year") or 2026,
        "citation_count": p.get("citation_count", 0),
    }
    for p in papers
]

df_raw = spark.createDataFrame(input_data)
print("--- Raw Input DataFrame Schema ---")
df_raw.printSchema()
display(df_raw)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🛠️ DataFrame Transformation & Feature Engineering
# MAGIC The `clean_papers_df` function applies data quality filters and structural transformations:
# MAGIC - Deduplicates records on `paper_id`.
# MAGIC - Filters out papers missing abstracts or with short abstract descriptions (< 50 chars).
# MAGIC - Computes abstract word/char length (`abstract_length`).
# MAGIC - Categorizes citation velocity into buckets (`highly_cited`, `well_cited`, `cited`, `emerging`).
# MAGIC - Computes `years_since_publication`.
# MAGIC - Constructs formatted prompt string `embed_text` directly within the DataFrame API before executor partition distribution.

# COMMAND ----------

from pyspark.sql.functions import count, avg
from src.spark_pipeline import clean_papers_df

df_clean = clean_papers_df(df_raw)

print("--- Cleaned & Transformed DataFrame Preview ---")
display(df_clean)

print("--- Citation Bucket Aggregation ---")
display(df_clean.groupBy("citation_bucket").agg(count("*").alias("paper_count"), avg("abstract_length").alias("avg_abstract_len")))

# COMMAND ----------

print("--- Formatted Physical Execution Plan ---")
df_clean.explain(mode="formatted")

# COMMAND ----------

from pyspark.sql.functions import col, slice
from src.spark_pipeline import embed_pandas_partition, EMBEDDING_SCHEMA

df_embeddings = df_clean.mapInPandas(embed_pandas_partition, schema=EMBEDDING_SCHEMA)

print(f"Embedding DataFrame RDD Partition Count: {df_embeddings.rdd.getNumPartitions()}")
print("--- Embedded Chunks (Previewing Vector Dimension Truncation) ---")
display(
    df_embeddings.select(
        "paper_id",
        "chunk_index",
        "chunk_text",
        slice(col("embedding"), 1, 8).alias("embedding_preview"),
        "model_name",
        "created_at",
    ).limit(5)
)

# COMMAND ----------

from src.db.repository import init_db, insert_paper_embeddings
from src.embedding import upsert_paper_metadata

init_db()
upsert_count = upsert_paper_metadata(papers)

pdf_res = df_embeddings.toPandas()
records = pdf_res.to_dict(orient="records") if not pdf_res.empty else []
chunks_inserted = insert_paper_embeddings(records) if records else 0

print(f"Lakebase Upsert Complete : {upsert_count} paper metadata records.")
print(f"Lakebase Insert Complete : {chunks_inserted} vector chunk embedding rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔍 Read-Back Verification & Lakebase pgvector RAG Search
# MAGIC Directly queries Lakebase PostgreSQL tables (`capstone.papers`, `capstone.paper_chunks`) and executes a vector similarity search probe using the search query embedding vector.

# COMMAND ----------

import pandas as pd
from sqlalchemy import text
from src.db.connection import get_engine
from src.db.repository import vector_search_papers
from src.embedding import generate_embedding

engine = get_engine()
with engine.connect() as conn:
    paper_count = conn.execute(text("SELECT count(*) FROM capstone.papers;")).scalar()
    chunk_count = conn.execute(text("SELECT count(*) FROM capstone.paper_chunks;")).scalar()

print(f"Lakebase DB Total Papers Saved : {paper_count}")
print(f"Lakebase DB Total Chunks Saved : {chunk_count}")

query_vec = generate_embedding(search_query)
search_results = vector_search_papers(query_vec, top_k=5)
print(f"Vector search returned {len(search_results)} top matches for query probe '{search_query}'.")

if search_results:
    res_df = pd.DataFrame([
        {
            "paper_id": r["paper_id"],
            "title": r["title"],
            "distance": round(r["distance"], 4),
            "similarity": round(r["similarity"], 4),
            "chunk_text": r["chunk_text"][:120] + "..."
        }
        for r in search_results
    ])
    display(res_df)

# COMMAND ----------

assert len(papers) > 0, "Assertion Failed: Zero papers fetched from OpenAlex."
assert chunks_inserted > 0, "Assertion Failed: Zero chunk embeddings inserted into Lakebase."
if records:
    sample_emb = records[0]["embedding"]
    from src.embedding import EMBEDDING_DIMS
    assert len(sample_emb) == EMBEDDING_DIMS, f"Assertion Failed: Embedding dimension {len(sample_emb)} != {EMBEDDING_DIMS}"
assert len(search_results) > 0, f"Assertion Failed: Empty vector search results for query probe '{search_query}'"

print("✅ Pipeline Verification Passed: Distributed ingestion, DataFrame transformation, mapInPandas embedding, Lakebase persistence, and RAG vector search all verified successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 Capstone Requirements Summary
# MAGIC 
# MAGIC | Capstone Requirement | Implementation & Verification Status |
# MAGIC |---|---|
# MAGIC | **Distributed PySpark Processing** | `clean_papers_df` DataFrame API transformations + `mapInPandas` distributed UDF embedding. |
# MAGIC | **DataFrame Transformations** | `dropDuplicates`, `filter`, `withColumn` (abstract length, citation bucket, years since publication, prompt string). |
# MAGIC | **Physical Execution Plan** | Demonstrated via `df_clean.explain(mode="formatted")`. |
# MAGIC | **Vector Embedding Generation** | 384-dimensional normalized vectors generated using `sentence-transformers/all-MiniLM-L6-v2`. |
# MAGIC | **Lakebase pgvector Persistence** | Idempotent transaction upserts into PostgreSQL `capstone.papers` and `capstone.paper_chunks`. |
# MAGIC | **Read-Back Vector Search** | Verified via `vector_search_papers` pgvector cosine distance queries. |
