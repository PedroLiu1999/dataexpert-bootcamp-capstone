# Architectural Decision Record: Separation of Spark Execution Context & Web Application Ingestion

## Status
Accepted

## Context
The deployed Databricks App crashed during interactive paper ingestion with:
```
pyspark.errors.exceptions.base.PySparkRuntimeError: [JAVA_GATEWAY_EXITED]
Java gateway process exited before sending its port number.
```
Databricks Apps run in a standard serverless Python container without Java (`JAVA_HOME`), a JVM runtime, or an attached Spark cluster. When `pyspark` was included as a runtime dependency in `pyproject.toml`, calling `SparkSession.builder.getOrCreate()` defaulted to local Spark mode and attempted to launch a non-existent Java gateway process.

## Options Considered

1. **Local Spark in Databricks App Container**:
   - Install JRE/JVM into the container runtime.
   - *Rejected*: Databricks Apps environment images are managed by the platform and do not allow custom system/JVM packages or heavy local Spark driver processes.

2. **Databricks Connect Serverless from App Container**:
   - Connect remotely to a Databricks serverless compute endpoint using `databricks-connect`.
   - *Rejected*: `databricks-connect` cannot coexist with standard `pyspark` in the same virtual environment, requires configuring a dedicated service principal with compute attach entitlements, and introduces 10–30 seconds of cold-start latency to every interactive user search request.

3. **Separated Dual-Path Execution (Selected)**:
   - Move interactive ingestion to pure-Python in-process execution in `src/embedding.py` with zero PySpark dependencies.
   - Restrict Spark execution to dedicated Databricks compute (scheduled job notebook `notebooks/spark_ingestion_pipeline.py` and `src/spark_pipeline.py`).

## Decision
We adopt **Option 3 (Separated Dual-Path Execution)**:
- **Interactive App Ingestion**: The Streamlit web application (`app.py`) imports `src.embedding`, performing sequential text chunking, `SentenceTransformer` vector encoding, and Lakebase upserts in-process. `pyspark` is completely removed from the App runtime container dependencies.
- **Scheduled Distributed Batch Job**: Scheduled ingestion runs via `notebooks/spark_ingestion_pipeline.py` on Databricks compute. `src/spark_pipeline.py` requires an active Spark session (`SparkSession.getActiveSession()`) and performs physical DataFrame cleaning (`clean_papers_df`) and distributed partition-level embedding (`mapInPandas`).

## Consequences
- **App Reliability**: Eliminates `PySparkRuntimeError` / `JAVA_GATEWAY_EXITED` crashes in Databricks Apps.
- **Performance**: Sub-second latency for interactive search and ingestion without Spark session startup overhead.
- **Capstone Compliance**: Full PySpark capstone requirements (DataFrame transformations, execution plan, `mapInPandas` UDFs, Lakebase write & read-back) are demonstrated cleanly in the scheduled Databricks job notebook.
