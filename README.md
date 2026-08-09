# Capstone Project: Academic Research & Personalized Study Plan Assistant

An end-to-end academic research discovery, vector embedding, and personalized study plan sequencing system built for **Databricks Apps** with **OpenAlex API**, **Serverless PySpark Ingestion Pipeline**, **Lakebase (PostgreSQL + pgvector)**, **Autonomous AI Research Agent**, and **Streamlit UI**.

---

## 1. Architecture & Core Components

- **Lakebase PostgreSQL & Connection Pooling**:
  - `src/db/connection.py`: Thread-safe OAuth database token rotation via `WorkspaceClient().postgres.generate_database_credential()` (`_CredentialCache`), process-wide pooled SQLAlchemy engine (`postgresql+psycopg://`, `pool_size=5`, `max_overflow=5`, `pool_recycle=1800`, `pool_pre_ping=True`).
  - `migrations/001_init.sql`: Idempotent relational DDL schema bootstrap with `TIMESTAMPTZ` timestamps, foreign key `ON DELETE CASCADE` constraints, composite unique indexes, and `events` analytics storage.
- **Serverless PySpark Data Ingestion Pipeline**:
  - `src/spark_pipeline.py`: Distributed paper metadata ingestion and vector embedding pipeline. Uses `mapPartitions` for per-partition model loading (`sentence-transformers/all-MiniLM-L6-v2`, $N=384$ vector dimensions) without hardcoded Spark master strings, collecting batch records on the driver for atomic Lakebase transactional upserts.
- **OpenAlex REST API Discovery Client**:
  - `src/openalex_client.py`: High-throughput academic research search with `primary_location.landing_page_url` fallback, payload size optimization via `select=`, empty query result caching, and LRU cache eviction.
- **Vector Search Precision & HNSW Index**:
  - `src/db/repository.py`: `vector_search_papers` using `DISTINCT ON (paper_id)` deduplication, configurable similarity threshold filtering (`similarity_threshold=0.3`), and HNSW vector index (`idx_paper_chunks_embedding USING hnsw (embedding vector_cosine_ops)`).
- **Pedagogical Sequenced Reading Plan Engine**:
  - `src/agent/tools.py`: Constructs a directed citation graph ($G=(V,E)$) from `referenced_works` metadata and executes a topological sort to order foundational prerequisite papers before derivative advanced papers in student reading plans.
- **Autonomous AI Research Agent**:
  - `src/agent/research_agent.py`: Integrates with Databricks Foundation Model Serving endpoints (e.g. `databricks-meta-llama-3-3-70b-instruct`) with strict tool parameter type validation (`validate_tool_call`) and fallback intent orchestration.
- **Persistent Event Analytics**:
  - `src/analytics/delta_cdf.py`: Logs domain operational events (tool calls, progress updates, collection saves) into Lakebase `capstone.events` table for persistent real-time metrics tracking.
- **Databricks Apps Streamlit Frontend**:
  - `app.py`: Multi-tenant UI with native Streamlit components (`st.container(border=True)`, `st.link_button`), XSS sanitization, `@st.cache_resource` database schema bootstrap, and signed-in user identity extraction from request headers (`X-Forwarded-Email`).

---

## 2. Lakebase Domain Schema (9 Tables + Analytics)

1. `users`: User profiles and identities (`user_id`, `email`, `full_name`, `role`).
2. `learning_goals`: Student learning objectives and target levels.
3. `papers`: Academic works metadata (DOI, title, abstract, citations, open-access link).
4. `authors`: Researcher details and affiliations.
5. `paper_authors`: Junction table linking papers to authors with position index.
6. `collections`: User paper libraries.
7. `collection_papers`: Papers saved to collections.
8. `reading_progress`: Sequenced reading status (`unread`, `in_progress`, `completed`).
9. `notes`: Student study notes linked to papers and learning goals.
10. `paper_chunks`: Overlapping text passages with 384-dim vector embeddings and HNSW index.
11. `events`: Persistent analytics audit log (`event_id`, `event_type`, `user_id`, `payload`).

---

## 3. Quickstart & Testing

### Local Environment Setup (`uv`)
```bash
cd /home/peter/databricks-ai-boot-camp/capstone
uv sync
```

### Run Automated Pytest Suite (PostgreSQL Testcontainers)
```bash
uv run pytest
```

### Validate Code Quality & Linting
```bash
uv run ruff check .
```

### Validate Databricks Asset Bundle
```bash
databricks bundle validate
```

### Launch Streamlit Application Locally
```bash
uv run streamlit run app.py
```

### Deploy to Databricks Apps
```bash
databricks bundle deploy --target dev
```

### GitHub Actions CI/CD Secrets Setup (`gh` CLI)
To configure workspace authentication secrets for GitHub Actions CI bundle validation:

```bash
# 1. Set Databricks Workspace Host URL
gh secret set DATABRICKS_HOST --repo PedroLiu1999/dataexpert-bootcamp-capstone --body "https://dbc-117d1e6a-753a.cloud.databricks.com"

# 2. Set Databricks Authentication Token
gh secret set DATABRICKS_TOKEN --repo PedroLiu1999/dataexpert-bootcamp-capstone --body "$(databricks auth token | jq -r .access_token)"

# 3. Verify Repository Secrets
gh secret list --repo PedroLiu1999/dataexpert-bootcamp-capstone
```

