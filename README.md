# Capstone Project: Academic Research & Personalized Study Plan Assistant

An end-to-end academic research discovery, vector embedding, and personalized study plan sequencing system built with **OpenAlex API**, **PySpark Data Ingestion Pipeline**, **Lakebase (Postgres + pgvector)**, **Autonomous AI Research Agent**, and **Streamlit / Databricks Apps**.

---

## 1. Architecture & Capstone Requirements Coverage

- **Spark Data Pipeline**: `src/spark_pipeline.py` runs PySpark batch transformations to clean raw OpenAlex paper metadata, chunk text, compute 384-dimensional vector embeddings, and write structured records into Lakebase pgvector.
- **Third-Party API Integration**: `src/openalex_client.py` connects to OpenAlex (`https://api.openalex.org/works`) for works, authors, institutions, topics, citations, and inverted index abstract reconstruction.
- **Processing Unstructured Data**: Text abstracts and student notes are vector-embedded using `sentence-transformers/all-MiniLM-L6-v2` (384-dim) for multi-paper RAG evidence retrieval.
- **9 Required Lakebase Domain Tables**:
  - `users`: User accounts and profiles.
  - `learning_goals`: Target topics, objectives, and target levels.
  - `papers`: Academic works metadata (DOI, title, abstract, citations, open-access link).
  - `authors`: Researcher details and affiliations.
  - `paper_authors`: Many-to-many paper author links.
  - `collections`: User paper libraries.
  - `collection_papers`: Papers saved to collections.
  - `reading_progress`: Sequenced paper reading status (`unread`, `in_progress`, `completed`).
  - `notes`: Student study notes.
  - `paper_embeddings`: 384-dim vector embeddings with HNSW cosine distance index (`CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`).
- **AI Agent Capabilities**:
  - `tool_search_openalex_papers`: Live API research search.
  - `tool_vector_search_papers`: RAG pgvector semantic search.
  - `tool_add_paper_to_collection`: Database write action saving papers to collections.
  - `tool_generate_sequenced_reading_plan`: Sequenced study plan generator.
  - `tool_track_reading_progress`: Database write action tracking paper progress.
  - `tool_add_user_note`: Database write action storing student study notes.
- **Databricks App Frontend**: Streamlit UI ([app.py](file:///home/peter/databricks-ai-boot-camp/capstone/app.py)) with interactive tabs for learning goal discovery, collections, sequenced reading plan progress, and AI assistant chatbot.

---

## 2. Quickstart & Execution

### Local Environment Setup (`uv`)
```bash
cd /home/peter/databricks-ai-boot-camp/capstone
uv sync
```

### Run Automated Pytest Suite
```bash
uv run pytest
```

### Validate Databricks Asset Bundle
```bash
databricks bundle validate
```

### Launch Streamlit Application Locally
```bash
uv run streamlit run app.py
```

### Deploy to Databricks
```bash
databricks bundle deploy --target dev
```
