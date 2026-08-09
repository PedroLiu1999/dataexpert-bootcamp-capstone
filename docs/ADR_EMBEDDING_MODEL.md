# Architectural Decision Record: Embedding Serving Model & Vector Dimension

## Context
The research paper vector search system requires text embeddings for paper titles, abstracts, and chunked text.
We evaluated two primary embedding serving options:
1. **SentenceTransformers (`all-MiniLM-L6-v2`)**: 384 dimensions. Local PyTorch execution.
2. **Databricks Model Serving Endpoint (`bge-large-en` / `gte-large-en`)**: 1024 dimensions. Serverless managed endpoint.

## Decision
We select **SentenceTransformers `all-MiniLM-L6-v2` with $N=384$ dimensions** as the primary single-source-of-truth embedding model for the capstone system.

### Single Source of Truth Constants:
- **Vector Dimension**: `384`
- **Python Constant**: `src.embedding.EMBEDDING_DIMS = 384`
- **PostgreSQL DDL**: `vector(384)` in `capstone.papers` and `capstone.paper_chunks`

### Rationale:
- **Performance & Cold-Start**: 384-dimensional vectors provide high precision with minimal memory footprint on Lakebase pgvector HNSW indexes.
- **Serverless Spark Compatibility**: Executing `mapInPandas` with a single model load per partition ensures high throughput without external HTTP network dependency per batch.
- **Single Source of Truth**: All schema DDL (`migrations/001_init.sql`), Spark transformations, and repository validation assert `EMBEDDING_DIMS == 384`.

## Consequences
- If upgrading to a 1024-dim Foundation Model endpoint in the future, a schema migration script must alter table columns from `vector(384)` to `vector(1024)` and rebuild the HNSW index.
