-- Capstone Minimal Schema Bootstrap & Identity Grants Migration
-- Vector dimension N=384 per ADR (docs/ADR_EMBEDDING_MODEL.md)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS capstone;

-- Users table
CREATE TABLE IF NOT EXISTS capstone.users (
    user_id VARCHAR(64) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(32) DEFAULT 'student',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Learning Goals
CREATE TABLE IF NOT EXISTS capstone.learning_goals (
    goal_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES capstone.users(user_id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    target_level VARCHAR(64) DEFAULT 'Intermediate',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Papers
CREATE TABLE IF NOT EXISTS capstone.papers (
    paper_id VARCHAR(64) PRIMARY KEY,
    doi VARCHAR(255),
    title TEXT NOT NULL,
    abstract TEXT,
    publication_year INT,
    citation_count INT DEFAULT 0,
    open_access_url TEXT,
    topics TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Authors
CREATE TABLE IF NOT EXISTS capstone.authors (
    author_id VARCHAR(64) PRIMARY KEY,
    display_name VARCHAR(255) NOT NULL,
    institution VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Paper Authors Junction
CREATE TABLE IF NOT EXISTS capstone.paper_authors (
    paper_id VARCHAR(64) NOT NULL REFERENCES capstone.papers(paper_id) ON DELETE CASCADE,
    author_id VARCHAR(64) NOT NULL REFERENCES capstone.authors(author_id) ON DELETE CASCADE,
    author_position INT DEFAULT 1,
    PRIMARY KEY (paper_id, author_id)
);

-- Collections
CREATE TABLE IF NOT EXISTS capstone.collections (
    collection_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES capstone.users(user_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Collection Papers Junction
CREATE TABLE IF NOT EXISTS capstone.collection_papers (
    collection_id VARCHAR(64) NOT NULL REFERENCES capstone.collections(collection_id) ON DELETE CASCADE,
    paper_id VARCHAR(64) NOT NULL REFERENCES capstone.papers(paper_id) ON DELETE CASCADE,
    added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (collection_id, paper_id)
);

-- Reading Progress
CREATE TABLE IF NOT EXISTS capstone.reading_progress (
    progress_id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES capstone.users(user_id) ON DELETE CASCADE,
    paper_id VARCHAR(64) NOT NULL REFERENCES capstone.papers(paper_id) ON DELETE CASCADE,
    status VARCHAR(32) DEFAULT 'unread',
    sequence_order INT DEFAULT 1,
    rating INT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_user_paper UNIQUE (user_id, paper_id)
);

-- Notes
CREATE TABLE IF NOT EXISTS capstone.notes (
    note_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES capstone.users(user_id) ON DELETE CASCADE,
    paper_id VARCHAR(64) REFERENCES capstone.papers(paper_id) ON DELETE CASCADE,
    goal_id VARCHAR(64) REFERENCES capstone.learning_goals(goal_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Paper Chunks
CREATE TABLE IF NOT EXISTS capstone.paper_chunks (
    chunk_id VARCHAR(64) PRIMARY KEY,
    paper_id VARCHAR(64) NOT NULL REFERENCES capstone.papers(paper_id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(384),
    model_name VARCHAR(128) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_paper_chunk UNIQUE (paper_id, chunk_index)
);

-- HNSW Vector Index for Cosine Similarity
CREATE INDEX IF NOT EXISTS idx_paper_chunks_embedding
ON capstone.paper_chunks USING hnsw (embedding vector_cosine_ops);
