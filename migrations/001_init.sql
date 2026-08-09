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
    topic VARCHAR(255) NOT NULL,
    target_date VARCHAR(64),
    status VARCHAR(32) DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Papers
CREATE TABLE IF NOT EXISTS capstone.papers (
    paper_id VARCHAR(64) PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    publication_year INT,
    citation_count INT DEFAULT 0,
    open_access_url TEXT,
    embedding vector(384),
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
    rating INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_user_paper UNIQUE (user_id, paper_id)
);

-- Notes
CREATE TABLE IF NOT EXISTS capstone.notes (
    note_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES capstone.users(user_id) ON DELETE CASCADE,
    paper_id VARCHAR(64) NOT NULL REFERENCES capstone.papers(paper_id) ON DELETE CASCADE,
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
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
