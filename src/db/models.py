"""
SQLAlchemy ORM models and DDL definitions for Lakebase PostgreSQL schema.
Includes 9 core business domain tables + paper_embeddings vector storage table.
"""

from datetime import datetime, timezone
import uuid
from typing import List, Optional
from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class UserRecord(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), default=lambda: datetime.now(timezone.utc).isoformat())

    learning_goals: Mapped[List["LearningGoalRecord"]] = relationship("LearningGoalRecord", back_populates="user", cascade="all, delete-orphan")
    collections: Mapped[List["CollectionRecord"]] = relationship("CollectionRecord", back_populates="user", cascade="all, delete-orphan")
    reading_progress: Mapped[List["ReadingProgressRecord"]] = relationship("ReadingProgressRecord", back_populates="user", cascade="all, delete-orphan")
    notes: Mapped[List["NoteRecord"]] = relationship("NoteRecord", back_populates="user", cascade="all, delete-orphan")


class LearningGoalRecord(Base):
    __tablename__ = "learning_goals"

    goal_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    target_level: Mapped[str] = mapped_column(String(64), default="Intermediate")
    created_at: Mapped[str] = mapped_column(String(64), default=lambda: datetime.now(timezone.utc).isoformat())

    user: Mapped["UserRecord"] = relationship("UserRecord", back_populates="learning_goals")
    notes: Mapped[List["NoteRecord"]] = relationship("NoteRecord", back_populates="goal")


class PaperRecord(Base):
    __tablename__ = "papers"

    paper_id: Mapped[str] = mapped_column(String(128), primary_key=True)  # OpenAlex Work ID (e.g. W2741809807)
    doi: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, nullable=False)
    publication_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    citation_count: Mapped[int] = mapped_column(Integer, default=0)
    open_access_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    topics: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Comma-separated or JSON list of topics
    created_at: Mapped[str] = mapped_column(String(64), default=lambda: datetime.now(timezone.utc).isoformat())

    paper_authors: Mapped[List["PaperAuthorRecord"]] = relationship("PaperAuthorRecord", back_populates="paper", cascade="all, delete-orphan")
    collection_papers: Mapped[List["CollectionPaperRecord"]] = relationship("CollectionPaperRecord", back_populates="paper", cascade="all, delete-orphan")
    reading_progress: Mapped[List["ReadingProgressRecord"]] = relationship("ReadingProgressRecord", back_populates="paper", cascade="all, delete-orphan")
    notes: Mapped[List["NoteRecord"]] = relationship("NoteRecord", back_populates="paper")


class AuthorRecord(Base):
    __tablename__ = "authors"

    author_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    institution: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    paper_authors: Mapped[List["PaperAuthorRecord"]] = relationship("PaperAuthorRecord", back_populates="author", cascade="all, delete-orphan")


class PaperAuthorRecord(Base):
    __tablename__ = "paper_authors"

    paper_id: Mapped[str] = mapped_column(String(128), ForeignKey("papers.paper_id", ondelete="CASCADE"), primary_key=True)
    author_id: Mapped[str] = mapped_column(String(128), ForeignKey("authors.author_id", ondelete="CASCADE"), primary_key=True)
    author_position: Mapped[int] = mapped_column(Integer, default=1)

    paper: Mapped["PaperRecord"] = relationship("PaperRecord", back_populates="paper_authors")
    author: Mapped["AuthorRecord"] = relationship("AuthorRecord", back_populates="paper_authors")


class CollectionRecord(Base):
    __tablename__ = "collections"

    collection_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(64), default=lambda: datetime.now(timezone.utc).isoformat())

    user: Mapped["UserRecord"] = relationship("UserRecord", back_populates="collections")
    collection_papers: Mapped[List["CollectionPaperRecord"]] = relationship("CollectionPaperRecord", back_populates="collection", cascade="all, delete-orphan")


class CollectionPaperRecord(Base):
    __tablename__ = "collection_papers"

    collection_id: Mapped[str] = mapped_column(String(64), ForeignKey("collections.collection_id", ondelete="CASCADE"), primary_key=True)
    paper_id: Mapped[str] = mapped_column(String(128), ForeignKey("papers.paper_id", ondelete="CASCADE"), primary_key=True)
    added_at: Mapped[str] = mapped_column(String(64), default=lambda: datetime.now(timezone.utc).isoformat())

    collection: Mapped["CollectionRecord"] = relationship("CollectionRecord", back_populates="collection_papers")
    paper: Mapped["PaperRecord"] = relationship("PaperRecord", back_populates="collection_papers")


class ReadingProgressRecord(Base):
    __tablename__ = "reading_progress"

    progress_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    paper_id: Mapped[str] = mapped_column(String(128), ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="unread")  # 'unread', 'in_progress', 'completed'
    sequence_order: Mapped[int] = mapped_column(Integer, default=1)
    rating: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1-5 rating scale
    updated_at: Mapped[str] = mapped_column(String(64), default=lambda: datetime.now(timezone.utc).isoformat())

    user: Mapped["UserRecord"] = relationship("UserRecord", back_populates="reading_progress")
    paper: Mapped["PaperRecord"] = relationship("PaperRecord", back_populates="reading_progress")


class NoteRecord(Base):
    __tablename__ = "notes"

    note_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    paper_id: Mapped[Optional[str]] = mapped_column(String(128), ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=True)
    goal_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("learning_goals.goal_id", ondelete="CASCADE"), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), default=lambda: datetime.now(timezone.utc).isoformat())

    user: Mapped["UserRecord"] = relationship("UserRecord", back_populates="notes")
    paper: Mapped[Optional["PaperRecord"]] = relationship("PaperRecord", back_populates="notes")
    goal: Mapped[Optional["LearningGoalRecord"]] = relationship("LearningGoalRecord", back_populates="notes")


# DDL statements for native PostgreSQL pgvector tables
CREATE_TABLES_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(64) PRIMARY KEY,
    email VARCHAR(128) UNIQUE NOT NULL,
    full_name VARCHAR(128) NOT NULL,
    created_at VARCHAR(64) NOT NULL
);

CREATE TABLE IF NOT EXISTS learning_goals (
    goal_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    target_level VARCHAR(64) DEFAULT 'Intermediate',
    created_at VARCHAR(64) NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id VARCHAR(128) PRIMARY KEY,
    doi VARCHAR(255),
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    publication_year INT,
    citation_count INT DEFAULT 0,
    open_access_url TEXT,
    topics TEXT,
    created_at VARCHAR(64) NOT NULL
);

CREATE TABLE IF NOT EXISTS authors (
    author_id VARCHAR(128) PRIMARY KEY,
    display_name VARCHAR(255) NOT NULL,
    institution VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id VARCHAR(128) NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    author_id VARCHAR(128) NOT NULL REFERENCES authors(author_id) ON DELETE CASCADE,
    author_position INT DEFAULT 1,
    PRIMARY KEY (paper_id, author_id)
);

CREATE TABLE IF NOT EXISTS collections (
    collection_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(128) NOT NULL,
    description TEXT,
    created_at VARCHAR(64) NOT NULL
);

CREATE TABLE IF NOT EXISTS collection_papers (
    collection_id VARCHAR(64) NOT NULL REFERENCES collections(collection_id) ON DELETE CASCADE,
    paper_id VARCHAR(128) NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    added_at VARCHAR(64) NOT NULL,
    PRIMARY KEY (collection_id, paper_id)
);

CREATE TABLE IF NOT EXISTS reading_progress (
    progress_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    paper_id VARCHAR(128) NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    status VARCHAR(32) DEFAULT 'unread',
    sequence_order INT DEFAULT 1,
    rating INT,
    updated_at VARCHAR(64) NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    note_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    paper_id VARCHAR(128) REFERENCES papers(paper_id) ON DELETE CASCADE,
    goal_id VARCHAR(64) REFERENCES learning_goals(goal_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at VARCHAR(64) NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_embeddings (
    id BIGSERIAL PRIMARY KEY,
    paper_id VARCHAR(128) NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    created_at VARCHAR(64) NOT NULL,
    CONSTRAINT uq_paper_chunk UNIQUE (paper_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_paper_embeddings_hnsw
ON paper_embeddings USING hnsw (embedding vector_cosine_ops);
"""
