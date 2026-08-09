"""Executable tool functions for the AI Research Agent.

Enables searching OpenAlex, executing pgvector RAG semantic search, writing to collections,
tracking reading progress, generating sequenced reading plans, and adding user notes.
Includes tool payload signature validation for secure execution.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from src.db.repository import (
    add_note,
    add_paper_to_collection,
    create_collection,
    get_user_collections,
    update_reading_progress,
    vector_search_papers,
)
from src.openalex_client import OpenAlexClient
from src.spark_pipeline import generate_embedding, process_and_embed_papers

logger = logging.getLogger(__name__)
openalex_client = OpenAlexClient()

VALID_PROGRESS_STATUSES = {"unread", "in_progress", "completed"}


def validate_tool_call(tool_name: str, kwargs: Dict[str, Any], user_id: str) -> None:
    """Validates parameter types, ranges, and user identity context before tool execution."""
    if not user_id or not isinstance(user_id, str) or not user_id.strip():
        raise ValueError(f"Unauthorized tool call '{tool_name}': Invalid or missing user_id context.")

    if tool_name == "tool_search_openalex_papers":
        limit = kwargs.get("limit", 5)
        if not isinstance(limit, int) or limit < 1 or limit > 50:
            raise ValueError(f"Invalid limit '{limit}' for tool_search_openalex_papers. Must be between 1 and 50.")

    elif tool_name == "tool_vector_search_papers":
        top_k = kwargs.get("top_k", 5)
        if not isinstance(top_k, int) or top_k < 1 or top_k > 50:
            raise ValueError(f"Invalid top_k '{top_k}' for tool_vector_search_papers. Must be between 1 and 50.")

    elif tool_name == "tool_track_reading_progress":
        status = kwargs.get("status", "in_progress")
        if status not in VALID_PROGRESS_STATUSES:
            raise ValueError(f"Invalid reading status '{status}'. Must be one of {VALID_PROGRESS_STATUSES}.")

    elif tool_name == "tool_add_user_note":
        content = kwargs.get("content", "")
        if not content or not isinstance(content, str) or not content.strip():
            raise ValueError("Note content cannot be empty.")


def tool_search_openalex_papers(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Tool: Searches OpenAlex API for papers matching query, ingests into Lakebase, and returns details."""
    logger.info("[Agent Tool] Searching OpenAlex for: '%s'", query)
    papers = openalex_client.search_works(query, limit=limit)
    if papers:
        process_and_embed_papers(papers)
    return papers


def tool_vector_search_papers(
    query: str, top_k: int = 5, similarity_threshold: Optional[float] = 0.3, collection_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Tool: Performs RAG pgvector cosine similarity search over paper embeddings in Lakebase."""
    logger.info("[Agent Tool] Vector searching Lakebase for: '%s'", query)
    query_vector = generate_embedding(query)
    return vector_search_papers(
        query_vector=query_vector, top_k=top_k, similarity_threshold=similarity_threshold, collection_id=collection_id
    )


def tool_add_paper_to_collection(user_id: str, collection_name: str, paper_id: str) -> Dict[str, Any]:
    """Tool: Adds a paper to a user collection in Lakebase (creates collection if not present)."""
    validate_tool_call("tool_add_paper_to_collection", {"collection_name": collection_name, "paper_id": paper_id}, user_id=user_id)
    logger.info("[Agent Tool] Adding paper '%s' to collection '%s' for user '%s'", paper_id, collection_name, user_id)
    collections = get_user_collections(user_id)
    target_coll = next((c for c in collections if c["name"].lower() == collection_name.lower()), None)
    if not target_coll:
        target_coll = create_collection(user_id=user_id, name=collection_name, description=f"Collection for {collection_name}")

    add_paper_to_collection(target_coll["collection_id"], paper_id)
    return {
        "status": "success",
        "collection_id": target_coll["collection_id"],
        "collection_name": target_coll["name"],
        "paper_id": paper_id,
    }


def tool_track_reading_progress(
    user_id: str, paper_id: str, status: str = "in_progress", sequence_order: int = 1
) -> Dict[str, Any]:
    """Tool: Updates reading progress state ('unread', 'in_progress', 'completed') in Lakebase."""
    validate_tool_call("tool_track_reading_progress", {"status": status, "paper_id": paper_id}, user_id=user_id)
    logger.info("[Agent Tool] Updating progress for user '%s', paper '%s' to status '%s'", user_id, paper_id, status)
    res = update_reading_progress(user_id=user_id, paper_id=paper_id, status=status, sequence_order=sequence_order)
    return {"status": "success", "progress": res}


def tool_generate_sequenced_reading_plan(user_id: str, goal_title: str) -> List[Dict[str, Any]]:
    """Tool: Constructs a sequenced reading plan for a learning goal using RAG paper evidence."""
    validate_tool_call("tool_generate_sequenced_reading_plan", {"goal_title": goal_title}, user_id=user_id)
    logger.info("[Agent Tool] Generating sequenced reading plan for goal: '%s'", goal_title)
    papers = tool_search_openalex_papers(goal_title, limit=5)
    if not papers:
        papers = tool_vector_search_papers(goal_title, top_k=5)

    sequenced_plan = []
    for idx, paper in enumerate(papers):
        seq_num = idx + 1
        p_id = paper["paper_id"]
        update_reading_progress(user_id=user_id, paper_id=p_id, status="unread" if idx > 0 else "in_progress", sequence_order=seq_num)
        sequenced_plan.append({
            "step": seq_num,
            "paper_id": p_id,
            "title": paper["title"],
            "abstract": paper["abstract"][:250] + "..." if len(paper.get("abstract", "")) > 250 else paper.get("abstract", ""),
            "citation_count": paper.get("citation_count", 0),
            "open_access_url": paper.get("open_access_url"),
            "status": "in_progress" if idx == 0 else "unread",
        })
    return sequenced_plan


def tool_add_user_note(
    user_id: str, content: str, paper_id: Optional[str] = None, goal_id: Optional[str] = None
) -> Dict[str, Any]:
    """Tool: Writes a student study note to Lakebase."""
    validate_tool_call("tool_add_user_note", {"content": content}, user_id=user_id)
    logger.info("[Agent Tool] Writing student note for user '%s'", user_id)
    record = add_note(user_id=user_id, content=content, paper_id=paper_id, goal_id=goal_id)
    return {"status": "success", "note": record}
