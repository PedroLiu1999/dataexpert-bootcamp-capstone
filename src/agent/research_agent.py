"""Autonomous AI Research Agent Orchestrator.

Integrates with Databricks Foundation Model Serving endpoints (e.g., databricks-meta-llama-3-3-70b-instruct)
for LLM tool calling when serving endpoints are configured, with secure payload validation and fallback execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from src.agent.tools import (
    tool_add_paper_to_collection,
    tool_add_user_note,
    tool_generate_sequenced_reading_plan,
    tool_search_openalex_papers,
    tool_track_reading_progress,
    tool_vector_search_papers,
    validate_tool_call,
)

logger = logging.getLogger(__name__)

# Default Databricks Foundation Model serving endpoint name
DEFAULT_SERVING_ENDPOINT = os.environ.get("DATABRICKS_SERVING_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")

# JSON Schema declaration for Databricks Model Serving tool calling
AGENT_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "tool_generate_sequenced_reading_plan",
            "description": "Generate a topologically sequenced reading plan for a learning goal topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_title": {"type": "string", "description": "The study topic or learning objective title"}
                },
                "required": ["goal_title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_add_paper_to_collection",
            "description": "Add a paper to a named user collection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "collection_name": {"type": "string", "description": "Target collection name"},
                    "query": {"type": "string", "description": "Search query or paper topic"}
                },
                "required": ["collection_name", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_track_reading_progress",
            "description": "Update reading progress status for a paper.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query or paper title"},
                    "status": {"type": "string", "enum": ["unread", "in_progress", "completed"]},
                },
                "required": ["query", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_add_user_note",
            "description": "Record a student study note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Note content to record"}
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_vector_search_papers",
            "description": "Search paper passages using pgvector RAG semantic search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query or research question"},
                    "top_k": {"type": "integer", "description": "Number of results to return"},
                },
                "required": ["query"],
            },
        },
    },
]


def is_databricks_fm_available() -> bool:
    """Checks if active Databricks workspace authentication environment is present."""
    return bool(
        (os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN"))
        or (os.environ.get("DATABRICKS_CLIENT_ID") and os.environ.get("DATABRICKS_CLIENT_SECRET"))
        or os.environ.get("DATABRICKS_RUNTIME_VERSION")
    )


def query_databricks_foundation_model(
    messages: List[Dict[str, Any]],
    endpoint_name: str,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Queries Databricks Model Serving endpoint via Databricks SDK WorkspaceClient with tool schemas."""
    if not is_databricks_fm_available():
        return None

    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        kwargs: Dict[str, Any] = {
            "name": endpoint_name,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.2,
        }
        if tools:
            kwargs["tools"] = tools

        response = w.serving_endpoints.query(**kwargs)
        if hasattr(response, "as_dict"):
            return response.as_dict()
        if isinstance(response, dict):
            return response
        return None
    except Exception as e:
        logger.warning("Databricks Foundation Model serving endpoint query error (%s)", e)
        return None


def classify_intent_with_entities(prompt: str) -> Dict[str, Any]:
    """Classifies user query intent and extracts target entities with robust pattern matching."""
    p_lower = prompt.lower()
    intent = "rag_synthesis"
    entities: Dict[str, Any] = {}

    if "collection" in p_lower or ("add" in p_lower and "save" in p_lower):
        intent = "add_to_collection"
        coll_name = "General Research"
        m = re.search(r"\b(?:to|in|into)(?:\s+my)?\s+(.*?)\s+collection\b", prompt, re.IGNORECASE)
        if m:
            raw_name = m.group(1).strip()
            if raw_name:
                coll_name = raw_name.title()
        else:
            m2 = re.search(r"\bcollection\s+([a-zA-Z0-9_\s]+)\b", prompt, re.IGNORECASE)
            if m2:
                raw_name = m2.group(1).strip()
                if raw_name:
                    coll_name = raw_name.title()
        entities["collection_name"] = coll_name

    elif "plan" in p_lower or "study" in p_lower or "sequence" in p_lower:
        intent = "generate_plan"
        target = prompt
        for kw in ["for", "about", "on"]:
            if f" {kw} " in p_lower:
                target = prompt.split(f" {kw} ", 1)[-1].strip()
                break
        entities["target_goal"] = target

    elif "progress" in p_lower or "complete" in p_lower or "finish" in p_lower or "mark" in p_lower:
        intent = "track_progress"

    elif "note" in p_lower or "remember" in p_lower:
        intent = "take_note"

    return {"intent": intent, "entities": entities}


class ResearchAgent:
    def __init__(self, user_id: str = "demo-user-123", serving_endpoint: Optional[str] = None):
        self.user_id = user_id
        self.serving_endpoint = serving_endpoint or DEFAULT_SERVING_ENDPOINT
        self.execution_history: List[Dict[str, Any]] = []

    def process_user_request(self, prompt: str, goal_title: Optional[str] = None) -> Dict[str, Any]:
        """Main agent invocation method. Evaluates prompt intent, invokes Foundation Model / tool orchestrator,

        validates user identity context, executes actions, and returns grounded response.
        """
        # 1. Attempt Databricks Foundation Model tool-calling invocation
        fm_response = None
        if is_databricks_fm_available():
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI Academic Research Assistant specializing in paper discovery, collection management, and study plan sequencing.",
                },
                {"role": "user", "content": prompt},
            ]
            fm_response = query_databricks_foundation_model(
                messages=messages,
                endpoint_name=self.serving_endpoint,
                tools=AGENT_TOOL_SCHEMAS,
            )

        # 2. Check if Foundation Model returned executable tool calls
        if fm_response and isinstance(fm_response, dict):
            choices = fm_response.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    return self._dispatch_fm_tool_calls(prompt, tool_calls)

        # 3. Fallback Intent & Entity Classification Tool Execution Engine
        return self._execute_classified_intent(prompt, goal_title=goal_title)

    def _dispatch_fm_tool_calls(self, prompt: str, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Dispatches and executes tool calls returned by Databricks Foundation Model endpoint."""
        actions_taken = []
        citations = []
        now_iso = datetime.now(timezone.utc).isoformat()
        responses = []

        for tc in tool_calls:
            fn = tc.get("function") or {}
            t_name = str(fn.get("name") or "tool_vector_search_papers")
            args_raw = fn.get("arguments") or "{}"

            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}

            validate_tool_call(t_name, args, user_id=self.user_id, session_user_id=self.user_id)

            if t_name == "tool_generate_sequenced_reading_plan":
                goal_title = args.get("goal_title", prompt)
                plan = tool_generate_sequenced_reading_plan(self.user_id, goal_title)
                actions_taken.append(f"Generated {len(plan)}-step sequenced study plan for '{goal_title}'.")
                self.execution_history.append({
                    "tool_name": t_name,
                    "timestamp": now_iso,
                    "parameters": {"goal_title": goal_title},
                    "status": "success",
                })
                lines = [f"### Sequenced Study Plan for *{goal_title}*\n"]
                for step in plan:
                    lines.append(f"**Step {step['step']}: {step['title']}**")
                    if step.get("open_access_url"):
                        lines.append(f"- [Read Open Access Paper]({step['open_access_url']})")
                    ab_str = (step.get("abstract") or "")[:200]
                    lines.append(f"- {ab_str}...\n")
                    citations.append({"citation_num": step["step"], "paper_id": step["paper_id"], "title": step["title"], "url": step.get("open_access_url")})
                responses.append("\n".join(lines))

            elif t_name == "tool_add_paper_to_collection":
                coll_name = args.get("collection_name", "General Research")
                query_str = args.get("query", prompt)
                search_res = tool_vector_search_papers(query_str, top_k=1) or tool_search_openalex_papers(query_str, limit=1)
                if search_res:
                    p = search_res[0]
                    tool_add_paper_to_collection(self.user_id, coll_name, p["paper_id"])
                    actions_taken.append(f"Saved paper '{p['title']}' to collection '{coll_name}'.")
                    self.execution_history.append({
                        "tool_name": t_name,
                        "timestamp": now_iso,
                        "parameters": {"collection": coll_name, "paper_id": p["paper_id"]},
                        "status": "success",
                    })
                    citations.append({"paper_id": p["paper_id"], "title": p["title"], "url": p.get("open_access_url")})
                    responses.append(f"Added **{p['title']}** to collection **'{coll_name}'**.")

            elif t_name == "tool_track_reading_progress":
                query_str = args.get("query", prompt)
                status_val = args.get("status", "completed")
                search_res = tool_vector_search_papers(query_str, top_k=1) or tool_search_openalex_papers(query_str, limit=1)
                if search_res:
                    p = search_res[0]
                    tool_track_reading_progress(self.user_id, p["paper_id"], status=status_val)
                    actions_taken.append(f"Marked paper '{p['title']}' as {status_val}.")
                    self.execution_history.append({
                        "tool_name": t_name,
                        "timestamp": now_iso,
                        "parameters": {"paper_id": p["paper_id"], "status": status_val},
                        "status": "success",
                    })
                    citations.append({"paper_id": p["paper_id"], "title": p["title"], "url": p.get("open_access_url")})
                    responses.append(f"Updated reading progress for **{p['title']}** to **{status_val.capitalize()}**.")

            elif t_name == "tool_add_user_note":
                content = args.get("content", prompt)
                tool_add_user_note(self.user_id, content)
                actions_taken.append("Recorded student study note in Lakebase.")
                self.execution_history.append({
                    "tool_name": t_name,
                    "timestamp": now_iso,
                    "parameters": {"content": content[:50]},
                    "status": "success",
                })
                responses.append(f"Recorded your study note: *\"{content}\"*.")

            else:
                query_str = args.get("query", prompt)
                rag_results = tool_vector_search_papers(query_str, top_k=4) or tool_search_openalex_papers(query_str, limit=4)
                actions_taken.append(f"Retrieved {len(rag_results)} evidence papers via pgvector semantic search.")
                lines = ["### Research Synthesis & Multi-Paper Evidence Analysis\n"]
                for idx, p in enumerate(rag_results):
                    c_num = idx + 1
                    abstract_text = p.get("abstract") or "No abstract provided."
                    lines.append(f"#### [{c_num}] {p['title']} ({p.get('publication_year', 'N/A')})")
                    lines.append(f"- **Evidence Summary**: {abstract_text[:250]}...")
                    if p.get("open_access_url"):
                        lines.append(f"- [Open Access Link]({p['open_access_url']})")
                    citations.append({"citation_num": c_num, "paper_id": p["paper_id"], "title": p["title"], "url": p.get("open_access_url")})
                responses.append("\n".join(lines))

        return {
            "query": prompt,
            "intent": "fm_tool_calling",
            "entities": {},
            "agent_rationale": f"Databricks Foundation Model endpoint '{self.serving_endpoint}' executed tool calling flow.",
            "response": "\n\n".join(responses),
            "actions_taken": actions_taken,
            "citations": citations,
            "tool_history": self.execution_history,
        }

    def _execute_classified_intent(self, prompt: str, goal_title: Optional[str] = None) -> Dict[str, Any]:
        """Fallback Intent & Entity Classification Tool Execution Engine."""
        classification = classify_intent_with_entities(prompt)
        intent = classification["intent"]
        entities = classification["entities"]

        actions_taken = []
        citations = []
        response_text = ""
        now_iso = datetime.now(timezone.utc).isoformat()
        rationale = ""

        # 1. Action: Add to collection intent
        if intent == "add_to_collection":
            coll_name = entities.get("collection_name", "General Research")
            rationale = f"Detected collection management intent for collection '{coll_name}'. Invoking tool_vector_search_papers to retrieve best matching paper."

            search_res = tool_vector_search_papers(prompt, top_k=1)
            if not search_res:
                search_res = tool_search_openalex_papers(prompt, limit=1)

            if search_res:
                p = search_res[0]
                tool_add_paper_to_collection(self.user_id, coll_name, p["paper_id"])
                act_msg = f"Saved paper '{p['title']}' to collection '{coll_name}'."
                actions_taken.append(act_msg)
                self.execution_history.append({
                    "tool_name": "tool_add_paper_to_collection",
                    "timestamp": now_iso,
                    "parameters": {"collection": coll_name, "paper_id": p["paper_id"]},
                    "status": "success",
                })
                citations.append({"paper_id": p["paper_id"], "title": p["title"], "url": p.get("open_access_url")})
                response_text = f"💡 *Agent Rationale*: {rationale}\n\nAdded **{p['title']}** to your collection **'{coll_name}'**.\n\n"
            else:
                response_text = f"💡 *Agent Rationale*: {rationale}\n\nCould not find a paper matching query '{prompt}' to add to collection.\n\n"

        # 2. Action: Sequenced Study / Reading Plan intent
        elif intent == "generate_plan":
            target_goal = goal_title or entities.get("target_goal", prompt)
            rationale = f"Classified goal sequence intent for topic '{target_goal}'. Invoking tool_generate_sequenced_reading_plan."
            plan = tool_generate_sequenced_reading_plan(self.user_id, target_goal)
            act_msg = f"Generated {len(plan)}-step sequenced reading plan for '{target_goal}'."
            actions_taken.append(act_msg)
            self.execution_history.append({
                "tool_name": "tool_generate_sequenced_reading_plan",
                "timestamp": now_iso,
                "parameters": {"goal_title": target_goal},
                "status": "success",
            })

            lines = [f"💡 *Agent Rationale*: {rationale}\n", f"### Sequenced Study Plan for *{target_goal}*\n"]
            for step in plan:
                lines.append(f"**Step {step['step']}: {step['title']}** [{step['step']}]")
                lines.append(f"- *Status*: `{step['status']}` | *Citations*: {step['citation_count']}")
                if step.get("open_access_url"):
                    lines.append(f"- [Read Open Access Paper]({step['open_access_url']})")
                ab_str = (step.get("abstract") or "")[:200]
                lines.append(f"- {ab_str}...\n")

                citations.append({
                    "citation_num": step["step"],
                    "paper_id": step["paper_id"],
                    "title": step["title"],
                    "url": step.get("open_access_url"),
                })
            response_text = "\n".join(lines)

        # 3. Action: Progress tracking intent
        elif intent == "track_progress":
            rationale = "Recognized reading progress update request. Searching Lakebase pgvector storage for target paper."
            matches = tool_vector_search_papers(prompt, top_k=1)
            if not matches:
                matches = tool_search_openalex_papers(prompt, limit=1)

            if matches:
                p = matches[0]
                tool_track_reading_progress(self.user_id, p["paper_id"], status="completed")
                act_msg = f"Marked paper '{p['title']}' as completed."
                actions_taken.append(act_msg)
                self.execution_history.append({
                    "tool_name": "tool_track_reading_progress",
                    "timestamp": now_iso,
                    "parameters": {"paper_id": p["paper_id"], "status": "completed"},
                    "status": "success",
                })
                response_text = f"💡 *Agent Rationale*: {rationale}\n\nUpdated reading progress: Marked **{p['title']}** as **Completed**! Recommended next paper in sequence is ready in your plan."
                citations.append({"paper_id": p["paper_id"], "title": p["title"], "url": p.get("open_access_url")})
            else:
                response_text = f"💡 *Agent Rationale*: {rationale}\n\nNo matching paper found for query '{prompt}'. Retrying search or checking user reading progress."

        # 4. Action: Note taking intent
        elif intent == "take_note":
            rationale = "Recognized note creation request. Executing tool_add_user_note to persist in Lakebase."
            tool_add_user_note(self.user_id, prompt)
            act_msg = "Recorded student study note in Lakebase."
            actions_taken.append(act_msg)
            self.execution_history.append({
                "tool_name": "tool_add_user_note",
                "timestamp": now_iso,
                "parameters": {"content": prompt[:50]},
                "status": "success",
            })
            response_text = f"💡 *Agent Rationale*: {rationale}\n\nRecorded your study note in Lakebase: *\"{prompt}\"*."

        # 5. Default RAG Research Evidence Retrieval & Synthesis with Grounding Rationale
        else:
            rationale = "Selected multi-paper RAG vector search & synthesis tool to retrieve evidence passages."
            rag_results = tool_vector_search_papers(prompt, top_k=4)
            if not rag_results:
                rag_results = tool_search_openalex_papers(prompt, limit=4)

            act_msg = f"Retrieved {len(rag_results)} evidence papers via pgvector semantic search."
            actions_taken.append(act_msg)
            self.execution_history.append({
                "tool_name": "tool_vector_search_papers",
                "timestamp": now_iso,
                "parameters": {"query": prompt, "top_k": 4},
                "status": "success",
            })

            lines = [f"💡 *Agent Rationale*: {rationale}\n", "### Research Synthesis & Multi-Paper Evidence Analysis\n"]
            for idx, p in enumerate(rag_results):
                c_num = idx + 1
                sim = p.get("similarity")
                abstract_text = p.get("abstract") or "No abstract provided."
                chunk_snippet = p.get("chunk_text") or abstract_text
                lines.append(f"#### [{c_num}] {p['title']} ({p.get('publication_year', 'N/A')})")
                if sim is not None:
                    lines.append(f"- **Similarity Score**: `{float(sim):.2f}`")
                lines.append(f"- **Grounding Rationale**: Matched passage: *\"{chunk_snippet[:180]}...\"*")
                lines.append(f"- **Evidence Summary**: {abstract_text[:250]}...")
                if p.get("open_access_url"):
                    lines.append(f"- [Open Access Link]({p['open_access_url']})")
                lines.append("")

                citations.append({
                    "citation_num": c_num,
                    "paper_id": p["paper_id"],
                    "title": p["title"],
                    "url": p.get("open_access_url"),
                    "similarity": sim,
                })

            response_text = "\n".join(lines)

        return {
            "query": prompt,
            "intent": intent,
            "entities": entities,
            "agent_rationale": rationale,
            "response": response_text,
            "actions_taken": actions_taken,
            "citations": citations,
            "tool_history": self.execution_history,
        }
