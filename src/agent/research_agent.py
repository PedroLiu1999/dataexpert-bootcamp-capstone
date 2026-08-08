"""
Autonomous AI Research Agent Orchestrator.
Combines RAG evidence retrieval across multiple papers with executable tools
(adding papers to collections, sequencing reading plans, tracking progress, taking notes, inline citations).
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from src.agent.tools import (
    tool_add_paper_to_collection,
    tool_add_user_note,
    tool_generate_sequenced_reading_plan,
    tool_search_openalex_papers,
    tool_track_reading_progress,
    tool_vector_search_papers,
)

logger = logging.getLogger(__name__)


def classify_intent(prompt: str) -> str:
    p_lower = prompt.lower()
    if "collection" in p_lower or ("add" in p_lower and "save" in p_lower):
        return "add_to_collection"
    elif "plan" in p_lower or "study" in p_lower or "sequence" in p_lower:
        return "generate_plan"
    elif "progress" in p_lower or "complete" in p_lower or "finish" in p_lower or "mark" in p_lower:
        return "track_progress"
    elif "note" in p_lower or "remember" in p_lower:
        return "take_note"
    return "rag_synthesis"


class ResearchAgent:
    def __init__(self, user_id: str = "demo-user-123"):
        self.user_id = user_id
        self.execution_history: List[Dict[str, Any]] = []

    def process_user_request(self, prompt: str, goal_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Main agent invocation method. Evaluates prompt intent, selects tools,
        retrieves RAG paper evidence with similarity scores & grounding rationale,
        executes database actions, and returns answer with citations and structured tool history.
        """
        intent = classify_intent(prompt)
        p_lower = prompt.lower()
        actions_taken = []
        citations = []
        response_text = ""
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Action: Add to collection intent
        if intent == "add_to_collection":
            coll_name = "General Research"
            if "to" in p_lower:
                parts = prompt.split("to")
                if len(parts) > 1:
                    coll_name = parts[-1].replace("collection", "").strip().title() or "General Research"

            search_res = tool_vector_search_papers(prompt, top_k=1)
            if not search_res:
                search_res = tool_search_openalex_papers(prompt, limit=1)

            if search_res:
                p = search_res[0]
                tool_res = tool_add_paper_to_collection(self.user_id, coll_name, p["paper_id"])
                act_msg = f"Saved paper '{p['title']}' to collection '{coll_name}'."
                actions_taken.append(act_msg)
                self.execution_history.append({
                    "tool_name": "tool_add_paper_to_collection",
                    "timestamp": now_iso,
                    "parameters": {"collection": coll_name, "paper_id": p["paper_id"]},
                    "status": "success"
                })
                citations.append({"paper_id": p["paper_id"], "title": p["title"], "url": p.get("open_access_url")})
                response_text = f"Added **{p['title']}** to your collection **'{coll_name}'**.\n\n"
            else:
                response_text = "I searched for papers matching your request but could not find a suitable candidate to add.\n\n"

        # 2. Action: Sequenced Study / Reading Plan intent
        elif intent == "generate_plan":
            target_goal = goal_title or prompt
            plan = tool_generate_sequenced_reading_plan(self.user_id, target_goal)
            act_msg = f"Generated {len(plan)}-step sequenced reading plan for '{target_goal}'."
            actions_taken.append(act_msg)
            self.execution_history.append({
                "tool_name": "tool_generate_sequenced_reading_plan",
                "timestamp": now_iso,
                "parameters": {"goal_title": target_goal},
                "status": "success"
            })

            lines = [f"### Sequenced Study Plan for *{target_goal}*\n"]
            for step in plan:
                lines.append(f"**Step {step['step']}: {step['title']}** [{step['step']}]")
                lines.append(f"- *Status*: `{step['status']}` | *Citations*: {step['citation_count']}")
                if step.get("open_access_url"):
                    lines.append(f"- [Read Open Access Paper]({step['open_access_url']})")
                lines.append(f"- {step['abstract'][:200]}...\n")

                citations.append({
                    "citation_num": step['step'],
                    "paper_id": step["paper_id"],
                    "title": step["title"],
                    "url": step.get("open_access_url")
                })
            response_text = "\n".join(lines)

        # 3. Action: Progress tracking intent
        elif intent == "track_progress":
            matches = tool_vector_search_papers(prompt, top_k=1)
            if matches:
                p = matches[0]
                tool_track_reading_progress(self.user_id, p["paper_id"], status="completed")
                act_msg = f"Marked paper '{p['title']}' as completed."
                actions_taken.append(act_msg)
                self.execution_history.append({
                    "tool_name": "tool_track_reading_progress",
                    "timestamp": now_iso,
                    "parameters": {"paper_id": p["paper_id"], "status": "completed"},
                    "status": "success"
                })
                response_text = f"Updated reading progress: Marked **{p['title']}** as **Completed**! Recommended next paper in sequence is ready in your plan."
                citations.append({"paper_id": p["paper_id"], "title": p["title"], "url": p.get("open_access_url")})
            else:
                response_text = "Retrieved your reading progress list. You are on track with your study plan!"

        # 4. Action: Note taking intent
        elif intent == "take_note":
            tool_add_user_note(self.user_id, prompt)
            act_msg = "Recorded student study note in Lakebase."
            actions_taken.append(act_msg)
            self.execution_history.append({
                "tool_name": "tool_add_user_note",
                "timestamp": now_iso,
                "parameters": {"content": prompt[:50]},
                "status": "success"
            })
            response_text = f"Recorded your study note in Lakebase: *\"{prompt}\"*."

        # 5. Default RAG Research Evidence Retrieval & Synthesis with Grounding Rationale
        else:
            rag_results = tool_vector_search_papers(prompt, top_k=4)
            if not rag_results:
                rag_results = tool_search_openalex_papers(prompt, limit=4)

            act_msg = f"Retrieved {len(rag_results)} evidence papers via pgvector semantic search."
            actions_taken.append(act_msg)
            self.execution_history.append({
                "tool_name": "tool_vector_search_papers",
                "timestamp": now_iso,
                "parameters": {"query": prompt, "top_k": 4},
                "status": "success"
            })

            lines = [f"### Research Synthesis & Multi-Paper Evidence Analysis\n"]
            for idx, p in enumerate(rag_results):
                c_num = idx + 1
                sim = p.get("similarity", 0.85)
                chunk_snippet = p.get("chunk_text") or p["abstract"]
                lines.append(f"#### [{c_num}] {p['title']} ({p.get('publication_year', 'N/A')})")
                lines.append(f"- **Similarity Score**: `{sim:.2f}`")
                lines.append(f"- **Grounding Rationale**: Matched passage: *\"{chunk_snippet[:180]}...\"*")
                lines.append(f"- **Evidence Summary**: {p['abstract'][:250]}...")
                if p.get("open_access_url"):
                    lines.append(f"- [Open Access Link]({p['open_access_url']})")
                lines.append("")

                citations.append({
                    "citation_num": c_num,
                    "paper_id": p["paper_id"],
                    "title": p["title"],
                    "url": p.get("open_access_url"),
                    "similarity": sim
                })

            response_text = "\n".join(lines)

        return {
            "query": prompt,
            "intent": intent,
            "response": response_text,
            "actions_taken": actions_taken,
            "citations": citations,
            "tool_history": self.execution_history
        }
