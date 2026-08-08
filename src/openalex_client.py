"""
OpenAlex API client for searching academic papers, authors, topics, and open-access URLs.
Reconstructs abstracts from OpenAlex abstract_inverted_index representation.
"""

import logging
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)

OPENALEX_BASE_URL = "https://api.openalex.org/works"
USER_AGENT = "CapstoneAcademicBot/1.0 (mailto:student@example.com)"


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """
    Reconstructs full text abstract from OpenAlex inverted index dictionary:
    {"word": [pos1, pos2], ...} -> "full text abstract string"
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""

    pos_map: Dict[int, str] = {}
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                pos_map[pos] = word

    if not pos_map:
        return ""

    max_pos = max(pos_map.keys())
    ordered_words = [pos_map.get(i, "") for i in range(max_pos + 1)]
    return " ".join(w for w in ordered_words if w).strip()


class OpenAlexClient:
    def __init__(self, user_agent: str = USER_AGENT, timeout: float = 10.0):
        self.headers = {"User-Agent": user_agent, "Accept": "application/json"}
        self.timeout = timeout

    def search_works(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Searches OpenAlex works matching search query.
        Returns normalized list of paper dictionaries.
        """
        if not query or not query.strip():
            return []

        url = f"{OPENALEX_BASE_URL}?search={requests.utils.quote(query.strip())}&per-page={min(50, limit)}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code != 200:
                logger.error(f"OpenAlex API search failed ({resp.status_code}): {resp.text}")
                return []

            data = resp.json()
            results = data.get("results", [])
            normalized_papers = []

            for item in results:
                raw_id = item.get("id", "")
                paper_id = raw_id.split("/")[-1] if "/" in raw_id else raw_id
                title = item.get("display_name") or item.get("title") or "Untitled Paper"

                # Reconstruct abstract
                inv_abstract = item.get("abstract_inverted_index")
                abstract = reconstruct_abstract(inv_abstract) or f"Abstract not provided for {title}."

                doi = item.get("doi")
                year = item.get("publication_year")
                citations = item.get("cited_by_count", 0)

                # Open access URL lookup
                oa_info = item.get("open_access", {})
                oa_url = oa_info.get("oa_url") or item.get("landing_page_url")

                # Topics parsing
                topics_list = []
                for topic_obj in item.get("topics", []):
                    t_name = topic_obj.get("display_name")
                    if t_name:
                        topics_list.append(t_name)
                topics_str = ", ".join(topics_list[:5]) if topics_list else "General Academic"

                # Authors parsing
                authors = []
                for idx, authorship in enumerate(item.get("authorships", [])):
                    author_obj = authorship.get("author", {})
                    a_id = author_obj.get("id", "").split("/")[-1]
                    a_name = author_obj.get("display_name", "Unknown Author")
                    inst_list = authorship.get("institutions", [])
                    inst_name = inst_list[0].get("display_name") if inst_list else None

                    if a_id and a_name:
                        authors.append({
                            "author_id": a_id,
                            "display_name": a_name,
                            "institution": inst_name,
                            "author_position": idx + 1
                        })

                normalized_papers.append({
                    "paper_id": paper_id,
                    "doi": doi,
                    "title": title,
                    "abstract": abstract,
                    "publication_year": year,
                    "citation_count": citations,
                    "open_access_url": oa_url,
                    "topics": topics_str,
                    "authors": authors
                })

            return normalized_papers
        except Exception as e:
            logger.error(f"Error querying OpenAlex API: {e}")
            return []
