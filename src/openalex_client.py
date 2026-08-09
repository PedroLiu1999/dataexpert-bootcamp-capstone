"""OpenAlex API client for searching academic papers, authors, topics, and open-access URLs.

Reconstructs abstracts from OpenAlex abstract_inverted_index representation.
"""

from __future__ import annotations

from functools import lru_cache
import logging
import os
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger(__name__)

OPENALEX_BASE_URL = "https://api.openalex.org/works"
DEFAULT_USER_AGENT = "CapstoneAcademicBot/1.0 (mailto:student@databricks.com)"

# Select only necessary fields from OpenAlex API to reduce network payload size.
# Includes referenced_works for citation graph sequencing in T13/T14.
OPENALEX_SELECT_FIELDS = (
    "id,doi,display_name,publication_year,cited_by_count,open_access,"
    "primary_location,abstract_inverted_index,authorships,topics,referenced_works"
)


def get_user_agent() -> str:
    return os.environ.get("OPENALEX_USER_AGENT", DEFAULT_USER_AGENT)


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Reconstructs full text abstract from OpenAlex inverted index dictionary:

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
    def __init__(
        self,
        user_agent: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        cache_maxsize: int = 256,
    ):
        ua = user_agent or get_user_agent()
        self.headers = {"User-Agent": ua, "Accept": "application/json"}
        self.timeout = timeout
        self.cache_maxsize = cache_maxsize
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

        self.session = requests.Session()
        self.session.headers.update(self.headers)
        retries = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def search_works(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Searches OpenAlex works matching search query.

        Returns normalized list of paper dictionaries. Includes caching & field selection.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        cache_key = f"{clean_query.lower()}::{limit}"
        if cache_key in self._cache:
            logger.info("Returning cached OpenAlex results for query: '%s'", clean_query)
            return self._cache[cache_key]

        quoted_query = requests.utils.quote(clean_query)
        url = (
            f"{OPENALEX_BASE_URL}?search={quoted_query}"
            f"&per-page={min(50, limit)}&select={OPENALEX_SELECT_FIELDS}"
        )
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 429:
                logger.warning(
                    "OpenAlex API rate limit exceeded (HTTP 429). Retries exhausted or backoff active."
                )
                return []
            elif resp.status_code != 200:
                logger.error("OpenAlex API search failed (%s): %s", resp.status_code, resp.text)
                return []

            data = resp.json()
            results = data.get("results", [])
            normalized_papers: List[Dict[str, Any]] = []

            for item in results:
                raw_id = item.get("id", "")
                paper_id = raw_id.split("/")[-1] if "/" in raw_id else raw_id
                title = item.get("display_name") or item.get("title") or "Untitled Paper"

                inv_abstract = item.get("abstract_inverted_index")
                abstract = reconstruct_abstract(inv_abstract) or None

                doi = item.get("doi")
                year = item.get("publication_year")
                citations = item.get("cited_by_count", 0)

                # Open access URL lookup: check open_access.oa_url first, then primary_location.landing_page_url
                oa_info = item.get("open_access", {}) or {}
                primary_loc = item.get("primary_location", {}) or {}
                oa_url = oa_info.get("oa_url") or primary_loc.get("landing_page_url")

                topics_list = []
                for topic_obj in item.get("topics", []) or []:
                    t_name = topic_obj.get("display_name")
                    if t_name:
                        topics_list.append(t_name)
                topics_str = ", ".join(topics_list[:5]) if topics_list else "General Academic"

                authors = []
                for idx, authorship in enumerate(item.get("authorships", []) or []):
                    author_obj = authorship.get("author", {}) or {}
                    a_id = author_obj.get("id", "").split("/")[-1]
                    a_name = author_obj.get("display_name", "Unknown Author")
                    inst_list = authorship.get("institutions", []) or []
                    inst_name = inst_list[0].get("display_name") if inst_list else None

                    if a_id and a_name:
                        authors.append(
                            {
                                "author_id": a_id,
                                "display_name": a_name,
                                "institution": inst_name,
                                "author_position": idx + 1,
                            }
                        )

                referenced_works = [
                    w.split("/")[-1] for w in (item.get("referenced_works", []) or [])
                ]

                normalized_papers.append(
                    {
                        "paper_id": paper_id,
                        "doi": doi,
                        "title": title,
                        "abstract": abstract,
                        "publication_year": year,
                        "citation_count": citations,
                        "open_access_url": oa_url,
                        "topics": topics_str,
                        "authors": authors,
                        "referenced_works": referenced_works,
                    }
                )

            # Eviction policy: trim cache if exceeding maxsize
            if len(self._cache) >= self.cache_maxsize:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            # Cache results (including empty result sets)
            self._cache[cache_key] = normalized_papers
            return normalized_papers
        except Exception as e:
            logger.error("Error querying OpenAlex API: %s", e)
            return []
