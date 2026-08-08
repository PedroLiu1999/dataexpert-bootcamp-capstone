"""
Pytest Test Suite for Capstone Academic Research & Study Plan Assistant.
Tests OpenAlex client abstract reconstruction, Lakebase repository CRUD across 9 domain tables,
PySpark ingestion & embedding pipeline, and AI Agent action tools.
"""

from unittest.mock import patch, MagicMock
import pytest
from src.agent.research_agent import ResearchAgent
from src.agent.tools import (
    tool_add_paper_to_collection,
    tool_add_user_note,
    tool_generate_sequenced_reading_plan,
    tool_search_openalex_papers,
    tool_track_reading_progress,
)
from src.db.repository import (
    add_note,
    add_paper_to_collection,
    create_collection,
    create_learning_goal,
    create_user,
    get_collection_papers,
    get_user_collections,
    get_user_learning_goals,
    get_user_notes,
    get_user_reading_progress,
    init_db,
    update_reading_progress,
    upsert_paper,
    vector_search_papers,
)
from src.openalex_client import OpenAlexClient, reconstruct_abstract
from src.spark_pipeline import chunk_text, generate_embedding, process_and_embed_papers


def test_openalex_abstract_reconstruction():
    inverted_index = {
        "Graph": [0],
        "neural": [1],
        "networks": [2],
        "enable": [3],
        "molecular": [4],
        "property": [5],
        "prediction.": [6],
    }
    abstract = reconstruct_abstract(inverted_index)
    assert abstract == "Graph neural networks enable molecular property prediction."


def test_openalex_client_search():
    client = OpenAlexClient()
    client._cache.clear()

    mock_openalex_response = {
        "results": [
            {
                "id": "https://openalex.org/W999999",
                "display_name": "Attention Is All You Need",
                "publication_year": 2017,
                "cited_by_count": 80000,
                "landing_page_url": "https://arxiv.org/abs/1706.03762",
                "abstract_inverted_index": {"Transformers": [0], "replace": [1], "recurrent": [2], "layers.": [3]},
                "authorships": [
                    {
                        "author": {"id": "https://openalex.org/A111", "display_name": "Ashish Vaswani"},
                        "institutions": [{"display_name": "Google Brain"}],
                    }
                ],
            }
        ]
    }

    with patch.object(client.session, "get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_openalex_response
        mock_get.return_value = mock_resp

        results = client.search_works("Transformers", limit=5)
        assert len(results) == 1
        paper = results[0]
        assert paper["paper_id"] == "W999999"
        assert paper["title"] == "Attention Is All You Need"
        assert "Transformers replace recurrent layers." in paper["abstract"]
        assert len(paper["authors"]) == 1
        assert paper["authors"][0]["display_name"] == "Ashish Vaswani"


def test_lakebase_repository_all_tables():
    init_db()

    # 1. User
    user = create_user("student@test.com", "Test Student")
    assert user["email"] == "student@test.com"

    # 2. Learning Goal
    goal = create_learning_goal(user["user_id"], "Deep Learning", "Master GNNs and Transformers")
    assert goal["title"] == "Deep Learning"

    # 3. Paper
    paper = upsert_paper("W12345", "GNN Paper", "Graph Neural Networks Abstract", citation_count=50)
    assert paper["paper_id"] == "W12345"

    # 4. Collection & Collection Paper
    coll = create_collection(user["user_id"], "GNN Collection")
    assert coll["name"] == "GNN Collection"
    added = add_paper_to_collection(coll["collection_id"], paper["paper_id"])
    assert added is True

    coll_papers = get_collection_papers(coll["collection_id"])
    assert len(coll_papers) >= 1

    # 5. Reading Progress
    prog = update_reading_progress(user["user_id"], paper["paper_id"], status="in_progress", sequence_order=1)
    assert prog["status"] == "in_progress"

    # 6. Notes
    note = add_note(user["user_id"], "Interesting node message passing mechanism.", paper_id=paper["paper_id"])
    assert note["content"] == "Interesting node message passing mechanism."


def test_spark_pipeline_and_embeddings():
    test_text = "PySpark pipeline ingesting academic literature and vector embeddings."
    chunks = chunk_text(test_text, chunk_size=800)
    assert len(chunks) == 1

    vec = generate_embedding(test_text)
    assert len(vec) == 384
    assert isinstance(vec[0], float)

    mock_papers = [
        {
            "paper_id": "W777777",
            "title": "Quantum Optimization Methods",
            "abstract": "Quantum computing algorithms for combinatorial optimization problems.",
            "publication_year": 2026,
            "citation_count": 12,
            "open_access_url": "https://arxiv.org/abs/quantum",
            "topics": "Quantum Computing",
        }
    ]

    count = process_and_embed_papers(mock_papers)
    assert count >= 1

    search_res = vector_search_papers(vec, top_k=5)
    assert len(search_res) >= 1


def test_agent_tools_and_orchestrator():
    user = create_user("agent_test@test.com", "Agent Tester")
    agent = ResearchAgent(user_id=user["user_id"])

    # Test Action: Generate study plan
    res_plan = agent.process_user_request("Build a sequenced study plan for Quantum Optimization")
    assert "Sequenced Study Plan" in res_plan["response"]
    assert len(res_plan["citations"]) >= 1

    # Test Action: Add to collection
    res_coll = agent.process_user_request("Add this paper to Quantum Collection")
    assert "collection" in res_coll["response"].lower() or "added" in res_coll["response"].lower()

    # Test Action: Track progress
    res_prog = agent.process_user_request("Mark Quantum Optimization paper as completed")
    assert "progress" in res_prog["response"].lower() or "completed" in res_prog["response"].lower()


def test_idempotent_embeddings_upsert():
    from src.db.repository import insert_paper_embeddings, _MOCK_EMBEDDINGS
    init_db()

    data_v1 = [{
        "paper_id": "W_IDEM_1",
        "chunk_index": 0,
        "chunk_text": "Original text v1",
        "embedding": [0.1] * 384,
        "model_name": "all-MiniLM-L6-v2",
        "created_at": "2026-08-08T00:00:00Z"
    }]
    insert_paper_embeddings(data_v1)

    data_v2 = [{
        "paper_id": "W_IDEM_1",
        "chunk_index": 0,
        "chunk_text": "Updated text v2",
        "embedding": [0.2] * 384,
        "model_name": "all-MiniLM-L6-v2",
        "created_at": "2026-08-08T00:01:00Z"
    }]
    insert_paper_embeddings(data_v2)

    # Verify no duplicate entries created for (paper_id, chunk_index)
    matches = [e for e in _MOCK_EMBEDDINGS if e["paper_id"] == "W_IDEM_1" and e["chunk_index"] == 0]
    assert len(matches) == 1
    assert matches[0]["chunk_text"] == "Updated text v2"


def test_openalex_retry_and_backoff():
    client = OpenAlexClient(max_retries=2, backoff_factor=0.01)

    # Test 429 rate limit fallback
    with patch.object(client.session, "get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_get.return_value = mock_resp

        res = client.search_works("retry test", limit=2)
        assert res == []
        assert mock_get.called

    # Test caching behavior
    client_cached = OpenAlexClient()
    with patch.object(client_cached.session, "get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": [{"id": "W11", "title": "Cached Title"}]}
        mock_get.return_value = mock_resp

        # First call fetches via session.get
        res1 = client_cached.search_works("caching query", limit=2)
        assert len(res1) == 1
        assert res1[0]["title"] == "Cached Title"
        assert mock_get.call_count == 1

        # Second call returns cached data without calling session.get
        res2 = client_cached.search_works("caching query", limit=2)
        assert len(res2) == 1
        assert res2[0]["title"] == "Cached Title"
        assert mock_get.call_count == 1


def test_delta_cdf_analytics():
    from src.analytics.delta_cdf import cdf_tracker

    user_id = "test-user-cdf"
    cdf_tracker.log_event("tool_call", user_id, {"tool_name": "tool_generate_sequenced_reading_plan"})
    cdf_tracker.log_event("tool_call", user_id, {"tool_name": "tool_add_paper_to_collection"})
    cdf_tracker.log_event("progress_update", user_id, {"paper_id": "W1", "status": "completed"})
    cdf_tracker.log_event("progress_update", user_id, {"paper_id": "W2", "status": "in_progress"})

    analytics = cdf_tracker.get_cdf_analytics(user_id=user_id)
    assert analytics["plans_generated"] >= 1
    assert analytics["papers_added"] >= 1
    assert analytics["completed_reading_count"] >= 1
    assert analytics["completion_rate_pct"] == 50.0
    assert analytics["cdf_enabled"] is True


def test_persist_authorship_data():
    from src.db.repository import upsert_author, upsert_paper_author, _MOCK_AUTHORS, _MOCK_PAPER_AUTHORS

    a_rec = upsert_author("A_TEST_1", "Geoffrey Hinton", "University of Toronto")
    assert a_rec["display_name"] == "Geoffrey Hinton"
    assert _MOCK_AUTHORS["A_TEST_1"]["institution"] == "University of Toronto"

    pa_rec = upsert_paper_author("W_TEST_PAPER", "A_TEST_1", author_position=1)
    assert pa_rec["paper_id"] == "W_TEST_PAPER"
    assert len(_MOCK_PAPER_AUTHORS) >= 1




