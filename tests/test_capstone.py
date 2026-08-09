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
    insert_paper_embeddings,
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
                "open_access": {"oa_url": None},
                "primary_location": {"landing_page_url": "https://arxiv.org/abs/1706.03762"},
                "abstract_inverted_index": {"Transformers": [0], "replace": [1], "recurrent": [2], "layers.": [3]},
                "authorships": [
                    {
                        "author": {"id": "https://openalex.org/A111", "display_name": "Ashish Vaswani"},
                        "institutions": [{"display_name": "Google Brain"}],
                    }
                ],
                "referenced_works": ["https://openalex.org/W111111"],
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
        assert paper["open_access_url"] == "https://arxiv.org/abs/1706.03762"
        assert paper["referenced_works"] == ["W111111"]
        assert "select=" in mock_get.call_args[0][0]

        # Verify caching works for second call
        cached_results = client.search_works("Transformers", limit=5)
        assert cached_results == results
        assert mock_get.call_count == 1


def test_openalex_client_empty_result_caching():
    client = OpenAlexClient()
    client._cache.clear()

    with patch.object(client.session, "get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}
        mock_get.return_value = mock_resp

        res1 = client.search_works("NonExistentQueryXYZ", limit=5)
        assert res1 == []
        res2 = client.search_works("NonExistentQueryXYZ", limit=5)
        assert res2 == []
        assert mock_get.call_count == 1


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

    quantum_vec = generate_embedding(mock_papers[0]["abstract"])
    search_res = vector_search_papers(quantum_vec, top_k=5)
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
    from src.db.repository import insert_paper_embeddings, upsert_paper, vector_search_papers
    init_db()

    upsert_paper("W_IDEM_1", "Idempotent Test Paper", "Abstract text")

    data_v1 = [{
        "paper_id": "W_IDEM_1",
        "chunk_index": 0,
        "chunk_text": "Original text v1",
        "embedding": [0.1] * 384,
        "model_name": "all-MiniLM-L6-v2",
        "created_at": "2026-08-08T00:00:00Z"
    }]
    count1 = insert_paper_embeddings(data_v1)
    assert count1 == 1

    data_v2 = [{
        "paper_id": "W_IDEM_1",
        "chunk_index": 0,
        "chunk_text": "Updated text v2",
        "embedding": [0.2] * 384,
        "model_name": "all-MiniLM-L6-v2",
        "created_at": "2026-08-08T00:01:00Z"
    }]
    count2 = insert_paper_embeddings(data_v2)
    assert count2 == 1

    matches = vector_search_papers([0.2] * 384, top_k=5)
    idem_match = [m for m in matches if m["paper_id"] == "W_IDEM_1"]
    assert len(idem_match) == 1
    assert idem_match[0]["chunk_text"] == "Updated text v2"



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
    from src.db.repository import upsert_author, upsert_paper_author, upsert_paper

    upsert_paper("W_TEST_PAPER", "Test Title", "Test Abstract")
    a_rec = upsert_author("A_TEST_1", "Geoffrey Hinton", "University of Toronto")
    assert a_rec["display_name"] == "Geoffrey Hinton"
    assert a_rec["institution"] == "University of Toronto"

    pa_rec = upsert_paper_author("W_TEST_PAPER", "A_TEST_1", author_position=1)
    assert pa_rec["paper_id"] == "W_TEST_PAPER"
    assert pa_rec["author_id"] == "A_TEST_1"




def test_connection_engine_missing_env():
    import pytest
    import src.db.connection as conn_mod

    old_engine = conn_mod._engine
    try:
        conn_mod._engine = None
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="PGHOST is not set"):
                conn_mod.get_engine()
    finally:
        conn_mod._engine = old_engine



def test_credential_cache_minting():
    from src.db.connection import _CredentialCache

    mock_ws = MagicMock()
    mock_cred = MagicMock()
    mock_cred.token = "mock_token_abc123"
    mock_cred.expire_time = "2030-01-01T00:00:00Z"
    mock_ws.postgres.generate_database_credential.return_value = mock_cred

    with patch("src.db.connection.WorkspaceClient", return_value=mock_ws):
        cache = _CredentialCache("projects/test/branches/dev/endpoints/primary")
        token = cache.token()
        assert token == "mock_token_abc123"
        mock_ws.postgres.generate_database_credential.assert_called_once_with(
            endpoint="projects/test/branches/dev/endpoints/primary"
        )



def test_html_xss_sanitization():
    import html

    malicious_title = "<img src=x onerror=alert(1)>"
    escaped_title = html.escape(malicious_title)
    assert "<img" not in escaped_title
    assert "&lt;img src=x onerror=alert(1)&gt;" == escaped_title


def test_user_identity_isolation():
    u1 = create_user("alice@company.com", "Alice Smith")
    u2 = create_user("bob@company.com", "Bob Jones")

    assert u1["user_id"] != u2["user_id"]

    c1 = create_collection(u1["user_id"], "Alice Collection")
    c2 = create_collection(u2["user_id"], "Bob Collection")

    alice_colls = get_user_collections(u1["user_id"])
    bob_colls = get_user_collections(u2["user_id"])

    assert len(alice_colls) == 1
    assert alice_colls[0]["name"] == "Alice Collection"
    assert len(bob_colls) == 1
    assert bob_colls[0]["name"] == "Bob Collection"


def test_vector_search_similarity_threshold_filtering():
    init_db()
    upsert_paper("W_NO_MATCH", "Orthogonal Paper", "Physics abstract")
    insert_paper_embeddings([{
        "paper_id": "W_NO_MATCH",
        "chunk_index": 0,
        "chunk_text": "Physics passage",
        "embedding": [1.0] + [0.0] * 383,
        "model_name": "all-MiniLM-L6-v2"
    }])

    # Search with an orthogonal query vector and high threshold (0.99)
    query_vec = [0.0] * 383 + [1.0]
    results = vector_search_papers(query_vec, top_k=5, similarity_threshold=0.99)
    assert len(results) == 0


def test_vector_search_distinct_paper_deduplication():
    init_db()
    upsert_paper("W_MULTI_CHUNK", "Multi Chunk Paper", "Long paper abstract")
    chunks = [
        {"paper_id": "W_MULTI_CHUNK", "chunk_index": 0, "chunk_text": "Chunk 0", "embedding": [0.5] * 384},
        {"paper_id": "W_MULTI_CHUNK", "chunk_index": 1, "chunk_text": "Chunk 1", "embedding": [0.51] * 384},
        {"paper_id": "W_MULTI_CHUNK", "chunk_index": 2, "chunk_text": "Chunk 2", "embedding": [0.49] * 384},
    ]
    insert_paper_embeddings(chunks)

    results = vector_search_papers([0.5] * 384, top_k=5, similarity_threshold=0.1)
    matching = [r for r in results if r["paper_id"] == "W_MULTI_CHUNK"]
    # DISTINCT ON (paper_id) ensures exactly 1 row returned per paper
    assert len(matching) == 1


def test_long_document_multi_chunking():
    long_abstract = "Graph Neural Networks (GNNs) represent a powerful class of deep learning models designed for non-Euclidean domain data. " * 45
    assert len(long_abstract) > 4000

    chunks = chunk_text(long_abstract, chunk_size=800, chunk_overlap=100)
    assert len(chunks) >= 4


