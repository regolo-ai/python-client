import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from regolo import RegoloClient


@pytest.fixture
def smoke_runner():
    path = Path(__file__).resolve().parents[1] / "scripts/smoke_inference.py"
    spec = importlib.util.spec_from_file_location("smoke_inference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "model,tagged",
    [
        ("Qwen3-Reranker-4B", True),
        ("Qwen/Qwen3-Reranker-8B", True),
        ("qwen3-reranker-4b", True),
        ("other-reranker", False),
    ],
)
def test_rerank_smoke_sends_model_specific_input(smoke_runner, model, tagged):
    seen = []

    def handler(request):
        seen.append(request)
        assert request.method == "POST"
        assert request.url.path == "/v1/rerank"
        body = json.loads(request.content)
        assert body["model"] == model
        assert body["top_n"] == 1
        query = "Which city is in Italy?"
        documents = ["Rome is in Italy.", "Paris is in France."]
        if tagged:
            query = (
                "<Instruct>: Given a web search query, retrieve relevant passages that answer the query"
                f"\n<Query>: {query}"
            )
            documents = [f"<Document>: {document}" for document in documents]
        assert body["query"] == query
        assert body["documents"] == documents
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.94}]})

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as http,
        RegoloClient(
            api_key="test-key",
            reranker_model=model,
            check_models=False,
            pre_existent_client=http,
            alternative_url="https://inference.test",
        ) as client,
    ):
        smoke_runner.check_rerank(client, model)
    assert len(seen) == 1


@pytest.mark.parametrize("index", [1, -1, None, False, "0"])
def test_rerank_smoke_still_rejects_wrong_or_invalid_ranking(smoke_runner, index):
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"results": [{"index": index, "relevance_score": 0.95}]}
                )
            )
        ) as http,
        RegoloClient(
            api_key="test-key",
            reranker_model="Qwen3-Reranker-4B",
            check_models=False,
            pre_existent_client=http,
        ) as client,
    ):
        with pytest.raises(ValueError, match="expected document index 0"):
            smoke_runner.check_rerank(client, "Qwen3-Reranker-4B")


@pytest.mark.parametrize("results", [[], [{"index": 0}, {"index": 1}], {}, ["invalid"]])
def test_rerank_smoke_rejects_malformed_or_unbounded_results(smoke_runner, results):
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"results": results})
            )
        ) as http,
        RegoloClient(
            api_key="test-key",
            reranker_model="Qwen3-Reranker-4B",
            check_models=False,
            pre_existent_client=http,
        ) as client,
    ):
        with pytest.raises(ValueError, match="exactly one result object"):
            smoke_runner.check_rerank(client, "Qwen3-Reranker-4B")
