from __future__ import annotations

import base64
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import regolo
from regolo.client.regolo_client import RegoloClient
from regolo.models.models import ModelsHandler


BASE_URL = "https://regolo.test"
CHAT_MODEL = "chat-model"
IMAGE_MODEL = "image-model"
EMBEDDER_MODEL = "embedder-model"
AUDIO_MODEL = "audio-model"
RERANKER_MODEL = "reranker-model"

ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


@pytest.fixture(autouse=True)
def regolo_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(regolo, "default_key", "test-key")
    monkeypatch.setattr(regolo, "default_chat_model", CHAT_MODEL)
    monkeypatch.setattr(regolo, "default_image_generation_model", IMAGE_MODEL)
    monkeypatch.setattr(regolo, "default_embedder_model", EMBEDDER_MODEL)
    monkeypatch.setattr(regolo, "default_audio_transcription_model", AUDIO_MODEL)
    monkeypatch.setattr(regolo, "default_reranker_model", RERANKER_MODEL)
    monkeypatch.setattr(regolo, "enable_model_checks", False)


def mock_client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def request_json(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.read().decode("utf-8"))


def json_response(request: httpx.Request, payload: dict[str, Any], status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status_code, json=payload, request=request)


def completion_payload(text: str) -> dict[str, Any]:
    return {"choices": [{"text": text}]}


def chat_payload(content: str, role: str = "assistant") -> dict[str, Any]:
    return {"choices": [{"message": {"role": role, "content": content}}]}


def embedding_payload(vectors: list[list[float]]) -> dict[str, Any]:
    return {
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ]
    }


def test_completions_sends_prompt_and_returns_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == f"{BASE_URL}/v1/completions"
        payload = request_json(request)
        assert payload["model"] == CHAT_MODEL
        assert payload["prompt"] == "Tell me something about Rome."
        assert payload["stream"] is False
        assert payload["max_tokens"] == 32
        assert payload["top_p"] == 0.9
        assert payload["top_k"] == 40
        assert request.headers["Authorization"] == "Bearer test-key"
        return json_response(request, completion_payload("Rome is the capital city of Italy."))

    client = mock_client(handler)

    response = RegoloClient.static_completions(
        prompt="Tell me something about Rome.",
        model=CHAT_MODEL,
        api_key="test-key",
        client=client,
        base_url=BASE_URL,
        max_tokens=32,
        top_p=0.9,
        top_k=40,
    )

    assert response == "Rome is the capital city of Italy."


def test_chat_completions_sends_messages_and_returns_role_content() -> None:
    messages = [{"role": "user", "content": "Tell me something about Rome."}]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request_json(request)
        assert str(request.url) == f"{BASE_URL}/v1/chat/completions"
        assert payload["model"] == CHAT_MODEL
        assert payload["messages"] == messages
        assert payload["stream"] is False
        return json_response(request, chat_payload("Rome is the capital city of Italy."))

    client = mock_client(handler)

    response = RegoloClient.static_chat_completions(
        messages=messages,
        model=CHAT_MODEL,
        api_key="test-key",
        client=client,
        base_url=BASE_URL,
    )

    assert response == ("assistant", "Rome is the capital city of Italy.")


def test_image_create_decodes_base64_images() -> None:
    encoded_png = base64.b64encode(ONE_PIXEL_PNG).decode("ascii")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request_json(request)
        assert str(request.url) == f"{BASE_URL}/v1/images/generations"
        assert payload == {
            "model": IMAGE_MODEL,
            "prompt": "A cat in Rome",
            "n": 1,
            "quality": "standard",
            "size": "1024x1024",
            "style": "realistic",
        }
        return json_response(request, {"data": [{"b64_json": encoded_png}]})

    client = mock_client(handler)

    images = RegoloClient.static_create_image(
        prompt="A cat in Rome",
        model=IMAGE_MODEL,
        api_key="test-key",
        client=client,
        base_url=BASE_URL,
    )

    assert len(images) == 1
    image = Image.open(BytesIO(images[0]))
    assert image.size == (1, 1)


def test_audio_transcription_posts_multipart_bytes_and_returns_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{BASE_URL}/v1/audio/transcriptions"
        assert request.headers["content-type"].startswith("multipart/form-data")
        body = request.read()
        assert b'name="model"' in body
        assert AUDIO_MODEL.encode("ascii") in body
        assert b'name="response_format"' in body
        assert b"json" in body
        assert b"fake-audio" in body
        return json_response(request, {"text": "transcribed text"})

    client = mock_client(handler)

    response = RegoloClient.static_audio_transcription(
        file=b"fake-audio",
        model=AUDIO_MODEL,
        api_key="test-key",
        client=client,
        base_url=BASE_URL,
    )

    assert response == "transcribed text"


def test_audio_transcription_instance_uses_audio_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        assert b'name="model"' in body
        assert AUDIO_MODEL.encode("ascii") in body
        assert CHAT_MODEL.encode("ascii") not in body
        return json_response(request, {"text": "audio model used"})

    transport_client = mock_client(handler)
    client = RegoloClient(
        chat_model=CHAT_MODEL,
        audio_transcription_model=AUDIO_MODEL,
        api_key="test-key",
        alternative_url=BASE_URL,
        pre_existent_client=transport_client,
    )

    assert client.audio_transcription(file=b"fake-audio") == "audio model used"


def test_audio_transcription_rejects_invalid_file_input() -> None:
    with pytest.raises(Exception, match="File must be a path string, bytes, or file-like object"):
        RegoloClient.static_audio_transcription(
            file=object(),
            model=AUDIO_MODEL,
            api_key="test-key",
            client=mock_client(lambda request: json_response(request, {"text": ""})),
            base_url=BASE_URL,
        )


def test_embeddings_batch_multiple_inputs_preserves_edge_case_text() -> None:
    inputs = ["", "\u4f60\u597d \U0001f30d", "x" * 10_000]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request_json(request)
        assert str(request.url) == f"{BASE_URL}/v1/embeddings"
        assert payload["model"] == EMBEDDER_MODEL
        assert payload["input"] == inputs
        return json_response(request, embedding_payload([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))

    client = mock_client(handler)

    embeddings = RegoloClient.static_embeddings(
        input_text=inputs,
        model=EMBEDDER_MODEL,
        api_key="test-key",
        client=client,
        base_url=BASE_URL,
    )

    assert [item["index"] for item in embeddings] == [0, 1, 2]
    assert [item["embedding"] for item in embeddings] == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]


def test_embedding_dimension_consistency_across_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        request_json(request)
        return json_response(request, embedding_payload([[0.1, 0.2, 0.3]]))

    client = mock_client(handler)

    first = RegoloClient.static_embeddings("first", EMBEDDER_MODEL, "test-key", client, BASE_URL)
    second = RegoloClient.static_embeddings("second", EMBEDDER_MODEL, "test-key", client, BASE_URL)

    assert len(first[0]["embedding"]) == len(second[0]["embedding"]) == 3


@pytest.mark.parametrize(
    ("documents", "top_n", "results"),
    [
        ([], None, []),
        (["only"], 5, [{"index": 0, "relevance_score": 0.99, "document": "only"}]),
        (
            ["a", "b", "c"],
            2,
            [
                {"index": 1, "relevance_score": 0.9, "document": "b"},
                {"index": 0, "relevance_score": 0.8, "document": "a"},
            ],
        ),
        (["a"], 0, []),
    ],
)
def test_rerank_handles_document_counts_and_top_n_boundaries(
    documents: list[str],
    top_n: int | None,
    results: list[dict[str, Any]],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = request_json(request)
        assert str(request.url) == f"{BASE_URL}/v1/rerank"
        assert payload["model"] == RERANKER_MODEL
        assert payload["query"] == "capital"
        assert payload["documents"] == documents
        assert payload["return_documents"] is True
        if top_n is None:
            assert "top_n" not in payload
        else:
            assert payload["top_n"] == top_n
        return json_response(request, {"results": results})

    client = mock_client(handler)

    response = RegoloClient.static_rerank(
        query="capital",
        documents=documents,
        model=RERANKER_MODEL,
        api_key="test-key",
        top_n=top_n,
        client=client,
        base_url=BASE_URL,
    )

    assert response == results


def test_rerank_sends_structured_document_options() -> None:
    documents = [{"title": "Rome", "body": "Capital of Italy"}]
    results = [{"index": 0, "relevance_score": 0.95}]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request_json(request)
        assert payload["documents"] == documents
        assert payload["rank_fields"] == ["title", "body"]
        assert payload["return_documents"] is False
        assert payload["max_chunks_per_doc"] == 3
        return json_response(request, {"results": results})

    client = mock_client(handler)

    response = RegoloClient.static_rerank(
        query="capital",
        documents=documents,
        model=RERANKER_MODEL,
        api_key="test-key",
        rank_fields=["title", "body"],
        return_documents=False,
        max_chunks_per_doc=3,
        client=client,
        base_url=BASE_URL,
    )

    assert response == results


def test_completion_stream_yields_sse_text_chunks() -> None:
    stream_body = b"""
data: {"choices": [{"text": "Hel"}]}

data: {"choices": [{"text": "lo"}]}

data: [DONE]
"""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request_json(request)
        assert payload["stream"] is True
        return httpx.Response(200, content=stream_body, request=request)

    client = mock_client(handler)

    stream = RegoloClient.static_completions(
        prompt="Say hello",
        model=CHAT_MODEL,
        api_key="test-key",
        stream=True,
        client=client,
        base_url=BASE_URL,
    )

    assert list(stream) == ["Hel", "lo"]


def test_chat_stream_yields_reasoning_and_content_chunks() -> None:
    stream_body = b"""
data: {"choices": [{"delta": {"reasoning_content": "thinking"}}]}

data: {"choices": [{"delta": {"content": "answer"}}]}

data: [DONE]
"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream_body, request=request)

    client = mock_client(handler)

    stream = RegoloClient.static_chat_completions(
        messages=[{"role": "user", "content": "Question"}],
        model=CHAT_MODEL,
        api_key="test-key",
        stream=True,
        client=client,
        base_url=BASE_URL,
    )

    assert list(stream) == [("thinking", "thinking"), ("", "answer")]


def test_stream_full_output_returns_parsed_chunks() -> None:
    chunk = {"choices": [{"text": "full"}]}
    stream_body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n".encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream_body, request=request)

    client = mock_client(handler)

    stream = RegoloClient.static_completions(
        prompt="full",
        model=CHAT_MODEL,
        api_key="test-key",
        stream=True,
        full_output=True,
        client=client,
        base_url=BASE_URL,
    )

    assert list(stream) == [chunk]


def test_stream_non_200_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"}, request=request)

    client = mock_client(handler)
    stream = RegoloClient.static_completions(
        prompt="fail",
        model=CHAT_MODEL,
        api_key="test-key",
        stream=True,
        client=client,
        base_url=BASE_URL,
    )

    with pytest.raises(Exception, match="unexpected status code 503"):
        list(stream)


@pytest.mark.parametrize("status_code", [401, 429])
def test_auth_and_rate_limit_errors_raise_without_retry(status_code: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return json_response(request, {"error": "failed"}, status_code=status_code)

    client = mock_client(handler)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        RegoloClient.static_completions(
            prompt="fail",
            model=CHAT_MODEL,
            api_key="test-key",
            client=client,
            base_url=BASE_URL,
        )

    assert exc_info.value.response.status_code == status_code
    assert calls == 1


def test_server_error_retries_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return json_response(request, {"error": "transient"}, status_code=503)
        return json_response(request, completion_payload("eventual success"))

    client = mock_client(handler)

    response = RegoloClient.static_completions(
        prompt="retry",
        model=CHAT_MODEL,
        api_key="test-key",
        client=client,
        base_url=BASE_URL,
    )

    assert response == "eventual success"
    assert calls == 3


def test_server_error_raises_after_retries_exhausted() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return json_response(request, {"error": "still down"}, status_code=500)

    client = mock_client(handler)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        RegoloClient.static_completions(
            prompt="retry",
            model=CHAT_MODEL,
            api_key="test-key",
            client=client,
            base_url=BASE_URL,
        )

    assert exc_info.value.response.status_code == 500
    assert calls == 3


def test_timeout_propagates_without_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    client = mock_client(handler)

    with pytest.raises(httpx.ReadTimeout):
        RegoloClient.static_completions(
            prompt="timeout",
            model=CHAT_MODEL,
            api_key="test-key",
            client=client,
            base_url=BASE_URL,
        )

    assert calls == 1


def test_malformed_json_response_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not-json", request=request)

    client = mock_client(handler)

    with pytest.raises(json.JSONDecodeError):
        RegoloClient.static_completions(
            prompt="bad json",
            model=CHAT_MODEL,
            api_key="test-key",
            client=client,
            base_url=BASE_URL,
        )


def test_model_not_available_raises_before_request(monkeypatch: pytest.MonkeyPatch) -> None:
    request_was_sent = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_was_sent
        request_was_sent = True
        return json_response(request, completion_payload("should not be called"))

    monkeypatch.setattr(regolo, "enable_model_checks", True)
    monkeypatch.setattr(
        ModelsHandler,
        "get_models",
        lambda base_url, api_key, model_info=False: ["known-model"],
    )

    with pytest.raises(RuntimeError, match="Model not found"):
        RegoloClient.static_completions(
            prompt="bad model",
            model="missing-model",
            api_key="test-key",
            client=mock_client(handler),
            base_url=BASE_URL,
        )

    assert request_was_sent is False


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(regolo, "default_key", None)

    with pytest.raises(RuntimeError, match="API key is required"):
        RegoloClient.static_completions(
            prompt="missing key",
            model=CHAT_MODEL,
            api_key=None,
            client=mock_client(lambda request: json_response(request, completion_payload("unused"))),
            base_url=BASE_URL,
        )


def test_run_chat_maintains_multi_turn_history() -> None:
    requests: list[list[dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request_json(request)
        messages = payload["messages"]
        requests.append(messages)
        return json_response(request, chat_payload(f"answer {len(requests)}"))

    transport_client = mock_client(handler)
    client = RegoloClient(
        chat_model=CHAT_MODEL,
        api_key="test-key",
        alternative_url=BASE_URL,
        pre_existent_client=transport_client,
    )

    first = client.run_chat("first question")
    second = client.run_chat("second question")

    assert first == ("assistant", "answer 1")
    assert second == ("assistant", "answer 2")
    assert requests[0] == [{"role": "user", "content": "first question"}]
    assert requests[1] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "second question"},
    ]
    assert client.instance.get_conversation() == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "answer 2"},
    ]


@pytest.mark.parametrize(
    "prompt",
    ["", "\u00c8 valido? \U0001f680", "x" * 10_000],
    ids=["empty", "unicode", "very-long"],
)
def test_completions_preserve_prompt_edge_cases(prompt: str) -> None:
    seen_prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request_json(request)
        seen_prompts.append(payload["prompt"])
        return json_response(request, completion_payload("ok"))

    client = mock_client(handler)

    assert RegoloClient.static_completions(prompt, CHAT_MODEL, "test-key", False, client=client, base_url=BASE_URL) == "ok"
    assert seen_prompts == [prompt]


def test_static_completions_supports_concurrent_requests_with_shared_client() -> None:
    lock = threading.Lock()
    seen_prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request_json(request)
        prompt = payload["prompt"]
        with lock:
            seen_prompts.append(prompt)
        return json_response(request, completion_payload(f"answer:{prompt}"))

    client = mock_client(handler)

    def call(index: int) -> str:
        return RegoloClient.static_completions(
            prompt=f"prompt-{index}",
            model=CHAT_MODEL,
            api_key="test-key",
            client=client,
            base_url=BASE_URL,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(call, range(20)))

    assert results == [f"answer:prompt-{index}" for index in range(20)]
    assert sorted(seen_prompts, key=lambda value: int(value.rsplit("-", 1)[1])) == [
        f"prompt-{index}" for index in range(20)
    ]
