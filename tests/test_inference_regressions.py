import json

import httpx
import pytest

import regolo
from regolo.client.regolo_client import RegoloClient, safe_post
from regolo.instance.structures.conversation_model import Conversation, ConversationLine
from regolo.keys.keys import KeysHandler
from regolo.models.models import ModelsHandler


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    monkeypatch.setattr(regolo, "enable_model_checks", False)
    monkeypatch.setattr(regolo, "default_key", "test")
    monkeypatch.setattr(regolo, "default_chat_model", "chat")
    monkeypatch.setattr(regolo, "default_reranker_model", "rerank")


def transport(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def stream_transport(events):
    return transport(lambda r: httpx.Response(200, content=events))


def test_low_level_stream_requires_a_handler_for_processed_output():
    with transport(lambda r: pytest.fail("Invalid stream must fail before HTTP")) as http:
        stream = RegoloClient.create_stream_generator(http, "https://inference.test", payload={})
        with pytest.raises(ValueError, match="output_handler is required"):
            next(stream)


def test_low_level_full_stream_needs_no_output_handler():
    event = {"choices": [{"delta": {"content": "hello"}}]}
    events = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()
    with stream_transport(events) as http:
        assert list(
            RegoloClient.create_stream_generator(
                http, "https://inference.test", payload={}, full_output=True
            )
        ) == [event]


def test_invalid_streamed_chat_content_does_not_enter_history():
    with stream_transport(
        b'data: {"choices":[{"delta":{"content":123}}]}\n\ndata: [DONE]\n\n'
    ) as http:
        client = RegoloClient(pre_existent_client=http)
        with pytest.raises(ValueError, match="Chat stream content must be text"):
            list(client.run_chat("hello", stream=True, full_output=True))
        assert client.instance.get_conversation() == [{"role": "user", "content": "hello"}]


def test_declared_tool_message_fields_preserve_omitted_and_extra_values():
    lines = [
        ConversationLine(role="user", content="hello"),
        ConversationLine(role="tool", content=None, tool_call_id="call-1", name="lookup"),
        ConversationLine.model_validate({"role": "assistant", "custom_metadata": {"id": 1}}),
    ]
    assert Conversation(lines=lines).get_lines() == [
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": None, "tool_call_id": "call-1", "name": "lookup"},
        {"role": "assistant", "custom_metadata": {"id": 1}},
    ]


@pytest.mark.parametrize("fmt", ["text", "srt", "vtt"])
def test_plain_transcription_formats(fmt):
    with transport(lambda r: httpx.Response(200, text="plain transcript")) as http:
        assert (
            RegoloClient.static_audio_transcription(
                b"audio", model="audio", response_format=fmt, client=http
            )
            == "plain transcript"
        )


def test_transcription_repeated_multipart_fields_and_path_input(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio-content")

    def handler(r):
        body = r.read()
        assert body.count(b'name="timestamp_granularities[]"') == 2
        assert b"word" in body and b"segment" in body
        assert body.count(b'name="include[]"') == 2
        assert b"audio-content" in body
        assert b'filename="sample.wav"' in body
        return httpx.Response(200, json={"text": "transcript", "segments": []})

    with transport(handler) as http:
        result = RegoloClient.static_audio_transcription(
            str(audio),
            model="audio",
            client=http,
            timestamp_granularities=["word", "segment"],
            include=["logprobs", "tokens"],
            response_format="verbose_json",
            full_output=True,
        )
    assert result["segments"] == []


def test_audio_http_error_is_preserved_and_missing_file_is_clear(tmp_path):
    with transport(lambda r: httpx.Response(401, json={"detail": "bad key"})) as http:
        with pytest.raises(httpx.HTTPStatusError) as e:
            RegoloClient.static_audio_transcription(b"audio", model="audio", client=http)
        assert e.value.response.status_code == 401
        with pytest.raises(FileNotFoundError):
            RegoloClient.static_audio_transcription(
                str(tmp_path / "missing"), model="audio", client=http
            )


def test_sse_comments_metadata_multiline_and_usage_event():
    events = b':keepalive\nevent: message\nid: 1\ndata: {"choices":\ndata: [{"delta":{"content":"hello"}}]}\n\ndata: {"choices":[],"usage":{"total_tokens":4}}\n\ndata: [DONE]\n\n'
    with stream_transport(events) as http:
        assert list(RegoloClient.static_chat_completions([], client=http, stream=True)) == [
            ("", "hello")
        ]
        chunks = list(
            RegoloClient.static_chat_completions([], client=http, stream=True, full_output=True)
        )
        assert chunks[-1]["usage"]["total_tokens"] == 4


@pytest.mark.parametrize(
    "body,expected",
    [
        (b"data: not-json\n\n", "Malformed"),
        (b"data: []\n\n", "object"),
        (b'data: {"error":{"message":"bad request"}}\n\n', "bad request"),
    ],
)
def test_sse_errors_are_explicit(body, expected):
    with stream_transport(body) as http:
        with pytest.raises((ValueError, RuntimeError), match=expected):
            list(RegoloClient.static_completions("prompt", client=http, stream=True))


@pytest.mark.parametrize("full", [True, False])
def test_streamed_chat_history_includes_completed_assistant(full):
    events = b'data: {"choices":[{"delta":{"reasoning_content":"private thought"}}]}\n\ndata: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: {"choices":[{"delta":{"content":" world"}}]}\n\ndata: [DONE]\n\n'
    with stream_transport(events) as http:
        c = RegoloClient(pre_existent_client=http)
        list(c.run_chat("question", stream=True, full_output=full))
        assert c.instance.get_conversation()[-1] == {
            "role": "assistant",
            "content": "hello world",
            "reasoning_content": "private thought",
        }


def test_cancelled_and_failed_streams_do_not_record_a_complete_answer():
    for body in [
        b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\ndata: {"error":"failed"}\n\n',
        b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\ndata: [DONE]\n\n',
    ]:
        with stream_transport(body) as http:
            c = RegoloClient(pre_existent_client=http)
            gen = c.run_chat("q", stream=True)
            next(gen)
            if b'"error"' in body:
                with pytest.raises(RuntimeError):
                    next(gen)
            else:
                gen.close()
            assert len(c.instance.get_conversation()) == 1


def test_from_instance_preserves_all_settings_and_borrows_transport():
    with transport(lambda r: httpx.Response(200, json={})) as http:
        c = RegoloClient(
            chat_model="chat",
            embedder_model="embed",
            image_generation_model="img",
            audio_transcription_model="audio",
            reranker_model="rank",
            alternative_url="https://custom.test",
            pre_existent_client=http,
        )
        clone = RegoloClient.from_instance(c.instance)
        for field in (
            "chat_model",
            "embedder_model",
            "image_generation_model",
            "audio_transcription_model",
            "reranker_model",
            "base_url",
            "client",
            "conversation",
        ):
            assert getattr(clone.instance, field) == getattr(c.instance, field)
        clone.close()
        c.close()
        assert not http.is_closed


def test_cloning_does_not_replace_stored_none_with_changed_defaults(monkeypatch):
    with transport(lambda r: httpx.Response(200, json={})) as http:
        c = RegoloClient(pre_existent_client=http)
        c.instance.embedder_model = None
        monkeypatch.setattr(regolo, "default_embedder_model", "new-default")
        assert RegoloClient.from_instance(c.instance).instance.embedder_model is None


def test_default_reranker_and_model_change_failure(monkeypatch):
    with transport(lambda r: httpx.Response(200, json={})) as http:
        c = RegoloClient(pre_existent_client=http)
        assert c.instance.reranker_model == "rerank"
        monkeypatch.setattr(regolo, "enable_model_checks", True)
        with pytest.raises(RuntimeError):
            # Deliberately invalid input exercises runtime validation.
            c.change_model(None)  # type: ignore[arg-type]


def test_model_checks_use_injected_transport():
    seen = []

    def handler(r):
        seen.append(r.url.path)
        assert r.headers["authorization"] == "Bearer test"
        if r.url.path == "/models":
            return httpx.Response(200, json={"data": [{"id": "chat"}]})
        return httpx.Response(200, json={"choices": [{"text": "ok"}]})

    regolo.enable_model_checks = True
    with transport(handler) as http:
        assert RegoloClient.static_completions("q", client=http) == "ok"
    assert seen == ["/models", "/v1/completions"]


@pytest.mark.parametrize("info", [False, True])
def test_model_discovery_and_errors(info):
    payload = [{"id": "model", "model_name": "model"}]
    with transport(lambda r: httpx.Response(200, json={"data": payload})) as http:
        assert ModelsHandler.get_models("https://api.test", "test", info, http) == (
            payload if info else ["model"]
        )
    for status in [401, 503]:
        with transport(lambda r: httpx.Response(status, json={})) as http:
            with pytest.raises(httpx.HTTPStatusError) as error:
                ModelsHandler.get_models("https://api.test", "test", info, http)
            assert error.value.response.status_code == status


def test_tools_and_stream_options_can_be_sent_without_overriding_identity():
    extra = {
        "tools": [{"type": "function", "function": {"name": "weather"}}],
        "tool_choice": "auto",
        "response_format": {"type": "json_object"},
    }

    def handler(r):
        payload = json.loads(r.content)
        for k, v in extra.items():
            assert payload[k] == v
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{"id": "call1"}],
                        }
                    }
                ]
            },
        )

    with transport(handler) as http:
        response = RegoloClient.static_chat_completions(
            [], client=http, extra_body=extra, full_output=True
        )
        assert response["choices"][0]["message"]["tool_calls"][0]["id"] == "call1"
        with pytest.raises(ValueError, match="override"):
            RegoloClient.static_chat_completions([], client=http, extra_body={"model": "other"})


def test_owned_static_clients_close_on_success_error_and_stream_cancel(monkeypatch):
    original = httpx.Client
    created = []

    def factory(**_kwargs):
        http = original(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"choices": [{"text": "ok"}]})
            )
        )
        created.append(http)
        return http

    monkeypatch.setattr("regolo.transport.httpx.Client", factory)
    assert RegoloClient.static_completions("q") == "ok"
    assert created[0].is_closed
    with pytest.raises(RuntimeError):
        RegoloClient.static_completions("q", api_key="")
    assert created[1].is_closed

    def stream_factory(**_kwargs):
        http = original(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200,
                    content=b'data: {"choices":[{"text":"one"}]}\n\ndata: {"choices":[{"text":"two"}]}\n\n',
                )
            )
        )
        created.append(http)
        return http

    monkeypatch.setattr("regolo.transport.httpx.Client", stream_factory)
    gen = RegoloClient.static_completions("q", stream=True)
    assert len(created) == 2  # unstarted stream opens nothing
    assert next(gen) == "one"
    gen.close()
    assert created[-1].is_closed


def test_owned_instance_close_and_context_manager(monkeypatch):
    original = httpx.Client
    http = original(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    monkeypatch.setattr("regolo.client.regolo_client.httpx.Client", lambda **kw: http)
    with RegoloClient() as c:
        assert c.instance.get_client() is http
    assert http.is_closed


@pytest.mark.parametrize("key", [None, "", "  ", "Bearer", "Bearer ", 1])
def test_empty_or_invalid_keys(key: object):
    with pytest.raises(RuntimeError, match="API key"):
        KeysHandler.check_key(key)  # type: ignore[arg-type]  # Deliberately invalid input.


@pytest.mark.parametrize("retries", [-1, True, 1.5])
def test_bad_retry_count(retries: object):
    with transport(lambda r: pytest.fail("must not send")) as http:
        with pytest.raises(ValueError):
            safe_post(http, "https://api.test", max_retries=retries)  # type: ignore[arg-type]


def test_all_default_env_loaders_and_conversation_operations(monkeypatch):
    for env, field in [
        ("API_KEY", "default_key"),
        ("LLM", "default_chat_model"),
        ("IMAGE_GENERATION_MODEL", "default_image_generation_model"),
        ("EMBEDDER_MODEL", "default_embedder_model"),
        ("AUDIO_TRANSCRIPTION_MODEL", "default_audio_transcription_model"),
        ("RERANKER_MODEL", "default_reranker_model"),
    ]:
        monkeypatch.setenv(env, "from-env")
    regolo.try_loading_from_env()
    assert regolo.default_reranker_model == "from-env"
    with transport(lambda r: httpx.Response(200, json={})) as http:
        c = RegoloClient(pre_existent_client=http)
        c.add_prompt_to_chat("system", "system")
        replacement = Conversation(lines=[ConversationLine(role="user", content="question")])
        c.instance.overwrite_conversation(replacement)
        assert c.instance.get_conversation() == [{"role": "user", "content": "question"}]
        c.clear_conversations()
        assert c.instance.get_conversation() == []


def test_model_discovery_public_entrypoint_uses_injected_client():
    with transport(lambda r: httpx.Response(200, json={"data": [{"id": "custom"}]})) as http:
        assert RegoloClient.get_available_models("key", client=http) == ["custom"]


def test_tool_messages_and_multimodal_content_are_preserved():
    initial = ConversationLine(role="user", content=[{"type": "text", "text": "question"}])
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "weather", "arguments": '{"city":"Rome"}'},
    }

    def handler(r):
        assert json.loads(r.content)["messages"][0]["content"] == initial.content
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": None, "tool_calls": [tool_call]}}
                ]
            },
        )

    with transport(handler) as http:
        c = RegoloClient(
            pre_existent_client=http, pre_existent_conversation=Conversation(lines=[initial])
        )
        c.run_chat(full_output=True)
        assert c.instance.get_conversation()[-1]["tool_calls"] == [tool_call]
        c.instance.add_line(ConversationLine(role="tool", content="24", tool_call_id="call-1"))
        assert c.instance.get_conversation()[-1]["tool_call_id"] == "call-1"


def test_streamed_tool_call_arguments_are_assembled():
    events = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call1",
                                "type": "function",
                                "function": {"name": "weather", "arguments": '{"city":'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Rome"}'}}]}}
            ]
        },
    ]
    body = (
        "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode() + b"data: [DONE]\n\n"
    )
    with stream_transport(body) as http:
        c = RegoloClient(pre_existent_client=http)
        list(c.run_chat("weather", stream=True, full_output=True))
        message = c.instance.get_conversation()[-1]
        assert message["content"] is None
        assert message["tool_calls"][0]["function"] == {
            "name": "weather",
            "arguments": '{"city":"Rome"}',
        }


def test_incomplete_stream_does_not_commit_partial_history():
    with stream_transport(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n') as http:
        c = RegoloClient(pre_existent_client=http)
        with pytest.raises(ValueError, match="before completion"):
            list(c.run_chat("q", stream=True))
        assert len(c.instance.get_conversation()) == 1


@pytest.mark.parametrize(
    "method,kwargs,key",
    [
        ("completions", {"prompt": "q"}, "choices"),
        ("create_image", {"prompt": "q"}, "data"),
        ("embeddings", {"input_text": "q"}, "data"),
        ("rerank", {"query": "q", "documents": ["a"]}, "results"),
    ],
)
def test_instance_wrappers_and_full_output_preserve_metadata(method, kwargs, key):
    payload = {"choices": [{"text": "ok"}], "data": [], "results": [], "usage": {"total_tokens": 3}}
    with transport(lambda r: httpx.Response(200, json=payload)) as http:
        c = RegoloClient(
            pre_existent_client=http, embedder_model="emb", image_generation_model="img"
        )
        assert getattr(c, method)(**kwargs, full_output=True)["usage"] == {"total_tokens": 3}
