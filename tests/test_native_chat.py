"""Native workflows against HTTP requests and SSE, without paid network calls."""

from __future__ import annotations

import base64
import json
import stat

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

import regolo
from regolo import (
    Conversation,
    FunctionTool,
    RegoloClient,
    audio_part,
    image_part,
    text_part,
    video_part,
)
from regolo.keys.keys import KeysHandler


def response(content="answer", *, calls=None, usage=None, finish="stop", **message):
    msg = {"role": "assistant", "content": content, **message}
    if calls is not None:
        msg["tool_calls"] = calls
    return {
        "id": "reply",
        "model": "chat",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": usage or {"total_tokens": 3},
    }


def client(handler, **kwargs):
    return RegoloClient(
        api_key="test",
        chat_model="chat",
        check_models=False,
        pre_existent_client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def call(name="add", arguments='{"a":2,"b":3}', id="c1"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


def add(a: int, b: int = 1) -> int:
    """Add two integers."""
    return a + b


def test_native_tools_preserve_metadata_even_in_simplified_chat():
    seen = []

    def handler(r):
        seen.append(json.loads(r.content))
        return httpx.Response(200, json=response(None, calls=[call()], reasoning_content="reason"))

    c = client(handler)
    assert c.run_chat("question", tools=[FunctionTool(add).schema()], max_completion_tokens=32) == (
        "assistant",
        None,
    )
    assert seen[0]["tools"][0]["function"]["parameters"]["properties"]["a"]["type"] == "integer"
    assert "max_tokens" not in seen[0]
    assert seen[0]["max_completion_tokens"] == 32
    assert c.conversation.get_lines()[-1]["tool_calls"] == [call()]
    assert c.conversation.get_lines()[-1]["reasoning_content"] == "reason"
    assert c.last_usage == {"total_tokens": 3}
    assert c.usage_totals == {"total_tokens": 3}


def test_tool_loop_executes_only_registered_function_and_roundtrips_result():
    seen = []

    def handler(r):
        body = json.loads(r.content)
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(200, json=response(None, calls=[call()]))
        assert body["messages"][-1] == {"role": "tool", "content": "5", "tool_call_id": "c1"}
        assert body["messages"][-2]["tool_calls"] == [call()]
        return httpx.Response(200, json=response("5"))

    c = client(handler)
    assert c.run_tools("add", [FunctionTool(add)])["choices"][0]["message"]["content"] == "5"
    assert len(seen) == 2
    assert c.usage_totals == {"total_tokens": 6}


@pytest.mark.parametrize(
    "calls",
    [
        [call(name="unregistered")],
        [call(arguments='{"a":"2"}')],
        [call(arguments="[1,2]")],
        [call(arguments='{"a":2,"unknown":1}')],
        [call(), call(id="c1")],
        [call(), call(arguments='{"a":true}', id="c2")],
    ],
)
def test_tool_batch_preflight_prevents_partial_execution(calls):
    executions = []

    def monitored(a: int, b: int = 1):
        executions.append(a)
        return a + b

    c = client(lambda r: httpx.Response(200, json=response(None, calls=calls)))
    with pytest.raises((ValueError, ValidationError)):
        c.run_tools("add", [FunctionTool(monitored, name="add")])
    assert executions == []


def test_tool_round_and_call_limits_are_bounded():
    executions = []

    def record(a: int, b: int = 1):
        executions.append(a)
        return a + b

    c = client(lambda r: httpx.Response(200, json=response(None, calls=[call()])))
    with pytest.raises(RuntimeError, match="round limit"):
        c.run_tools("q", [FunctionTool(record, name="add")], max_rounds=1)
    assert not executions
    c = client(lambda r: httpx.Response(200, json=response(None, calls=[call(), call(id="c2")])))
    with pytest.raises(RuntimeError, match="call limit"):
        c.run_tools("q", [FunctionTool(record, name="add")], max_calls=1)
    assert not executions


def test_tool_result_is_retained_after_followup_network_failure_without_replaying():
    executions = []
    requests = []

    def record(a: int, b: int = 1):
        executions.append(a)
        return a + b

    def handler(r):
        requests.append(r)
        if len(requests) == 1:
            return httpx.Response(200, json=response(None, calls=[call()]))
        raise httpx.ReadTimeout("offline", request=r)

    c = client(handler)
    with pytest.raises(httpx.ReadTimeout):
        c.run_tools("q", [FunctionTool(record, name="add")])
    assert executions == [2]
    assert c.conversation.get_lines()[-1]["role"] == "tool"
    assert len(requests) == 2


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


def test_structured_output_uses_schema_and_validates_locally():
    def handler(r):
        body = json.loads(r.content)
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
        return httpx.Response(200, json=response('{"value":7}'))

    c = client(handler)
    assert c.run_structured("json", Item) == Item(value=7)
    assert c.last_response["id"] == "reply"


@pytest.mark.parametrize(
    "payload",
    [
        response('{"value":"wrong"}'),
        response("{}", refusal="no"),
        response('{"value":7}', finish="length"),
    ],
)
def test_structured_output_rejects_invalid_refused_and_truncated_data(payload):
    c = client(lambda r: httpx.Response(200, json=payload))
    with pytest.raises(ValueError):
        c.run_structured("json", Item)
    assert c.last_response == payload


def sse(events, done=True):
    return (
        "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        + ("data: [DONE]\n\n" if done else "")
    ).encode()


@pytest.mark.parametrize("full", [False, True])
def test_stream_keeps_tools_reasoning_usage_and_choice_zero(full):
    events = [
        {
            "id": "stream1",
            "choices": [
                {"index": 1, "delta": {"content": "ignored"}},
                {
                    "index": 0,
                    "delta": {
                        "content": "yes",
                        "reasoning_content": "think",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "add", "arguments": '{"a":'},
                            }
                        ],
                    },
                },
            ],
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "2}"}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"choices": [], "usage": {"total_tokens": 8}},
    ]
    c = client(lambda r: httpx.Response(200, content=sse(events)))
    output = list(c.run_chat("q", stream=True, full_output=full))
    if not full:
        assert output == [("thinking", "think"), ("", "yes")]
    assert c.conversation.get_lines()[-1]["tool_calls"][0]["function"]["arguments"] == '{"a":2}'
    assert c.last_usage == {"total_tokens": 8}
    assert c.last_response["id"] == "stream1"
    assert c.conversation.get_lines()[-1]["content"] == "yes"


def test_stream_is_lazy_and_cannot_overlap_or_mutate_active_history():
    c = client(
        lambda r: httpx.Response(200, content=sse([{"choices": [{"delta": {"content": "hi"}}]}]))
    )
    unused = c.run_chat("unused", stream=True)
    unused.close()
    assert c.conversation.get_lines() == []
    source = c.run_chat("q", stream=True)
    assert next(source) == ("", "hi")
    with pytest.raises(RuntimeError, match="active chat"):
        c.run_chat("overlap")
    with pytest.raises(RuntimeError, match="active chat"):
        c.add_message("user", "overlap")
    with pytest.raises(RuntimeError):
        c.clear_conversations()
    source.close()
    assert c.conversation.get_lines() == [{"role": "user", "content": "q"}]
    list(c.run_chat("new", stream=True))
    assert c.conversation.get_lines()[-1]["content"] == "hi"


def test_injected_client_timeout_is_respected_for_normal_and_streaming_requests():
    seen = []

    def handler(r):
        seen.append(r.extensions["timeout"])
        if json.loads(r.content).get("stream"):
            return httpx.Response(200, content=sse([{"choices": [{"delta": {"content": "hi"}}]}]))
        return httpx.Response(200, json=response())

    http = httpx.Client(transport=httpx.MockTransport(handler), timeout=httpx.Timeout(2, read=3))
    c = RegoloClient(
        api_key="test", chat_model="chat", check_models=False, pre_existent_client=http
    )
    c.run_chat("q")
    list(c.run_chat("q2", stream=True))
    assert all(value["connect"] == 2 and value["read"] == 3 for value in seen)
    c.close()
    assert not http.is_closed
    with pytest.raises(RuntimeError, match="closed"):
        c.run_chat("q3")


@pytest.mark.parametrize(
    "base,expected",
    [
        ("https://x.test/", "/v1/chat/completions"),
        ("https://x.test/v1/", "/v1/chat/completions"),
        ("https://x.test/custom-model/v1/", "/custom-model/v1/chat/completions"),
    ],
)
def test_root_versioned_and_custom_urls(base, expected):
    def handler(r):
        assert r.url.path == expected
        return httpx.Response(200, json=response())

    c = client(handler, alternative_url=base)
    assert c.run_chat("q") == ("assistant", "answer")


@pytest.mark.parametrize(
    "base",
    [
        "https://user:password@x.test",
        "file:///a",
        "https://x.test?token=bad",
        "https://x.test/#part",
    ],
)
def test_invalid_base_url_is_rejected_before_authentication(base):
    with pytest.raises(ValueError, match="base URL"):
        RegoloClient(api_key="test", alternative_url=base)


def test_model_catalog_preserves_new_stats_filters_and_cache_independence():
    calls = []
    rows = [
        {
            "model_group": "chat",
            "mode": "chat",
            "supports_vision": True,
            "input_cost_per_token": 0.000001,
            "future_limit": None,
        },
        {"model_group": "emb", "mode": "embedding"},
    ]

    def handler(r):
        calls.append(r)
        assert r.url.path == "/model_group/info"
        return httpx.Response(200, json={"data": rows})

    c = client(handler)
    found = c.get_model_catalog(mode="chat", supports=["vision"])
    assert found == [rows[0]]
    found[0]["future_limit"] = "mutated"
    assert c.get_model_info("chat")["future_limit"] is None
    assert len(calls) == 1
    c.get_model_catalog(refresh=True)
    assert len(calls) == 2


def test_multimodal_helpers_and_ocr_send_real_content_parts(tmp_path):
    path = tmp_path / "doc.png"
    path.write_bytes(b"PNG")
    part = image_part(path)
    assert base64.b64decode(part["image_url"]["url"].split(",")[1]) == b"PNG"
    assert (
        image_part("https://images.test/a.png")["image_url"]["url"] == "https://images.test/a.png"
    )
    assert audio_part(b"sound")["input_audio"]["format"] == "wav"
    assert video_part("https://videos.test/clip.mp4")["type"] == "video_url"
    assert text_part("q") == {"type": "text", "text": "q"}

    def handler(r):
        body = json.loads(r.content)
        assert body["messages"][0]["content"][1] == part
        assert body["skip_special_tokens"] is False
        return httpx.Response(200, json=response("extracted"))

    c = client(handler)
    assert c.ocr(path) == ("assistant", "extracted")


@pytest.mark.parametrize(
    "f,args",
    [(image_part, (b"no MIME",)), (audio_part, (b"audio",)), (video_part, ("file:///video.mp4",))],
)
def test_invalid_multimodal_inputs(f, args):
    if f is audio_part:
        with pytest.raises(ValueError):
            f(*args, format="exe")
    else:
        with pytest.raises(ValueError):
            f(*args)


def test_conversation_roundtrip_deep_fork_and_turn_trimming(tmp_path):
    c = (
        Conversation()
        .append("system", "guide")
        .append("user", [{"type": "text", "text": "one"}])
        .append("assistant", None, tool_calls=[call()])
        .append("tool", "5", tool_call_id="c1")
        .append("user", "two")
        .append("assistant", "answer")
    )
    fork = c.fork()
    assert isinstance(fork.lines[1].content, list)
    assert isinstance(c.lines[1].content, list)
    fork.lines[1].content[0]["text"] = "changed"
    assert c.lines[1].content[0]["text"] == "one"
    path = tmp_path / "nested/conversation.json"
    c.save(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert Conversation.load(path).get_lines() == c.get_lines()
    c.trim(1)
    assert [line.role for line in c.lines] == ["system", "user", "assistant"]
    assert c.lines[1].content == "two"


@pytest.mark.parametrize("key", ["bad\nkey", "bad key", "Bearer ", "bearer"])
def test_keys_reject_invalid_whitespace_without_echoing_key(key):
    with pytest.raises(RuntimeError):
        KeysHandler.check_key(key)


def test_bearer_prefix_is_case_insensitive():
    assert KeysHandler.check_key("bearer test") == "Bearer test"


def test_per_client_model_checks_do_not_change_global_settings(monkeypatch):
    monkeypatch.setattr(regolo, "enable_model_checks", True)
    c = client(lambda r: httpx.Response(200, json=response()))
    c.run_chat("q")
    assert regolo.enable_model_checks is True


def test_modern_options_work_for_embeddings_images_and_audio(tmp_path):
    seen = []

    def handler(r):
        seen.append(r)
        if r.url.path.endswith("embeddings"):
            assert json.loads(r.content)["dimensions"] == 3
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 2, 3]}]})
        if r.url.path.endswith("generations"):
            assert json.loads(r.content)["response_format"] == "b64_json"
            return httpx.Response(
                200, json={"data": [{"b64_json": base64.b64encode(b"image").decode()}]}
            )
        assert b'name="vad_filter"' in r.content and b"true" in r.content
        return httpx.Response(200, json={"text": "words"})

    c = client(
        handler,
        embedder_model="emb",
        image_generation_model="img",
        audio_transcription_model="audio",
    )
    assert c.embeddings("text", dimensions=3)[0]["embedding"] == [1, 2, 3]
    assert c.create_image("image", response_format="b64_json") == [b"image"]
    path = tmp_path / "a.wav"
    path.write_bytes(b"audio")
    assert c.audio_transcription(path, vad_filter=True) == "words"
    assert len(seen) == 3


def test_url_only_images_do_not_trigger_automatic_download_or_credential_forwarding():
    seen = []

    def handler(r):
        seen.append(r)
        return httpx.Response(200, json={"data": [{"url": "http://internal.test/image"}]})

    c = client(handler, image_generation_model="img")
    with pytest.raises(ValueError, match="image URLs"):
        c.create_image("q")
    assert len(seen) == 1


def test_usage_only_stream_does_not_create_empty_assistant_reply():
    c = client(
        lambda r: httpx.Response(200, content=sse([{"choices": [], "usage": {"total_tokens": 2}}]))
    )
    with pytest.raises(ValueError, match="no assistant choice"):
        list(c.run_chat("q", stream=True))
    assert [line.role for line in c.conversation.lines] == ["user"]
    assert c.last_response is None


def test_forced_tool_selection_applies_only_to_first_round():
    seen = []

    def handler(r):
        body = json.loads(r.content)
        seen.append(body)
        return httpx.Response(
            200, json=response(None, calls=[call()]) if len(seen) == 1 else response("5")
        )

    c = client(handler)
    c.run_tools(
        "q", [FunctionTool(add)], tool_choice={"type": "function", "function": {"name": "add"}}
    )
    assert seen[0]["tool_choice"]["function"]["name"] == "add"
    assert seen[1]["tool_choice"] == "auto"


def test_completed_tool_ids_cannot_be_reexecuted_on_resume():
    executions = []

    def monitored(a: int, b: int = 1):
        executions.append(a)
        return a + b

    c = client(lambda r: httpx.Response(200, json=response(None, calls=[call()])))
    c.add_tool_result("c1", "previous result")
    with pytest.raises(ValueError, match="repeated tool-call"):
        c.run_tools("q", [FunctionTool(monitored, name="add")])
    assert not executions


def test_catalog_accepts_common_versioned_base_url_and_closed_clients_fail():
    def handler(r):
        assert r.url.path == "/model_group/info"
        return httpx.Response(200, json={"data": []})

    c = client(handler, alternative_url="https://x.test/v1/")
    assert c.get_model_catalog() == []
    c.close()
    with pytest.raises(RuntimeError, match="closed"):
        c.get_model_catalog(refresh=True)


def test_from_instance_preserves_per_client_discovery_and_timeout_settings():
    c = client(lambda r: httpx.Response(200, json=response()))
    clone = RegoloClient.from_instance(c.instance)
    assert clone._check_models is False
    assert clone.request_timeout == c.request_timeout
    assert clone.run_chat("q") == ("assistant", "answer")
    clone.close()
    assert c.run_chat("q2") == ("assistant", "answer")


def test_truncated_tool_decision_cannot_execute_even_valid_partial_arguments():
    executed = []

    def monitor(a: int, b: int = 1):
        executed.append(a)
        return a + b

    c = client(lambda r: httpx.Response(200, json=response(None, calls=[call()], finish="length")))
    with pytest.raises(ValueError, match="did not complete the tool decision"):
        c.run_tools("q", [FunctionTool(monitor, name="add")])
    assert not executed
