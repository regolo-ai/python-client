import json

import httpx
import pytest
from click.testing import CliRunner
from PIL import Image

import regolo
import regolo.cli as commands
from regolo.client.regolo_client import RegoloClient

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="


@pytest.fixture
def gateway(monkeypatch):
    seen = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        if r.url.path == "/model/info":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"model_name": "chat", "model_info": {"mode": "chat"}},
                        {"model_name": "unknown", "model_info": {"mode": None}},
                    ]
                },
            )
        if r.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                content=b'data: {"choices":[{"delta":{"reasoning_content":"thought"}}]}\n\ndata: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n',
            )
        if r.url.path == "/v1/images/generations":
            return httpx.Response(200, json={"data": [{"b64_json": PNG}]})
        if r.url.path == "/v1/audio/transcriptions":
            if b'name="stream"\r\n\r\ntrue' in r.content:
                return httpx.Response(
                    200, content=b'data: {"text":"transcript"}\n\ndata: [DONE]\n\n'
                )
            if b'name="response_format"\r\n\r\ntext' in r.content:
                return httpx.Response(200, text="transcript")
            return httpx.Response(200, json={"text": "transcript", "usage": {"seconds": 1}})
        if r.url.path == "/v1/rerank":
            body = json.loads(r.content)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 0, "relevance_score": 0.8, "document": body["documents"][0]}
                    ],
                    "model": "rank",
                },
            )
        raise AssertionError(f"Unexpected gateway request: {r.url}")

    http = httpx.Client(transport=httpx.MockTransport(handler))
    original_audio = RegoloClient.static_audio_transcription
    original_discovery = RegoloClient.get_available_models

    class GatewayClient(RegoloClient):
        def __init__(self, **kwargs):
            super().__init__(
                pre_existent_client=http, alternative_url="https://inference.test", **kwargs
            )

    def audio(**kwargs):
        return original_audio(client=http, base_url="https://inference.test", **kwargs)

    monkeypatch.setattr(GatewayClient, "static_audio_transcription", staticmethod(audio))

    monkeypatch.setattr(regolo, "enable_model_checks", False)
    monkeypatch.setattr(commands, "RegoloClient", GatewayClient)
    monkeypatch.setattr(
        RegoloClient,
        "get_available_models",
        staticmethod(
            lambda api_key, model_info=False: original_discovery(
                api_key, model_info=model_info, client=http, base_url="https://inference.test"
            )
        ),
    )
    return CliRunner(), seen


def test_gateway_discovery_default_and_chat_filter(gateway):
    runner, _ = gateway
    result = runner.invoke(commands.cli, ["get-available-models", "--api-key", "key"])
    assert result.exit_code == 0 and "unknown" in result.output and "chat" in result.output
    result = runner.invoke(
        commands.cli, ["get-available-models", "--api-key", "key", "--model-type", "chat"]
    )
    assert result.exit_code == 0 and "unknown" not in result.output


def test_interactive_chat_reasoning_answer_and_exit(gateway):
    runner, seen = gateway
    result = runner.invoke(commands.cli, ["chat", "--api-key", "key"], input="1\nhello\n/bye\n")
    assert result.exit_code == 0, result.output
    assert "Thinking" in result.output and "answer" in result.output
    request = next(r for r in seen if r.url.path == "/v1/chat/completions")
    assert json.loads(request.content)["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize("fmt", ["png", "jpg", "webp"])
def test_image_generation_cli_saves_valid_files(gateway, tmp_path, fmt):
    runner, _ = gateway
    result = runner.invoke(
        commands.cli,
        [
            "create-image",
            "--api-key",
            "key",
            "--model",
            "image",
            "--save-path",
            str(tmp_path),
            "--output-file-format",
            fmt,
        ],
    )
    assert result.exit_code == 0, result.output
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    with Image.open(files[0]) as image:
        assert image.size == (1, 1)


@pytest.mark.parametrize(
    "stream,save,full,fmt",
    [
        (False, False, False, "json"),
        (False, True, True, "json"),
        (False, False, True, "text"),
        (True, True, False, "json"),
        (True, False, False, "json"),
    ],
)
def test_transcription_cli_formats_streams_and_files(gateway, tmp_path, stream, save, full, fmt):
    runner, seen = gateway
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    dest = tmp_path / "transcript.txt"
    args = [
        "transcribe-audio",
        "--api-key",
        "key",
        "--model",
        "audio",
        "--file-path",
        str(audio),
        "--response-format",
        fmt,
        "--language",
        "it",
        "--prompt",
        "hint",
        "--temperature",
        ".2",
        "--include-logprobs",
    ]
    args += (
        (["--stream"] if stream else [])
        + (["--full-output"] if full else [])
        + (["--save-path", str(dest)] if save else [])
    )
    result = runner.invoke(commands.cli, args)
    assert result.exit_code == 0, result.output
    text = dest.read_text() if save else result.output
    assert "transcript" in text
    if fmt == "text":
        assert '"transcript"' not in text
    assert len(seen) == 1
    assert b'name="timestamp_granularities[]"' not in seen[0].content


@pytest.mark.parametrize(
    "full,fmt,save", [(False, "table", False), (True, "table", False), (True, "json", True)]
)
def test_reranking_cli_displays_metadata_and_saves_json(gateway, tmp_path, full, fmt, save):
    runner, _ = gateway
    dest = tmp_path / "rank.json"
    args = [
        "rerank",
        "--api-key",
        "key",
        "--model",
        "rank",
        "--query",
        "q",
        "--documents",
        "document",
        "--top-n",
        "1",
        "--rank-fields",
        "text",
        "--max-chunks-per-doc",
        "2",
        "--format",
        fmt,
    ]
    args += (["--full-output"] if full else []) + (["--save-path", str(dest)] if save else [])
    result = runner.invoke(commands.cli, args)
    assert result.exit_code == 0, result.output
    if save:
        assert json.loads(dest.read_text())["model"] == "rank"
    else:
        assert "80.00%" in result.output
        if full:
            assert "Metadata" in result.output
