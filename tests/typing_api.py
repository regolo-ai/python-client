"""Consumer examples checked by mypy; these functions never make live requests."""

from collections.abc import Generator
from typing import Any, assert_type

import httpx

from regolo import RegoloClient
from regolo.models.models import ModelsHandler


def inference_results(client: RegoloClient, http: httpx.Client) -> None:
    assert_type(client.run_chat("hello"), tuple[str, str | None])
    assert_type(client.run_chat("hello", full_output=True), dict[str, Any])
    assert_type(client.run_chat("hello", stream=True), Generator[tuple[str, str], None, None])
    assert_type(
        client.run_chat("hello", stream=True, full_output=True),
        Generator[dict[str, Any], None, None],
    )
    assert_type(client.completions("hello"), str)
    assert_type(client.completions("hello", full_output=True), dict[str, Any])
    assert_type(client.completions("hello", stream=True), Generator[str, None, None])
    assert_type(
        client.completions("hello", stream=True, full_output=True),
        Generator[dict[str, Any], None, None],
    )
    assert_type(RegoloClient.static_completions("hello", client=http), str)
    assert_type(
        RegoloClient.static_chat_completions([{"role": "user", "content": "hello"}]),
        tuple[str, str | None],
    )
    assert_type(RegoloClient.get_available_models("key"), list[str])
    assert_type(RegoloClient.get_available_models("key", model_info=True), list[dict[str, Any]])
    assert_type(ModelsHandler.get_models("https://inference.test", "key"), list[str])
    assert_type(
        ModelsHandler.get_models("https://inference.test", "key", True), list[dict[str, Any]]
    )
    assert_type(client.audio_transcription(b"audio"), str)
    assert_type(client.audio_transcription(b"audio", full_output=True), dict[str, Any])
    assert_type(client.audio_transcription(b"audio", stream=True), Generator[str, None, None])
    assert_type(
        client.audio_transcription(b"audio", stream=True, full_output=True),
        Generator[dict[str, Any], None, None],
    )
    assert_type(client.create_image("a cat"), list[bytes])
    assert_type(client.create_image("a cat", full_output=True), dict[str, Any])
    assert_type(client.rerank("hello", ["hello"]), list[dict[str, Any]])
    assert_type(client.rerank("hello", ["hello"], full_output=True), dict[str, Any])
    assert_type(client.embeddings("hello"), list[Any])
    assert_type(client.embeddings("hello", full_output=True), dict[str, Any])


def dynamic_flags(client: RegoloClient, stream: bool, full: bool) -> None:
    # Real applications can choose response modes dynamically, including positionally.
    client.run_chat("hello", stream, full_output=full)
    client.completions("hello", stream, full_output=full)
    RegoloClient.get_available_models("key", "https://inference.test", True)
