"""Small native chat, multimodal and tool helpers; no agent framework required."""

from __future__ import annotations

import base64
import inspect
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, get_type_hints
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model

if TYPE_CHECKING:
    from regolo.client.regolo_client import RegoloClient


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_part(
    source: str | Path | bytes, *, mime_type: str | None = None, detail: str = "auto"
) -> dict[str, Any]:
    """Remote URL, data URL or explicitly supplied local image; no URL download."""
    if detail not in {"auto", "low", "high"}:
        raise ValueError("Image detail must be auto, low or high")
    if isinstance(source, str) and source.startswith(("https://", "http://", "data:image/")):
        url = source
    else:
        if isinstance(source, bytes):
            if mime_type is None:
                raise ValueError("mime_type is required for image bytes")
            data = source
        else:
            path = Path(source)
            mime_type = mime_type or mimetypes.guess_type(path.name)[0]
            data = path.read_bytes()
        if mime_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
            raise ValueError("Supported images are PNG, JPEG, WebP and GIF")
        url = f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"
    return {"type": "image_url", "image_url": {"url": url, "detail": detail}}


def audio_part(data: bytes, *, format: str = "wav") -> dict[str, Any]:
    if format not in {"wav", "mp3"}:
        raise ValueError("Chat audio format must be wav or mp3")
    return {
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(data).decode("ascii"), "format": format},
    }


def video_part(url: str) -> dict[str, Any]:
    """Pass a public video URL to a model advertising video support."""
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError("Video input must be an HTTP(S) URL")
    return {"type": "video_url", "video_url": {"url": url}}


class ChatAccumulator:
    """Assemble choice zero without mixing other choices, tools or reasoning."""

    def __init__(self):
        self.message: dict[str, Any] = {"role": "assistant", "content": ""}
        self.calls: dict[int, dict] = {}
        self.metadata: dict[str, Any] = {}
        self.finish_reason: str | None = None
        self.seen_choice = False

    def add(self, chunk: dict) -> None:
        for key, value in chunk.items():
            if key not in {"choices", "object"}:
                self.metadata[key] = value
        choices = [choice for choice in chunk.get("choices", []) if choice.get("index", 0) == 0]
        if not choices:
            return
        choice = choices[0]
        self.seen_choice = True
        self.finish_reason = choice.get("finish_reason") or self.finish_reason
        delta = choice.get("delta") or {}
        for field in ("content", "reasoning_content", "reasoning", "refusal"):
            value = delta.get(field)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError("Chat stream content must be text")
                self.message[field] = self.message.get(field, "") + value
        for call in delta.get("tool_calls") or []:
            index = call.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("Invalid streamed tool-call index")
            saved = self.calls.setdefault(index, {"function": {"name": "", "arguments": ""}})
            for key in ("id", "type"):
                if key in call:
                    saved[key] = call[key]
            for key in ("name", "arguments"):
                saved["function"][key] += call.get("function", {}).get(key) or ""

    def response(self) -> dict:
        if not self.seen_choice:
            raise ValueError("Chat stream contained no assistant choice")
        message = dict(self.message)
        if self.calls:
            message["tool_calls"] = [self.calls[i] for i in sorted(self.calls)]
            if not message["content"]:
                message["content"] = None
        return {
            **self.metadata,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": self.finish_reason}],
        }


def text_chunks(chunk: dict):
    """Compatibility tuples; reasoning and answer can coexist in one event."""
    for choice in chunk.get("choices") or []:
        if choice.get("index", 0) != 0:
            continue
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            yield "thinking", reasoning
        content = delta.get("content")
        if content:
            yield "", content


@dataclass(frozen=True)
class FunctionTool:
    """A caller-registered Python function with a schema from its type hints.

    Constructing or passing a tool never executes it. Only run_tools executes
    explicitly registered callables, with strict Pydantic argument validation.
    """

    function: Callable[..., Any]
    name: str | None = None
    description: str | None = None

    @property
    def tool_name(self) -> str:
        name = self.name or self.function.__name__
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ValueError("Tool name must contain 1–64 letters, digits, underscores or hyphens")
        return name

    def argument_model(self) -> type[BaseModel]:
        if inspect.iscoroutinefunction(self.function):
            raise ValueError("Synchronous tools cannot be coroutine functions")
        for param in inspect.signature(self.function).parameters.values():
            if param.kind in {param.POSITIONAL_ONLY, param.VAR_POSITIONAL, param.VAR_KEYWORD}:
                raise ValueError("Tool functions need named, typed parameters")
            if param.annotation is param.empty:
                raise ValueError("Tool parameters need type annotations")
        hints = get_type_hints(self.function)
        fields: dict[str, Any] = {
            p.name: (hints[p.name], ... if p.default is p.empty else p.default)
            for p in inspect.signature(self.function).parameters.values()
        }
        return create_model("ToolArguments", __config__=ConfigDict(extra="forbid"), **fields)

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.description or inspect.getdoc(self.function) or "",
                "parameters": self.argument_model().model_json_schema(),
            },
        }


def run_tools(
    client: RegoloClient,
    prompt: str | list[dict] | None,
    tools: list[FunctionTool],
    *,
    max_rounds: int = 5,
    max_calls: int = 32,
    **options,
) -> dict:
    """Bounded model/tool exchange. Never retries tools or executes unknown names."""
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError("max_rounds must be a positive integer")
    if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
        raise ValueError("max_calls must be a positive integer")
    if any(key in options for key in {"stream", "full_output", "tools", "n"}):
        raise ValueError("run_tools controls stream, full_output, tools and n")
    registered = {tool.tool_name: tool for tool in tools}
    if len(registered) != len(tools) or not tools:
        raise ValueError("Provide at least one tool, with unique names")
    schemas = [tool.schema() for tool in tools]
    seen: set[str] = {
        message["tool_call_id"]
        for message in client.conversation.get_lines()
        if isinstance(message.get("tool_call_id"), str)
    }
    remaining_calls = max_calls
    for round_number in range(max_rounds):
        round_options = dict(options)
        if round_number and "tool_choice" in round_options:
            round_options["tool_choice"] = "auto"
        response = client.run_chat(
            prompt if round_number == 0 else None, full_output=True, tools=schemas, **round_options
        )
        choice = response["choices"][0]
        if choice.get("finish_reason") in {"length", "content_filter"} or choice["message"].get(
            "refusal"
        ):
            raise ValueError("Model did not complete the tool decision; no functions were executed")
        calls = response["choices"][0]["message"].get("tool_calls") or []
        if not calls:
            return response
        if round_number == max_rounds - 1:
            raise RuntimeError("Tool round limit reached; inspect conversation before continuing")
        if len(calls) > remaining_calls:
            raise RuntimeError("Tool call limit reached; no additional tools were executed")
        remaining_calls -= len(calls)
        # Preflight the entire batch before executing any local function.
        pending = []
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise ValueError("Malformed function tool call")
            call_id = call.get("id")
            name = call.get("function", {}).get("name")
            if (
                call.get("type") != "function"
                or not isinstance(name, str)
                or name not in registered
            ):
                raise ValueError("Model requested an unregistered function tool")
            if not isinstance(call_id, str) or not call_id or call_id in seen:
                raise ValueError("Missing or repeated tool-call ID; no tool was replayed")
            seen.add(call_id)
            encoded_arguments = call["function"].get("arguments")
            if not isinstance(encoded_arguments, str):
                raise ValueError("Tool arguments must be a JSON string")
            arguments = json.loads(encoded_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            tool = registered[name]
            validated = tool.argument_model().model_validate(arguments, strict=True)
            pending.append(
                (
                    tool,
                    call_id,
                    {key: getattr(validated, key) for key in type(validated).model_fields},
                )
            )
        for tool, call_id, arguments in pending:
            result = tool.function(**arguments)
            if inspect.isawaitable(result):
                raise ValueError("Synchronous tool returned an awaitable")
            content = (
                result
                if isinstance(result, str)
                else TypeAdapter(type(result)).dump_json(result).decode()
            )
            client.add_tool_result(call_id, content)
    raise AssertionError("Unreachable")


def parse_structured[T: BaseModel](response: dict, response_model: type[T]) -> T:
    choice = response["choices"][0]
    message = choice["message"]
    if message.get("refusal") or choice.get("finish_reason") in {"length", "content_filter"}:
        raise ValueError("Model refused or could not complete the structured response")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Structured response has no text content")
    return response_model.model_validate_json(content, strict=True)
