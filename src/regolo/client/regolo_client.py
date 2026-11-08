import json
import os
from base64 import b64decode
from collections.abc import Generator
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, TypeAlias, overload

import httpx
from pydantic import BaseModel

import regolo
from regolo.chat import (
    ChatAccumulator,
    FunctionTool,
    image_part,
    parse_structured,
    run_tools,
    text_chunks,
    text_part,
)
from regolo.instance.regolo_instance import RegoloInstance
from regolo.instance.structures.conversation_model import Conversation, ConversationLine
from regolo.keys.keys import KeysHandler
from regolo.models.models import ModelsHandler
from regolo.transport import DEFAULT_TIMEOUT, api_url, json_object, managed_http

REGOLO_URL = "https://api.regolo.ai"
COMPLETIONS_URL_PATH = "/v1/completions"
CHAT_COMPLETIONS_URL_PATH = "/v1/chat/completions"
IMAGE_GENERATION_URL_PATH = "/v1/images/generations"
EMBEDDINGS_URL_PATH = "/v1/embeddings"
AUDIO_TRANSCRIPTION_URL_PATH = "/v1/audio/transcriptions"
RERANK_URL_PATH = "/v1/rerank"


Role: TypeAlias = str
Content: TypeAlias = str | None
JSONDict: TypeAlias = dict[str, Any]
TextStream: TypeAlias = Generator[str, None, None]
ChatStream: TypeAlias = Generator[tuple[Role, str], None, None]
JSONStream: TypeAlias = Generator[JSONDict, None, None]


def _merge_payload(payload: dict, options: dict, extra_body: dict | None = None) -> dict:
    """Native endpoint options, with no accidental identity/body overrides."""
    options = dict(options)
    nested = options.pop("extra_body", None)
    if nested is not None:
        if extra_body is not None:
            raise ValueError("extra_body was supplied twice")
        extra_body = nested
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("extra_body must be an object")
    extra_body = extra_body or {}
    conflict = (set(extra_body) & set(payload)) | (set(options) & (set(payload) | set(extra_body)))
    if conflict:
        raise ValueError(f"extra_body cannot override named parameters: {sorted(conflict)}")
    result = {**payload, **extra_body, **options}
    if result.get("max_completion_tokens") is not None and result.get("max_tokens") is not None:
        if result["max_tokens"] != 200:
            raise ValueError("Use max_tokens or max_completion_tokens, not both")
        result.pop("max_tokens")
    return result


def safe_post(
    client: httpx.Client,
    url_to_query: str,
    json_to_query: Optional[dict] = None,
    headers_to_query: Optional[dict] = None,
    max_retries: int = 0,
) -> httpx.Response:
    """
    Sends a POST request using the provided HTTPX client.

    :param client: The instance of an HTTPX client to use for sending the request.
    :param url_to_query: The URL to which the POST request is sent.
    :param json_to_query: The JSON payload to include in the request body. (Optional)
    :param headers_to_query: The headers to include in the request. (Optional)
    :param max_retries: Opt-in retries for HTTP 5xx responses. Defaults to zero.

    :return: The HTTPX response object.

    :raises httpx.HTTPStatusError: If the request fails with an HTTP error status.
    """

    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer")

    def post() -> httpx.Response:
        return client.post(url=url_to_query, json=json_to_query, headers=headers_to_query)

    response = post()
    for _ in range(max_retries):
        if not 500 <= response.status_code < 600:
            break
        response = post()
    response.raise_for_status()
    return response


class RegoloClient:
    def __init__(
        self,
        chat_model: Optional[str] = None,
        embedder_model: Optional[str] = None,
        image_generation_model: Optional[str] = None,
        audio_transcription_model: Optional[str] = None,
        reranker_model: Optional[str] = None,
        api_key: Optional[str] = None,
        alternative_url: Optional[str] = None,
        pre_existent_conversation: Optional[Conversation] = None,
        pre_existent_client: httpx.Client | None = None,
        *,
        request_timeout: float | httpx.Timeout = DEFAULT_TIMEOUT,
        check_models: bool | None = None,
    ) -> None:
        """
        Initialize the client for regolo.ai HTTP API.

        :param chat_model: The regolo.ai chat model to use.
            (Defaults to regolo.default_model)
        :param embedder_model: The regolo.ai embedder model to use for text embeddings.
            (Optional)
        :param image_generation_model: The regolo.ai model to use for image generation.
            (Optional)
        :param audio_transcription_model: The regolo.ai model to use for audio transcription.
            (Optional)
        :param reranker_model: The regolo.ai model to use for reranking search results.
            (Optional)
        :param api_key: The API key for regolo.ai.
            (Defaults to regolo.default_key)
        :param alternative_url: Base URL of the regolo HTTP server.
            (Optional)
        :param pre_existent_conversation: An existing conversation instance to continue chatting with.
            (Optional)
        :param pre_existent_client: An existing httpx.Client instance to use.
            (Optional)
        """

        model = regolo.default_chat_model if chat_model is None else chat_model
        embedder_model = regolo.default_embedder_model if embedder_model is None else embedder_model
        image_generation_model = (
            regolo.default_image_generation_model
            if image_generation_model is None
            else image_generation_model
        )
        audio_transcription_model = (
            regolo.default_audio_transcription_model
            if audio_transcription_model is None
            else audio_transcription_model
        )
        reranker_model = regolo.default_reranker_model if reranker_model is None else reranker_model
        api_key = KeysHandler.check_key(regolo.default_key if api_key is None else api_key)
        base_url = (
            os.getenv("REGOLO_URL", REGOLO_URL) if alternative_url is None else alternative_url
        )
        api_url(base_url, "")
        self._check_models = check_models
        self._chat_active = False
        self._catalog: list[JSONDict] | None = None
        self.last_response: JSONDict | None = None
        self.usage_totals: dict[str, int] = {}
        self.request_timeout = request_timeout
        client = (
            httpx.Client(
                base_url=base_url,
                timeout=request_timeout,
            )
            if pre_existent_client is None
            else pre_existent_client
        )

        self.instance = RegoloInstance(
            chat_model=model,
            embedder_model=embedder_model,
            image_generation_model=image_generation_model,
            audio_transcription_model=audio_transcription_model,
            api_key=api_key,
            previous_conversations=pre_existent_conversation,
            reranker_model=reranker_model,
            client=client,
            base_url=base_url,
            check_models=check_models,
            request_timeout=request_timeout,
        )
        self.instance._owns_client = pre_existent_client is None

    def close(self) -> None:
        """Close an owned transport. Caller-provided transports remain open."""
        self.instance.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @classmethod
    def from_instance(
        cls, instance: RegoloInstance, alternative_url: Optional[str] = None
    ) -> "RegoloClient":
        """Creates RegoloClient from instance."""
        cloned = cls(
            api_key=instance.api_key,
            chat_model=instance.chat_model,
            embedder_model=instance.embedder_model,
            image_generation_model=instance.image_generation_model,
            audio_transcription_model=instance.audio_transcription_model,
            reranker_model=instance.reranker_model,
            alternative_url=alternative_url if alternative_url is not None else instance.base_url,
            pre_existent_client=instance.client,
            pre_existent_conversation=instance.conversation,
            check_models=instance.check_models,
            request_timeout=instance.request_timeout,
        )
        # None is a stored setting too; constructor defaults may have changed
        # since the original instance was created.
        for name in (
            "chat_model",
            "embedder_model",
            "image_generation_model",
            "audio_transcription_model",
            "reranker_model",
        ):
            setattr(cloned.instance, name, getattr(instance, name))
        return cloned

    def change_model(self, model: str) -> None:
        """Change the model used in this instance of regolo_client"""
        self.instance.chat_model = ModelsHandler.check_model(
            model,
            self.instance.base_url,
            self.instance.api_key,
            self.instance.client,
            enabled=self._check_models,
        )

    @property
    def conversation(self) -> Conversation:
        return self.instance.conversation

    @property
    def last_usage(self) -> JSONDict | None:
        return (self.last_response or {}).get("usage")

    def _remember_response(self, response: JSONDict) -> None:
        self.last_response = deepcopy(response)
        usage = response.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                self.usage_totals[key] = self.usage_totals.get(key, 0) + value

    def add_message(
        self, role: str, content: str | list[JSONDict] | None = None, **fields: Any
    ) -> None:
        if self._chat_active:
            raise RuntimeError("Consume or close the active chat stream before editing history")
        self.conversation.append(role, content, **fields)

    def add_tool_result(self, tool_call_id: str, content: Any) -> None:
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError("tool_call_id is required")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        self.add_message("tool", content, tool_call_id=tool_call_id)

    def fork(self) -> "RegoloClient":
        """Independent conversation and transport, retaining selected models."""
        cloned = RegoloClient(
            api_key=self.instance.api_key,
            chat_model=self.instance.chat_model,
            embedder_model=self.instance.embedder_model,
            image_generation_model=self.instance.image_generation_model,
            audio_transcription_model=self.instance.audio_transcription_model,
            reranker_model=self.instance.reranker_model,
            alternative_url=self.instance.base_url,
            pre_existent_conversation=self.conversation.fork(),
            request_timeout=self.request_timeout,
            check_models=self._check_models,
        )
        for name in (
            "chat_model",
            "embedder_model",
            "image_generation_model",
            "audio_transcription_model",
            "reranker_model",
        ):
            setattr(cloned.instance, name, getattr(self.instance, name))
        return cloned

    def get_model_catalog(
        self, *, mode: str | None = None, supports: list[str] | None = None, refresh: bool = False
    ) -> list[JSONDict]:
        """Full catalog with opt-in filters; retain all server metadata."""
        if self._catalog is None or refresh:
            self._catalog = ModelsHandler.get_catalog(
                self.instance.base_url or REGOLO_URL,
                self.instance.api_key,
                self.instance.get_client(),
            )
        return deepcopy(
            [
                row
                for row in self._catalog
                if (mode is None or row.get("mode") == mode)
                and all(
                    row.get(key if key.startswith("supports_") else "supports_" + key) is True
                    for key in supports or []
                )
            ]
        )

    def get_model_info(self, model: str | None = None, *, refresh: bool = False) -> JSONDict:
        model = model or self.instance.chat_model
        rows = [
            row
            for row in self.get_model_catalog(refresh=refresh)
            if row.get("model_group", row.get("id")) == model
        ]
        if len(rows) != 1:
            raise LookupError("Model was not found uniquely in the authenticated catalog")
        return rows[0]

    def run_tools(
        self,
        user_prompt: str | list[JSONDict] | None,
        tools: list[FunctionTool],
        *,
        max_rounds: int = 5,
        **options: Any,
    ) -> JSONDict:
        return run_tools(self, user_prompt, tools, max_rounds=max_rounds, **options)

    def run_structured[T: BaseModel](
        self, user_prompt: str | list[JSONDict] | None, response_model: type[T], **options: Any
    ) -> T:
        if any(key in options for key in {"response_format", "stream", "full_output"}):
            raise ValueError("run_structured controls response_format, stream and full_output")
        response = self.run_chat(
            user_prompt,
            full_output=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            },
            **options,
        )
        return parse_structured(response, response_model)

    def vision(
        self,
        prompt: str,
        images: list[str | Path | bytes],
        *,
        mime_type: str | None = None,
        **options: Any,
    ):
        return self.run_chat(
            [text_part(prompt), *[image_part(i, mime_type=mime_type) for i in images]], **options
        )

    def ocr(
        self,
        image: str | Path | bytes,
        *,
        prompt: str = "Convert the document to markdown.",
        mime_type: str | None = None,
        **options: Any,
    ):
        """OCR through chat; select a catalog model with mode=ocr first."""
        options.setdefault("skip_special_tokens", False)
        return self.vision(prompt, [image], mime_type=mime_type, **options)

    @staticmethod
    @overload
    def get_available_models(
        api_key: str,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        model_info: Literal[False] = False,
        client: httpx.Client | None = None,
    ) -> list[str]: ...

    @staticmethod
    @overload
    def get_available_models(
        api_key: str,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        *,
        model_info: Literal[True],
        client: httpx.Client | None = None,
    ) -> list[JSONDict]: ...

    @staticmethod
    @overload
    def get_available_models(
        api_key: str,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        model_info: bool = False,
        client: httpx.Client | None = None,
    ) -> List[str] | List[dict]: ...

    @staticmethod
    def get_available_models(
        api_key: str,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        model_info: bool = False,
        client: httpx.Client | None = None,
    ) -> List[str] | List[dict]:
        """
        Gets all available models on regolo.ai.

        :param base_url: Base URL of the regolo HTTP server.
        :param api_key: The API key for regolo.ai (Defaults to regolo.default_key if it exists)
        :param model_info: Whether to retrieve information about the model (Defaults to False)
        :param client: Caller-owned HTTPX transport, or None to create a temporary transport.

        :return model_info=False: A list of available models (list[str])
        :return model_info=True: A list of available models and their information (list[dict])
        """

        # Validate API key
        api_key = KeysHandler.check_key(api_key)

        return ModelsHandler.get_models(
            base_url=base_url, api_key=api_key, model_info=model_info, client=client
        )

    @staticmethod
    def create_stream_generator(
        client: httpx.Client,
        base_url: str,
        payload: Optional[dict] = None,
        files: Optional[dict] = None,
        data: Optional[dict] = None,
        headers: dict | None = None,
        full_output: bool = False,
        search_url: str = "",
        output_handler: Callable[[Dict], Any] | None = None,
    ) -> Generator[Any, Any, None]:
        """
        Yields generators for streams from regolo.ai (generalized for JSON and multipart requests).

        :param client: The httpx.Client instance to use.
        :param base_url: Base URL of the regolo HTTP server.
        :param payload: The JSON request payload to send (for JSON requests).
            (Optional)
        :param files: The files dict for multipart requests.
            (Optional)
        :param data: The form data dict for multipart requests.
            (Optional)
        :param headers: The request headers.
        :param full_output: Whether to return the full response.
        :param search_url: The URL for the search request.
        :param output_handler: A function that processes responses if full_output=False.
        :return: A generator that yields streamed responses from regolo.ai.
        """
        if not full_output and output_handler is None:
            raise ValueError("output_handler is required when full_output=False")
        # Determine the request type and prepare arguments
        request_kwargs: dict[str, Any] = {"url": api_url(base_url, search_url), "headers": headers}

        if payload is not None:
            # JSON request
            request_kwargs["json"] = payload
        elif files is not None and data is not None:
            # Multipart form data request
            request_kwargs["files"] = files
            request_kwargs["data"] = data
        else:
            raise ValueError(
                "Either payload (for JSON) or both files and data (for multipart) must be provided"
            )

        def events(lines):
            parts: list[str] = []
            for line in lines:
                if not line:
                    if parts:
                        yield "\n".join(parts)
                        parts = []
                elif line.startswith("data:"):
                    parts.append(line[5:].lstrip(" "))
                # Comments, id, retry and event fields are SSE metadata.
            if parts:
                yield "\n".join(parts)

        with client.stream("POST", **request_kwargs) as response:
            response.raise_for_status()
            finished = False
            for event in events(response.iter_lines()):
                if event.strip() == "[DONE]":
                    return
                try:
                    data_chunk = json.loads(event)
                except ValueError as exc:
                    raise ValueError("Malformed JSON in inference stream") from exc
                if not isinstance(data_chunk, dict):
                    raise ValueError("Inference stream event must be a JSON object")
                if data_chunk.get("error"):
                    raise RuntimeError(f"Inference stream failed: {data_chunk['error']}")
                if any(
                    c.get("finish_reason") is not None for c in data_chunk.get("choices", [])
                ) or data_chunk.get("type", "").endswith(".done"):
                    finished = True
                if full_output:
                    yield data_chunk
                else:
                    assert output_handler is not None
                    output = output_handler(data_chunk)
                    if output is not None:
                        yield output
            if not finished:
                raise ValueError("Inference stream ended before completion")

    # Completions

    @staticmethod
    @overload
    def static_completions(
        prompt: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: Literal[False] = False,
        max_tokens: int | None = 200,
        temperature: Optional[float] = 0.5,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: Literal[False] = False,
        extra_body: dict | None = None,
        check_models: bool | None = None,
        **options: Any,
    ) -> str: ...

    @staticmethod
    @overload
    def static_completions(
        prompt: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: Literal[False] = False,
        max_tokens: int | None = 200,
        temperature: Optional[float] = 0.5,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        *,
        full_output: Literal[True],
        extra_body: dict | None = None,
        check_models: bool | None = None,
        **options: Any,
    ) -> JSONDict: ...

    @staticmethod
    @overload
    def static_completions(
        prompt: str,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: Literal[True],
        max_tokens: int | None = 200,
        temperature: Optional[float] = 0.5,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: Literal[False] = False,
        extra_body: dict | None = None,
        check_models: bool | None = None,
        **options: Any,
    ) -> TextStream: ...

    @staticmethod
    @overload
    def static_completions(
        prompt: str,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: Literal[True],
        max_tokens: int | None = 200,
        temperature: Optional[float] = 0.5,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: Literal[True],
        extra_body: dict | None = None,
        check_models: bool | None = None,
        **options: Any,
    ) -> JSONStream: ...

    @staticmethod
    @overload
    def static_completions(
        prompt: str,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: Literal[True],
        max_tokens: int | None = 200,
        temperature: Optional[float] = 0.5,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: bool = False,
        extra_body: dict | None = None,
        check_models: bool | None = None,
        **options: Any,
    ) -> TextStream | JSONStream: ...

    @staticmethod
    @overload
    def static_completions(
        prompt: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: bool = False,
        max_tokens: int | None = 200,
        temperature: Optional[float] = 0.5,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: bool = False,
        extra_body: dict | None = None,
        check_models: bool | None = None,
        **options: Any,
    ) -> str | JSONDict | TextStream | JSONStream: ...

    @staticmethod
    @managed_http
    def static_completions(
        prompt: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: bool = False,
        max_tokens: int | None = 200,
        temperature: Optional[float] = 0.5,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: bool = False,
        extra_body: dict | None = None,
        check_models: bool | None = None,
        **options: Any,
    ) -> str | JSONDict | TextStream | JSONStream:
        """
        Will return generators for stream=True and values for stream=False
        Send a prompt to regolo server and get the generated response.

        :param prompt: The prompt in input to the LLM.
        :param model: The regolo.ai model to use. (Optional)
        :param api_key: The API key for regolo.ai. (Optional)
        :param stream: Whether to stream the prompt from regolo.ai. (Defaults to False)
        :param max_tokens: Maximum number of tokens to generate. (Defaults to 200)
        :param temperature: Sampling temperature for randomness. (Optional)
        :param top_p: Nucleus sampling parameter. (Optional)
        :param top_k: Top-k sampling parameter. (Optional)
        :param client: httpx client to use. (Optional)
        :param base_url: Base URL of the regolo HTTP server. (Defaults to REGOLO_URL)
        :param full_output: Whether to return the full response. (Defaults to False)

        :return for stream=True, full_output=False: Generator yielding text chunks (completions) or role/content tuples (chat).
        :return for stream=True, full_output=True: Generator yielding full JSON event dictionaries.
        :return for stream=False, full_output=False: String with response from regolo.ai.
        :return for stream=False, full_output=True: Full JSON response dictionary.
        :param extra_body: Additional endpoint fields; cannot override named parameters.
        """

        def handle_search_text_completions(data: dict) -> str | None:
            """
            Internal method, describes how RegoloClient.create_stream_generator() should handle output from completions.
            """
            choices = data.get("choices") or []
            return choices[0].get("text") if choices else None

        # Use the default API key if none is provided
        if api_key is None:
            api_key = regolo.default_key

        # Validate API key
        api_key = KeysHandler.check_key(api_key)

        # Use the default model if none is specified
        if model is None:
            model = regolo.default_chat_model

        # Validate the selected model
        ModelsHandler.check_model(
            model=model, api_key=api_key, base_url=base_url, client=client, enabled=check_models
        )

        # managed_http supplies an owned client when none was passed.
        assert client is not None

        # Construct the payload for the API request
        payload = {
            "prompt": prompt,
            "stream": stream,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

        payload = _merge_payload(payload, options, extra_body)
        # Remove None values from the payload to avoid unnecessary parameters
        payload = {k: v for k, v in payload.items() if v is not None}

        # Set authorization header
        headers = {"Authorization": api_key}

        if stream:
            # If streaming is enabled, return a generator for handling streamed responses
            return RegoloClient.create_stream_generator(
                client=client,
                base_url=base_url,
                payload=payload,
                headers=headers,
                full_output=full_output,
                search_url=os.getenv("COMPLETIONS_URL_PATH", COMPLETIONS_URL_PATH),
                output_handler=handle_search_text_completions,
            )
        else:
            # Send a synchronous POST request
            response = safe_post(
                client=client,
                url_to_query=api_url(
                    base_url, os.getenv("COMPLETIONS_URL_PATH", COMPLETIONS_URL_PATH)
                ),
                json_to_query=payload,
                headers_to_query=headers,
            )

            if full_output:
                # Return the full JSON response if requested
                return json_object(response)
            else:
                # Extract and return only the generated text
                return json_object(response)["choices"][0]["text"]

    @overload
    def completions(
        self,
        prompt: str,
        stream: Literal[False] = False,
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        full_output: Literal[False] = False,
        **options: Any,
    ) -> str: ...

    @overload
    def completions(
        self,
        prompt: str,
        stream: Literal[False] = False,
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        *,
        full_output: Literal[True],
        **options: Any,
    ) -> JSONDict: ...

    @overload
    def completions(
        self,
        prompt: str,
        stream: Literal[True],
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        full_output: Literal[False] = False,
        **options: Any,
    ) -> TextStream: ...

    @overload
    def completions(
        self,
        prompt: str,
        stream: Literal[True],
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        *,
        full_output: Literal[True],
        **options: Any,
    ) -> JSONStream: ...

    @overload
    def completions(
        self,
        prompt: str,
        stream: Literal[True],
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        full_output: bool = False,
        **options: Any,
    ) -> TextStream | JSONStream: ...

    @overload
    def completions(
        self,
        prompt: str,
        stream: bool = False,
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        full_output: bool = False,
        **options: Any,
    ) -> str | JSONDict | TextStream | JSONStream: ...

    def completions(
        self,
        prompt: str,
        stream: bool = False,
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        full_output: bool = False,
        **options: Any,
    ) -> str | JSONDict | TextStream | JSONStream:
        """
        Will return generators for stream=True and values for stream=False
        Performs requests to completions endpoint from the RegoloClient instance.

        :param prompt: The input prompt to the LLM.
        :param stream: Whether to stream the prompt from regolo.ai. (Defaults to False)
        :param max_tokens: Maximum number of tokens to generate. (Defaults to 200)
        :param temperature: Sampling temperature for randomness. (Optional)
        :param top_p: Nucleus sampling parameter. (Optional)
        :param top_k: Top-k sampling parameter. (Optional)
        :param full_output: Whether to return the full response. (Defaults to False)

        :return for stream=True, full_output=False: Generator yielding text chunks (completions) or role/content tuples (chat).
        :return for stream=True, full_output=True: Generator yielding full JSON event dictionaries.
        :return for stream=False, full_output=False: String with response from regolo.ai.
        :return for stream=False, full_output=True: Full JSON response dictionary.
        """
        base_url = self.instance.get_base_url()
        if base_url is None:
            base_url = os.getenv("REGOLO_URL", REGOLO_URL)
        response = self.static_completions(
            prompt=prompt,
            model=self.instance.get_model(),
            api_key=self.instance.get_api_key(),
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            client=self.instance.get_client(),
            base_url=base_url,
            full_output=full_output,
            check_models=self._check_models,
            **options,
        )

        return response

    # Chat completions

    @staticmethod
    @overload
    def static_chat_completions(
        messages: Conversation | list[JSONDict],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: Literal[False] = False,
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: Literal[False] = False,
        extra_body: dict | None = None,
        check_models: bool | None = None,
        **options: Any,
    ) -> tuple[Role, Content]: ...

    @staticmethod
    @overload
    def static_chat_completions(
        messages: Conversation | list[JSONDict],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: Literal[False] = False,
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        *,
        full_output: Literal[True],
        extra_body: dict | None = None,
        check_models: bool | None = None,
        **options: Any,
    ) -> JSONDict: ...

    @staticmethod
    @overload
    def static_chat_completions(
        messages: Conversation | list[JSONDict],
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: Literal[True],
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: Literal[False] = False,
        extra_body: dict | None = None,
        check_models: bool | None = None,
        **options: Any,
    ) -> ChatStream: ...

    @staticmethod
    @overload
    def static_chat_completions(
        messages: Conversation | list[JSONDict],
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: Literal[True],
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: Literal[True],
        extra_body: dict | None = None,
        check_models: bool | None = None,
        **options: Any,
    ) -> JSONStream: ...

    @staticmethod
    @overload
    def static_chat_completions(
        messages: Conversation | list[JSONDict],
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: Literal[True],
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: bool = False,
        extra_body: dict | None = None,
        check_models: bool | None = None,
        **options: Any,
    ) -> ChatStream | JSONStream: ...

    @staticmethod
    @overload
    def static_chat_completions(
        messages: Conversation | list[JSONDict],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: bool = False,
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: bool = False,
        extra_body: dict | None = None,
        check_models: bool | None = None,
        **options: Any,
    ) -> ChatStream | JSONStream | tuple[Role, Content] | JSONDict: ...

    @staticmethod
    @managed_http
    def static_chat_completions(
        messages: Conversation | list[JSONDict],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: bool = False,
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: bool = False,
        extra_body: dict | None = None,
        check_models: bool | None = None,
        **options: Any,
    ) -> ChatStream | JSONStream | tuple[Role, Content] | JSONDict:
        """
        Sends a series of chat messages to the vLLM server and gets the response.

        :param messages: A list of messages in the format [{"role": "user"|"assistant", "content": "message"}].
        :param model: The regolo.ai model to use. (Optional)
        :param api_key: The API key for regolo.ai. (Optional)
        :param stream: Whether to stream the prompt from regolo.ai. (Defaults to False)
        :param max_tokens: Maximum number of tokens to generate. (Defaults to 200)
        :param temperature: Sampling temperature for randomness. (Optional)
        :param top_p: Nucleus sampling parameter. (Optional)
        :param top_k: Top-k sampling parameter. (Optional)
        :param client: httpx client to use. (Optional)
        :param base_url: Base URL of the regolo HTTP server. (Defaults to REGOLO_URL)
        :param full_output: Whether to return full response. (Defaults to False)

        :return for stream=True, full_output=False: Generator yielding text chunks (completions) or role/content tuples (chat).
        :return for stream=True, full_output=True: Generator yielding full JSON event dictionaries.
        :return for stream=False, full_output=False: String, with response from regolo.ai.
        :return for stream=False, full_output=True: Full JSON response dictionary.
        :param extra_body: Additional endpoint fields; cannot override named parameters.
        """

        # Use the default API key if not provided
        if api_key is None:
            api_key = regolo.default_key

        # Validate the API key
        api_key = KeysHandler.check_key(api_key)

        # Use the default model if not specified
        if model is None:
            model = regolo.default_chat_model

        # Validate the model
        ModelsHandler.check_model(
            model=model, base_url=base_url, api_key=api_key, client=client, enabled=check_models
        )

        # Convert the Conversation object to a list of message dictionaries
        if isinstance(messages, Conversation):
            messages = messages.get_lines()

        # managed_http supplies an owned client when none was passed.
        assert client is not None

        # Construct the payload for the API request
        payload = {
            "model": model,
            "stream": stream,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

        payload = _merge_payload(payload, options, extra_body)
        # Remove None values from the payload to avoid unnecessary parameters
        payload = {k: v for k, v in payload.items() if v is not None}

        # Set authorization header
        headers = {"Authorization": api_key}

        if stream:
            # If streaming, return a generator for handling the response
            events = RegoloClient.create_stream_generator(
                client=client,
                base_url=base_url,
                payload=payload,
                headers=headers,
                full_output=True,
                search_url=os.getenv("CHAT_COMPLETIONS_URL_PATH", CHAT_COMPLETIONS_URL_PATH),
            )
            if full_output:
                return events

            def simplified():
                try:
                    for event in events:
                        yield from text_chunks(event)
                finally:
                    events.close()

            return simplified()
        else:
            # Send a synchronous POST request
            response = json_object(
                safe_post(
                    client=client,
                    url_to_query=api_url(
                        base_url, os.getenv("CHAT_COMPLETIONS_URL_PATH", CHAT_COMPLETIONS_URL_PATH)
                    ),
                    json_to_query=payload,
                    headers_to_query=headers,
                )
            )

            if full_output:
                # Return the full response if requested
                return response
            else:
                # Extract role and content from response
                output_role = response["choices"][0]["message"]["role"]
                output_content = response["choices"][0]["message"]["content"]
                return output_role, output_content

    def add_prompt_to_chat(self, prompt: str, role: str):
        """
        Adds a prompt to the chat as the role specified

        :param prompt: The prompt to be added.
        :param role: The role of the prompt to be added.

        :example:
            client = RegoloClient()
            client.add_prompt_to_chat(prompt="how are you?", role="user")
            print(client.run_chat())
        """

        self.add_message(role, prompt)

    def clear_conversations(self) -> None:
        """clear all prompts to start new conversations."""
        if self._chat_active:
            raise RuntimeError("Consume or close the active chat stream before clearing history")
        self.instance.clear_conversation()

    @overload
    def run_chat(
        self,
        user_prompt: str | list[JSONDict] | None = None,
        stream: Literal[False] = False,
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        full_output: Literal[False] = False,
        extra_body: dict | None = None,
        **options: Any,
    ) -> tuple[Role, Content]: ...

    @overload
    def run_chat(
        self,
        user_prompt: str | list[JSONDict] | None = None,
        stream: Literal[False] = False,
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        *,
        full_output: Literal[True],
        extra_body: dict | None = None,
        **options: Any,
    ) -> JSONDict: ...

    @overload
    def run_chat(
        self,
        user_prompt: str | list[JSONDict] | None = None,
        *,
        stream: Literal[True],
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        full_output: Literal[False] = False,
        extra_body: dict | None = None,
        **options: Any,
    ) -> ChatStream: ...

    @overload
    def run_chat(
        self,
        user_prompt: str | list[JSONDict] | None = None,
        *,
        stream: Literal[True],
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        full_output: Literal[True],
        extra_body: dict | None = None,
        **options: Any,
    ) -> JSONStream: ...

    @overload
    def run_chat(
        self,
        user_prompt: str | list[JSONDict] | None = None,
        *,
        stream: Literal[True],
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        full_output: bool = False,
        extra_body: dict | None = None,
        **options: Any,
    ) -> ChatStream | JSONStream: ...

    @overload
    def run_chat(
        self,
        user_prompt: str | list[JSONDict] | None = None,
        stream: bool = False,
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        full_output: bool = False,
        extra_body: dict | None = None,
        **options: Any,
    ) -> ChatStream | JSONStream | tuple[Role, Content] | JSONDict: ...

    def run_chat(
        self,
        user_prompt: str | list[JSONDict] | None = None,
        stream: bool = False,
        max_tokens: int | None = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        full_output: bool = False,
        extra_body: dict | None = None,
        **options: Any,
    ) -> ChatStream | JSONStream | tuple[Role, Content] | JSONDict:
        """Run a text/multimodal turn, retaining tools, reasoning and token usage.

        A stream starts lazily. Closing/canceling it records no assistant reply.
        Completed replies retain full metadata even with simplified output.
        One stateful conversation selects choice zero; use stateless calls for n>1.
        """
        if options.get("n", (extra_body or {}).get("n", 1)) != 1:
            raise ValueError(
                "Stateful chat requires n=1; use static_chat_completions for multiple choices"
            )

        def begin():
            if self._chat_active:
                raise RuntimeError(
                    "Consume or close the active chat stream before starting another turn"
                )
            self.instance.get_client()
            self._chat_active = True
            if user_prompt is not None:
                try:
                    self.instance.add_line(ConversationLine(role="user", content=user_prompt))
                except Exception:
                    self._chat_active = False
                    raise

        def request(is_stream: bool):
            return self.static_chat_completions(
                messages=self.instance.get_conversation(),
                model=self.instance.get_model(),
                stream=is_stream,
                api_key=self.instance.get_api_key(),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                client=self.instance.get_client(),
                base_url=self.instance.base_url or REGOLO_URL,
                full_output=True,
                extra_body=extra_body,
                check_models=self._check_models,
                **options,
            )

        if stream:

            def remember() -> Generator[Any, None, None]:
                begin()
                source = None
                try:
                    source = request(True)
                    assert isinstance(source, Generator)
                    accumulated = ChatAccumulator()
                    for chunk in source:
                        assert isinstance(chunk, dict)
                        accumulated.add(chunk)
                        if full_output:
                            yield chunk
                        else:
                            yield from text_chunks(chunk)
                    completed = accumulated.response()
                    self.instance.add_line(
                        ConversationLine.model_validate(completed["choices"][0]["message"])
                    )
                    self._remember_response(completed)
                finally:
                    if isinstance(source, Generator):
                        source.close()
                    self._chat_active = False

            return remember()
        begin()
        try:
            response = request(False)
            assert isinstance(response, dict)
            message = response["choices"][0]["message"]
            self.instance.add_line(ConversationLine.model_validate(message))
            self._remember_response(response)
            return (
                response
                if full_output
                else (message.get("role", "assistant"), message.get("content"))
            )
        finally:
            self._chat_active = False

    # Create images
    @staticmethod
    @overload
    def static_create_image(
        prompt: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        n: int = 1,
        quality: str = "standard",
        size: str = "1024x1024",
        style: str = "realistic",
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: Literal[False] = False,
        check_models: bool | None = None,
        **options: Any,
    ) -> list[bytes]: ...

    @staticmethod
    @overload
    def static_create_image(
        prompt: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        n: int = 1,
        quality: str = "standard",
        size: str = "1024x1024",
        style: str = "realistic",
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        *,
        full_output: Literal[True],
        check_models: bool | None = None,
        **options: Any,
    ) -> JSONDict: ...

    @staticmethod
    @overload
    def static_create_image(
        prompt: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        n: int = 1,
        quality: str = "standard",
        size: str = "1024x1024",
        style: str = "realistic",
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: bool = False,
        check_models: bool | None = None,
        **options: Any,
    ) -> list[bytes] | dict: ...

    @staticmethod
    @managed_http
    def static_create_image(
        prompt: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        n: int = 1,
        quality: str = "standard",
        size: str = "1024x1024",
        style: str = "realistic",
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: bool = False,
        check_models: bool | None = None,
        **options: Any,
    ) -> list[bytes] | dict:
        """
        Generates an image based on the given prompt using the regolo.ai image model.

        :param prompt: The text prompt to be used for image generation.
        :param model: The regolo.ai image model to use. (Optional)
        :param api_key: The API key for regolo.ai. (Optional)
        :param n: The number of images to generate. (Defaults to 1)
        :param quality: The quality of the image that will be generated. The "hd" value creates images with finer details and greater consistency across the image. (Defaults to "standard")
        :param size: The size of the generated images.
        :param style: The style of the generated images. (Defaults to "realistic")
        :param client: The HTTP client for making requests. (Optional)
        :param base_url: Base URL of the regolo HTTP server. (Defaults to REGOLO_URL)
        :param full_output: Whether to return full response. (Defaults to False)

        :return full_output=True: Dict containing the text of the response.
        :return full_output=False: List containing the images decoded as bytes.
        """

        # Use the default API key if not provided
        if api_key is None:
            api_key = regolo.default_key

        # Validate the API key
        api_key = KeysHandler.check_key(api_key)

        # Use the default model if not specified
        if model is None:
            model = regolo.default_image_generation_model

        # Validate the model
        ModelsHandler.check_model(
            model=model, base_url=base_url, api_key=api_key, client=client, enabled=check_models
        )

        # managed_http supplies an owned client when none was passed.
        assert client is not None

        # Construct the payload for the API request
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "quality": quality,
            "size": size,
            "style": style,
        }

        payload = _merge_payload(payload, options)
        # Remove None values from the payload
        payload = {k: v for k, v in payload.items() if v is not None}

        # Set authorization header
        headers = {"Authorization": api_key}

        # Send a synchronous POST request
        response = json_object(
            safe_post(
                client=client,
                url_to_query=api_url(
                    base_url, os.getenv("IMAGE_GENERATION_URL_PATH", IMAGE_GENERATION_URL_PATH)
                ),
                json_to_query=payload,
                headers_to_query=headers,
            )
        )

        if full_output:
            return response
        else:
            # Extract the image URL from the response
            if any("b64_json" not in item for item in response["data"]):
                raise ValueError(
                    "Server returned image URLs; use full_output=True or request response_format='b64_json'"
                )
            return [b64decode(img_info["b64_json"], validate=True) for img_info in response["data"]]

    @overload
    def create_image(
        self,
        prompt: str,
        n: int = 1,
        quality: str = "standard",
        size: str = "1024x1024",
        style: str = "realistic",
        full_output: Literal[False] = False,
        **options: Any,
    ) -> list[bytes]: ...

    @overload
    def create_image(
        self,
        prompt: str,
        n: int = 1,
        quality: str = "standard",
        size: str = "1024x1024",
        style: str = "realistic",
        *,
        full_output: Literal[True],
        **options: Any,
    ) -> JSONDict: ...

    @overload
    def create_image(
        self,
        prompt: str,
        n: int = 1,
        quality: str = "standard",
        size: str = "1024x1024",
        style: str = "realistic",
        full_output: bool = False,
        **options: Any,
    ) -> list[bytes] | dict: ...

    def create_image(
        self,
        prompt: str,
        n: int = 1,
        quality: str = "standard",
        size: str = "1024x1024",
        style: str = "realistic",
        full_output: bool = False,
        **options: Any,
    ) -> list[bytes] | dict:
        """
        Generates an image based on the given prompt using the regolo.ai image model.

        :param prompt: The text prompt to be used for image generation.
        :param n: The number of images to generate. (Defaults to 1)
        :param quality: The quality of the image that will be generated.
            The "hd" value creates images with finer details and greater consistency across the image.
            (Defaults to "standard")
        :param size: The size of the generated images.
        :param style: The style of the generated images. (Defaults to "vivid")
        :param full_output: Whether to return full response. (Defaults to False)

        :return full_output=True: Dict containing the text of the response.
        :return full_output=False: List containing the images decoded as bytes.
        """

        base_url = self.instance.get_base_url()
        if base_url is None:
            base_url = os.getenv("REGOLO_URL", REGOLO_URL)

        response = self.static_create_image(
            prompt=prompt,
            model=self.instance.get_image_model(),
            api_key=self.instance.get_api_key(),
            n=n,
            quality=quality,
            size=size,
            style=style,
            client=self.instance.get_client(),
            base_url=base_url,
            full_output=full_output,
            check_models=self._check_models,
            **options,
        )

        return response

    # Generate embeddings
    @staticmethod
    @overload
    def static_embeddings(
        input_text: list[str] | str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: Literal[False] = False,
        check_models: bool | None = None,
        **options: Any,
    ) -> list[Any]: ...

    @staticmethod
    @overload
    def static_embeddings(
        input_text: list[str] | str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        *,
        full_output: Literal[True],
        check_models: bool | None = None,
        **options: Any,
    ) -> JSONDict: ...

    @staticmethod
    @overload
    def static_embeddings(
        input_text: list[str] | str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: bool = False,
        check_models: bool | None = None,
        **options: Any,
    ) -> dict | list: ...

    @staticmethod
    @managed_http
    def static_embeddings(
        input_text: list[str] | str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: bool = False,
        check_models: bool | None = None,
        **options: Any,
    ) -> dict | list:
        """

        :param input_text: The text to be embedded.
        :param model: The regolo.ai image model to use. (Optional)
        :param api_key: The API key for regolo.ai. (Optional)
        :param client: HTTP client for making requests. (Optional)
        :param base_url: Base URL of the regolo HTTP server. (Optional)
        :param full_output: Whether to return a full response. (Defaults to False)
        """
        # Use the default API key if none is provided
        if api_key is None:
            api_key = regolo.default_key

        # Validate API key
        api_key = KeysHandler.check_key(api_key)

        if model is None:
            model = regolo.default_embedder_model

        # Validate the selected model
        ModelsHandler.check_model(
            model=model, api_key=api_key, base_url=base_url, client=client, enabled=check_models
        )

        # managed_http supplies an owned client when none was passed.
        assert client is not None

        # Construct the payload for the API request
        payload = {
            "input": input_text,
            "model": model,
        }

        payload = _merge_payload(payload, options)
        # Remove None values from the payload to avoid unnecessary parameters
        payload = {k: v for k, v in payload.items() if v is not None}

        # Set authorization header
        headers = {"Authorization": api_key}

        response = safe_post(
            client=client,
            url_to_query=api_url(base_url, os.getenv("EMBEDDINGS_URL_PATH", EMBEDDINGS_URL_PATH)),
            json_to_query=payload,
            headers_to_query=headers,
        )

        if full_output:
            return json_object(response)
        else:
            return json_object(response)["data"]

    @overload
    def embeddings(
        self,
        input_text: list[str] | str,
        full_output: Literal[False] = False,
        **options: Any,
    ) -> list[Any]: ...

    @overload
    def embeddings(
        self,
        input_text: list[str] | str,
        *,
        full_output: Literal[True],
        **options: Any,
    ) -> JSONDict: ...

    @overload
    def embeddings(
        self,
        input_text: list[str] | str,
        full_output: bool = False,
        **options: Any,
    ) -> dict | list: ...

    def embeddings(
        self,
        input_text: list[str] | str,
        full_output: bool = False,
        **options: Any,
    ) -> dict | list:
        """
        :param input_text: The text to be embedded.
        :param full_output: Whether to return a full response. (Defaults to False)
        """
        base_url = self.instance.get_base_url()
        if base_url is None:
            base_url = os.getenv("REGOLO_URL", REGOLO_URL)

        return self.static_embeddings(
            input_text=input_text,
            full_output=full_output,
            model=self.instance.get_embedder_model(),
            api_key=self.instance.get_api_key(),
            client=self.instance.get_client(),
            base_url=base_url,
            check_models=self._check_models,
            **options,
        )

    @staticmethod
    @overload
    def static_audio_transcription(
        file,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        chunking_strategy: Optional[str | dict] = None,
        include: Optional[List[str]] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        stream: Literal[False] = False,
        temperature: Optional[float] = 0,
        timestamp_granularities: Optional[List[str]] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: Literal[False] = False,
        check_models: bool | None = None,
        **options: Any,
    ) -> str: ...

    @staticmethod
    @overload
    def static_audio_transcription(
        file,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        chunking_strategy: Optional[str | dict] = None,
        include: Optional[List[str]] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Literal["json", "verbose_json"] = "json",
        stream: Literal[False] = False,
        temperature: Optional[float] = 0,
        timestamp_granularities: Optional[List[str]] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: Literal[True],
        check_models: bool | None = None,
        **options: Any,
    ) -> JSONDict: ...

    @staticmethod
    @overload
    def static_audio_transcription(
        file,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        chunking_strategy: Optional[str | dict] = None,
        include: Optional[List[str]] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        stream: Literal[True],
        temperature: Optional[float] = 0,
        timestamp_granularities: Optional[List[str]] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: Literal[False] = False,
        check_models: bool | None = None,
        **options: Any,
    ) -> TextStream: ...

    @staticmethod
    @overload
    def static_audio_transcription(
        file,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        chunking_strategy: Optional[str | dict] = None,
        include: Optional[List[str]] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        stream: Literal[True],
        temperature: Optional[float] = 0,
        timestamp_granularities: Optional[List[str]] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: Literal[True],
        check_models: bool | None = None,
        **options: Any,
    ) -> JSONStream: ...

    @staticmethod
    @overload
    def static_audio_transcription(
        file,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        chunking_strategy: Optional[str | dict] = None,
        include: Optional[List[str]] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        stream: bool = False,
        temperature: Optional[float] = 0,
        timestamp_granularities: Optional[List[str]] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: bool = False,
        check_models: bool | None = None,
        **options: Any,
    ) -> str | JSONDict | TextStream | JSONStream: ...

    @staticmethod
    @managed_http
    def static_audio_transcription(
        file,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        chunking_strategy: Optional[str | dict] = None,
        include: Optional[List[str]] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        stream: bool = False,
        temperature: Optional[float] = 0,
        timestamp_granularities: Optional[List[str]] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: bool = False,
        check_models: bool | None = None,
        **options: Any,
    ) -> str | JSONDict | TextStream | JSONStream:
        """
        Transcribes audio using the regolo.ai audio transcription model.

        :param file: The audio file object (bytes, file-like an object, or path string) to transcribe,
            in formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
        :param model: The name of the model to use.
            Example: faster-whisper-large-v3
            (Optional)
        :param api_key: The API key for regolo.ai.
            (Optional)
        :param chunking_strategy: Controls how audio is cut into chunks.
            Auto or object.
            (Optional)
        :param include: Additional information to include in response.
            (Optional)
        :param language: The language of the input audio in ISO-639-1 format.
            (Optional)
        :param prompt: An optional text to guide the model's style or continue a previous audio segment.
            (Optional)
        :param response_format: The format of the output: ``json``, ``text``, ``srt``, ``verbose_json``, or ``vtt``.
            (Defaults to "json")
        :param stream: If true, stream the response using server-sent events.
            Note: Not supported for whisper-1.
            (Defaults to False)
        :param temperature: The sampling temperature, between 0 and 1.
            (Defaults to 0)
        :param timestamp_granularities: Timestamp granularities: word or segment.
            Requires the ``verbose_json`` format.
            (Optional)
        :param client: httpx client to use.
            (Optional)
        :param base_url: Base URL of the regolo HTTP server.
            (Defaults to REGOLO_URL)
        :param full_output: Whether to return the full response.
            (Defaults to False)

        :return for stream=True: Generator yielding streaming responses.
        :return for stream=False, full_output=True: Dict containing the full response from regolo.ai.
        :return for stream=False, full_output=False: String with transcribed text.
        """

        def handle_search_audio_transcription(output_data: dict) -> Optional[str]:
            """
            Internal method, describes how RegoloClient.create_stream_generator() should handle
            output from audio transcriptions.
            """
            if isinstance(output_data, dict):
                # Extract text content from streaming chunks
                return output_data.get("text", "")
            return ""

        # Use the default API key if none is provided
        if api_key is None:
            api_key = regolo.default_key

        # Validate API key
        api_key = KeysHandler.check_key(api_key)

        if model is None:
            model = regolo.default_audio_transcription_model

        # Validate the selected model
        ModelsHandler.check_model(
            model=model, api_key=api_key, base_url=base_url, client=client, enabled=check_models
        )

        # managed_http supplies an owned client when none was passed.
        assert client is not None

        if response_format not in {"json", "verbose_json", "text", "srt", "vtt"}:
            raise ValueError("Unsupported transcription response_format")
        if timestamp_granularities and response_format != "verbose_json":
            raise ValueError("timestamp_granularities requires verbose_json")
        # Handle different file input types
        try:
            if isinstance(file, (str, Path)):
                # File path provided
                with open(file, "rb") as audio_file:
                    file_content = audio_file.read()
                    file_name = os.path.basename(file)
            elif isinstance(file, bytes):
                # Bytes provided
                file_content = file
                file_name = "audio_file"
            elif hasattr(file, "read"):
                # File-like object provided
                file_content = file.read()
                file_name = os.path.basename(str(getattr(file, "name", "audio_file")))
            else:
                raise ValueError("File must be a path string, bytes, or file-like object")

            files = {"file": (file_name, file_content, "audio/*")}

            # Construct the data payload (form data, not JSON for file uploads)
            data: dict[str, Any] = {
                "model": model,
                "response_format": response_format,
                "stream": str(stream).lower(),
            }

            # Add optional parameters if provided
            if chunking_strategy is not None:
                if isinstance(chunking_strategy, str):
                    data["chunking_strategy"] = chunking_strategy
                else:
                    # Handle object case - convert to JSON string
                    data["chunking_strategy"] = json.dumps(chunking_strategy)

            if include is not None:
                data["include[]"] = include

            if language is not None:
                data["language"] = language
            if prompt is not None:
                data["prompt"] = prompt
            if temperature is not None:
                data["temperature"] = str(temperature)
            if timestamp_granularities is not None:
                data["timestamp_granularities[]"] = timestamp_granularities

            extra_form = options.pop("extra_body", {})
            if not isinstance(extra_form, dict):
                raise ValueError("extra_body must be an object")
            for key, value in _merge_payload({}, options, extra_form).items():
                if key in data or key in {"file", "model", "stream"}:
                    raise ValueError(f"Audio options cannot override named parameter: {key}")
                data[key] = (
                    json.dumps(value) if isinstance(value, (dict, list, bool)) else str(value)
                )
            # Set authorization header
            headers = {"Authorization": api_key}

            if stream:
                # Use the generalized create_stream_generator for streaming
                return RegoloClient.create_stream_generator(
                    client=client,
                    base_url=base_url,
                    files=files,
                    data=data,
                    headers=headers,
                    full_output=full_output,
                    search_url=os.getenv(
                        "AUDIO_TRANSCRIPTION_URL_PATH", AUDIO_TRANSCRIPTION_URL_PATH
                    ),
                    output_handler=handle_search_audio_transcription,
                )

            # Non-streaming case
            response = client.post(
                url=api_url(
                    base_url,
                    os.getenv("AUDIO_TRANSCRIPTION_URL_PATH", AUDIO_TRANSCRIPTION_URL_PATH),
                ),
                files=files,
                data=data,
                headers=headers,
            )
            response.raise_for_status()

        except FileNotFoundError:
            raise FileNotFoundError(f"Audio file not found: {file}")
        except (httpx.HTTPError, ValueError):
            raise

        if response_format in {"text", "srt", "vtt"}:
            return response.text
        response_json = json_object(response)

        if full_output:
            return response_json
        else:
            # Return just the transcribed text
            return response_json.get("text", "")

    @overload
    def audio_transcription(
        self,
        file,
        *,
        model: Optional[str] = None,
        chunking_strategy: Optional[str | dict] = None,
        include: Optional[List[str]] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        stream: Literal[False] = False,
        temperature: Optional[float] = 0,
        timestamp_granularities: Optional[List[str]] = None,
        full_output: Literal[False] = False,
        **options: Any,
    ) -> str: ...

    @overload
    def audio_transcription(
        self,
        file,
        *,
        model: Optional[str] = None,
        chunking_strategy: Optional[str | dict] = None,
        include: Optional[List[str]] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Literal["json", "verbose_json"] = "json",
        stream: Literal[False] = False,
        temperature: Optional[float] = 0,
        timestamp_granularities: Optional[List[str]] = None,
        full_output: Literal[True],
        **options: Any,
    ) -> JSONDict: ...

    @overload
    def audio_transcription(
        self,
        file,
        *,
        model: Optional[str] = None,
        chunking_strategy: Optional[str | dict] = None,
        include: Optional[List[str]] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        stream: Literal[True],
        temperature: Optional[float] = 0,
        timestamp_granularities: Optional[List[str]] = None,
        full_output: Literal[False] = False,
        **options: Any,
    ) -> TextStream: ...

    @overload
    def audio_transcription(
        self,
        file,
        *,
        model: Optional[str] = None,
        chunking_strategy: Optional[str | dict] = None,
        include: Optional[List[str]] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        stream: Literal[True],
        temperature: Optional[float] = 0,
        timestamp_granularities: Optional[List[str]] = None,
        full_output: Literal[True],
        **options: Any,
    ) -> JSONStream: ...

    @overload
    def audio_transcription(
        self,
        file,
        model: Optional[str] = None,
        chunking_strategy: Optional[str | dict] = None,
        include: Optional[List[str]] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        stream: bool = False,
        temperature: Optional[float] = 0,
        timestamp_granularities: Optional[List[str]] = None,
        full_output: bool = False,
        **options: Any,
    ) -> str | JSONDict | TextStream | JSONStream: ...

    def audio_transcription(
        self,
        file,
        model: Optional[str] = None,
        chunking_strategy: Optional[str | dict] = None,
        include: Optional[List[str]] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        stream: bool = False,
        temperature: Optional[float] = 0,
        timestamp_granularities: Optional[List[str]] = None,
        full_output: bool = False,
        **options: Any,
    ) -> str | JSONDict | TextStream | JSONStream:
        """
        Transcribes audio using the regolo.ai audio transcription model from the RegoloClient instance.

        :param file: The audio file object (bytes, file-like an object, or path string) to transcribe,
            in formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
        :param model: The name of the model to use.
            Example: faster-whisper-large-v3
            (Optional)
        :param chunking_strategy: Controls how audio is cut into chunks.
            Auto or object.
            (Optional)
        :param include: Additional information to include in response.
            (Optional)
        :param language: The language of the input audio in ISO-639-1 format.
            (Optional)
        :param prompt: An optional text to guide the model's style or continue a previous audio segment.
            (Optional)
        :param response_format: The format of the output: ``json``, ``text``, ``srt``, ``verbose_json``, or ``vtt``.
            (Defaults to "json")
        :param stream: If true, stream the response using server-sent events.
            Note: Not supported for whisper-1.
            (Defaults to False)
        :param temperature: The sampling temperature, between 0 and 1.
            (Defaults to 0)
        :param timestamp_granularities: Timestamp granularities: word or segment.
            Requires the ``verbose_json`` format.
            (Optional)
        :param full_output: Whether to return the full response.
            (Defaults to False)

        :return for stream=True: Generator yielding streaming responses.
        :return for stream=False, full_output=True: Dict containing the full response from regolo.ai.
        :return for stream=False, full_output=False: String with transcribed text.
        """

        base_url = self.instance.get_base_url()
        if base_url is None:
            base_url = os.getenv("REGOLO_URL", REGOLO_URL)

        # Use the instance model if not specified
        if model is None:
            model = self.instance.get_audio_model()

        return self.static_audio_transcription(
            file=file,
            model=model,
            api_key=self.instance.get_api_key(),
            chunking_strategy=chunking_strategy,
            include=include,
            language=language,
            prompt=prompt,
            response_format=response_format,
            stream=stream,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            client=self.instance.get_client(),
            base_url=base_url,
            full_output=full_output,
            check_models=self._check_models,
            **options,
        )

    @staticmethod
    @overload
    def static_rerank(
        query: str,
        documents: List[str] | List[Dict[str, Any]],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: bool = True,
        max_chunks_per_doc: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: Literal[False] = False,
        check_models: bool | None = None,
        **options: Any,
    ) -> list[JSONDict]: ...

    @staticmethod
    @overload
    def static_rerank(
        query: str,
        documents: List[str] | List[Dict[str, Any]],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: bool = True,
        max_chunks_per_doc: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        *,
        full_output: Literal[True],
        check_models: bool | None = None,
        **options: Any,
    ) -> JSONDict: ...

    @staticmethod
    @overload
    def static_rerank(
        query: str,
        documents: List[str] | List[Dict[str, Any]],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: bool = True,
        max_chunks_per_doc: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: bool = False,
        check_models: bool | None = None,
        **options: Any,
    ) -> List[Dict[str, Any]] | dict: ...

    @staticmethod
    @managed_http
    def static_rerank(
        query: str,
        documents: List[str] | List[Dict[str, Any]],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: bool = True,
        max_chunks_per_doc: Optional[int] = None,
        client: Optional[httpx.Client] = None,
        base_url: str = os.getenv("REGOLO_URL", REGOLO_URL),
        full_output: bool = False,
        check_models: bool | None = None,
        **options: Any,
    ) -> List[Dict[str, Any]] | dict:
        """
        Reranks a list of documents based on their relevance to a query using the regolo.ai reranking model.

        :param query: The search query to compare documents against.
        :param documents: List of documents to rerank.
        Can be strings or dicts with text fields.
        :param model: The regolo.ai reranking model to use.
        (Optional)
        :param api_key: The API key for regolo.ai.
        (Optional)
        :param top_n: Number of most relevant documents to return.
        Returns all if not specified.
        (Optional)
        :param rank_fields: For structured documents, specify which fields to rank by.
        (Optional)
        :param return_documents: Whether to return document content in results.
        (Defaults to True)
        :param max_chunks_per_doc: Maximum number of chunks per document.
        (Optional)
        :param client: httpx client to use.
        (Optional)
        :param base_url: Base URL of the regolo HTTP server.
        (Defaults to REGOLO_URL)
        :param full_output: Whether to return the full response.
        (Defaults to False)

        :return full_output=True: Dict containing the full response from regolo.ai.
        :return full_output=False: List of dicts with 'index', 'relevance_score', and optionally 'document'.
        """

        # Use the default API key if none is provided
        if api_key is None:
            api_key = regolo.default_key

        # Validate API key
        api_key = KeysHandler.check_key(api_key)

        # Use the default reranking model if not specified
        if model is None:
            model = regolo.default_reranker_model

        # Validate the selected model
        ModelsHandler.check_model(
            model=model, api_key=api_key, base_url=base_url, client=client, enabled=check_models
        )

        # managed_http supplies an owned client when none was passed.
        assert client is not None

        # Construct the payload for the API request
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "rank_fields": rank_fields,
            "return_documents": return_documents,
            "max_chunks_per_doc": max_chunks_per_doc,
        }

        payload = _merge_payload(payload, options)
        # Remove None values from the payload to avoid unnecessary parameters
        payload = {k: v for k, v in payload.items() if v is not None}

        # Set authorization header
        headers = {"Authorization": api_key}

        # Send a synchronous POST request
        response = safe_post(
            client=client,
            url_to_query=api_url(base_url, os.getenv("RERANK_URL_PATH", RERANK_URL_PATH)),
            json_to_query=payload,
            headers_to_query=headers,
        )

        response_json = json_object(response)

        if full_output:
            return response_json
        else:
            # Return just the ranked results
            return response_json.get("results", [])

    @overload
    def rerank(
        self,
        query: str,
        documents: List[str] | List[Dict[str, Any]],
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: bool = True,
        max_chunks_per_doc: Optional[int] = None,
        full_output: Literal[False] = False,
        **options: Any,
    ) -> list[JSONDict]: ...

    @overload
    def rerank(
        self,
        query: str,
        documents: List[str] | List[Dict[str, Any]],
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: bool = True,
        max_chunks_per_doc: Optional[int] = None,
        *,
        full_output: Literal[True],
        **options: Any,
    ) -> JSONDict: ...

    @overload
    def rerank(
        self,
        query: str,
        documents: List[str] | List[Dict[str, Any]],
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: bool = True,
        max_chunks_per_doc: Optional[int] = None,
        full_output: bool = False,
        **options: Any,
    ) -> List[Dict[str, Any]] | dict: ...

    def rerank(
        self,
        query: str,
        documents: List[str] | List[Dict[str, Any]],
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: bool = True,
        max_chunks_per_doc: Optional[int] = None,
        full_output: bool = False,
        **options: Any,
    ) -> List[Dict[str, Any]] | dict:
        """
        Reranks a list of documents based on their relevance to a query using the regolo.ai reranking model
        from the RegoloClient instance.

        :param query: The search query to compare documents against.
        :param documents: List of documents to rerank.
        Can be strings or dicts with text fields.
        :param top_n: Number of most relevant documents to return.
        Returns all if not specified.
        (Optional)
        :param rank_fields: For structured documents, specify which fields to rank by.
        (Optional)
        :param return_documents: Whether to return document content in results.
        (Defaults to True)
        :param max_chunks_per_doc: Maximum number of chunks per document.
        (Optional)
        :param full_output: Whether to return the full response.
        (Defaults to False)

        :return full_output=True: Dict containing the full response from regolo.ai.
        :return full_output=False: List of dicts with 'index', 'relevance_score', and optionally 'document'.
        """

        base_url = self.instance.get_base_url()
        if base_url is None:
            base_url = os.getenv("REGOLO_URL", REGOLO_URL)

        return self.static_rerank(
            query=query,
            documents=documents,
            model=self.instance.get_reranker_model(),
            api_key=self.instance.get_api_key(),
            top_n=top_n,
            rank_fields=rank_fields,
            return_documents=return_documents,
            max_chunks_per_doc=max_chunks_per_doc,
            client=self.instance.get_client(),
            base_url=base_url,
            full_output=full_output,
            check_models=self._check_models,
            **options,
        )
