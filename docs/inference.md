# Native inference and conversations

`RegoloClient` uses an inference API key independently of Devmid's Keycloak
login. Choose serving model IDs from `/models` or the full catalog, rather than
assuming a Devmid registration/session ID is the gateway's serving ID.

| Method | Endpoint or behavior | Default result |
| --- | --- | --- |
| `run_chat(prompt=None, ...)` | `/v1/chat/completions`; keeps history | `(role, content)` or streamed tuples |
| `static_chat_completions(messages, ...)` | Stateless chat | Same; `full_output=True` keeps raw JSON/events |
| `embeddings(input_text, dimensions=..., encoding_format=...)` | `/v1/embeddings` | Embedding data array |
| `rerank(query, documents, ...)` | `/v1/rerank` | Ranking results |
| `create_image(prompt, ...)` | `/v1/images/generations` | Decoded image bytes |
| `audio_transcription(file, ...)` | `/v1/audio/transcriptions` | Text or streaming text |
| `get_model_catalog(mode=..., supports=..., refresh=False)` | `/model_group/info` | Full model capability, limit and price dictionaries |
| `get_model_info(model=None)` | Lookup in cached full catalog | One full model dictionary |
| `get_available_models(...)` | Legacy `/models` or `/model/info` | IDs or legacy metadata |
| `vision(prompt, images, ...)` | Image content in chat | Chat result |
| `ocr(image, ...)` | OCR-model chat; preserves special tokens by default | Chat result |
| `run_structured(prompt, Model, ...)` | JSON-schema request plus strict local validation | Pydantic model |
| `run_tools(prompt, tools, max_rounds=5, max_calls=32, ...)` | Bounded native function exchange | Final raw chat response |
| `completions(prompt, ...)` | Legacy `/v1/completions` | Text or streamed text |

Legacy completions remain for compatible custom servers. Current Regolo docs
recommend chat; this SDK does not assume a public legacy-completion model.
Static/top-level equivalents remain available for embeddings, reranking, image
creation and transcription. Existing positional parameters are preserved.

## Transport and options

Pass endpoint options directly, or use `extra_body`. Duplicate options and
attempted overrides of model/messages/input/stream are rejected. Additional
parameters are model-specific; catalog `supported_openai_params` helps select
appropriate settings. The SDK does not silently discard unsupported parameters.

```python
from regolo import RegoloClient

with RegoloClient(api_key="<KEY>", chat_model="<MODEL>", request_timeout=90) as client:
    reply = client.run_chat(
        "Return a JSON greeting", response_format={"type": "json_object"},
        max_completion_tokens=64, full_output=True,
    )
    print(reply.get("usage"))
```

`max_completion_tokens` replaces the legacy default 200-token cap. An explicit
non-default `max_tokens` together with `max_completion_tokens` is rejected.
`max_tokens=None` omits that cap. Parameters are not retried automatically on
5xx, rate limits, network errors or malformed success responses.

Instance-owned transports use `request_timeout` (default 500 seconds). A caller's
injected HTTPX client retains its own timeout/proxy/TLS settings and remains open
when the SDK client closes. Static owned calls use 500 seconds. Close/exhaust
streams; a closed SDK client cannot start another request.

Both root URLs and URLs ending in `/v1` or `/custom-model/v1` work without
appending `/v1` twice. Explicitly disable discovery for a custom gateway that
has no model-list endpoint:

```python
client = RegoloClient(
    api_key="<INFERENCE_KEY>", chat_model="<EXPLICIT_SERVING_MODEL_ID>",
    alternative_url="https://api.regolo.ai/custom-model/v1", check_models=False,
)
```

`check_models` is per client/static call. The legacy process-wide
`regolo.enable_model_checks` remains the default when the option is omitted.
A nonempty model ID is always required. Client construction performs no network
requests. `regolo.try_loading_from_env()` retains its explicit legacy defaults
loader; importing the package no longer populates endpoint environment variables.

## History, reasoning, streaming and usage

```python
from regolo import Conversation

client.add_message("system", "Answer concisely")
client.run_chat("First question")
client.conversation.save("conversation.json")  # Explicit, atomic, owner-only JSON.
client.instance.overwrite_conversation(Conversation.load("conversation.json"))
client.conversation.trim(max_turns=10)  # Keeps initial system/developer messages and whole tool exchanges.
branch = client.fork()                # Independent history and owned transport.
branch.close()
```

`run_chat` accepts either text or a content-part array as its user turn. It keeps
full assistant/tool/reasoning metadata even when returning simplified tuples.
`last_response`, `last_usage` and `usage_totals` expose reported chat usage;
missing usage stays missing. Counters cover completed replies only and do not
estimate costs. A streaming response usually needs
`stream_options={"include_usage": True}` for usage metadata.

Simplified streams yield `("thinking", text)` and `("", answer)`; when one event
contains both, both are emitted. Choice zero is used consistently. Stateful chat
requires `n=1`; stateless full output supports multiple choices. Final empty
usage events do not erase content or tools. SSE errors, malformed events and
incomplete termination raise; a canceled/failed stream never records a completed
assistant response. An unstarted stateful stream changes no history. A failed
started turn retains its user message for inspection.

A conversation can have only one active generation. Close/finish its stream
before editing history or starting another turn. The low-level `instance`
remains available for compatibility; callers directly mutating it are responsible
for synchronization. `from_instance` deliberately shares history and borrows
its original transport; `fork()` is the independent alternative.

## Tools and structured replies

```python
from regolo import FunctionTool

def weather(city: str) -> dict[str, str]:
    """Get demo weather for a city."""
    return {"city": city, "weather": "sunny"}

reply = client.run_tools(
    "Use weather for Rome", [FunctionTool(weather)], max_rounds=3, max_calls=4,
)
```

Constructing tools or calling ordinary `run_chat(tools=[...])` never executes
functions. `run_tools` runs only caller-registered synchronous callables. It
creates schemas from annotations, validates the entire returned batch strictly
before executing any function, rejects unknown names/extra arguments/duplicate
IDs, and caps model rounds/local calls. An explicit forced `tool_choice` applies
to the first round; subsequent rounds use `auto` so the model can finish.
Previously recorded tool-result IDs cannot be replayed. Function/serialization/
network errors propagate; successful results already in history remain there.
There is no automatic rollback or retry of a function with side effects.

For manual handling, inspect `client.last_response` and call
`client.add_tool_result(call_id, result)` before the next turn. `run_structured`
uses the supplied Pydantic model's JSON schema and strict local validation. It
rejects refusal/truncated responses; its raw response remains in `last_response`
if local validation fails. No paid repair/retry request is made.

## Multimodal inputs and other families

```python
from regolo import audio_part, image_part, text_part, video_part

parts = [text_part("Describe this"), image_part("image.png")]
client.run_chat(parts)
# A suitable model must advertise the corresponding capability:
audio_input = audio_part(b"<WAV_BYTES>", format="wav")
video_input = video_part("https://example.test/clip.mp4")
```

`image_part` accepts an image path, a public/data image URL, or bytes with an
explicit MIME type. It does not download a remote URL. Audio chat accepts WAV/MP3
bytes; `video_part` builds a video URL part. Media limits and backend support are
model-specific. OCR processes images through chat; PDF rendering is deliberately
left to the caller, so there is no PDF framework dependency.

Embedding dimensions/encoding and extra reranking options are forwarded natively.
Image `full_output=True` preserves URL-based providers; byte output needs
`b64_json`. The SDK never follows a generated image URL with the API key.
Transcription accepts `str`/`Path`, bytes, or binary file-like input and keeps
caller-owned handles open. Text/SRT/VTT return raw text even in full-output mode;
timestamps require `verbose_json`. Extra multipart options encode objects/lists/
booleans as JSON and cannot replace named fields or the file.

Official references: [catalog](https://docs.regolo.ai/models/catalog/),
[tools](https://docs.regolo.ai/models/features/function-calling/),
[structured output](https://docs.regolo.ai/models/features/structured-outputs/),
[vision](https://docs.regolo.ai/models/features/vision/),
[OCR](https://docs.regolo.ai/models/families/ocr/).
