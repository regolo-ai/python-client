# Regolo Python client

A small Python SDK and CLI for **Regolo inference and Devmid's public user API**.
Version **1.14.0** adds native tools, structured replies, multimodal chat, full
model metadata, scaling history and richer custom-model inspection. No agent
or inference framework is required.

```bash
pip install regolo
```

Python 3.12 or newer is required. From this checkout:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

## Credentials

| Task | Credential | Client |
| --- | --- | --- |
| Chat, embeddings, images, audio, ranking and OCR | Regolo inference API key | `RegoloClient` |
| Custom model registration, deployment, history and hosting usage | Keycloak access token/login | `ModelManagementClient` |
| Private/gated Hugging Face source | HF token passed at registration | `register_model(..., api_key=...)` |

Credentials remain separate. SDK token persistence is opt-in; the CLI stores
server-scoped tokens in an owner-only file. Passwords and inference keys are not
saved by the SDK.

## Model discovery and inference

```python
import os
from regolo import RegoloClient

with RegoloClient(api_key=os.environ["REGOLO_API_KEY"], chat_model="<MODEL_ID>") as client:
    catalog = client.get_model_catalog(mode="chat", supports=["function_calling"])
    print(catalog)  # Full limits, capability flags, pricing and future server fields.
    print(client.run_chat("Tell me about Rome", max_tokens=128))
    for role, text in client.run_chat("And its history?", stream=True, max_tokens=128):
        print(text, end="")
    print(client.last_usage)
    print(client.usage_totals)  # Only token counts actually reported by the server.
```

All model families retain `full_output=True` for raw responses. Native keyword
options work across families: `dimensions`/`encoding_format` for embeddings,
`response_format` for images, and `tools`, `tool_choice`, `reasoning_effort`,
`response_format` or `stream_options` for chat. Model support still matters.
`extra_body` remains available; duplicate/identity overrides are rejected.

## Native tools and typed replies

```python
from pydantic import BaseModel, ConfigDict
from regolo import FunctionTool


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


# Only this explicitly registered callable can run. Arguments are validated
# before any function in the returned batch executes. No automatic tool retry.
reply = client.run_tools("Use add for 2 plus 3", [FunctionTool(add)], max_rounds=3)


class City(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


city = client.run_structured("Return the name of Italy's capital", City)
print(city.name)
```

These snippets use an open `client` configured for a suitable model. Ordinary
`run_chat` never executes tools. It retains tool calls and reasoning in history
even when returning simplified text. `run_tools` is a separate, explicit action.

Use `client.vision(prompt, images=[...])` or `client.ocr(image)` with a suitable
catalog model. Local paths, image bytes with `mime_type`, and remote image URLs
are accepted. Helpers also build audio/video chat content parts without
additional frameworks. Conversation history can be forked, trimmed by complete
user turns, and explicitly saved/loaded as JSON. See [inference](docs/inference.md).

## Custom model lifecycle and all user statistics

```python
from regolo import GPUPlanConfig, ModelManagementClient

with ModelManagementClient(token="<KEYCLOAK_ACCESS_TOKEN>") as models:
    models.register_model("my-model", url="organization/model", revision="release-v3")
    print(models.get_model("my-model"))  # Scan, revision, capacity, trust and owner actions.
    loaded = models.load_and_wait(
        "my-model",
        gpu="<AVAILABLE_GPU_INSTANCE_TYPE>",
        gpu_plan_config=GPUPlanConfig(
            min_replicas=0,
            max_replicas=3,
            target_requests=250,
            scale_down_delay_seconds=45,
        ),
        vllm_config={"max_model_len": 4096},
    )
    session_id = loaded["session_id"]
    print(models.get_session_details(session_id, include_scaling=True))
    print(models.get_dashboard_snapshot(include_scaling=True))
    for session in models.iter_inference_sessions(
        include_status=True,
        include_error=True,
        include_billing=True,
        include_scaling=True,
    ):
        print(session)
    models.export_sessions_json("history.json", include_billing=True, include_scaling=True)
    # Explicitly stop hosting/billing when finished:
    models.unload_model_from_inference(session_id)
```

Responses preserve every server field, including nullable/future statistics.
Usage aggregation follows all session pages; raw runtime/billing responses
remain available. Scaling intervals, assumptions and frozen finalized data come
from Devmid, without local reconstruction.

Available engines, scaling options and deployment settings depend on your
server's public API. For schema-based engine configuration and paginated scaling
history, provide a public OpenAPI export or let the client fetch `/openapi.json`:

```python
with ModelManagementClient(
    token="<KEYCLOAK_ACCESS_TOKEN>", public_openapi="public-openapi.json"
) as models:
    models.load_model_for_inference(
        "my-model",
        "<GPU>",
        engine="sglang",
        engine_config={"context_length": 4096},
    )
    page = models.get_scaling_history(123, limit=100)
    print(page)  # Includes server pagination/finalization metadata.
```

Engine values, config keys, GGUF selection fields and scaling route/cursor names
are taken from that public schema. Unsupported settings fail locally. See the
[custom model guide](docs/custom-models.md) for LlamaCpp and GGUF options.

## CLI

```bash
regolo catalog --mode chat --supports vision
regolo auth login
regolo models details my-model --format json
regolo inference loaded --include-scaling --format json
regolo inference details 123 --include-scaling --include-error
regolo inference sessions --include-status --include-billing --include-scaling --all
regolo inference sessions --include-scaling --json-file history.json
regolo inference overview --month 112026 --include-scaling
regolo --public-openapi public-openapi.json inference scaling 123 --limit 100
regolo inference load my-model --gpu <GPU> --target-requests 250 --scale-down-delay-seconds 0
```

`--devmid-url`, `--config-file` and `--public-openapi` also accept `DEVMID_URL`,
`REGOLO_CONFIG_FILE` and `DEVMID_PUBLIC_OPENAPI`. Inference commands can use
`REGOLO_API_KEY`; credentials for each service remain distinct.

## Documentation

- [Public endpoint/SDK/CLI mapping](docs/api-reference.md)
- [Custom model lifecycle and current-schema engines](docs/custom-models.md)
- [Billing, scaling history and exports](docs/usage.md)
- [Native inference and conversation guide](docs/inference.md)
- [Offline tests and bounded opt-in live tests](docs/development.md)
- [Upgrade notes](CHANGELOG.md)
