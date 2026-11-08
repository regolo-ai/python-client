# Custom models with the public Devmid user API

## Authentication and ownership

`ModelManagementClient(base_url="https://devmid.regolo.ai", token=None,
refresh_token=None, timeout=60, http_client=None, config_file=None)` accepts an
existing Keycloak token or a subsequent `authenticate(username, password)` call.
Without `config_file`, SDK credentials stay in memory. Passwords/Hugging Face
keys are not written to the token file. CLI login/refresh never print tokens.

The client refreshes known expiring credentials before an authenticated request
and attempts one refresh/replay after HTTP 401. `/auth/refresh` requires the old
bearer header. If Devmid rejects an already-expired access token there, log in
again. Refresh failures remain visible; there is no infinite retry loop.

The client reuses one transport. `close()`/a context manager closes only an
internally owned HTTP client; injected clients remain the caller's responsibility.
Network/5xx failures do not automatically replay mutations: the server may have
already committed a deployment/registration. Inspect state before retrying.

## Registration and identity

`register_model(name, provider="huggingface", url=..., api_key=..., revision=...)`
returns an acknowledgement, not a model object or ready deployment. The server
resolves a branch/tag to an immutable SHA. Omitted revision selects latest.
Private/gated repositories require their Hugging Face key.

Devmid normalizes names and may rewrite reserved names. Use simple lowercase
names such as `my-model`, read the acknowledgement and use `get_models()` to
discover the stored name. The combined CLI workflow requires canonical lowercase
letters/digits/single hyphens. Duplicate live names return HTTP 409.

`get_model(name)` and `delete_model(name)` use the owner's **name**, not numeric
registration ID. Retirement preserves shared scan history and does not replace
an explicit inference unload.

Model dictionaries preserve all server fields:

| Fields                                                        | Meaning                                                             |
|---------------------------------------------------------------|---------------------------------------------------------------------|
| `registration_id`, `scanned_artifact_id`, revision/digest     | Immutable registration/artifact identity                            |
| `scan_status`, `scan_verdict`, `is_safe`                      | Security lifecycle and effective verdict                            |
| Detail/report URL, scanner timestamps/version                 | Explanation and scan evidence                                       |
| `capacity_status`, `capacity_reason`, `weight_bytes_estimate` | Hostability, separate from security                                 |
| `verdict_source`, trust reason/evidence                       | Effective administrative/scanner trust information                  |
| `scan_retry_allowed`, `scan_retry_reason`                     | Whether owner retry is permitted                                    |
| `recommended_action`                                          | `wait`, `load`, `retry_scan`, `replace_credential`, `contact_admin` |

`wait_for_scan(name, timeout=3600, poll_interval=5)` polls `pending`, `scanning`
and retryable `error` until `safe`. It stops on `unsafe`, `inconclusive`,
`awaiting_credentials`, `capacity_blocked_state` or an unknown state with
`ModelNotReadyError`; `.state` preserves the full response.

`retry_model_scan(name)` calls the public retry route. Read guidance first; server
policy is authoritative and denial returns HTTP 409. Trust decisions/capacity
overrides remain administrative operations.

To use another revision, register it under a new name, wait for readiness, and
retire the old registration when ready. `update_model_revision` is unsupported
and raises `NotImplementedError`; there is no revision-update CLI command.

## Deployment options

```python
from regolo import GPUPlanConfig

gpus = client.get_available_gpus().get("gpus") or []
available = [gpu for gpu in gpus if gpu["Available"]]
loaded = client.load_model_for_inference(
    "my-model",
    gpu=available[0]["InstanceType"],
    gpu_plan_config=GPUPlanConfig(min_replicas=0, max_replicas=3),
    vllm_config={"max_model_len": 4096, "reasoning_parser": "deepseek_r1"},
)
client.wait_for_inference(loaded["session_id"], timeout=1800)
```

Check `Available`/`Region`: catalog entries can be unavailable. `MemoryGiB` is
system RAM; `Details.TotalGpuMemoryInMiB` describes GPU memory. `PriceEUR` is the
customer price per replica-hour. Interactive CLI GPU selection omits unavailable
plans; the server determines actual model fit.

Autoscaling defaults to one replica. `min_replicas >= 0`, `max_replicas >= 1`, and
minimum cannot exceed maximum. Zero minimum enables scale-to-zero; an idle
deployment can have no worker. Runtime polling does not warm up the model.

The load acknowledgement includes `session_id` and effective `max_model_len` /
`max_model_len_source` when supplied by Devmid. HTTP 200 means preparation, not
readiness; the historical response type name containing `202` is not its status.
`force` does not bypass scan/capacity/billing policy.

`wait_for_inference(session_id, timeout=1800, poll_interval=5)` returns on
`running`. `failed`, `stopped`, `completed` or unknown states raise
`ModelNotReadyError` with full state/error text. Polling validates nonnegative
timeouts/positive intervals and caps sleeps at the deadline. An individual HTTP
call can extend past the polling deadline, bounded by the HTTP timeout.

`load_and_wait(name, gpu, scan_timeout=3600, timeout=1800, poll_interval=5, ...)`
combines scan/capacity guidance, load and readiness. Timeout leaves the session
reserved and reports its ID. Inspect or unload it explicitly.

## vLLM configuration

The following vLLM fields are supported:

```text
max_model_len, max_num_batched_tokens, max_num_seqs, dtype, kv_cache_dtype,
override_attention_dtype, enable_prefix_caching, enable_chunked_prefill,
block_size, kv_sharing_fast_prefill, optimization_level, performance_mode,
enable_auto_tool_choice, tool_call_parser, chat_template, reasoning_parser,
async_scheduling, stream_interval, speculative_config, seed
```

Values are sent unchanged, including `False`/zero. Server validation and effective
settings remain authoritative. Access fields without CLI flags through a JSON
config file; do not combine it with individual vLLM flags:

```bash
regolo inference load my-model --gpu <GPU_INSTANCE_TYPE> --vllm-config-file vllm.json
```

`chat_template` is template content/name recognized by the engine, not an uploaded
client-local path. Revision fields and `tensor_parallel_size` are server-controlled.
`gpu_memory_utilization` and `disable_log_requests` are absent from the public
schema. These and unknown keys are rejected instead of silently ignored.

## SSH keys and errors

`add_ssh_key(title, key)`, `get_ssh_keys()` and `delete_ssh_key(key_id)` cover all
public SSH-key operations. Supply OpenSSH **public** key material.

HTTP failures raise `RegoloAPIError` (an `httpx.HTTPStatusError`) with `.request`,
`.response` and `.detail`, including FastAPI's 422 list. Transport failures retain
their `httpx.RequestError` type. Invalid successful JSON/pagination raises
`RegoloProtocolError`. Readiness failures expose `ModelNotReadyError.state`;
polling exhaustion raises `TimeoutError`. Failure text is not truncated.

## Newer engines, GGUF selection and public schema discovery

`ModelManagementClient(..., public_openapi="public-openapi.json")` accepts a
current public export. Without one, new engine/scaling-detail features lazily
fetch `/openapi.json` once from the configured server; old operations do not
perform discovery. Explicitly refresh with `get_public_contract(refresh=True)`.
This export supplies config names, types, ranges and authentication metadata.

```python
from regolo import ModelManagementClient

with ModelManagementClient(token="<TOKEN>", public_openapi="public-openapi.json") as models:
    # Values/config field names are resolved against this server's public schema.
    models.load_model_for_inference(
        "my-model", "<GPU>", engine="sglang", engine_config={"context_length": 4096},
    )
    # Use only tuning keys actually present in the current public export.
    # Example shape; n_ctx must be declared by your server's LlamaCpp config.
    models.load_model_for_inference(
        "my-gguf", "<GPU>", engine="llamacpp", engine_config={"n_ctx": 4096},
    )
```

The client normalizes engine spelling/case to a unique declared enum value and
uses the exact corresponding config property from the schema. `deployment_options`
can carry additional **declared writable** top-level fields, such as the GGUF
filename selector if your server exposes one. Do not place local `--model` paths,
raw args/env, revisions, credentials or model sources in an engine config; undeclared
and read-only fields fail locally. Devmid remains responsible for scanned-revision
pinning, security/capacity policy and generating the actual local GGUF path.

Use the configuration field names and allowed values declared by your server's
public OpenAPI schema. Available LlamaCpp/SGLang options vary by server version.

`GPUPlanConfig` also accepts `target_requests > 0` and
`scale_down_delay_seconds >= 0`. Integers are strict (booleans/strings are rejected).
Omitted/null target/delay fields are omitted from JSON; zero delay is preserved.
They retain API snake_case names; Devmid performs manifest-name conversion.

CLI equivalents: `inference load ... --engine sglang --engine-config tuning.json`,
`--deployment-options selection.json`, `--target-requests 250`, and
`--scale-down-delay-seconds 0`. Use global `--public-openapi public-openapi.json`
or `DEVMID_PUBLIC_OPENAPI` when the export cannot be fetched from the server.

`get_dashboard_snapshot` combines only models, GPU catalog, loaded models and
user windowed status. `get_session_details` combines exact owned history with
runtime and billing; all raw server metrics remain intact. It does not guess
GPU performance/request metrics that the user API has not supplied.
