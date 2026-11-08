# Public Devmid API coverage

The client supports the following Devmid user operations. Responses remain
dictionaries to preserve new fields and the distinction between omitted and null
values.

| HTTP   | Path                         | SDK method                                | CLI                           |
|--------|------------------------------|-------------------------------------------|-------------------------------|
| POST   | `/auth/login`                | `authenticate(username, password)`        | `auth login`                  |
| POST   | `/auth/refresh`              | `refresh_access_token()`                  | `auth refresh`                |
| GET    | `/models`                    | `get_models()`                            | `models list`                 |
| GET    | `/models/{model}`            | `get_model(model_name)`                   | `models details NAME`         |
| DELETE | `/models/{model}`            | `delete_model(model_name)`                | `models delete NAME`          |
| POST   | `/models/load`               | `register_model(...)`                     | `models register`             |
| POST   | `/models/{model}/scan/retry` | `retry_model_scan(model_name)`            | `models retry-scan NAME`      |
| GET    | `/ssh-keys`                  | `get_ssh_keys()`                          | `ssh list`                    |
| POST   | `/ssh-keys`                  | `add_ssh_key(title, key)`                 | `ssh add`                     |
| DELETE | `/ssh-keys/{keyId}`          | `delete_ssh_key(key_id)`                  | `ssh delete ID`               |
| GET    | `/inference/gpus`            | `get_available_gpus()`                    | `inference gpus`              |
| POST   | `/inference/load`            | `load_model_for_inference(...)`           | `inference load NAME`         |
| POST   | `/inference/unload`          | `unload_model_from_inference(session_id)` | `inference unload ID`         |
| GET    | `/inference/loaded-models`   | `get_loaded_models()`                     | `inference loaded` / `status` |
| GET    | `/inference/user-status`     | `get_user_inference_status(...)`          | `inference user-status`       |
| GET    | `/inference/sessions`        | `get_inference_sessions(...)`             | `inference sessions`          |
| GET    | `/runtime/status`            | `get_runtime_status(session_id)`          | `inference session-status ID` |
| GET    | `/runtime/billing`           | `get_runtime_billing(session_id)`         | `inference billing ID`        |

All routes except login require Keycloak bearer authentication. Read/write scopes
are declared by the public API. User-status requires
`model:write` even though it is GET; refresh declares both `model:read` and
`model:write`.

History supports date, model, status and cursor filters plus optional scaling summaries. Runtime uses the exact
query name `model_inference_id`. Lookup/deletion/retry use model **names**; unload
and runtime use positive numeric **session IDs**. SSH-key IDs are positive integers.

Higher-level helpers: `wait_for_scan`, `wait_for_inference`, `load_and_wait`,
`iter_inference_sessions`, `get_usage_summary`, and `export_sessions_csv`.
`GPUPlanConfig` validates autoscaling bounds. Load accepts every writable public
vLLM field and rejects server-controlled/unknown fields.

To use a different model revision, register it under a new name and retire the
old registration when ready. `update_model_revision` raises `NotImplementedError`;
there is no corresponding CLI command.

## Additional helpers

| Capability | SDK | CLI | Usage |
| --- | --- | --- | --- |
| Full user overview | `get_dashboard_snapshot` | `inference overview` | Models, GPUs, loaded deployments and windowed usage |
| Exact owned history row | `get_inference_session` | Via `inference details` | Look up one session by ID |
| Combined session details | `get_session_details` | `inference details ID` | Session history, runtime state and billing |
| Inline scaling | `include_scaling=True` on loaded/history methods | `--include-scaling` | Requires server support |
| Scaling detail/pages | `get_scaling_history`, `iter_scaling_history` | `inference scaling ID` | Requires the server's public OpenAPI schema |
| SGLang/LlamaCpp and GGUF options | `engine`, `engine_config`, `deployment_options` | `--engine`, `--engine-config`, `--deployment-options` | Uses configuration fields declared by the server |
| Autoscaling targets/delays | `GPUPlanConfig` | Target/delay load options | Requires server support |
| Complete nested JSON export | `export_sessions_json` | `sessions --json-file` | Export all matching history pages |

Engine configuration and scaling-detail discovery require a public OpenAPI
schema from the server you use. The server controls feature availability and
access to models and sessions. HTTP 401/403/404 errors are returned to the caller.

See [custom models](custom-models.md), [usage](usage.md) and
[inference](inference.md) for lifecycle, billing and serving API details.
