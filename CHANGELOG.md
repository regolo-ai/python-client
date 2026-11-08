# Changelog

## 1.14.0 — Public Devmid API and native inference

### Added

- Native typed function tools, bounded tool execution and structured replies
  validated with Pydantic.
- Image, OCR, audio and video chat helpers; conversation JSON persistence,
  independent forks and turn-based history trimming.
- Full model catalog discovery with capability, pricing and limit filters.
- `ModelManagementClient` for model registration, scan retry, SSH public keys,
  GPU discovery, deployment lifecycle, session history, runtime state and billing.
- Scan/readiness polling, load-and-wait workflows, all-page usage summaries,
  complete JSON history exports and spreadsheet-safe CSV exports.
- Autoscaling targets and delays, optional scaling summaries, and paginated
  scaling history using the server's public OpenAPI schema.
- Schema-based engine configuration and GGUF selection for compatible servers.
- CLI commands for model discovery, session details, usage, scaling and exports.
- Offline tests, type checks, CI checks and an opt-in live inference smoke runner.

### Improved

- Preserved tool calls, reasoning, usage and multimodal messages in conversation
  history, including simplified and streamed responses.
- Added overlapping-turn protection, explicit stream cancellation and error
  handling, and validation of incomplete or malformed responses.
- Forwarded model-specific inference options and normalized root, versioned and
  custom gateway URLs.
- Respected caller-owned HTTP transports, timeouts and resource lifetimes.
- Added opt-in, server-scoped token persistence with atomic writes, token refresh
  and credential-safe error display.
- Improved CLI aliases, GPU selection, JSON output and raw audio/SRT/VTT handling.
- Corrected Qwen3 reranker inputs in the live smoke test and clarified reranking
  input formats and failure messages.
- Added `py.typed`, response-mode overloads and clearer nullable response types.
- Removed unused requests, dotenv and JSON-repair runtime dependencies.

### Upgrade notes

- Run `regolo auth login` after upgrading an older CLI token file. New token files
  are scoped to their server; old unscoped files are ignored.
- New SDK code should import `ModelManagementClient` from `regolo`. Importing it
  from `regolo.cli` remains compatible and enables persistence by default.
- Deployment configuration rejects unknown and server-controlled fields.
  Tensor-parallel size and revision fields are controlled by the server.
- `update_model_revision` raises `NotImplementedError`. Register another model
  name to use a different revision, then retire the old registration when ready.
- Inference requests are not automatically replayed after network or 5xx errors.
  Explicit retries remain available through `safe_post(..., max_retries=...)`.
- Malformed or incomplete SSE streams raise errors. Completed streamed assistant
  replies are added to conversation history.
- `change_model()` propagates model validation errors.
