# Regolo.ai Python client

Python SDK and CLI for Regolo inference APIs and the devmid model-management
API. Client 1.13 is aligned with devmid 3.6.

## Installation

```bash
pip install regolo
```

Python 3.12 or newer is required.

## Inference

```python
import regolo

regolo.default_key = "<API_KEY>"
regolo.default_chat_model = "<MODEL>"

role, content = regolo.static_chat_completions(
    messages=[{"role": "user", "content": "Tell me about Rome"}]
)
print(role, content)
```

The SDK also provides completions, streaming chat, image generation, audio
transcription, embeddings, and reranking through `RegoloClient`.

```python
client = regolo.RegoloClient(
    api_key="<API_KEY>",
    chat_model="<MODEL>",
)
print(client.run_chat(user_prompt="Hello"))
```

## Authentication for model management

```bash
regolo auth login
```

Or in Python:

```python
from regolo.cli import ModelManagementClient

client = ModelManagementClient("https://devmid.regolo.ai")
client.authenticate("username", "password")
```

## Registering a Hugging Face model

Hugging Face is currently the only supported registration provider. The server
checks repository access and resolves the requested branch, tag, or commit to
an immutable commit SHA before creating the registration.

```bash
regolo models register \
  --name my-model \
  --url organization/model \
  --revision release-v2 \
  --api-key hf_xxxxxxxxx
```

`--revision` is optional and defaults to the repository's latest default
revision. `--api-key` is needed for private or gated repositories.

Python:

```python
result = client.register_model(
    name="my-model",
    provider="huggingface",
    url="organization/model",
    api_key="hf_xxxxxxxxx",
    revision="release-v2",
)
```

For source compatibility, `register_model(..., force=...)` is still accepted,
but registration no longer sends or uses that obsolete option.

## Scan state and model information

```bash
regolo models list
regolo models details my-model
regolo models list --format json
```

Model responses expose:

- `registration_id`: immutable ID for this user's registration;
- `scanned_artifact_id`: shared artifact for the exact URL and revision;
- `revision` / `source_revision`: immutable upstream commit;
- `content_digest`: digest calculated from downloaded model content;
- `scan_status`, `scan_verdict`, `scan_detail`, and `scan_report_url`;
- `scanner_version`, `scan_claimed_at`, and `scan_updated_at`.

Scan statuses are:

- `pending`, `scanning`, or `error`: asynchronous work is pending or retrying;
- `safe`: inference is allowed;
- `unsafe`: the content failed scanning;
- `inconclusive`: scanning could not produce a safe verdict;
- `awaiting_credentials`: the scan credential failed and the revision must be
  registered or updated with a newly validated Hugging Face key.

The same URL and immutable revision share one artifact and one terminal
verdict, even across registrations and users. Physical retry attempts remain
unique.

## Updating a registered revision

Revision updates address the immutable registration ID—not its display name.
The server soft-deletes the old registration and returns a new registration.

```bash
# Explicit branch, tag, or commit
regolo models update-revision 42 --revision release-v3 --api-key hf_xxxxxxxxx

# Resolve latest
regolo models update-revision 42
```

Python:

```python
updated = client.update_model_revision(
    registration_id=42,
    revision="release-v3",  # omit for latest
    api_key="hf_xxxxxxxxx",
)
print(updated["registration_id"], updated["scan_status"])
```

## Waiting for scanning and loading inference

```python
client.wait_for_scan("my-model", timeout=3600, poll_interval=5)
client.load_model_for_inference("my-model", gpu="ECS1GPU11")
```

`wait_for_scan` returns only when the model is `safe`. It raises for unsafe,
inconclusive, credential-blocked, or timed-out scans.

The CLI workflow waits for scanning before `--auto-load`:

```bash
regolo workflow workflow my-model \
  --url organization/model \
  --revision release-v3 \
  --api-key hf_xxxxxxxxx \
  --auto-load \
  --scan-timeout 3600
```

Inference operations:

```bash
regolo inference gpus
regolo inference load my-model --gpu ECS1GPU11
regolo inference loaded
regolo inference unload <SESSION_ID>
```

## Development

```bash
python -m pytest
python -m build
```

Tests are offline and use HTTP mock transports.
