# Contributing

## Setup

Use Python 3.12 or newer and install the checkout with development dependencies:

```bash
python -m pip install -e '.[dev]'
python -c 'import regolo; print(regolo.__file__)'
```

The module path should point into this checkout's `src/regolo` directory. For
PyCharm, open the directory containing `pyproject.toml`, select the same virtual
environment, and mark `src` as a Sources Root if it is not detected automatically.
With uv, use `uv pip install -e '.[dev]'`.

`requirements.txt` contains runtime dependencies only. To regenerate it:

```bash
python -m piptools compile --strip-extras --output-file requirements.txt pyproject.toml
```

## Checks

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest --cov=regolo --cov-report=term-missing
python scripts/check_devmid_contract.py tests/fixtures/devmid-3.9-public-openapi.json
```

The test suite uses HTTPX MockTransport and rejects real HTTP requests. It uses
test credentials and temporary token files. Tests cover management requests,
authentication, pagination, billing, exports, CLI behavior, inference, streaming,
tools, structured replies, media inputs and transport ownership.

Schema discovery tests use simulated API extensions. These test documents are
not server exports. The SDK's `py.typed` marker and overloads support typed
consumer code; examples are in `tests/typing_api.py`.

GitHub CI runs lint, formatting, type checks, tests, contract checks and builds on
Python 3.12, 3.13 and 3.14. Package, bumpver and module versions must agree.

## Updating the public API contract

`tests/fixtures/devmid-3.9-public-openapi.json` contains the 3.9 public user API
contract. Compare another public export against it with:

```bash
python scripts/check_devmid_contract.py public-openapi.json
```

Added or removed operations, parameter changes and request/response schema
changes produce a nonzero exit status. Descriptions, titles and examples do not
affect the comparison. Update the implementation, tests and documentation for
API changes before replacing the fixture. Include only public user operations
and their schemas; use test data instead of production records or credentials.

## Optional live inference tests

Live inference tests make paid API requests and run separately from pytest:

```bash
REGOLO_API_KEY='<KEY>' python scripts/smoke_inference.py --run-paid --max-requests 16 --report live-smoke-report.json
```

Omit `REGOLO_API_KEY` to enter the key at a password prompt. The runner selects
accessible catalog models, caps requests at 16, generates at most one 1024×1024
image and limits chat calls to 128 output tokens. It makes no automatic retries
and excludes credentials from its report. Audio inputs can use generated silence
to check upload/response formats. The runner does not exercise custom deployments
or video input.

Use `--only structured_json function_tools` to select workflows. Incomplete or
truncated replies are failures. Test custom deployment lifecycles separately
with authorized test accounts and explicitly unload billable resources afterward.

## Building

```bash
python -m build
```

Builds are written to `dist/`. Wheels include runtime modules and the CLI entry
point. Source distributions also include tests, the public schema fixture,
scripts and documentation. Rebuild release files after changing the source.
