"""Public, synchronous Devmid user API and custom-model workflows.

Responses remain dictionaries so new server fields and omitted-versus-null
values are preserved. An injected HTTPX client remains owned by its caller.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator, TextIO
from urllib.parse import quote

import httpx
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, model_validator

from regolo.public_contract import PublicContract

DEFAULT_DEVMID_URL = "https://devmid.regolo.ai"


class RegoloAPIError(httpx.HTTPStatusError):
    """HTTP failure preserving status, response, and Devmid's detail envelope."""

    def __init__(self, response: httpx.Response):
        try:
            payload = response.json()
        except ValueError:
            payload = None
        self.detail = payload.get("detail") if isinstance(payload, dict) else None
        # Do not include arbitrary HTML, request bodies, or auth credentials.
        detail = self.detail
        if detail is None:
            detail = response.reason_phrase

        def sanitized(value):
            if isinstance(value, dict):
                return {
                    key: (
                        "<redacted>"
                        if key.lower()
                        in {
                            "input",
                            "password",
                            "api_key",
                            "token",
                            "access_token",
                            "refresh_token",
                            "authorization",
                            "secret",
                            "hf_token",
                        }
                        else sanitized(item)
                    )
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [sanitized(item) for item in value]
            return value

        rendered = str(sanitized(detail))
        secrets: list[str] = []

        def collect_secrets(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if (
                        any(
                            word in key.lower()
                            for word in ("password", "token", "api_key", "secret")
                        )
                        and isinstance(item, str)
                        and item
                    ):
                        secrets.append(item)
                    else:
                        collect_secrets(item)
            elif isinstance(value, list):
                for item in value:
                    collect_secrets(item)

        try:
            collect_secrets(json.loads(response.request.content))
        except (ValueError, httpx.RequestNotRead):
            pass
        # Some upstream errors echo credentials inside a free-text message.
        authorization = response.request.headers.get("authorization", "")
        if authorization:
            rendered = rendered.replace(authorization, "<redacted>")
            credential = authorization.removeprefix("Bearer ")
            if credential:
                rendered = rendered.replace(credential, "<redacted>")
        for secret in secrets:
            rendered = rendered.replace(secret, "<redacted>")
        super().__init__(
            f"Devmid HTTP {response.status_code}: {rendered[:4000]}",
            request=response.request,
            response=response,
        )


class RegoloProtocolError(ValueError):
    """The server returned an invalid successful response or pagination cursor."""


class ModelNotReadyError(RuntimeError):
    """A scan or deployment needs user intervention instead of further polling."""

    def __init__(self, message: str, state: dict[str, Any]):
        super().__init__(message)
        self.state = state


class GPUPlanConfig(BaseModel):
    """Autoscaling bounds; zero minimum enables scale-to-zero."""

    model_config = ConfigDict(extra="forbid", strict=True)
    min_replicas: int = Field(default=1, ge=0)
    max_replicas: int = Field(default=1, ge=1)
    target_requests: int | None = Field(default=None, gt=0)
    scale_down_delay_seconds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.min_replicas > self.max_replicas:
            raise ValueError("min_replicas must not exceed max_replicas")
        return self


# All writable fields in Devmid 3.9's public VLLMEngineConfig. Tensor parallel
# size and revision fields belong to the server; unsupported fields must never
# appear to work while being silently dropped by Devmid's request model.
VLLM_FIELDS = {
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "dtype",
    "kv_cache_dtype",
    "override_attention_dtype",
    "enable_prefix_caching",
    "enable_chunked_prefill",
    "block_size",
    "kv_sharing_fast_prefill",
    "optimization_level",
    "performance_mode",
    "enable_auto_tool_choice",
    "tool_call_parser",
    "chat_template",
    "reasoning_parser",
    "async_scheduling",
    "stream_interval",
    "speculative_config",
    "seed",
}


def _positive_id(value: int, name: str = "session_id") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _segment(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or value in {".", ".."}:
        raise ValueError("A nonempty model name or key ID is required")
    if any(c in value for c in "/\\\r\n"):
        raise ValueError("Model names and key IDs must be single path segments")
    return quote(value, safe="")


def _window(month=None, time_range_start=None, time_range_end=None) -> dict[str, str]:
    if month is not None:
        if time_range_start is not None or time_range_end is not None:
            raise ValueError("month and time ranges are mutually exclusive")
        if not isinstance(month, str) or not re.fullmatch(r"(0[1-9]|1[0-2])\d{4}", month):
            raise ValueError("month must use MMYYYY format")
        # Validate the calendar, including year zero and December overflow.
        datetime(int(month[2:]), int(month[:2]), 1)
        if month == "129999":
            raise ValueError("month's end exceeds the supported calendar")
        return {"month": month}
    if time_range_start is None and time_range_end is None:
        return {}
    if time_range_start is None or time_range_end is None:
        raise ValueError("Both time range start and end are required")
    values = []
    for value in (time_range_start, time_range_end):
        dt = value if isinstance(value, datetime) else datetime.fromisoformat(value)
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError("Time range values must include a timezone")
        values.append(dt)
    if values[1] <= values[0]:
        raise ValueError("Time range end must be after start")
    return dict(zip(("time_range_start", "time_range_end"), (dt.isoformat() for dt in values)))


class ModelManagementClient:
    """Keycloak-authenticated Devmid client, independent of the inference API key.

    Token persistence is opt-in via ``config_file``. The CLI enables it. Use
    ``token`` for an existing Keycloak access token, or call ``authenticate``.
    No HTTP or server mutation happens during construction.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_DEVMID_URL,
        *,
        token: str | None = None,
        refresh_token: str | None = None,
        timeout: float = 60,
        http_client: httpx.Client | None = None,
        config_file: str | Path | None = None,
        public_openapi: dict | str | Path | None = None,
    ):
        url = httpx.URL(base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.host
            or url.query
            or url.fragment
            or url.userinfo
        ):
            raise ValueError(
                "base_url must be an HTTP(S) server URL without credentials, query, or fragment"
            )
        self.base_url = str(url).rstrip("/")
        self.token = token
        self.refresh_token = refresh_token
        self.timeout = timeout
        self.config_file = Path(config_file).expanduser() if config_file is not None else None
        self._expires_at: float | None = None
        self._http = http_client
        self._owns_http = http_client is None
        self._closed = False
        self._public_contract = (
            PublicContract(public_openapi) if public_openapi is not None else None
        )
        if token is None:
            self._load_config()

    @property
    def is_closed(self) -> bool:
        """Whether this SDK client has been closed."""
        return self._closed

    @property
    def http_client(self) -> httpx.Client:
        if self._closed:
            raise RuntimeError("Client is closed")
        if self._http is None:
            self._http = httpx.Client()
        return self._http

    def close(self) -> None:
        if self._owns_http and self._http is not None:
            self._http.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _config(self) -> dict:
        if self.config_file is None or not self.config_file.exists():
            return {"version": 1, "servers": {}}
        try:
            data = json.loads(self.config_file.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ValueError("Cannot read Regolo token configuration; repair or remove it") from exc
        if not isinstance(data, dict) or not isinstance(data.get("servers", {}), dict):
            raise ValueError("Invalid Regolo token configuration")
        if "servers" not in data:
            # Old files have no server binding. Require a new login rather
            # than guessing where these credentials may be transmitted.
            warnings.warn(
                "Legacy unscoped token file ignored; log in again", UserWarning, stacklevel=3
            )
            return {"version": 1, "servers": {}}
        return data

    def _load_config(self) -> None:
        entry = self._config()["servers"].get(self.base_url, {})
        if not isinstance(entry, dict):
            raise ValueError("Invalid Regolo server token configuration")
        for key in ("access_token", "refresh_token"):
            if entry.get(key) is not None and not isinstance(entry[key], str):
                raise ValueError(f"Invalid {key} in token configuration")
        expires = entry.get("expires_at")
        if expires is not None and (
            isinstance(expires, bool)
            or not isinstance(expires, (int, float))
            or not math.isfinite(expires)
        ):
            raise ValueError("Invalid expires_at in token configuration")
        self.token = entry.get("access_token")
        self.refresh_token = entry.get("refresh_token")
        self._expires_at = entry.get("expires_at")

    def _save_config(self) -> None:
        if self.config_file is None:
            return
        data = self._config()
        if self.token:
            data["servers"][self.base_url] = {
                "access_token": self.token,
                "refresh_token": self.refresh_token,
                "expires_at": self._expires_at,
            }
        else:
            data["servers"].pop(self.base_url, None)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=".regolo-", dir=self.config_file.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            os.replace(temp, self.config_file)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def logout(self) -> None:
        """Clear this server's local tokens. This does not revoke Keycloak sessions."""
        self.token = self.refresh_token = self._expires_at = None
        self._save_config()

    def _headers(self) -> dict[str, str]:
        if not self.token:
            raise RuntimeError("Not authenticated. Call authenticate() or run 'regolo auth login'.")
        token = self.token.removeprefix("Bearer ")
        return {"Authorization": f"Bearer {token}"}

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        if not response.is_success:
            raise RegoloAPIError(response)
        if not response.content:
            return {}
        try:
            data = response.json()
        except ValueError as exc:
            raise RegoloProtocolError("Devmid returned non-JSON success data") from exc
        if not isinstance(data, dict):
            raise RegoloProtocolError("Devmid returned a non-object JSON response")
        return data

    def _set_tokens(self, data: dict, *, retain_refresh: bool = True) -> None:
        if not isinstance(data.get("access_token"), str) or not data["access_token"]:
            raise RegoloProtocolError("Authentication response is missing access_token")
        refresh = data.get("refresh_token")
        if refresh is not None and not isinstance(refresh, str):
            raise RegoloProtocolError("Authentication response has invalid refresh_token")
        expires = data.get("expires_in")
        if expires is not None and (
            isinstance(expires, bool)
            or not isinstance(expires, (int, float))
            or not math.isfinite(expires)
            or expires < 0
        ):
            raise RegoloProtocolError("Authentication response has invalid expires_in")
        self.token = data["access_token"]
        self.refresh_token = refresh or (self.refresh_token if retain_refresh else None)
        self._expires_at = time.time() + expires if expires is not None else None
        self._save_config()

    def authenticate(self, username: str, password: str) -> dict[str, Any]:
        """Exchange Keycloak credentials; the password is never persisted."""
        data = self._decode(
            self.http_client.post(
                f"{self.base_url}/auth/login",
                json={"username": username, "password": password},
                timeout=self.timeout,
            )
        )
        # A login is a new account/session; do not retain another user's refresh token.
        self._set_tokens(data, retain_refresh=False)
        return data

    def refresh_access_token(self) -> dict[str, Any]:
        """Exchange refresh_token, including the bearer header required by Devmid."""
        if not self.refresh_token:
            raise RuntimeError("No refresh token. Log in again.")
        data = self._decode(
            self.http_client.post(
                f"{self.base_url}/auth/refresh",
                headers=self._headers(),
                json={"refresh_token": self.refresh_token},
                timeout=self.timeout,
            )
        )
        self._set_tokens(data)
        return data

    def _refresh_token(self) -> bool:
        """Legacy boolean refresh helper; explicit refresh preserves errors."""
        try:
            self.refresh_access_token()
        except (httpx.HTTPError, RuntimeError, ValueError):
            return False
        return True

    def _make_request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        if (
            self.refresh_token
            and self._expires_at is not None
            and time.time() >= self._expires_at - 30
        ):
            self.refresh_access_token()
        headers = self._headers()
        response = self.http_client.request(
            method,
            f"{self.base_url}{endpoint}",
            headers=headers,
            timeout=self.timeout,
            **kwargs,
        )
        if response.status_code == 401 and self.refresh_token:
            self.refresh_access_token()
            response = self.http_client.request(
                method,
                f"{self.base_url}{endpoint}",
                headers=self._headers(),
                timeout=self.timeout,
                **kwargs,
            )
        # Never automatically repeat a load/register/unload on network or 5xx
        # failure: the server may already have committed the operation.
        return self._decode(response)

    def register_model(
        self,
        name: str,
        provider: str = "huggingface",
        url: str | None = None,
        api_key: str | None = None,
        force: bool = False,
        revision: str | None = None,
    ) -> dict:
        """Register a repository/revision. ``force`` is an ignored legacy argument."""
        # Retained for callers of the old SDK; Devmid has no registration force field.
        del force
        if provider != "huggingface" or not url:
            raise ValueError("A huggingface provider and repository URL are required")
        if not name or len(name) > 20:
            raise ValueError("Model names must contain 1–20 characters")
        data = {"name": name, "provider": provider, "url": url}
        data.update(
            {k: v for k, v in {"api_key": api_key, "revision": revision}.items() if v is not None}
        )
        return self._make_request("POST", "/models/load", json=data)

    def update_model_revision(
        self, registration_id: int, revision: str | None = None, api_key: str | None = None
    ) -> dict:
        """Compatibility stub: revision replacement has no supported public route."""
        raise NotImplementedError(
            "Revision replacement is not in the supported public user API. "
            "Register the new revision under a new name, then retire the old registration."
        )

    def get_models(self) -> dict:
        """List the caller's live registrations with scan/capacity/action fields."""
        return self._make_request("GET", "/models")

    def get_model(self, model_name: str) -> dict:
        """Fetch by owner-scoped name, not numeric registration ID."""
        return self._make_request("GET", f"/models/{_segment(model_name)}")

    def delete_model(self, model_name: str) -> dict:
        """Retire this registration; the shared immutable artifact remains."""
        return self._make_request("DELETE", f"/models/{_segment(model_name)}")

    def retry_model_scan(self, model_name: str) -> dict:
        """Request a scan retry. Server policy/409 remains authoritative."""
        return self._make_request("POST", f"/models/{_segment(model_name)}/scan/retry")

    def add_ssh_key(self, title: str, key: str) -> dict:
        return self._make_request("POST", "/ssh-keys", json={"title": title, "key": key})

    def get_ssh_keys(self) -> dict:
        return self._make_request("GET", "/ssh-keys")

    def delete_ssh_key(self, key_id: str) -> dict:
        if isinstance(key_id, bool) or not isinstance(key_id, (str, int)):
            raise ValueError("key_id must be a positive integer")
        return self._make_request("DELETE", f"/ssh-keys/{_positive_id(int(key_id), 'key_id')}")

    def get_available_gpus(self) -> dict:
        """Return the catalog, including availability, region and hourly EUR prices."""
        return self._make_request("GET", "/inference/gpus")

    def load_model_for_inference(
        self,
        model_name: str,
        gpu: str,
        force: bool = False,
        vllm_config: dict | None = None,
        gpu_plan_config: GPUPlanConfig | dict | None = None,
        *,
        engine: str | None = None,
        engine_config: dict | None = None,
        deployment_options: dict | None = None,
    ) -> dict:
        """Start a deployment; use returned session_id to poll readiness, not HTTP 200."""
        data: dict[str, Any] = {"model_name": model_name, "gpu": gpu, "force": force}
        if vllm_config is not None:
            unsupported = set(vllm_config) - VLLM_FIELDS
            if unsupported:
                raise ValueError(
                    f"Unsupported or server-controlled vLLM fields: {', '.join(sorted(unsupported))}"
                )
            data["vllm_config"] = vllm_config
        if gpu_plan_config is not None:
            plan = (
                gpu_plan_config
                if isinstance(gpu_plan_config, GPUPlanConfig)
                else GPUPlanConfig.model_validate(gpu_plan_config)
            )
            data["gpu_plan_config"] = plan.model_dump(exclude_none=True)
        if engine_config is not None and engine is None:
            raise ValueError("engine is required with engine_config")
        if engine is not None or deployment_options is not None:
            contract = self.get_public_contract()
            if engine is not None:
                if vllm_config is not None:
                    raise ValueError("Use engine_config or vllm_config, not both")
                data.update(contract.engine_fields(engine, engine_config))
            if deployment_options is not None:
                conflicts = set(data) & set(deployment_options)
                if conflicts:
                    raise ValueError(
                        f"deployment_options cannot override named fields: {sorted(conflicts)}"
                    )
                data.update(deployment_options)
            contract.validate_load(data)
        return self._make_request("POST", "/inference/load", json=data)

    def get_public_contract(self, *, refresh: bool = False) -> PublicContract:
        """Use an exported public schema, or fetch /openapi.json once when needed.

        This is read-only discovery. Older workflows need no discovery request.
        New engine fields and scaling-detail routes are never guessed.
        """
        if self._public_contract is None or refresh:
            response = self.http_client.get(f"{self.base_url}/openapi.json", timeout=self.timeout)
            self._public_contract = PublicContract(self._decode(response))
        return self._public_contract

    def unload_model_from_inference(self, session_id: int) -> dict:
        return self._make_request(
            "POST", "/inference/unload", json={"session_id": _positive_id(session_id)}
        )

    def get_loaded_models(self, *, include_scaling: bool = False) -> dict:
        """Open sessions, including pending deployments; failed closed sessions need history."""
        if include_scaling:
            return self._make_request(
                "GET", "/inference/loaded-models", params={"include_scaling": True}
            )
        return self._make_request("GET", "/inference/loaded-models")

    def get_user_inference_status(
        self, month=None, time_range_start=None, time_range_end=None
    ) -> dict:
        """Windowed billing/status; omitted filters select the current UTC month."""
        return self._make_request(
            "GET", "/inference/user-status", params=_window(month, time_range_start, time_range_end)
        )

    def get_inference_sessions(
        self,
        *,
        limit: int = 50,
        before_session_id: int | None = None,
        month=None,
        time_range_start=None,
        time_range_end=None,
        model_name: str | None = None,
        status: str | None = None,
        include_status: bool = False,
        include_error: bool = False,
        include_billing: bool = False,
        include_scaling: bool = False,
    ) -> dict:
        """One history page; total_cost is for this page, not the whole query.

        With no dates this includes all history. Stored failures use ``error``
        (runtime status maps them to ``failed``). Error text is opt-in/unbounded.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ValueError("limit must be an integer between 1 and 200")
        params: dict[str, Any] = _window(month, time_range_start, time_range_end)
        params.update(
            limit=limit,
            include_status=include_status,
            include_error=include_error,
            include_billing=include_billing,
        )
        if before_session_id is not None:
            params["before_session_id"] = _positive_id(before_session_id, "before_session_id")
        if model_name is not None:
            params["model_name"] = model_name
        if status is not None:
            params["status"] = status
        if include_scaling:
            params["include_scaling"] = True
        return self._make_request("GET", "/inference/sessions", params=params)

    def get_inference_session(self, session_id: int, **include_options) -> dict:
        """Read an exact owned history row, including retired/closed sessions.

        Uses the public keyset query; a gap or another owner's ID cannot return
        a neighboring session as though it were the requested session.
        """
        session_id = _positive_id(session_id)
        for key in include_options:
            if key not in {"include_status", "include_error", "include_billing", "include_scaling"}:
                raise ValueError(f"Unsupported session detail option: {key}")
        options = {"include_status": True, "include_billing": True, **include_options}
        page = self.get_inference_sessions(limit=1, before_session_id=session_id + 1, **options)
        rows = page.get("sessions")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise RegoloProtocolError("History response has invalid sessions")
        if (
            len(rows) != 1
            or type(rows[0].get("session_id")) is not int
            or rows[0].get("session_id") != session_id
        ):
            raise LookupError("Session was not found in the authenticated user's history")
        return rows[0]

    def get_session_details(
        self, session_id: int, *, include_scaling: bool = False, include_error: bool = False
    ) -> dict:
        """Preserve complete server status, billing and historical fields."""
        return {
            "session": self.get_inference_session(
                session_id, include_scaling=include_scaling, include_error=include_error
            ),
            "runtime": self.get_runtime_status(session_id),
            "billing": self.get_runtime_billing(session_id),
        }

    def get_scaling_history(
        self, session_id: int, *, limit: int = 100, cursor: str | int | None = None
    ) -> dict:
        """One raw scaling page from the route declared by the public schema.

        Preserve next_cursor, finalization, assumptions and frozen intervals.
        No billing timeline or scaling events are reconstructed by the SDK.
        """
        session_id = _positive_id(session_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        contract = self.get_public_contract()
        path, op = contract.scaling_operation()
        params: dict[str, Any] = {}
        cursor_used = False
        for p in op["parameters"]:
            name = p["name"]
            if name in {"session_id", "model_inference_id", "inference_session_id"}:
                if p["in"] == "path":
                    path = path.replace("{" + name + "}", str(session_id))
                else:
                    params[name] = session_id
            elif name == "limit":
                params[name] = limit
            elif cursor is not None and (
                "cursor" in name or name.startswith(("before_", "after_"))
            ):
                if cursor_used:
                    raise ValueError("Public scaling route has ambiguous pagination parameters")
                params[name] = cursor
                cursor_used = True
            elif p.get("required"):
                raise ValueError(f"Unsupported required public scaling parameter: {name}")
            if name in params:
                schema = contract.resolve(p.get("schema", {}))
                # HTTP query types: accept a textual CLI cursor for an integer.
                choices = [schema, *schema.get("anyOf", [])]
                if name != "limit" and any(s.get("type") == "integer" for s in choices):
                    value = params[name]
                    if isinstance(value, str) and value.isdecimal():
                        params[name] = int(value)
                if not Draft202012Validator(schema).is_valid(params[name]):
                    raise ValueError(f"Invalid public scaling parameter: {name}")
        if cursor is not None and not cursor_used:
            raise ValueError("Public scaling route declares no pagination cursor")
        if "{" in path or "}" in path:
            raise ValueError("Public scaling route has an unsupported path parameter")
        return self._make_request("GET", path, params=params)

    def iter_scaling_history(
        self,
        session_id: int,
        *,
        collection: str = "intervals",
        limit: int = 100,
        cursor: str | int | None = None,
    ) -> Iterator[dict]:
        """Follow scaling cursors; choose the response array named by your schema.

        Use get_scaling_history for page-level metadata and frozen/live flags.
        Repeated cursors are protocol failures, never infinite fetch loops.
        """
        seen = {str(cursor)} if cursor is not None else set()
        while True:
            page = self.get_scaling_history(session_id, limit=limit, cursor=cursor)
            rows = page.get(collection)
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise RegoloProtocolError(f"Scaling response has no object array: {collection}")
            yield from rows
            cursor = page.get("next_cursor")
            if cursor is None:
                return
            if (
                isinstance(cursor, bool)
                or not isinstance(cursor, (str, int))
                or not rows
                or str(cursor) in seen
            ):
                raise RegoloProtocolError(
                    "Scaling response returned an invalid or repeated next_cursor"
                )
            seen.add(str(cursor))

    def get_dashboard_snapshot(
        self,
        *,
        include_scaling: bool = False,
        month=None,
        time_range_start=None,
        time_range_end=None,
    ) -> dict:
        """User-only overview, retaining all model/GPU/runtime/windowed stats."""
        return {
            "models": self.get_models(),
            "gpus": self.get_available_gpus(),
            "loaded_models": self.get_loaded_models(include_scaling=include_scaling),
            "usage": self.get_user_inference_status(month, time_range_start, time_range_end),
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }

    def iter_inference_sessions(self, **filters) -> Iterator[dict]:
        """Follow all history cursors lazily. Stop on inconsistent pagination."""
        cursor = filters.pop("before_session_id", None)
        previous_id = cursor
        while True:
            page = self.get_inference_sessions(before_session_id=cursor, **filters)
            rows = page.get("sessions")
            if not isinstance(rows, list):
                raise RegoloProtocolError("History response is missing sessions")
            for row in rows:
                if not isinstance(row, dict):
                    raise RegoloProtocolError("History contains a non-object session")
                sid = row.get("session_id")
                if (
                    isinstance(sid, bool)
                    or not isinstance(sid, int)
                    or sid < 1
                    or (previous_id is not None and sid >= previous_id)
                ):
                    raise RegoloProtocolError("History session IDs must strictly decrease")
                previous_id = sid
                yield row
            next_cursor = page.get("next_cursor")
            if next_cursor is None:
                return
            if not rows or next_cursor != previous_id or isinstance(next_cursor, bool):
                raise RegoloProtocolError("History returned an invalid next_cursor")
            cursor = next_cursor

    def get_runtime_status(self, session_id: int) -> dict:
        return self._make_request(
            "GET", "/runtime/status", params={"model_inference_id": _positive_id(session_id)}
        )

    def get_runtime_billing(self, session_id: int) -> dict:
        """Server-calculated lifetime wall-clock/replica metrics and cost in EUR."""
        return self._make_request(
            "GET", "/runtime/billing", params={"model_inference_id": _positive_id(session_id)}
        )

    @staticmethod
    def _wait(fetch, ready, blocked, timeout: float, poll_interval: float) -> dict:
        if (
            not math.isfinite(timeout)
            or not math.isfinite(poll_interval)
            or timeout < 0
            or poll_interval <= 0
        ):
            raise ValueError("timeout must be nonnegative and poll_interval positive")
        deadline = time.monotonic() + timeout
        while True:
            state = fetch()
            if ready(state):
                return state
            reason = blocked(state)
            if reason:
                raise ModelNotReadyError(reason, state)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out after {timeout} seconds; last state: {state}")
            time.sleep(min(poll_interval, remaining))

    def wait_for_scan(
        self, model_name: str, timeout: float = 3600, poll_interval: float = 5
    ) -> dict:
        """Wait only for a safe verdict; stop on security or capacity blockers."""

        def blocked(state):
            status = state.get("scan_status")
            if status in {
                "unsafe",
                "inconclusive",
                "awaiting_credentials",
                "capacity_blocked_state",
            }:
                return f"Security scan stopped in {status}: {state.get('scan_detail') or state.get('recommended_action')}"
            if status not in {"pending", "scanning", "error", "safe"}:
                return f"Unknown scan status: {status}"
            return None

        return self._wait(
            lambda: self.get_model(model_name),
            lambda s: s.get("scan_status") == "safe",
            blocked,
            timeout,
            poll_interval,
        )

    def wait_for_inference(
        self, session_id: int, timeout: float = 1800, poll_interval: float = 5
    ) -> dict:
        """Wait for running, preserving full failure/stop text in exception.state."""

        def blocked(state):
            status = state.get("status")
            if status in {"failed", "error", "stopped", "completed"}:
                return f"Inference ended in {status}: {state.get('error') or 'no detail supplied'}"
            if status not in {"pending", "running"}:
                return f"Unknown runtime status: {status}"
            return None

        _positive_id(session_id)
        return self._wait(
            lambda: self.get_runtime_status(session_id),
            lambda s: s.get("status") == "running",
            blocked,
            timeout,
            poll_interval,
        )

    def load_and_wait(
        self,
        model_name: str,
        gpu: str,
        *,
        scan_timeout=3600,
        timeout=1800,
        poll_interval=5,
        **load_options,
    ) -> dict:
        """Wait for scan, load, then poll runtime. Timeout never auto-unloads."""
        model = self.wait_for_scan(model_name, scan_timeout, poll_interval)
        if model.get("capacity_status") in {
            "requires_approval",
            "unsupported",
            "hard_limit_exceeded",
            "denied",
            "blocked",
        }:
            raise ModelNotReadyError(
                model.get("capacity_reason") or "Model capacity requires intervention", model
            )
        loaded = self.load_model_for_inference(model_name, gpu, **load_options)
        if loaded.get("success") is False or not loaded.get("session_id"):
            raise RegoloProtocolError("Load did not return a successful session_id")
        try:
            state = self.wait_for_inference(loaded["session_id"], timeout, poll_interval)
        except TimeoutError as exc:
            raise TimeoutError(
                f"Session {loaded['session_id']} is still reserved; inspect status or unload explicitly"
            ) from exc
        return {**loaded, "runtime": state}

    def get_usage_summary(
        self,
        *,
        month=None,
        time_range_start=None,
        time_range_end=None,
        model_name=None,
        status=None,
        page_size=200,
    ) -> dict:
        """Aggregate every page using server billing, with model/GPU breakdowns.

        Uses cost_euro (reporting window), not cost_euro_from_start. No local
        price calculation or re-rounding. Open-session costs can change while
        pages are fetched; this is an observation, not an invoice snapshot.
        """

        def empty():
            return {
                "sessions": 0,
                "open_sessions": 0,
                "duration_seconds": Decimal(0),
                "replica_seconds": Decimal(0),
                "billable_replica_hours": Decimal(0),
                "cost_euro": Decimal(0),
            }

        total = empty()
        models: dict[str, dict[str, Any]] = {}
        gpus: dict[str, dict[str, Any]] = {}
        statuses: Counter[str] = Counter()
        for row in self.iter_inference_sessions(
            limit=page_size,
            month=month,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            model_name=model_name,
            status=status,
            include_status=True,
            include_billing=True,
        ):
            buckets = [
                total,
                models.setdefault(row["model_name"], empty()),
                gpus.setdefault(row["gpu"], empty()),
            ]
            for bucket in buckets:
                bucket["sessions"] += 1
                bucket["open_sessions"] += bool(row.get("is_open"))
                for field in (
                    "duration_seconds",
                    "replica_seconds",
                    "billable_replica_hours",
                    "cost_euro",
                ):
                    try:
                        number = Decimal(str(row[field]))
                        if not number.is_finite() or number < 0:
                            raise InvalidOperation
                    except (KeyError, InvalidOperation, ValueError) as exc:
                        raise RegoloProtocolError(
                            f"Session {row.get('session_id') or 'unknown'} has missing/invalid {field}"
                        ) from exc
                    bucket[field] += number
            statuses[row.get("status") or "unknown"] += 1

        def output(values):
            return {k: float(v) if isinstance(v, Decimal) else v for k, v in values.items()}

        return {
            "currency": "EUR",
            "filters": _window(month, time_range_start, time_range_end)
            | {"model_name": model_name, "status": status},
            **output(total),
            "status_counts": dict(statuses),
            "by_model": {k: output(v) for k, v in models.items()},
            "by_gpu": {k: output(v) for k, v in gpus.items()},
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }

    def export_sessions_csv(
        self, destination: str | Path | TextIO, *, spreadsheet_safe: bool = True, **filters
    ) -> int:
        """Export all pages, preserving nullable fields. Return exported row count."""
        fields = [
            "session_id",
            "session_number",
            "registration_id",
            "model_name",
            "gpu",
            "status",
            "error",
            "load_timestamp",
            "unload_timestamp",
            "is_open",
            "duration_seconds",
            "duration_hours",
            "replica_seconds",
            "billable_replica_hours",
            "hourly_price",
            "cost_euro",
            "cost_euro_from_start",
            "max_model_len",
            "max_model_len_source",
        ]
        filters.setdefault("include_status", True)
        filters.setdefault("include_billing", True)
        handle: TextIO
        if isinstance(destination, (str, Path)):
            handle = open(destination, "w", encoding="utf-8", newline="")
            owned = True
        else:
            handle = destination
            owned = False
        try:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            count = 0
            for row in self.iter_inference_sessions(**filters):
                if spreadsheet_safe:
                    row = {
                        key: (
                            "'" + value
                            if isinstance(value, str)
                            and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r"))
                            else value
                        )
                        for key, value in row.items()
                    }
                writer.writerow(row)
                count += 1
            return count
        finally:
            if owned:
                handle.close()

    def export_sessions_json(self, destination: str | Path, **filters) -> int:
        """Atomic JSON export of all fields/pages, including nested scaling data."""
        path = Path(destination).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".regolo-history-", dir=path.parent)
        count = 0
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write("[")
                for row in self.iter_inference_sessions(**filters):
                    if count:
                        handle.write(",\n")
                    json.dump(row, handle, ensure_ascii=False)
                    count += 1
                handle.write("]\n")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return count
