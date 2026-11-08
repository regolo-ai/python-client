from __future__ import annotations

import io
import json
import stat
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator

from regolo import (
    GPUPlanConfig,
    ModelManagementClient,
    ModelNotReadyError,
    RegoloAPIError,
    RegoloProtocolError,
)
from regolo.management import VLLM_FIELDS

SCHEMA = json.loads((Path(__file__).parent / "fixtures/devmid-3.9-public-openapi.json").read_text())


def client(handler, **kwargs):
    return ModelManagementClient(
        "https://devmid.test",
        token="access",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


# Invokes the SDK method for every operation from the bundled public
# OpenAPI. Outgoing bodies are validated against its schemas, not SDK mocks.
OPERATIONS = [
    ("POST", "/auth/login", "authenticate", ("user", "password"), {}),
    ("POST", "/auth/refresh", "refresh_access_token", (), {}),
    ("GET", "/models", "get_models", (), {}),
    ("GET", "/models/{model}", "get_model", ("custom",), {}),
    ("DELETE", "/models/{model}", "delete_model", ("custom",), {}),
    (
        "POST",
        "/models/load",
        "register_model",
        ("custom", "huggingface", "org/model"),
        {"api_key": "hf-token", "revision": "v2"},
    ),
    ("POST", "/models/{model}/scan/retry", "retry_model_scan", ("custom",), {}),
    ("GET", "/ssh-keys", "get_ssh_keys", (), {}),
    ("POST", "/ssh-keys", "add_ssh_key", ("laptop", "ssh-ed25519 abc"), {}),
    ("DELETE", "/ssh-keys/{keyId}", "delete_ssh_key", ("42",), {}),
    ("GET", "/inference/gpus", "get_available_gpus", (), {}),
    (
        "POST",
        "/inference/load",
        "load_model_for_inference",
        ("custom", "ECS1GPU11"),
        {
            "gpu_plan_config": {"min_replicas": 0, "max_replicas": 3},
            "vllm_config": {"max_model_len": 4096, "enable_prefix_caching": False},
        },
    ),
    ("POST", "/inference/unload", "unload_model_from_inference", (42,), {}),
    ("GET", "/inference/loaded-models", "get_loaded_models", (), {}),
    ("GET", "/inference/user-status", "get_user_inference_status", (), {"month": "102026"}),
    (
        "GET",
        "/inference/sessions",
        "get_inference_sessions",
        (),
        {
            "limit": 200,
            "before_session_id": 42,
            "month": "102026",
            "model_name": "custom",
            "status": "error",
            "include_status": True,
            "include_error": True,
            "include_billing": True,
        },
    ),
    ("GET", "/runtime/status", "get_runtime_status", (42,), {}),
    ("GET", "/runtime/billing", "get_runtime_billing", (42,), {}),
]


def test_public_contract_has_no_unimplemented_operation():
    public = {
        (m.upper(), p)
        for p, ops in SCHEMA["paths"].items()
        for m in ops
        if m in {"get", "post", "delete", "patch", "put"}
    }
    assert {(m, p) for m, p, *_ in OPERATIONS} == public
    assert all(not path.startswith(("/admin", "/security", "/observability")) for _, path in public)


@pytest.mark.parametrize("method,path,name,args,kwargs", OPERATIONS, ids=[o[2] for o in OPERATIONS])
def test_public_operation_matches_authoritative_contract(method, path, name, args, kwargs):
    requests = []
    op = SCHEMA["paths"][path][method.lower()]

    def handler(request):
        requests.append(request)
        assert request.method == method
        expected_path = path.replace("{model}", "custom").replace("{keyId}", "42")
        assert request.url.path == expected_path
        if path != "/auth/login":
            assert request.headers["authorization"] == "Bearer access"
        else:
            assert "authorization" not in request.headers
        content = op.get("requestBody", {}).get("content", {})
        if content:
            body_schema = content["application/json"]["schema"]
            Draft202012Validator({**body_schema, "components": SCHEMA["components"]}).validate(
                json.loads(request.content)
            )
        allowed = {p["name"] for p in op.get("parameters", []) if p["in"] == "query"}
        assert set(request.url.params) <= allowed
        for parameter in op.get("parameters", []):
            if parameter["in"] != "query":
                continue
            value = request.url.params.get(parameter["name"])
            if value is None:
                assert not parameter.get("required")
                continue
            schema = parameter["schema"]
            kind = schema.get("type") or schema.get("anyOf", [{}])[0].get("type")
            if kind == "integer":
                value = int(value)
            elif kind == "boolean":
                assert value in {"true", "false"}
                value = value == "true"
            Draft202012Validator(schema).validate(value)
        return httpx.Response(
            200, json={"access_token": "new", "refresh_token": "rotated", "extra": "preserved"}
        )

    c = client(handler, refresh_token="refresh")
    assert getattr(c, name)(*args, **kwargs)["extra"] == "preserved"
    assert len(requests) == 1


@pytest.mark.parametrize("code", [400, 401, 402, 403, 404, 409, 422, 429, 500, 503])
def test_api_error_keeps_status_detail_and_does_not_repeat_mutations(code):
    calls = []
    detail = [{"loc": ["body", "gpu"], "msg": "required"}] if code == 422 else "failure detail"

    def handler(r):
        calls.append(r)
        return httpx.Response(code, json={"detail": detail})

    c = client(handler)
    with pytest.raises(RegoloAPIError) as error:
        c.load_model_for_inference("custom", "gpu")
    assert error.value.response.status_code == code
    assert error.value.detail == detail
    assert len(calls) == 1


def test_html_errors_and_network_errors_do_not_mask_original_failure():
    c = client(lambda r: httpx.Response(502, text="<html>proxy</html>"))
    with pytest.raises(RegoloAPIError, match="502") as e:
        c.get_models()
    assert e.value.detail is None

    def offline(r):
        raise httpx.ConnectError("offline", request=r)

    c = client(offline)
    with pytest.raises(httpx.ConnectError, match="offline"):
        c.get_models()


@pytest.mark.parametrize("body", [b"not JSON", b"[]", b"null"])
def test_invalid_success_response_is_explicit(body):
    with pytest.raises(RegoloProtocolError):
        client(lambda r: httpx.Response(200, content=body)).get_models()


def test_empty_success_is_allowed():
    assert client(lambda r: httpx.Response(204)).delete_ssh_key("42") == {}


def test_auth_refresh_has_bearer_rotates_tokens_and_retries_once():
    requests = []

    def handler(r):
        requests.append(r)
        if r.url.path == "/auth/refresh":
            assert r.headers["authorization"] == "Bearer access"
            assert json.loads(r.content) == {"refresh_token": "refresh"}
            return httpx.Response(200, json={"access_token": "new", "refresh_token": "rotated"})
        if r.headers["authorization"] == "Bearer access":
            return httpx.Response(401, json={"detail": "expired"})
        return httpx.Response(200, json={"models": []})

    c = client(handler, refresh_token="refresh")
    assert c.get_models() == {"models": []}
    assert [r.url.path for r in requests] == ["/models", "/auth/refresh", "/models"]
    assert c.refresh_token == "rotated"


def test_repeated_401_or_failed_refresh_does_not_loop():
    seen = []

    def handler(r):
        seen.append(r.url.path)
        return httpx.Response(401, json={"detail": "denied"})

    c = client(handler, refresh_token="r")
    with pytest.raises(RegoloAPIError):
        c.get_models()
    assert seen == ["/models", "/auth/refresh"]
    assert c._refresh_token() is False


def test_known_expiry_refreshes_before_request(monkeypatch):
    seen = []

    def handler(r):
        seen.append(r.url.path)
        if r.url.path == "/auth/login":
            return httpx.Response(
                200, json={"access_token": "access", "refresh_token": "r", "expires_in": 60}
            )
        if r.url.path == "/auth/refresh":
            return httpx.Response(
                200, json={"access_token": "new", "refresh_token": "rr", "expires_in": 300}
            )
        assert r.headers["authorization"] == "Bearer new"
        return httpx.Response(200, json={})

    c = client(handler)
    monkeypatch.setattr("regolo.management.time.time", lambda: 100)
    c.authenticate("u", "p")
    monkeypatch.setattr("regolo.management.time.time", lambda: 131)
    c.get_models()
    assert seen == ["/auth/login", "/auth/refresh", "/models"]


def test_server_scoped_config_permissions_logout_and_password_not_saved(tmp_path):
    path = tmp_path / "nested/tokens.json"
    http = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, json={"access_token": "access", "refresh_token": "refresh", "expires_in": 300}
            )
        )
    )
    c = ModelManagementClient("https://one.test/", config_file=path, http_client=http)
    c.authenticate("user", "SECRET-PASSWORD")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "SECRET-PASSWORD" not in path.read_text()
    assert ModelManagementClient("https://one.test", config_file=path).token == "access"
    assert ModelManagementClient("https://other.test", config_file=path).token is None
    other = ModelManagementClient("https://other.test", config_file=path, http_client=http)
    other.authenticate("other", "password")
    c.logout()
    assert c.token is None and c.refresh_token is None
    assert ModelManagementClient("https://one.test", config_file=path).token is None
    assert ModelManagementClient("https://other.test", config_file=path).token == "access"


def test_legacy_corrupt_and_invalid_config(tmp_path):
    path = tmp_path / "tokens.json"
    path.write_text('{"access_token": "unscoped"}')
    with pytest.warns(UserWarning, match="Legacy"):
        assert ModelManagementClient(config_file=path).token is None
    for payload in (
        "bad JSON",
        "[]",
        '{"servers": []}',
        '{"servers": {"https://devmid.regolo.ai": []}}',
    ):
        path.write_text(payload)
        with pytest.raises(ValueError):
            ModelManagementClient(config_file=path)


def test_client_lifecycle_injected_transport_and_not_authenticated():
    c = client(lambda r: httpx.Response(200, json={}))
    assert not c.is_closed
    http = c.http_client
    with c:
        c.get_models()
    assert not http.is_closed
    assert c.is_closed
    with pytest.raises(RuntimeError, match="closed"):
        c.get_models()
    c = ModelManagementClient(http_client=http)
    with pytest.raises(RuntimeError, match="Not authenticated"):
        c.get_models()
    with pytest.raises(RuntimeError, match="No refresh"):
        c.refresh_access_token()
    assert c._refresh_token() is False


@pytest.mark.parametrize(
    "url", ["file:///tmp/a", "https://x.test?secret=1", "https://u:p@x.test", "https://x.test#frag"]
)
def test_bad_base_urls(url):
    with pytest.raises(ValueError):
        ModelManagementClient(url)


@pytest.mark.parametrize("value", [0, -1, True, "42", 1.5])
def test_session_id_validation(value: object):
    c = client(lambda r: pytest.fail("should not send invalid ID"))
    for method in (
        c.get_runtime_status,
        c.get_runtime_billing,
        c.unload_model_from_inference,
        c.wait_for_inference,
    ):
        with pytest.raises(ValueError):
            method(value)  # type: ignore[arg-type]  # Deliberately invalid input.


@pytest.mark.parametrize("value", [0, 201, -1, True, 1.5])
def test_page_limit_validation(value: object):
    with pytest.raises(ValueError):
        client(lambda r: pytest.fail("no HTTP")).get_inference_sessions(limit=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "filters",
    [
        {"month": "132026"},
        {"month": "2026-10"},
        {"month": "010000"},
        {"month": "129999"},
        {"month": "102026", "time_range_start": "2026-10-01Z"},
        {"time_range_start": "2026-10-01T00:00:00Z"},
        {"time_range_start": "2026-10-01", "time_range_end": "2026-10-02"},
        {"time_range_start": "2026-10-02T00:00:00Z", "time_range_end": "2026-10-01T00:00:00Z"},
    ],
)
def test_invalid_windows(filters):
    c = client(lambda r: pytest.fail("no HTTP"))
    with pytest.raises(ValueError):
        c.get_inference_sessions(**filters)
    with pytest.raises(ValueError):
        c.get_user_inference_status(**filters)


def test_timezone_ranges_and_explicit_false_flags_preserved():
    seen = []

    def handler(r):
        seen.append(dict(r.url.params))
        return httpx.Response(200, json={})

    c = client(handler)
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    c.get_inference_sessions(
        time_range_start=start, time_range_end="2026-11-01T01:00:00+01:00", include_status=False
    )
    assert seen[0]["time_range_start"] == start.isoformat()
    assert seen[0]["time_range_end"].endswith("+01:00")
    assert seen[0]["include_status"] == "false"
    assert "month" not in seen[0]


@pytest.mark.parametrize("name", ["", "..", ".", "x/y", "x\\y", "x\n"])
def test_invalid_path_segments(name):
    with pytest.raises(ValueError):
        client(lambda r: pytest.fail("no HTTP")).get_model(name)


def test_model_path_is_encoded_without_changing_query():
    seen = []

    def handler(r):
        seen.append(r)
        return httpx.Response(200, json={})

    client(handler).get_model("name?#%")
    assert seen[0].url.query == b""
    assert seen[0].url.raw_path == b"/models/name%3F%23%25"


def test_vllm_writable_fields_match_schema():
    props = SCHEMA["components"]["schemas"]["VLLMEngineConfig"]["properties"]
    assert VLLM_FIELDS == {k for k, v in props.items() if not v.get("readOnly")}


@pytest.mark.parametrize(
    "field",
    [
        "revision",
        "code_revision",
        "tokenizer_revision",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "disable_log_requests",
        "typo",
    ],
)
def test_unsupported_config_cannot_be_silently_ignored(field):
    with pytest.raises(ValueError, match="Unsupported"):
        client(lambda r: pytest.fail("no HTTP")).load_model_for_inference(
            "m", "gpu", vllm_config={field: 1}
        )


@pytest.mark.parametrize(
    "plan",
    [
        {"min_replicas": -1},
        {"max_replicas": 0},
        {"min_replicas": 3, "max_replicas": 2},
        {"max_replicas": True},
        {"typo": 1},
    ],
)
def test_invalid_autoscaling(plan):
    with pytest.raises(ValueError):
        GPUPlanConfig.model_validate(plan)


def row(sid, cost, **kwargs):
    return {
        "session_id": sid,
        "registration_id": 1,
        "model_name": "m",
        "gpu": "gpu",
        "status": "completed",
        "is_open": False,
        "duration_seconds": 100,
        "replica_seconds": 200,
        "billable_replica_hours": 1,
        "cost_euro": cost,
        "cost_euro_from_start": 99,
        **kwargs,
    }


def history_handler(requests):
    def handler(r):
        requests.append(r)
        if "before_session_id" not in r.url.params:
            return httpx.Response(
                200,
                json={
                    "sessions": [row(3, 0.1), row(2, 0.2, model_name="other", status="error")],
                    "count": 2,
                    "total": 3,
                    "next_cursor": 2,
                    "total_cost": 0.3,
                },
            )
        assert r.url.params["before_session_id"] == "2"
        return httpx.Response(
            200,
            json={
                "sessions": [row(1, 0.3, is_open=True, status="running")],
                "count": 1,
                "total": 3,
                "next_cursor": None,
                "total_cost": 0.3,
            },
        )

    return handler


def test_all_page_stats_use_window_cost_and_decimal_sum():
    requests: list[httpx.Request] = []
    summary = client(history_handler(requests)).get_usage_summary(month="102026", page_size=2)
    assert summary["sessions"] == 3
    assert summary["open_sessions"] == 1
    assert summary["cost_euro"] == 0.6
    assert summary["replica_seconds"] == 600
    assert summary["duration_seconds"] == 300
    assert summary["billable_replica_hours"] == 3
    assert summary["by_model"]["m"]["cost_euro"] == 0.4
    assert summary["by_gpu"]["gpu"]["cost_euro"] == 0.6
    assert summary["status_counts"] == {"completed": 1, "error": 1, "running": 1}
    assert all(r.url.params["month"] == "102026" for r in requests)
    assert all(r.url.params["include_billing"] == "true" for r in requests)
    assert all(r.url.params["include_error"] == "false" for r in requests)


@pytest.mark.parametrize(
    "page",
    [
        {},
        {"sessions": [], "next_cursor": 1},
        {"sessions": [row(3, 1)], "next_cursor": 4},
        {"sessions": [row(3, 1), row(3, 1)]},
        {"sessions": [row(2, 1), row(3, 1)]},
        {"sessions": ["bad"]},
        {"sessions": [row(True, 1)]},
    ],
)
def test_invalid_pagination_fails_instead_of_looping_or_double_counting(page):
    with pytest.raises(RegoloProtocolError):
        list(client(lambda r: httpx.Response(200, json=page)).iter_inference_sessions())


def test_empty_stats_and_missing_billing_are_distinct():
    c = client(lambda r: httpx.Response(200, json={"sessions": [], "next_cursor": None}))
    assert c.get_usage_summary()["cost_euro"] == 0
    for value in (None, "NaN", -1):
        c = client(lambda r: httpx.Response(200, json={"sessions": [row(1, value)]}))
        with pytest.raises(RegoloProtocolError, match="cost_euro"):
            c.get_usage_summary()


def test_csv_export_all_pages_nulls_and_caller_handle(tmp_path):
    c = client(history_handler([]))
    dest = io.StringIO()
    assert c.export_sessions_csv(dest, month="102026") == 3
    assert not dest.closed
    assert "cost_euro_from_start" in dest.getvalue().splitlines()[0]
    assert len(dest.getvalue().splitlines()) == 4
    file = tmp_path / "history.csv"
    assert c.export_sessions_csv(file) == 3
    assert file.read_text().startswith("session_id,")


@pytest.mark.parametrize(
    "status",
    ["unsafe", "inconclusive", "awaiting_credentials", "capacity_blocked_state", "new_unknown"],
)
def test_scan_blockers_stop_immediately(status, monkeypatch):
    c = client(lambda r: httpx.Response(200, json={"scan_status": status, "scan_detail": "reason"}))
    monkeypatch.setattr("regolo.management.time.sleep", lambda t: pytest.fail("must not sleep"))
    with pytest.raises(ModelNotReadyError) as e:
        c.wait_for_scan("m")
    assert e.value.state["scan_status"] == status


def test_scan_and_runtime_poll_then_succeed_without_real_sleep(monkeypatch):
    states = iter(
        [
            {"scan_status": "pending"},
            {"scan_status": "error"},
            {"scan_status": "safe"},
            {"status": "pending"},
            {"status": "running"},
        ]
    )
    c = client(lambda r: httpx.Response(200, json=next(states)))
    monkeypatch.setattr("regolo.management.time.sleep", lambda t: None)
    assert c.wait_for_scan("m")["scan_status"] == "safe"
    assert c.wait_for_inference(1)["status"] == "running"


@pytest.mark.parametrize("status", ["failed", "error", "stopped", "completed", "new_unknown"])
def test_runtime_terminal_states_preserve_error(status):
    error = "full error\n" * 500
    c = client(lambda r: httpx.Response(200, json={"status": status, "error": error}))
    with pytest.raises(ModelNotReadyError) as e:
        c.wait_for_inference(1)
    assert e.value.state["error"] == error


def test_wait_timeouts_and_invalid_options():
    c = client(lambda r: httpx.Response(200, json={"status": "pending", "scan_status": "scanning"}))
    with pytest.raises(TimeoutError):
        c.wait_for_scan("m", timeout=0)
    with pytest.raises(TimeoutError):
        c.wait_for_inference(1, timeout=0)
    with pytest.raises(ValueError):
        c.wait_for_scan("m", poll_interval=0)
    with pytest.raises(ValueError):
        c.wait_for_inference(1, timeout=-1)


def test_load_and_wait_correct_order_and_capacity_gate():
    seen = []

    def handler(r):
        seen.append(r.url.path)
        if r.url.path == "/models/m":
            return httpx.Response(200, json={"scan_status": "safe", "capacity_status": "supported"})
        if r.url.path == "/inference/load":
            return httpx.Response(200, json={"success": True, "session_id": 42})
        return httpx.Response(200, json={"status": "running"})

    assert client(handler).load_and_wait("m", "gpu")["session_id"] == 42
    assert seen == ["/models/m", "/inference/load", "/runtime/status"]
    c = client(
        lambda r: httpx.Response(
            200,
            json={
                "scan_status": "safe",
                "capacity_status": "requires_approval",
                "capacity_reason": "too large",
            },
        )
    )
    with pytest.raises(ModelNotReadyError, match="too large"):
        c.load_and_wait("m", "gpu")


def test_load_wait_timeout_retains_session_id_and_never_unloads():
    seen = []

    def handler(r):
        seen.append(r.url.path)
        if r.url.path.startswith("/models/"):
            return httpx.Response(200, json={"scan_status": "safe"})
        if r.url.path == "/inference/load":
            return httpx.Response(200, json={"session_id": 42})
        return httpx.Response(200, json={"status": "pending"})

    with pytest.raises(TimeoutError, match="Session 42"):
        client(handler).load_and_wait("m", "gpu", timeout=0)
    assert "/inference/unload" not in seen


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"access_token": None},
        {"access_token": "new", "refresh_token": 1},
        {"access_token": "new", "expires_in": "60"},
        {"access_token": "new", "expires_in": True},
        {"access_token": "new", "expires_in": -1},
    ],
)
def test_malformed_auth_response_does_not_corrupt_current_credentials(data):
    c = client(lambda r: httpx.Response(200, json=data), refresh_token="old")
    with pytest.raises(RegoloProtocolError):
        c.authenticate("u", "p")
    assert c.token == "access" and c.refresh_token == "old"


def test_new_login_does_not_keep_previous_user_refresh_token():
    c = client(lambda r: httpx.Response(200, json={"access_token": "new"}), refresh_token="old")
    c.authenticate("new-user", "password")
    assert c.token == "new" and c.refresh_token is None


@pytest.mark.parametrize(
    "entry",
    [{"access_token": 1}, {"refresh_token": []}, {"expires_at": "tomorrow"}, {"expires_at": True}],
)
def test_invalid_saved_token_fields(entry, tmp_path):
    file = tmp_path / "tokens.json"
    file.write_text(json.dumps({"servers": {"https://devmid.regolo.ai": entry}}))
    with pytest.raises(ValueError):
        ModelManagementClient(config_file=file)


def test_redirect_response_is_not_a_success():
    with pytest.raises(RegoloAPIError) as e:
        client(
            lambda r: httpx.Response(302, headers={"location": "https://other.test"})
        ).get_models()
    assert e.value.response.status_code == 302


def test_owned_management_transport_closes_and_missing_load_id_fails(monkeypatch):
    http = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"scan_status": "safe"}))
    )
    with monkeypatch.context() as patch:
        patch.setattr("regolo.management.httpx.Client", lambda: http)
        with ModelManagementClient(token="key") as c:
            c.get_models()
    assert http.is_closed
    with pytest.raises(RegoloProtocolError, match="session_id"):
        client(lambda r: httpx.Response(200, json={"scan_status": "safe"})).load_and_wait(
            "m", "gpu"
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"provider": "other", "url": "org/model"},
        {"provider": "huggingface", "url": None},
        {"name": "x" * 21, "url": "org/model"},
    ],
)
def test_invalid_registration_fails_locally(kwargs):
    kwargs.setdefault("name", "m")
    with pytest.raises(ValueError):
        client(lambda r: pytest.fail("no HTTP")).register_model(**kwargs)


def test_atomic_save_failure_preserves_old_file_and_removes_temporary_secret(tmp_path, monkeypatch):
    file = tmp_path / "tokens.json"
    file.write_text('{"servers":{"https://devmid.test":{"access_token":"old"}}}')
    original = file.read_text()
    c = client(lambda r: httpx.Response(200, json={"access_token": "new"}), config_file=file)

    def fail(*_args):
        raise OSError("replace failed")

    monkeypatch.setattr("regolo.management.os.replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        c.authenticate("u", "p")
    assert file.read_text() == original
    assert list(tmp_path.glob(".regolo-*")) == []


def test_legacy_boolean_refresh_succeeds():
    c = client(lambda r: httpx.Response(200, json={"access_token": "new"}), refresh_token="r")
    assert c._refresh_token() is True and c.token == "new"
