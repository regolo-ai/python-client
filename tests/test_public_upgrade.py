"""Schema discovery tests using simulated public API extensions."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from regolo import GPUPlanConfig, ModelManagementClient, RegoloAPIError, RegoloProtocolError

BASE = json.loads((Path(__file__).parent / "fixtures/devmid-3.9-public-openapi.json").read_text())


def newer_schema():
    schema = copy.deepcopy(BASE)
    schema["info"]["description"] = "SYNTHETIC SDK DISCOVERY TEST; not a deployed schema"
    schemas = schema["components"]["schemas"]
    props = schemas["LoadInferenceRequest"]["properties"]
    props.update(
        {
            "engine": {"type": "string", "enum": ["VLLM", "SGLang", "LlamaCpp"]},
            "sglang_config": {
                "anyOf": [{"$ref": "#/components/schemas/TestSGLangConfig"}, {"type": "null"}]
            },
            "llama_cpp_config": {"$ref": "#/components/schemas/TestLlamaConfig"},
            "gguf_file": {"type": "string"},
            "revision": {"type": "string", "readOnly": True},
        }
    )
    schemas["TestSGLangConfig"] = {
        "type": "object",
        "properties": {"context_length": {"type": "integer", "minimum": 1}},
        "additionalProperties": False,
    }
    schemas["TestLlamaConfig"] = {
        "type": "object",
        "properties": {"n_ctx": {"type": "integer", "minimum": 1}},
        "additionalProperties": False,
    }
    schemas["GPU_Plan_Configuration"]["properties"].update(
        {
            "target_requests": {"type": "integer", "minimum": 1},
            "scale_down_delay_seconds": {"type": "integer", "minimum": 0},
        }
    )
    schema["paths"]["/runtime/sessions/{session_id}/scaling"] = {
        "get": {
            "parameters": [
                {
                    "name": "session_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "integer", "minimum": 1},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "minimum": 1, "maximum": 150},
                },
                {
                    "name": "before_observation_id",
                    "in": "query",
                    "schema": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]},
                },
            ],
            "security": [{"OAuth2PasswordBearer": ["model:read"]}],
            "responses": {"200": {"description": "test"}},
        }
    }
    return schema


def client(handler, **kwargs):
    return ModelManagementClient(
        "https://devmid.test",
        token="access",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def test_autoscaling_target_and_zero_delay_are_transmitted_without_yaml_name_conversion():
    seen = []

    def handler(r):
        seen.append(r)
        return httpx.Response(202, json={"session_id": 1})

    c = client(handler)
    c.load_model_for_inference(
        "custom",
        "gpu",
        gpu_plan_config=GPUPlanConfig(
            min_replicas=0, max_replicas=3, target_requests=250, scale_down_delay_seconds=0
        ),
    )
    assert json.loads(seen[0].content)["gpu_plan_config"] == {
        "min_replicas": 0,
        "max_replicas": 3,
        "target_requests": 250,
        "scale_down_delay_seconds": 0,
    }
    c.load_model_for_inference(
        "custom", "gpu", gpu_plan_config={"target_requests": None, "scale_down_delay_seconds": None}
    )
    assert json.loads(seen[1].content)["gpu_plan_config"] == {"min_replicas": 1, "max_replicas": 1}


@pytest.mark.parametrize(
    "config",
    [
        {"target_requests": 0},
        {"target_requests": True},
        {"target_requests": "250"},
        {"scale_down_delay_seconds": -1},
        {"scale_down_delay_seconds": True},
    ],
)
def test_invalid_new_autoscaling_values_fail_locally(config):
    c = client(lambda r: pytest.fail("No invalid request should reach Devmid"))
    with pytest.raises(ValidationError):
        c.load_model_for_inference("m", "gpu", gpu_plan_config=config)


@pytest.mark.parametrize(
    "method,args,path",
    [
        ("get_loaded_models", (), "/inference/loaded-models"),
        ("get_inference_sessions", (), "/inference/sessions"),
    ],
)
def test_inline_scaling_is_opt_in_and_preserves_frozen_and_assumed_metadata(method, args, path):
    seen = []
    raw = {
        "sessions": [],
        "scaling": {
            "finalized": True,
            "assumed": False,
            "intervals": [{"replicas": 0, "cost_euro": 0}],
            "future_metric": None,
        },
    }

    def handler(r):
        seen.append(r)
        assert r.url.path == path
        return httpx.Response(200, json=raw)

    c = client(handler)
    assert getattr(c, method)(*args) == raw
    assert "include_scaling" not in seen[-1].url.params
    assert getattr(c, method)(*args, include_scaling=True) == raw
    assert seen[-1].url.params["include_scaling"] == "true"
    assert all(r.headers["authorization"] == "Bearer access" for r in seen)


@pytest.mark.parametrize(
    "engine,config,key,value",
    [
        ("sglang", {"context_length": 4096}, "sglang_config", "SGLang"),
        ("llamacpp", {"n_ctx": 4096}, "llama_cpp_config", "LlamaCpp"),
    ],
)
def test_engines_use_exact_public_schema_fields(engine, config, key, value):
    seen = []

    def handler(r):
        seen.append(r)
        return httpx.Response(202, json={"session_id": 12})

    c = client(handler, public_openapi=newer_schema())
    c.load_model_for_inference("custom", "gpu", engine=engine, engine_config=config)
    body = json.loads(seen[0].content)
    assert body["engine"] == value and body[key] == config
    assert len(seen) == 1


def test_schema_is_fetched_once_and_only_when_a_new_contract_feature_is_used():
    seen = []

    def handler(r):
        seen.append(r)
        if r.url.path == "/openapi.json":
            return httpx.Response(200, json=newer_schema())
        return httpx.Response(202, json={"session_id": 1})

    c = client(handler)
    c.load_model_for_inference("m", "gpu")
    assert [r.url.path for r in seen] == ["/inference/load"]
    for _ in range(2):
        c.load_model_for_inference("m", "gpu", engine="sglang", engine_config={"context_length": 5})
    assert [r.url.path for r in seen].count("/openapi.json") == 1
    assert [r.url.path for r in seen].count("/inference/load") == 3


@pytest.mark.parametrize(
    "options",
    [
        {"engine": "unknown"},
        {"engine_config": {"n_ctx": 5}},
        {"engine": "sglang", "engine_config": {"trust_remote_code": True}},
        {"engine": "sglang", "engine_config": {"context_length": 0}},
        {"engine": "sglang", "vllm_config": {}},
        {"deployment_options": {"model_name": "override"}},
        {"deployment_options": {"revision": "secret-revision-value"}},
        {"deployment_options": {"env": {"TOKEN": "secret-value"}}},
    ],
)
def test_unknown_server_controlled_and_unsafe_fields_cannot_be_silently_sent(options):
    c = client(
        lambda r: pytest.fail("Invalid config must not reach the server"),
        public_openapi=newer_schema(),
    )
    with pytest.raises(ValueError) as error:
        c.load_model_for_inference("m", "gpu", **options)
    assert "secret-revision-value" not in str(error.value)
    assert "secret-value" not in str(error.value)


def test_old_public_schema_fails_explicitly_instead_of_guessing_engine_contract():
    c = client(
        lambda r: pytest.fail("Unsupported engine must not reach the server"), public_openapi=BASE
    )
    with pytest.raises(ValueError, match="not declared"):
        c.load_model_for_inference("m", "gpu", engine="sglang", engine_config={"context_length": 5})


def test_public_schema_files_are_supported_and_do_not_leak_references(tmp_path):
    path = tmp_path / "openapi.json"
    path.write_text(json.dumps(newer_schema()))
    c = client(lambda r: httpx.Response(200, json={}), public_openapi=path)
    schema = c.get_public_contract()
    assert schema.load_schema()["properties"]["engine"]["enum"] == ["VLLM", "SGLang", "LlamaCpp"]
    with pytest.raises(ValueError):
        schema.resolve({"$ref": "https://evil.test/schema"})


def test_scaling_route_and_cursor_are_discovered_not_hardcoded():
    seen = []
    page = {
        "session_id": 7,
        "intervals": [{"replicas": 2, "frozen": True}],
        "next_cursor": 10,
        "assumed": False,
    }

    def handler(r):
        seen.append(r)
        assert r.url.path == "/runtime/sessions/7/scaling"
        assert dict(r.url.params) == {"limit": "50", "before_observation_id": "20"}
        return httpx.Response(200, json=page)

    c = client(handler, public_openapi=newer_schema())
    assert c.get_scaling_history(7, limit=50, cursor="20") == page
    assert len(seen) == 1


@pytest.mark.parametrize("changes", ["missing", "admin_only", "basic_only", "ambiguous"])
def test_missing_or_nonuser_scaling_routes_never_fall_back_to_privileged_apis(changes):
    schema = newer_schema()
    path = "/runtime/sessions/{session_id}/scaling"
    op = schema["paths"].pop(path)
    if changes == "admin_only":
        schema["paths"]["/admin/sessions/{session_id}/scaling"] = op
    if changes == "basic_only":
        schema["components"]["securitySchemes"]["internal"] = {"type": "http", "scheme": "basic"}
        op["get"]["security"] = [{"internal": []}]
        schema["paths"][path] = op
    if changes == "ambiguous":
        schema["paths"][path] = op
        schema["paths"]["/inference/sessions/{session_id}/scaling"] = op
    c = client(
        lambda r: pytest.fail("No discovered user route; no HTTP is allowed"), public_openapi=schema
    )
    with pytest.raises(ValueError, match="one owner-scoped"):
        c.get_scaling_history(7)


@pytest.mark.parametrize(
    "limit,cursor", [(151, None), (0, None), (True, None), (5, False), (5, "../admin")]
)
def test_scaling_params_obey_the_server_schema(limit, cursor):
    c = client(
        lambda r: pytest.fail("Invalid params must be local errors"), public_openapi=newer_schema()
    )
    with pytest.raises(ValueError):
        c.get_scaling_history(7, limit=limit, cursor=cursor)


def test_scaling_iterator_follows_pages_and_stops_repeated_cursors():
    seen = []

    def handler(r):
        seen.append(r)
        return httpx.Response(
            200,
            json={
                "intervals": [{"replicas": len(seen)}],
                "next_cursor": 10 if len(seen) == 1 else None,
            },
        )

    c = client(handler, public_openapi=newer_schema())
    assert list(c.iter_scaling_history(7)) == [{"replicas": 1}, {"replicas": 2}]
    assert seen[-1].url.params["before_observation_id"] == "10"
    c = client(
        lambda r: httpx.Response(200, json={"intervals": [{}], "next_cursor": 10}),
        public_openapi=newer_schema(),
    )
    with pytest.raises(RegoloProtocolError, match="repeated"):
        list(c.iter_scaling_history(7))


@pytest.mark.parametrize("rows", [[], [{"session_id": 6}], [{"session_id": True}], None])
def test_exact_session_lookup_never_returns_a_neighbors_history(rows):
    c = client(lambda r: httpx.Response(200, json={"sessions": rows}))
    with pytest.raises((LookupError, RegoloProtocolError)):
        c.get_inference_session(7)


def test_full_session_details_preserve_every_runtime_billing_and_scaling_field():
    def handler(r):
        assert r.headers["authorization"] == "Bearer access"
        if r.url.path == "/inference/sessions":
            assert (
                r.url.params["before_session_id"] == "8"
                and r.url.params["include_scaling"] == "true"
            )
            return httpx.Response(
                200,
                json={
                    "sessions": [{"session_id": 7, "scaling": {"frozen": True}, "retired": True}]
                },
            )
        return httpx.Response(200, json={"future_stat": 17, "nullable_stat": None})

    details = client(handler).get_session_details(7, include_scaling=True)
    assert details["session"]["retired"] is True
    assert details["runtime"]["future_stat"] == 17
    assert details["billing"]["nullable_stat"] is None


def test_dashboard_overview_uses_only_public_owner_token_routes():
    paths = []

    def handler(r):
        paths.append(r.url.path)
        assert r.headers["authorization"] == "Bearer access"
        return httpx.Response(200, json={"new_stat": 1})

    c = client(handler)
    overview = c.get_dashboard_snapshot(month="112026", include_scaling=True)
    assert set(paths) == {
        "/models",
        "/inference/gpus",
        "/inference/loaded-models",
        "/inference/user-status",
    }
    assert overview["usage"]["new_stat"] == 1


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500])
def test_scaling_failures_keep_status_and_never_try_admin_fallback(status):
    calls = []

    def handler(r):
        calls.append(r)
        return httpx.Response(status, json={"detail": "refused"})

    c = client(handler, public_openapi=newer_schema())
    with pytest.raises(RegoloAPIError) as error:
        c.get_scaling_history(7)
    assert error.value.response.status_code == status and len(calls) == 1


def test_error_display_redacts_validation_input_and_bearer_but_preserves_raw_detail():
    detail = [{"loc": ["body", "api_key"], "msg": "invalid", "input": "hf-private-token"}]
    c = client(lambda r: httpx.Response(422, json={"detail": detail}))
    with pytest.raises(RegoloAPIError) as error:
        c.register_model("m", url="org/model", api_key="hf-private-token")
    assert "hf-private-token" not in str(error.value)
    assert error.value.detail == detail
    c = client(lambda r: httpx.Response(403, json={"detail": "invalid Bearer access"}))
    with pytest.raises(RegoloAPIError) as error:
        c.get_models()
    assert "Bearer access" not in str(error.value)


@pytest.mark.parametrize(
    "timeout,interval", [(float("nan"), 1), (1, float("nan")), (float("inf"), 1)]
)
def test_wait_rejects_nonfinite_deadlines(timeout, interval):
    c = client(lambda r: pytest.fail("Invalid polling options must fail before requests"))
    with pytest.raises(ValueError):
        c.wait_for_scan("m", timeout=timeout, poll_interval=interval)


def test_history_json_export_is_complete_atomic_and_private(tmp_path):
    seen = []

    def handler(r):
        seen.append(r)
        sid = 2 if len(seen) == 1 else 1
        return httpx.Response(
            200,
            json={
                "sessions": [{"session_id": sid, "scaling": {"future_stat": None}}],
                "next_cursor": 1 + 1 if sid == 2 else None,
            },
        )

    c = client(handler)
    output = tmp_path / "history.json"
    assert c.export_sessions_json(output, include_scaling=True) == 2
    assert json.loads(output.read_text()) == [
        {"session_id": 2, "scaling": {"future_stat": None}},
        {"session_id": 1, "scaling": {"future_stat": None}},
    ]
    assert output.stat().st_mode & 0o777 == 0o600
    assert all(r.url.params["include_scaling"] == "true" for r in seen)
    c = client(lambda r: httpx.Response(500, json={"detail": "error"}))
    before = output.read_bytes()
    with pytest.raises(RegoloAPIError):
        c.export_sessions_json(output)
    assert output.read_bytes() == before
    assert not list(tmp_path.glob(".regolo-history-*"))


def test_csv_export_escapes_spreadsheet_formulas_with_explicit_raw_optout():
    import io

    row = {"session_id": 1, "model_name": "=DANGEROUS()", "error": "+formula"}
    c = client(lambda r: httpx.Response(200, json={"sessions": [row], "next_cursor": None}))
    safe = io.StringIO()
    c.export_sessions_csv(safe, include_error=True)
    assert "'=DANGEROUS()" in safe.getvalue() and "'+formula" in safe.getvalue()
    raw = io.StringIO()
    c.export_sessions_csv(raw, spreadsheet_safe=False)
    assert "'=DANGEROUS()" not in raw.getvalue()


def test_auth_error_does_not_echo_submitted_password():
    c = client(lambda r: httpx.Response(403, json={"detail": "failed for private-password"}))
    with pytest.raises(RegoloAPIError) as error:
        c.authenticate("user", "private-password")
    assert "private-password" not in str(error.value)


def test_discovered_scaling_rejects_unauthenticated_and_nested_admin_paths():
    for mode in ("anonymous", "nested_admin"):
        schema = newer_schema()
        route = schema["paths"].pop("/runtime/sessions/{session_id}/scaling")
        if mode == "anonymous":
            route["get"]["security"] = []
        path = (
            "/runtime/admin/{session_id}/scaling"
            if mode == "nested_admin"
            else "/runtime/sessions/{session_id}/scaling"
        )
        schema["paths"][path] = route
        c = client(
            lambda r: pytest.fail("No privileged or anonymous fallback"), public_openapi=schema
        )
        with pytest.raises(ValueError):
            c.get_scaling_history(1)


def test_public_contract_never_retrieves_external_schema_references():
    schema = newer_schema()
    schema["components"]["schemas"]["TestSGLangConfig"]["properties"]["context_length"] = {
        "$ref": "https://private.test/schema"
    }
    with pytest.raises(ValueError, match="remote schema retrieval is disabled"):
        client(lambda r: pytest.fail("No schema HTTP retrieval allowed"), public_openapi=schema)
