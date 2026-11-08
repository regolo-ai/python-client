import json

import httpx
import pytest
from click.testing import CliRunner

import regolo.cli as commands
from regolo import ModelManagementClient


@pytest.fixture
def cli(monkeypatch, tmp_path):
    calls = []

    def handler(r):
        calls.append(r)
        path = r.url.path
        if path.startswith("/auth/"):
            return httpx.Response(
                200, json={"access_token": "new", "refresh_token": "refresh", "expires_in": 300}
            )
        if path == "/inference/sessions":
            return httpx.Response(
                200, json={"sessions": [], "count": 0, "total": 0, "next_cursor": None}
            )
        if path == "/inference/gpus":
            return httpx.Response(
                200,
                json={
                    "gpus": [
                        {"InstanceType": "unavailable", "Available": False},
                        {"InstanceType": "available", "Available": True},
                    ],
                    "total": 2,
                },
            )
        if path == "/inference/loaded-models":
            return httpx.Response(
                200,
                json={
                    "loaded_models": [
                        {
                            "session_id": 42,
                            "model_name": "m",
                            "gpu_id": "gpu",
                            "status": "running",
                            "cost": 1,
                        }
                    ],
                    "total": 1,
                    "user_id": 1,
                },
            )
        if path in {"/runtime/status", "/runtime/billing"}:
            return httpx.Response(200, json={"status": "running", "cost_euro": 1})
        if path == "/models/m" or path.endswith("/scan/retry"):
            return httpx.Response(
                200,
                json={
                    "name": "m",
                    "provider": "huggingface",
                    "email": "u@example.test",
                    "created_at": "now",
                    "scan_status": "safe",
                },
            )
        return httpx.Response(
            200,
            json={
                "success": True,
                "session_id": 42,
                "models": [],
                "ssh_keys": [],
                "inferences": [],
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    c = ModelManagementClient(
        "https://devmid.test",
        token="access",
        refresh_token="r",
        config_file=tmp_path / "tokens.json",
        http_client=http,
    )
    monkeypatch.setattr(commands, "model_client", c)
    monkeypatch.setattr(commands, "CONFIG_FILE", str(tmp_path / "tokens.json"))
    return CliRunner(), calls, c


@pytest.mark.parametrize(
    "args,method,path",
    [
        (["auth", "login", "--username", "u", "--password", "p"], "POST", "/auth/login"),
        (["auth", "refresh"], "POST", "/auth/refresh"),
        (["models", "register", "--name", "m", "--url", "org/model"], "POST", "/models/load"),
        (["models", "list", "--format", "json"], "GET", "/models"),
        (["models", "details", "m", "--format", "json"], "GET", "/models/m"),
        (["models", "delete", "m", "--confirm"], "DELETE", "/models/m"),
        (["models", "retry-scan", "m"], "POST", "/models/m/scan/retry"),
        (["models", "wait", "m"], "GET", "/models/m"),
        (["ssh", "add", "--title", "laptop", "--key", "ssh-ed25519 abc"], "POST", "/ssh-keys"),
        (["ssh", "list", "--format", "json"], "GET", "/ssh-keys"),
        (["ssh", "delete", "42", "--confirm"], "DELETE", "/ssh-keys/42"),
        (["inference", "gpus", "--format", "json"], "GET", "/inference/gpus"),
        (["inference", "loaded", "--format", "json"], "GET", "/inference/loaded-models"),
        (["inference", "status", "--format", "json"], "GET", "/inference/loaded-models"),
        (["inference", "unload", "42"], "POST", "/inference/unload"),
        (["inference", "unload", "--session-id", "42"], "POST", "/inference/unload"),
        (["inference", "unload", "--model-name", "m"], "POST", "/inference/unload"),
        (
            ["inference", "user-status", "--month", "102026", "--format", "json"],
            "GET",
            "/inference/user-status",
        ),
        (
            ["inference", "sessions", "--include-status", "--include-error"],
            "GET",
            "/inference/sessions",
        ),
        (["inference", "sessions", "--all", "--include-billing"], "GET", "/inference/sessions"),
        (
            ["inference", "usage", "--month", "102026", "--format", "json"],
            "GET",
            "/inference/sessions",
        ),
        (["inference", "session-status", "42"], "GET", "/runtime/status"),
        (["inference", "session-status", "42", "--wait"], "GET", "/runtime/status"),
        (["inference", "billing", "42"], "GET", "/runtime/billing"),
    ],
)
def test_management_cli_routes(cli, args, method, path):
    runner, calls, _ = cli
    result = runner.invoke(commands.cli, args)
    assert result.exit_code == 0, result.output
    assert any(r.method == method and r.url.path == path for r in calls)


def test_cli_load_autoscaling_wait_and_only_available_gpu(cli):
    runner, calls, _ = cli
    result = runner.invoke(
        commands.cli,
        [
            "inference",
            "load",
            "m",
            "--min-replicas",
            "0",
            "--max-replicas",
            "3",
            "--enable-auto-tool-choice",
            "--wait",
        ],
        input="0\n",
    )
    assert result.exit_code == 0, result.output
    body = json.loads(next(r.content for r in calls if r.url.path == "/inference/load"))
    assert body["gpu"] == "available"
    assert body["gpu_plan_config"] == {"min_replicas": 0, "max_replicas": 3}
    assert body["vllm_config"] == {"enable_auto_tool_choice": True}
    assert any(r.url.path == "/runtime/status" for r in calls)


@pytest.mark.parametrize(
    "option,value",
    [
        ("--gpu-memory-utilization", ".8"),
        ("--tensor-parallel-size", "2"),
        ("--disable-log-requests", None),
    ],
)
def test_unsupported_cli_options_fail_before_loading(cli, option, value):
    runner, calls, _ = cli
    args = ["inference", "load", "m", "--gpu", "gpu", option] + ([] if value is None else [value])
    result = runner.invoke(commands.cli, args)
    assert result.exit_code != 0
    assert "unsupported or server-controlled" in result.output
    assert not calls


def test_default_cli_load_does_not_send_false_flags(cli):
    runner, calls, _ = cli
    result = runner.invoke(commands.cli, ["inference", "load", "m", "--gpu", "gpu"])
    assert result.exit_code == 0, result.output
    body = json.loads(calls[-1].content)
    assert "vllm_config" not in body


def test_cli_config_file_preserves_false_and_rejects_mixed_options(cli, tmp_path):
    runner, calls, _ = cli
    config = tmp_path / "vllm.json"
    config.write_text('{"enable_prefix_caching": false, "max_model_len": 4096}')
    args = ["inference", "load", "m", "--gpu", "gpu", "--vllm-config-file", str(config)]
    assert runner.invoke(commands.cli, args).exit_code == 0
    assert json.loads(calls[-1].content)["vllm_config"]["enable_prefix_caching"] is False
    calls.clear()
    result = runner.invoke(commands.cli, args + ["--max-model-len", "10"])
    assert result.exit_code != 0 and not calls


def test_cli_bad_configs_and_windows(cli, tmp_path):
    runner, calls, _ = cli
    for text in ("invalid", "[]"):
        path = tmp_path / "bad.json"
        path.write_text(text)
        result = runner.invoke(
            commands.cli,
            ["inference", "load", "m", "--gpu", "gpu", "--vllm-config-file", str(path)],
        )
        assert result.exit_code != 0
    result = runner.invoke(commands.cli, ["inference", "user-status", "--month", "2026-10"])
    assert result.exit_code != 0
    result = runner.invoke(
        commands.cli,
        [
            "inference",
            "sessions",
            "--month",
            "102026",
            "--time-range-start",
            "2026-10-01T00:00:00Z",
        ],
    )
    assert result.exit_code != 0
    assert not calls


def test_cli_empty_json_outputs_and_usage_table(cli):
    runner, _, _ = cli
    assert (
        json.loads(runner.invoke(commands.cli, ["models", "list", "--format", "json"]).output)[
            "models"
        ]
        == []
    )
    assert (
        json.loads(runner.invoke(commands.cli, ["ssh", "list", "--format", "json"]).output)[
            "ssh_keys"
        ]
        == []
    )
    result = runner.invoke(commands.cli, ["inference", "usage"])
    assert result.exit_code == 0 and "Cost: €0.00" in result.output


def test_cli_csv_export(cli, tmp_path):
    runner, calls, _ = cli
    path = tmp_path / "history.csv"
    result = runner.invoke(commands.cli, ["inference", "sessions", "--csv", str(path)])
    assert result.exit_code == 0, result.output
    assert path.read_text().startswith("session_id,")
    assert calls[-1].url.params["include_billing"] == "true"


def test_logout_clears_memory_and_saved_tokens(cli):
    runner, calls, c = cli
    c._save_config()
    assert runner.invoke(commands.cli, ["auth", "logout"]).exit_code == 0
    assert c.token is None and c.refresh_token is None
    assert not calls


def test_deletion_cancelled_without_request(cli):
    runner, calls, _ = cli
    for args in (["models", "delete", "m"], ["ssh", "delete", "42"]):
        result = runner.invoke(commands.cli, args, input="n\n")
        assert result.exit_code == 0 and "cancelled" in result.output.lower()
    assert not calls


def test_unload_ambiguous_options_are_rejected(cli):
    runner, calls, _ = cli
    for args in (["42", "--session-id", "42"], ["--session-id", "42", "--model-name", "m"]):
        assert runner.invoke(commands.cli, ["inference", "unload", *args]).exit_code != 0
    assert not calls


def test_workflow_register_scan_and_load(cli):
    runner, calls, _ = cli
    result = runner.invoke(
        commands.cli, ["workflow", "workflow", "m", "--url", "org/model", "--auto-load"]
    )
    assert result.exit_code == 0, result.output
    assert [r.url.path for r in calls] == [
        "/models/load",
        "/models/m",
        "/inference/gpus",
        "/inference/load",
    ]


def test_rerank_documents_file_alone_is_accepted(monkeypatch, tmp_path):
    file = tmp_path / "documents.json"
    file.write_text('["doc one", "doc two"]')

    class FakeClient:
        def __init__(self, **kw):
            pass

        def close(self):
            pass

        @staticmethod
        def rerank(**kw):
            assert kw["documents"] == ["doc one", "doc two"]
            return [{"index": 0, "relevance_score": 0.8, "document": "doc one"}]

    monkeypatch.setattr(commands, "RegoloClient", FakeClient)
    result = CliRunner().invoke(
        commands.cli,
        [
            "rerank",
            "--api-key",
            "test",
            "--model",
            "rank",
            "--query",
            "q",
            "--documents-file",
            str(file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "80.00%" in result.output


def test_cli_server_and_config_options_do_not_reuse_other_tokens(cli):
    runner, _, c = cli
    c._save_config()
    result = runner.invoke(
        commands.cli,
        [
            "--devmid-url",
            "https://other.test",
            "--config-file",
            str(c.config_file),
            "models",
            "list",
        ],
    )
    assert result.exit_code != 0
    assert "Not authenticated" in result.output


def test_management_cli_failure_has_nonzero_exit(cli, monkeypatch):
    runner, _, c = cli

    def fail(*_args, **_kwargs):
        raise RuntimeError("meaningful API error")

    for method, args in [
        ("get_inference_sessions", ["inference", "sessions"]),
        ("get_usage_summary", ["inference", "usage"]),
        ("get_runtime_status", ["inference", "session-status", "42"]),
        ("get_runtime_billing", ["inference", "billing", "42"]),
        ("retry_model_scan", ["models", "retry-scan", "m"]),
        ("wait_for_scan", ["models", "wait", "m"]),
    ]:
        monkeypatch.setattr(c, method, fail)
        result = runner.invoke(commands.cli, args)
        assert result.exit_code != 0 and "meaningful API error" in result.output


def test_unload_selected_model_without_session_id_has_clear_error(cli, monkeypatch):
    runner, _, c = cli
    monkeypatch.setattr(c, "get_loaded_models", lambda: {"loaded_models": [{"model_name": "m"}]})
    result = runner.invoke(commands.cli, ["inference", "unload", "--model-name", "m"])
    assert result.exit_code != 0
    assert "Selected model has no session ID" in result.output


def test_management_command_without_initialized_client_has_clear_error(monkeypatch):
    monkeypatch.setattr(commands, "model_client", None)
    # Calling a subgroup directly bypasses the root group's initialization.
    result = CliRunner().invoke(commands.models, ["list"])
    assert result.exit_code != 0
    assert "Management client is not initialized" in result.output
