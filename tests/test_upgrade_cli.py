import json

import httpx
import pytest
from click.testing import CliRunner
from test_public_upgrade import newer_schema

from regolo import ModelManagementClient, RegoloClient
from regolo import cli as commands


@pytest.fixture
def gateway(monkeypatch):
    seen = []

    def handler(r):
        seen.append(r)
        if r.url.path == "/inference/load":
            return httpx.Response(202, json={"session_id": 7})
        if r.url.path == "/inference/sessions":
            return httpx.Response(
                200,
                json={
                    "sessions": [{"session_id": 7, "scaling": {"frozen": True}}],
                    "next_cursor": None,
                },
            )
        if r.url.path.endswith("/scaling"):
            return httpx.Response(200, json={"session_id": 7, "intervals": [], "next_cursor": None})
        return httpx.Response(200, json={"future_stat": True})

    client = ModelManagementClient(
        "https://devmid.test",
        token="access",
        public_openapi=newer_schema(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(commands, "model_client", client)
    return CliRunner(), seen


@pytest.mark.parametrize(
    "arguments",
    [
        ["inference", "loaded", "--include-scaling", "--format", "json"],
        ["inference", "sessions", "--include-scaling", "--include-billing"],
        ["inference", "details", "7", "--include-scaling"],
        ["inference", "scaling", "7", "--limit", "50", "--cursor", "20"],
        ["inference", "overview", "--month", "112026", "--include-scaling"],
    ],
)
def test_new_user_commands_forward_only_public_requests(gateway, arguments):
    runner, seen = gateway
    result = runner.invoke(commands.cli, arguments)
    assert result.exit_code == 0, result.output
    assert seen and all(
        r.url.path.startswith(("/inference/", "/models", "/runtime/")) for r in seen
    )
    assert all(r.headers["authorization"] == "Bearer access" for r in seen)


def test_new_engine_and_autoscaling_cli_options(gateway, tmp_path):
    runner, seen = gateway
    config = tmp_path / "engine.json"
    config.write_text(json.dumps({"context_length": 4096}))
    result = runner.invoke(
        commands.cli,
        [
            "inference",
            "load",
            "custom",
            "--gpu",
            "gpu",
            "--engine",
            "sglang",
            "--engine-config",
            str(config),
            "--target-requests",
            "250",
            "--scale-down-delay-seconds",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(seen[-1].content)
    assert body["engine"] == "SGLang"
    assert body["sglang_config"] == {"context_length": 4096}
    assert body["gpu_plan_config"]["scale_down_delay_seconds"] == 0


def test_removed_revision_command_cannot_call_hidden_route(gateway):
    runner, seen = gateway
    result = runner.invoke(commands.cli, ["models", "update-revision", "7"])
    assert result.exit_code == 2 and "No such command" in result.output
    assert not seen


def test_json_history_cli_export_retains_nested_scaling(gateway, tmp_path):
    runner, seen = gateway
    output = tmp_path / "history.json"
    result = runner.invoke(
        commands.cli, ["inference", "sessions", "--include-scaling", "--json-file", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())[0]["scaling"]["frozen"] is True


def test_catalog_cli_exposes_raw_metadata_without_printing_key(monkeypatch):
    def factory(api_key, alternative_url):
        return RegoloClient(
            api_key=api_key,
            alternative_url=alternative_url,
            pre_existent_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(
                        200,
                        json={
                            "data": [
                                {
                                    "model_group": "chat",
                                    "mode": "chat",
                                    "supports_vision": True,
                                    "input_cost_per_token": 0.001,
                                }
                            ]
                        },
                    )
                )
            ),
        )

    monkeypatch.setattr(commands, "RegoloClient", factory)
    result = CliRunner().invoke(
        commands.cli,
        ["catalog", "--api-key", "private-key", "--mode", "chat", "--supports", "vision"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["input_cost_per_token"] == 0.001
    assert "private-key" not in result.output
