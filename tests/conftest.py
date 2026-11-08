"""The entire SDK suite is offline: accidental real HTTP fails immediately."""

import httpx
import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)

    def blocked(*_args, **_kwargs):
        raise AssertionError("Real HTTP is forbidden in tests; inject httpx.MockTransport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
