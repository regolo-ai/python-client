"""Ownership handling for optional HTTPX transports in static inference calls."""

from functools import wraps
from inspect import signature
from typing import Any, Callable

import httpx

DEFAULT_TIMEOUT = 500


def api_url(base_url: str, path: str) -> str:
    """Join root, /v1 and /custom-model/v1 URLs without duplicate /v1."""
    url = httpx.URL(base_url)
    if (
        url.scheme not in {"http", "https"}
        or not url.host
        or url.userinfo
        or url.query
        or url.fragment
    ):
        raise ValueError("API base URL must be HTTP(S), without credentials, query or fragment")
    if path and (not path.startswith("/") or path.startswith("//") or "?" in path or "#" in path):
        raise ValueError("API endpoint must be an absolute path without query or fragment")
    base = str(url).rstrip("/")
    if base.endswith("/v1") and path.startswith("/v1/"):
        path = path[3:]
    return base + path


def json_object(response: httpx.Response) -> dict[str, Any]:
    """Keep status errors and identify malformed successful inference data."""
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError as exc:
        raise ValueError("Inference API returned non-JSON success data") from exc
    if not isinstance(data, dict):
        raise ValueError("Inference API returned a non-object JSON response")
    if data.get("error"):
        # The raw response remains available on an HTTP error; successful
        # error envelopes are protocol failures, not valid model results.
        raise ValueError("Inference API returned an error envelope with HTTP success")
    return data


def managed_http[**P, R](method: Callable[P, R]) -> Callable[P, R]:
    """Close internally created clients, including generator cancellation/errors."""
    sig = signature(method)
    # inspect reconstructs arguments dynamically. Keep that boundary internal;
    # the decorator's public Callable[P, R] still preserves the original signature.
    invoke: Callable[..., Any] = method

    @wraps(method)
    def call(*args: P.args, **kwargs: P.kwargs) -> Any:
        bound = sig.bind(*args, **kwargs)
        if bound.arguments.get("client") is not None:
            return method(*args, **kwargs)
        if bound.arguments.get("stream", False):
            # Creation happens inside the generator: an unstarted generator
            # owns no resources, and close() releases a partially read stream.
            def stream():
                with httpx.Client(timeout=DEFAULT_TIMEOUT) as stream_client:
                    bound.arguments["client"] = stream_client
                    yield from invoke(*bound.args, **bound.kwargs)

            return stream()
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            bound.arguments["client"] = client
            return invoke(*bound.args, **bound.kwargs)

    return call
