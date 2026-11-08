"""Explicit, bounded paid smoke tests; never invoked by offline pytest.

Uses REGOLO_API_KEY or a hidden terminal prompt. Generated inputs are synthetic.
The report contains model IDs, statuses and token usage, never credentials.
"""

from __future__ import annotations

import argparse
import getpass
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import wave
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict

from regolo import FunctionTool, RegoloClient

PROBES = (
    "model_ids",
    "chat",
    "streaming_multiturn",
    "structured_json",
    "function_tools",
    "embeddings",
    "rerank",
    "images",
    "transcription",
    "ocr",
    "vision",
    "reasoning",
)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


def complete_text(reply: dict) -> str:
    choice = reply["choices"][0]
    if choice.get("finish_reason") in {"length", "content_filter"}:
        raise ValueError("Model did not finish within the explicit output cap")
    text = choice["message"].get("content")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Model returned no final text")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-paid", action="store_true")
    parser.add_argument("--base-url", default="https://api.regolo.ai")
    parser.add_argument("--report", type=Path, default=Path("live-smoke-report.json"))
    parser.add_argument("--max-requests", type=int, default=16)
    parser.add_argument("--only", nargs="+", choices=PROBES, help="Retest selected workflows only")
    args = parser.parse_args()
    if not args.run_paid:
        parser.error("No requests made; pass --run-paid to enable paid inference")
    if not 1 <= args.max_requests <= 16:
        parser.error("max-requests must be between 1 and 16")
    key = os.environ.get("REGOLO_API_KEY") or getpass.getpass("Regolo API key (hidden): ")
    if not key:
        parser.error("API key is required")
    selected = set(args.only or PROBES)
    report: dict[str, Any] = {
        "tests": [],
        "request_cap": args.max_requests,
        "limits": {
            "generated_images": int("images" in selected),
            "image_size": "1024x1024",
            "max_tokens_per_chat_call": 128,
            "tool_rounds": 2,
            "automatic_retries": 0,
        },
        "notes": [
            "No Devmid/custom deployments: those require separate owner credentials.",
            "Video and legacy completions are offline-only without suitable live fixtures.",
            "Prefer non-beta, non-reasoning models for ordinary short-output tests; reasoning has its own capped probe.",
        ],
    }
    budget = {"attempted": 0}

    def count_request(request):
        if budget["attempted"] >= args.max_requests:
            raise RuntimeError("Explicit live request cap reached")
        budget["attempted"] += 1

    def record(name, function, owner=None):
        started = time.monotonic()
        try:
            function()
            report["tests"].append(
                {
                    "name": name,
                    "status": "passed",
                    "model": owner.instance.chat_model if owner else None,
                    "usage": owner.last_usage if owner else None,
                    "seconds": round(time.monotonic() - started, 3),
                }
            )
            print(f"PASS {name}")
            return True
        except Exception as exc:
            report["tests"].append(
                {
                    "name": name,
                    "status": "failed",
                    "model": owner.instance.chat_model if owner else None,
                    "error_type": type(exc).__name__,
                    "error": str(exc).replace(key, "<redacted>")[:700],
                }
            )
            print(f"FAIL {name}: {type(exc).__name__}")
            return False

    try:
        with (
            httpx.Client(
                timeout=httpx.Timeout(25, connect=8),
                follow_redirects=False,
                event_hooks={"request": [count_request]},
            ) as http,
            ExitStack() as stack,
        ):
            root = stack.enter_context(
                RegoloClient(
                    api_key=key,
                    pre_existent_client=http,
                    check_models=False,
                    alternative_url=args.base_url,
                )
            )
            catalog = []

            def discover():
                catalog.extend(root.get_model_catalog(refresh=True))
                if not catalog:
                    raise ValueError("No accessible catalog models")

            if record("catalog", discover):
                if "model_ids" in selected:
                    record(
                        "model_ids",
                        lambda: root.get_available_models(key, args.base_url, client=http),
                    )

                def pick(mode, capability=None, param=None):
                    rows = [
                        r
                        for r in catalog
                        if r.get("mode") == mode
                        and (capability is None or r.get(capability) is True)
                        and (param is None or param in (r.get("supported_openai_params") or []))
                    ]
                    stable = [
                        r
                        for r in rows
                        if not any(
                            label in str(r.get("model_group", "")).lower()
                            for label in ("beta", "preview", "experimental")
                        )
                    ]
                    rows = stable or rows
                    if mode == "chat" and capability != "supports_reasoning":
                        ordinary = [r for r in rows if r.get("supports_reasoning") is not True]
                        rows = ordinary or rows

                    def price(r):
                        value = r.get("output_cost_per_token")
                        return float(value) if isinstance(value, (int, float)) else float("inf")

                    return min(rows, key=price) if rows else None

                def make(row):
                    model = row.get("model_group") or row.get("id")
                    return stack.enter_context(
                        RegoloClient(
                            api_key=key,
                            chat_model=model,
                            embedder_model=model,
                            image_generation_model=model,
                            audio_transcription_model=model,
                            reranker_model=model,
                            check_models=False,
                            pre_existent_client=http,
                            alternative_url=args.base_url,
                        )
                    )

                def options(row):
                    params = row.get("supported_openai_params") or []
                    limit: dict[str, Any] = {
                        "max_completion_tokens"
                        if "max_completion_tokens" in params
                        else "max_tokens": 128
                    }
                    if row.get("supports_reasoning") and "reasoning_effort" in params:
                        limit["reasoning_effort"] = "low"
                    return limit

                def skip(name):
                    report["tests"].append(
                        {
                            "name": name,
                            "status": "skipped",
                            "reason": "No suitable accessible catalog model",
                        }
                    )

                chat = pick("chat")
                if chat:
                    c = make(chat)
                    if "chat" in selected:

                        def plain():
                            text = complete_text(
                                c.run_chat("Reply with OK only.", full_output=True, **options(chat))
                            )
                            if "OK" not in text.upper():
                                raise ValueError("Unexpected demo answer")

                        record("chat", plain, c)
                    if "streaming_multiturn" in selected:

                        def streamed():
                            list(
                                c.run_chat(
                                    "Reply with OK only.",
                                    stream=True,
                                    full_output=True,
                                    stream_options={"include_usage": True},
                                    **options(chat),
                                )
                            )
                            complete_text(c.last_response)

                        record("streaming_multiturn", streamed, c)
                for name in ("structured_json", "function_tools"):
                    if name not in selected:
                        continue
                    row = (
                        pick("chat", param="response_format")
                        if name == "structured_json"
                        else pick("chat", "supports_function_calling")
                    )
                    if not row:
                        skip(name)
                        continue
                    c = make(row)
                    if name == "structured_json":

                        def structured():
                            answer = c.run_structured(
                                "Return JSON with value equal to 7.", Answer, **options(row)
                            )
                            if answer.value != 7:
                                raise ValueError("Structured demo value is not 7")

                        record(name, structured, c)
                    else:
                        invocations = []

                        def add(a: int, b: int) -> int:
                            """Add two integers."""
                            invocations.append((a, b))
                            return a + b

                        def tools():
                            reply = c.run_tools(
                                "Use add with a=2 and b=3, then answer 5.",
                                [FunctionTool(add)],
                                max_rounds=2,
                                tool_choice={"type": "function", "function": {"name": "add"}},
                                **options(row),
                            )
                            if invocations != [(2, 3)] or "5" not in complete_text(reply):
                                raise ValueError(
                                    "Tool demo did not execute the expected call and return 5"
                                )

                        record(name, tools, c)
                for mode, name in (
                    ("embedding", "embeddings"),
                    ("rerank", "rerank"),
                    ("image_generation", "images"),
                    ("audio_transcription", "transcription"),
                    ("ocr", "ocr"),
                    ("chat", "vision"),
                    ("chat", "reasoning"),
                ):
                    if name not in selected:
                        continue
                    capability = {
                        "vision": "supports_vision",
                        "reasoning": "supports_reasoning",
                    }.get(name)
                    row = pick(mode, capability)
                    if not row:
                        skip(name)
                        continue
                    c = make(row)

                    def exercise():
                        if name == "embeddings":
                            data = c.embeddings(["Rome", "Paris"])
                            if len(data) != 2 or any(not item.get("embedding") for item in data):
                                raise ValueError("Invalid embedding batch")
                        elif name == "rerank":
                            data = c.rerank(
                                "Italy", ["Rome is in Italy.", "Paris is in France."], top_n=1
                            )
                            if len(data) != 1 or data[0]["index"] != 0:
                                raise ValueError("Unexpected demo ranking")
                        elif name == "images":
                            images = c.create_image(
                                "A plain blue square on white.",
                                n=1,
                                size="1024x1024",
                                response_format="b64_json",
                            )
                            if len(images) != 1:
                                raise ValueError("Expected one generated image")
                            with Image.open(io.BytesIO(images[0])) as img:
                                img.verify()
                        elif name == "transcription":
                            with tempfile.TemporaryDirectory() as folder:
                                audio = Path(folder) / "sample.wav"
                                if shutil.which("espeak"):
                                    subprocess.run(
                                        ["espeak", "-w", str(audio), "Hello Regolo"],
                                        check=True,
                                        capture_output=True,
                                    )
                                else:
                                    with wave.open(str(audio), "wb") as wav:
                                        wav.setnchannels(1)
                                        wav.setsampwidth(2)
                                        wav.setframerate(16000)
                                        wav.writeframes(b"\0" * 16000)
                                    report["notes"].append(
                                        "Transcription uses synthetic silence: upload/format verified, not word-recognition accuracy."
                                    )
                                if not isinstance(c.audio_transcription(audio), str):
                                    raise ValueError("Invalid transcription type")
                        elif name in {"ocr", "vision"}:
                            buf = io.BytesIO()
                            generated_image = (
                                Image.new("RGB", (256, 96), "white")
                                if name == "ocr"
                                else Image.new("RGB", (64, 64), "red")
                            )
                            if name == "ocr":
                                ImageDraw.Draw(generated_image).text((20, 30), "TEST", fill="black")
                            generated_image.save(buf, format="PNG")
                            reply = (
                                c.ocr(
                                    buf.getvalue(),
                                    mime_type="image/png",
                                    full_output=True,
                                    **options(row),
                                )
                                if name == "ocr"
                                else c.vision(
                                    "Name this color in one word.",
                                    [buf.getvalue()],
                                    mime_type="image/png",
                                    full_output=True,
                                    **options(row),
                                )
                            )
                            text = complete_text(reply).lower()
                            if ("test" if name == "ocr" else "red") not in text:
                                raise ValueError("Unexpected demo visual answer")
                        else:
                            complete_text(
                                c.run_chat(
                                    "What is 2 plus 3? Reply briefly.",
                                    full_output=True,
                                    **options(row),
                                )
                            )

                    record(name, exercise, c)
            else:
                report["notes"].append("Discovery failed; no generation requests were attempted.")
    except Exception as exc:
        report["tests"].append(
            {
                "name": "connection_setup",
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc).replace(key, "<redacted>")[:500],
            }
        )
        print(f"FAIL connection_setup: {type(exc).__name__}")
    report["requests_attempted"] = budget["attempted"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return int(any(test["status"] == "failed" for test in report["tests"]))


if __name__ == "__main__":
    raise SystemExit(main())
