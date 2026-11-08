"""Resolve engine fields and scaling routes from a server's public OpenAPI schema.

Accept a local JSON export or retrieve it from the server's /openapi.json.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator


class PublicContract:
    def __init__(self, source: Mapping[str, Any] | str | Path):
        data = json.loads(Path(source).read_text()) if isinstance(source, (str, Path)) else source
        if not isinstance(data, Mapping) or not isinstance(data.get("paths"), Mapping):
            raise ValueError("Expected a public Devmid OpenAPI object with paths")
        self.data: dict[str, Any] = copy.deepcopy(dict(data))
        pending = [self.data]
        visited: set[int] = set()
        while pending:
            value = pending.pop()
            if not isinstance(value, (dict, list)) or id(value) in visited:
                continue
            visited.add(id(value))
            if isinstance(value, dict):
                ref = value.get("$ref")
                if ref is not None and (not isinstance(ref, str) or not ref.startswith("#/")):
                    raise ValueError(
                        "Public OpenAPI must use local references; remote schema retrieval is disabled"
                    )
                pending.extend(value.values())
            else:
                pending.extend(value)

    def resolve(self, schema: dict) -> dict:
        seen = set()
        while "$ref" in schema:
            ref = schema["$ref"]
            if (
                not isinstance(ref, str)
                or not ref.startswith("#/components/schemas/")
                or ref in seen
            ):
                raise ValueError("Only local, non-cyclic OpenAPI schema references are supported")
            seen.add(ref)
            schema = self.data.get("components", {}).get("schemas", {}).get(ref.rsplit("/", 1)[-1])
            if not isinstance(schema, dict):
                raise ValueError(f"Unresolved public schema reference: {ref}")
        return schema

    def operation(self, path: str, method: str = "get") -> dict:
        # Reading an OpenAPI document never grants access to administrative,
        # scanner, internal or observability endpoints.
        if not path.startswith(("/inference/", "/runtime/", "/models")):
            raise ValueError("Only model/inference/runtime user operations are supported")
        if any(
            segment.lower() in {"admin", "internal", "security", "observability"}
            for segment in path.split("/")
        ):
            raise ValueError("Privileged operation paths cannot be used by this client")
        op = self.data["paths"].get(path, {}).get(method)
        if not isinstance(op, dict) or op.get("deprecated"):
            raise ValueError(f"Public contract does not expose {method.upper()} {path}")
        security = op.get("security", self.data.get("security", []))
        schemes = self.data.get("components", {}).get("securitySchemes", {})
        if any(
            schemes.get(name, {}).get("scheme", "").lower() == "basic"
            for requirement in security
            for name in requirement
        ):
            raise ValueError("Internal Basic-auth operations cannot be used by this client")
        if not security or any(not requirement for requirement in security):
            raise ValueError(
                "User operation must declare authenticated access in the public schema"
            )
        if any(
            schemes.get(name, {}).get("type") not in {"oauth2", "openIdConnect"}
            and not (
                schemes.get(name, {}).get("type") == "http"
                and schemes[name].get("scheme", "").lower() == "bearer"
            )
            for requirement in security
            for name in requirement
        ):
            raise ValueError("User operation must use OAuth or Bearer authentication")
        return op

    def load_schema(self) -> dict:
        op = self.operation("/inference/load", "post")
        try:
            return self.resolve(op["requestBody"]["content"]["application/json"]["schema"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Public contract has no inference load request schema") from exc

    def validate_load(self, body: dict) -> None:
        schema = self.load_schema()
        self._known_fields(body, schema)
        validator = Draft202012Validator({**schema, "components": self.data.get("components", {})})
        error = next(iter(validator.iter_errors(body)), None)
        if error:
            # jsonschema's default message embeds the invalid instance. Never
            # echo configuration/credential values in a validation exception.
            location = ".".join(map(str, error.absolute_path)) or "request"
            raise ValueError(
                f"Invalid public deployment configuration at {location} ({error.validator})"
            )

    def _known_fields(self, body: dict, schema: dict) -> None:
        schema = self.resolve(schema)
        props = schema.get("properties", {})
        unknown = set(body) - set(props)
        if unknown:
            raise ValueError(f"Fields absent from public deployment schema: {sorted(unknown)}")
        for key, value in body.items():
            prop = self.resolve(props[key])
            if prop.get("readOnly"):
                raise ValueError(f"Server-controlled deployment field: {key}")
            # Check explicit config objects, including nullable references.
            for branch in [prop, *prop.get("anyOf", [])]:
                branch = self.resolve(branch)
                if isinstance(value, dict) and "properties" in branch:
                    self._known_fields(value, branch)

    def engine_fields(self, engine: str, config: dict | None) -> dict:
        def normalized(value: str) -> str:
            return "".join(c for c in value.lower() if c.isalnum())

        props = self.load_schema().get("properties", {})
        engine_prop = self.resolve(props.get("engine", {}))
        choices = list(engine_prop.get("enum", []))
        for branch in engine_prop.get("anyOf", []):
            choices.extend(self.resolve(branch).get("enum", []))
        selected = [
            v for v in choices if isinstance(v, str) and normalized(v) == normalized(engine)
        ]
        if len(selected) != 1:
            raise ValueError("Requested engine is not declared by this server's public load schema")
        result: dict[str, Any] = {"engine": selected[0]}
        if config is not None:
            fields = [
                key
                for key in props
                if key.endswith("_config") and normalized(key[:-7]) == normalized(engine)
            ]
            if len(fields) != 1:
                raise ValueError(
                    "Public schema does not identify a unique config field for this engine"
                )
            result[fields[0]] = config
        return result

    def scaling_operation(self) -> tuple[str, dict]:
        candidates = []
        for path, methods in self.data["paths"].items():
            if not path.startswith(("/inference/", "/runtime/")) or "scaling" not in path.lower():
                continue
            if "get" not in methods:
                continue
            try:
                op = self.operation(path)
            except ValueError:
                continue
            parameters = [*methods.get("parameters", []), *op.get("parameters", [])]
            if any(
                p.get("name") in {"session_id", "model_inference_id", "inference_session_id"}
                for p in parameters
            ):
                candidates.append((path, {**op, "parameters": parameters}))
        if len(candidates) != 1:
            raise ValueError(
                "Public schema must declare one owner-scoped session scaling GET route"
            )
        return candidates[0]
