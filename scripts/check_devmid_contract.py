"""Compare a public Devmid OpenAPI export with the bundled 3.9 contract fixture."""

import argparse
import json
from pathlib import Path
from typing import Any


def structural(value: Any, preserve_keys: bool = False) -> Any:
    """Ignore prose while preserving types, fields, constraints and API behavior."""
    ignored = {
        "description",
        "summary",
        "title",
        "example",
        "examples",
        "operationId",
        "tags",
        "externalDocs",
    }
    if isinstance(value, dict):
        return {
            k: structural(
                v,
                preserve_keys=k
                in {"properties", "schemas", "paths", "responses", "content", "securitySchemes"},
            )
            for k, v in value.items()
            if preserve_keys or k not in ignored
        }
    if isinstance(value, list):
        return [structural(v) for v in value]
    return value


def compare(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    errors = []
    for section in ("paths", "components"):
        before = structural(expected.get(section, {}))
        after = structural(actual.get(section, {}))
        if not isinstance(before, dict) or not isinstance(after, dict):
            errors.append(f"Invalid {section}: expected an object")
            continue
        for key in sorted(set(before) | set(after)):
            if before.get(key) != after.get(key):
                errors.append(f"Changed {section}: {key}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("openapi", type=Path, help="Exported public Devmid openapi.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / "tests/fixtures/devmid-3.9-public-openapi.json").read_text())
    actual = json.loads(args.openapi.read_text())
    errors = compare(expected, actual)
    if errors:
        print("\n".join(errors))
        return 1
    print("Public Devmid contract matches the bundled 3.9 contract fixture (18 operations).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
