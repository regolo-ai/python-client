import ast
import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import regolo

ROOT = Path(__file__).resolve().parents[1]


def test_pytest_collects_checkout_even_when_an_old_package_is_on_pythonpath(tmp_path):
    old_package = tmp_path / "regolo"
    old_package.mkdir()
    (old_package / "__init__.py").write_text(
        'raise ImportError("STALE_INSTALLED_CLIENT_IMPORTED")\n'
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "tests/test_cli.py",
            "tests/test_management.py",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STALE_INSTALLED_CLIENT_IMPORTED" not in result.stdout + result.stderr


def test_versions_and_every_public_operation_are_documented():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert (
        metadata["project"]["version"]
        == metadata["tool"]["bumpver"]["current_version"]
        == regolo.__version__
    )
    schema = json.loads((ROOT / "tests/fixtures/devmid-3.9-public-openapi.json").read_text())
    docs = (ROOT / "docs/api-reference.md").read_text()
    documented = {
        tuple(cell.strip() for cell in line.strip().strip("|").split("|")[:2])
        for line in docs.splitlines()
        if line.strip().startswith("|")
    }
    for path, methods in schema["paths"].items():
        for method in methods:
            assert (method.upper(), f"`{path}`") in documented


def test_documentation_links_and_python_snippets_are_valid():
    for path in [ROOT / "README.md", ROOT / "CHANGELOG.md", *ROOT.glob("docs/*.md")]:
        text = path.read_text()
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if not target.startswith(("https:", "http:", "#")):
                assert (path.parent / target.split("#")[0]).exists(), (path, target)
        for snippet in re.findall(r"```python\n(.*?)```", text, re.S):
            ast.parse(snippet)


def test_contract_checker_accepts_current_schema_and_rejects_changes(tmp_path):
    snapshot = ROOT / "tests/fixtures/devmid-3.9-public-openapi.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_devmid_contract.py"), str(snapshot)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "18 operations" in result.stdout
    schema = json.loads(snapshot.read_text())
    del schema["paths"]["/runtime/billing"]
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(schema))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_devmid_contract.py"), str(changed)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and "/runtime/billing" in result.stdout


def test_contract_checker_ignores_prose_but_preserves_properties_named_title():
    spec = importlib.util.spec_from_file_location(
        "contract_checker", ROOT / "scripts/check_devmid_contract.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schema = json.loads((ROOT / "tests/fixtures/devmid-3.9-public-openapi.json").read_text())
    edited = copy.deepcopy(schema)
    edited["components"]["schemas"]["AddSSHKeyRequest"]["description"] = "New prose"
    assert module.compare(schema, edited) == []
    edited["components"]["schemas"]["AddSSHKeyRequest"]["properties"]["title"]["type"] = "integer"
    assert module.compare(schema, edited)
    assert module.compare(schema, {"paths": [], "components": {}})[0] == (
        "Invalid paths: expected an object"
    )
