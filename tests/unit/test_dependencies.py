"""Every third-party module imported by src/ must be declared in pyproject.toml. This is the class of bug that hid a
missing `lib/` directory and an undeclared `prometheus-client`: it works on the author's machine and breaks in CI/Docker."""
import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_TO_DIST = {"llama_index": "llama-index-core", "qdrant_client": "qdrant-client", "pydantic_settings": "pydantic-settings",
                  "pymupdf": "pymupdf", "pymupdf4llm": "pymupdf4llm", "yaml": "pyyaml", "prometheus_client": "prometheus-client",
                  "fastapi": "fastapi", "starlette": "fastapi", "uvicorn": "uvicorn", "httpx": "httpx", "numpy": "numpy",
                  "pydantic": "pydantic", "litellm": "litellm", "fastembed": "fastembed", "celery": "celery", "docx": "python-docx", "pptx": "python-pptx",
                  "openpyxl": "openpyxl", "PIL": "pillow"}
OPTIONAL = {"celery"}  # ingest extra: imported only by ingestion/tasks.py


def declared() -> set[str]:
    proj = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    reqs = list(proj["dependencies"]) + [r for v in proj["optional-dependencies"].values() for r in v]
    return {re.split(r"[<>=\[ ;]", r)[0].lower().replace("_", "-") for r in reqs}


def test_all_imports_are_declared():
    decl, missing = declared(), set()
    for f in (ROOT / "src" / "orca_rag").rglob("*.py"):
        for n in ast.walk(ast.parse(f.read_text())):
            mods = ([a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import)
                    else [n.module.split(".")[0]] if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module else [])
            for m in mods:
                if m in sys.stdlib_module_names or m == "orca_rag":
                    continue
                dist = MODULE_TO_DIST.get(m)
                if dist is None or dist not in decl:
                    missing.add((m, dist))
    assert not missing, f"imported but not declared (or unmapped): {sorted(missing, key=str)}"


def test_core_dependencies_do_not_pull_in_torch():
    assert "torch" not in declared() and "sentence-transformers" not in declared()
