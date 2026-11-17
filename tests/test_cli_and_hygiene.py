"""CLI smoke tests, AST SyntaxWarning hygiene checks, and local Markdown link verification."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote
import warnings

import pytest

from vld.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_info_verify_and_navigate(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    info_out = json.loads(capsys.readouterr().out)
    assert info_out["package"] == "virtual-layer-dynamics"

    assert main(["verify"]) == 0
    verify_out = json.loads(capsys.readouterr().out)
    assert verify_out["status"] == "PASS"
    assert verify_out["total_assertions"] >= 50

    assert main(["navigate", "--seed", "73", "--coordinate", "1000000000000"]) == 0
    nav_out = json.loads(capsys.readouterr().out)
    assert nav_out["coordinate"] == 1000000000000
    assert 0 <= nav_out["value"] < 2**32

    with pytest.raises(SystemExit) as exc_info:
        main(["navigate", "--seed", "-1", "--coordinate", "10"])
    assert exc_info.value.code == 2
    capsys.readouterr()


def test_zero_syntax_warnings_in_all_python_sources() -> None:
    py_files = list((REPO_ROOT / "src").rglob("*.py")) + list((REPO_ROOT / "tests").rglob("*.py"))
    assert len(py_files) >= 10
    for path in py_files:
        source = path.read_text(encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("error", SyntaxWarning)
            ast.parse(source, filename=str(path))


def test_tracked_markdown_links_resolve() -> None:
    md_files = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md"))]
    link_pattern = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)")
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        for match in link_pattern.finditer(text):
            target = match.group(2).strip().strip("<>")
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                continue
            rel = unquote(target.split("#", 1)[0].split("?", 1)[0])
            resolved = (md_path.parent / rel).resolve()
            assert resolved.exists(), f"Broken link in {md_path.name}: {target!r} -> {resolved}"
            # Ensure tracked docs never link into gitignored directories
            rel_to_root = resolved.relative_to(REPO_ROOT).as_posix()
            ignored_prefixes = (
                "E-CPU/",
                "E-CPU-Virtual-Processor/",
                "In-Formation-Space/",
                "Virtual-Ground-Interpreter/",
                "agent-navigation/",
                "AGENT_NAVIGATION.md",
            )
            assert not rel_to_root.startswith(ignored_prefixes), (
                f"Tracked doc {md_path.name} links to gitignored path {rel_to_root}"
            )
