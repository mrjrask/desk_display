#!/usr/bin/env python3
"""Validate that tracked, non-image files are accounted for by the project.

The script focuses on Python reachability: starting from known entry points
(the standalone ``main.py``, ``display_server.py``, ``display_client.py``,
the config UI, the installers' ``install_modes.py``, ``scripts/`` and
``screens/``), it builds an import graph and flags any tracked Python modules
that are not reachable from those roots. Images are explicitly excluded from
the scan.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
PYTHON_EXT = ".py"
EXEMPT_PATH_PREFIXES = ("tests/", "vendor/")
# Tracked modules nothing imports yet, on purpose, with the reason.
OPTIONAL_MODULES: dict[str, str] = {
    "schema_migrations": "the schema migration contract (tests/test_protocol_compatibility.py); "
    "used once a v2 playlist or config schema ships",
}


@dataclass
class ModuleNode:
    name: str
    path: Path
    imports: set[str]


def _git_ls(args: list[str]) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def run_git_ls_files() -> list[Path]:
    tracked = _git_ls([])
    untracked = _git_ls(["--others", "--exclude-standard"])

    all_paths = {path for path in tracked + untracked if (REPO_ROOT / path).exists()}
    return sorted(all_paths)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS or "images" in path.parts


def module_name_from_path(path: Path) -> str:
    relative = path.with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def resolve_relative_module(current: str, module: str | None, level: int) -> str:
    if not level:
        return module or ""

    base_parts = current.split(".")
    if base_parts:
        base_parts = base_parts[:-1]  # remove current module name
    base_parts = [] if level > len(base_parts) else base_parts[:len(base_parts) - level + 1]

    prefix = ".".join(base_parts)
    if module:
        return f"{prefix}.{module}" if prefix else module
    return prefix


def parse_imports(path: Path, module_name: str) -> set[str]:
    imports: set[str] = set()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = resolve_relative_module(module_name, node.module, node.level)
            if base:
                imports.add(base)
            for alias in node.names:
                if alias.name != "*" and base:
                    imports.add(f"{base}.{alias.name}")
    return imports


def build_module_graph(py_files: Iterable[Path]) -> dict[str, ModuleNode]:
    graph: dict[str, ModuleNode] = {}
    for path in py_files:
        module = module_name_from_path(path)
        imports = parse_imports(REPO_ROOT / path, module)
        graph[module] = ModuleNode(name=module, path=path, imports=imports)
    return graph


ENTRY_POINTS = {
    "main.py", "config_ui.py", "feed_server.py", "schedule_migrations.py", "display_server.py",
    "display_client.py", "install_modes.py", "soak.py", "runtime.py",
}


def determine_seeds(graph: dict[str, ModuleNode]) -> set[str]:
    seeds: set[str] = set()
    for name, node in graph.items():
        top_level = node.path.parts[0]
        # Entry points run by services, installers or operators, and screens,
        # which screens/registry.py loads by name.
        if node.path.name in ENTRY_POINTS or top_level in {"scripts", "screens"}:
            seeds.add(name)
    return seeds


def _parent_modules(name: str) -> list[str]:
    parts = name.split(".")
    parents: list[str] = []
    while len(parts) > 1:
        parts = parts[:-1]
        parents.append(".".join(parts))
    return parents


def find_reachable_modules(graph: dict[str, ModuleNode], seeds: set[str]) -> set[str]:
    reachable: set[str] = set()
    stack = list(seeds)

    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        for parent in _parent_modules(current):
            if parent in graph:
                reachable.add(parent)
        imports = graph.get(current)
        if not imports:
            continue
        for target in imports.imports:
            if target in graph and target not in reachable:
                stack.append(target)
    return reachable


def format_unreachable(nodes: Iterable[ModuleNode]) -> str:
    lines = ["⚠️ Unreachable Python modules detected:"]
    for node in sorted(nodes, key=lambda n: n.path):
        lines.append(f"  - {node.path}")
    return "\n".join(lines)


def main() -> int:
    tracked = run_git_ls_files()
    non_image_files = [p for p in tracked if not is_image(p)]
    python_files = [p for p in non_image_files if p.suffix == PYTHON_EXT]

    graph = build_module_graph(python_files)
    seeds = determine_seeds(graph)
    reachable = find_reachable_modules(graph, seeds)

    unreachable = []
    optional = []
    for name, node in graph.items():
        if name in reachable:
            continue
        rel_str = str(node.path)
        if rel_str.startswith(EXEMPT_PATH_PREFIXES):
            continue
        if name in OPTIONAL_MODULES:
            optional.append(node)
            continue
        unreachable.append(node)

    print(f"Tracked non-image files: {len(non_image_files)}")
    print(f"Tracked Python modules: {len(python_files)}")
    entry = sorted(name for name in seeds if graph[name].path.parts[0] not in {"scripts", "screens"})
    bundled = len(seeds) - len(entry)
    print(f"Entry modules (seeds): {', '.join(entry) or 'none'}, plus {bundled} in scripts/ and screens/")

    if optional:
        print("ℹ️  Optional modules not linked from entry points:")
        for node in sorted(optional, key=lambda n: n.path):
            print(f"  - {node.path}: {OPTIONAL_MODULES[node.name]}")

    if unreachable:
        print(format_unreachable(unreachable))
        return 1

    print("✅ All tracked Python modules are reachable from the entry points.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
