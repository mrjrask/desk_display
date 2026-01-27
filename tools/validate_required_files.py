#!/usr/bin/env python3
"""Validate that tracked, non-image files are accounted for by the project.

The script focuses on Python reachability: starting from known entry points
(main, config UI, scripts/, tools/), it builds an import graph and flags any
tracked Python modules that are not reachable from those roots. Images are
explicitly excluded from the scan.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
PYTHON_EXT = ".py"
EXEMPT_PATH_PREFIXES = ("tests/", "vendor/")
OPTIONAL_MODULES = {"services.network"}


@dataclass
class ModuleNode:
    name: str
    path: Path
    imports: Set[str]


def _git_ls(args: List[str]) -> List[Path]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def run_git_ls_files() -> List[Path]:
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
    if level > len(base_parts):
        base_parts = []
    else:
        base_parts = base_parts[: len(base_parts) - level + 1]

    prefix = ".".join(base_parts)
    if module:
        return f"{prefix}.{module}" if prefix else module
    return prefix


def parse_imports(path: Path, module_name: str) -> Set[str]:
    imports: Set[str] = set()
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


def build_module_graph(py_files: Iterable[Path]) -> Dict[str, ModuleNode]:
    graph: Dict[str, ModuleNode] = {}
    for path in py_files:
        module = module_name_from_path(path)
        imports = parse_imports(REPO_ROOT / path, module)
        graph[module] = ModuleNode(name=module, path=path, imports=imports)
    return graph


def determine_seeds(graph: Dict[str, ModuleNode]) -> Set[str]:
    seeds: Set[str] = set()
    for name, node in graph.items():
        top_level = node.path.parts[0]
        if node.path.name in {"main.py", "config_ui.py", "schedule_migrations.py", "storage_overrides.py"}:
            seeds.add(name)
        elif top_level in {"scripts", "tools"}:
            seeds.add(name)
    return seeds


def _parent_modules(name: str) -> List[str]:
    parts = name.split(".")
    parents: List[str] = []
    while len(parts) > 1:
        parts = parts[:-1]
        parents.append(".".join(parts))
    return parents


def find_reachable_modules(graph: Dict[str, ModuleNode], seeds: Set[str]) -> Set[str]:
    reachable: Set[str] = set()
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
    print(f"Entry modules (seeds): {', '.join(sorted(seeds)) or 'none'}")

    if optional:
        print("ℹ️  Optional modules not linked from entry points:")
        for node in sorted(optional, key=lambda n: n.path):
            print(f"  - {node.path}")

    if unreachable:
        print(format_unreachable(unreachable))
        return 1

    print("✅ All tracked Python modules are reachable from the entry points.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
