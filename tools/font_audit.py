#!/usr/bin/env python3
"""
Font audit for desk_display.

Scans:
- screens/**/*.py for PIL ImageFont.truetype calls and common style/font lookup patterns
- screens_style.json for font definitions
- screens_layouts.json for per-display layout variants

Outputs a readable report + optional JSON export.

Usage:
  python3 tools/font_audit.py
  python3 tools/font_audit.py --json out_fonts.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

DISPLAY_TARGETS = [
    ("display_hat_mini", (320, 240)),
    ("hyperpixel_4", (800, 480)),
    ("hyperpixel_4_square", (720, 720)),
]


@dataclass
class TruetypeCall:
    lineno: int
    col: int
    font_arg: str
    size_arg: str


@dataclass
class FontLookupHint:
    lineno: int
    col: int
    expression: str


@dataclass
class ScreenFontReport:
    screen_file: str
    truetype_calls: List[TruetypeCall]
    lookup_hints: List[FontLookupHint]
    notes: List[str]


@dataclass
class RepoFontAudit:
    repo_root: str
    screens_style_fonts: Dict[str, Any]
    screens_layouts_summary: Dict[str, Any]
    per_screen: List[ScreenFontReport]


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_json_if_exists(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(safe_read_text(path))
    except Exception as exc:  # noqa: BLE001 - best-effort diagnostics
        return {"__error__": f"Failed to parse JSON: {exc}", "__path__": str(path)}


def expr_to_source(node: ast.AST, fallback: str = "") -> str:
    try:
        return ast.unparse(node)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - best-effort unparse
        return fallback


def looks_like_truetype_call(func_node: ast.AST) -> bool:
    """Detect ImageFont.truetype(...) or truetype(...)."""
    if isinstance(func_node, ast.Attribute):
        return func_node.attr == "truetype"
    if isinstance(func_node, ast.Name):
        return func_node.id == "truetype"
    return False


def node_loc(node: ast.AST) -> Tuple[int, int]:
    return (getattr(node, "lineno", 0) or 0, getattr(node, "col_offset", 0) or 0)


def is_string_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def stringify_arg(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    if is_string_literal(node):
        return repr(node.value)
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return expr_to_source(node, fallback=type(node).__name__)


class FontCallScanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.truetype_calls: List[TruetypeCall] = []
        self.lookup_hints: List[FontLookupHint] = []

    def visit_Call(self, node: ast.Call) -> Any:  # noqa: ANN401 - AST visitor signature
        if looks_like_truetype_call(node.func):
            lineno, col = node_loc(node)
            font_arg = stringify_arg(node.args[0]) if len(node.args) >= 1 else ""
            size_arg = stringify_arg(node.args[1]) if len(node.args) >= 2 else ""

            for kw in node.keywords or []:
                if kw.arg == "font" and not font_arg:
                    font_arg = stringify_arg(kw.value)
                if kw.arg == "size" and not size_arg:
                    size_arg = stringify_arg(kw.value)

            self.truetype_calls.append(
                TruetypeCall(
                    lineno=lineno,
                    col=col,
                    font_arg=font_arg,
                    size_arg=size_arg,
                )
            )

        call_src = expr_to_source(node, fallback="")
        if call_src:
            if re.search(r"\bget_font\s*\(", call_src):
                lineno, col = node_loc(node)
                self.lookup_hints.append(FontLookupHint(lineno, col, call_src))
            if re.search(r"\bfonts?\s*\[", call_src) or re.search(r"\bstyle\s*\[", call_src):
                lineno, col = node_loc(node)
                if "font" in call_src.lower():
                    self.lookup_hints.append(FontLookupHint(lineno, col, call_src))

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> Any:  # noqa: ANN401 - AST visitor signature
        src = expr_to_source(node, fallback="")
        if src and ("font" in src.lower()):
            lineno, col = node_loc(node)
            self.lookup_hints.append(FontLookupHint(lineno, col, src))
        self.generic_visit(node)


def extract_fonts_from_screens_style(style_json: Any) -> Dict[str, Any]:
    """Best-effort: return anything that looks like a 'fonts' block."""
    if not isinstance(style_json, dict):
        return {}

    fonts_found: Dict[str, Any] = {}

    def walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                next_path = f"{path}.{key}" if path else key
                if key.lower() == "fonts" and isinstance(value, dict):
                    fonts_found[next_path] = value
                walk(value, next_path)
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                walk(value, f"{path}[{index}]")

    walk(style_json)
    return fonts_found


def summarize_layouts(layout_json: Any) -> Dict[str, Any]:
    """Best-effort summary of screens_layouts.json and per-display variants."""
    if not isinstance(layout_json, dict):
        return {}

    summary: Dict[str, Any] = {
        "top_level_keys": sorted(list(layout_json.keys()))[:200],
        "display_targets": {},
    }

    blob = json.dumps(layout_json)[:2_000_000]
    for name, (width, height) in DISPLAY_TARGETS:
        hits: List[str] = []
        if name in blob:
            hits.append(f"mentions '{name}'")
        if str(width) in blob and str(height) in blob:
            hits.append(f"mentions {width}x{height} somewhere")
        summary["display_targets"][name] = {
            "expected_size": {"w": width, "h": height},
            "heuristic_hits": hits,
        }

    return summary


def audit_repo(repo_root: Path) -> RepoFontAudit:
    screens_dir = repo_root / "screens"
    if not screens_dir.exists():
        raise SystemExit(f"Expected screens/ directory at: {screens_dir}")

    style_path = repo_root / "screens_style.json"
    layouts_path = repo_root / "screens_layouts.json"

    style_json = load_json_if_exists(style_path) or {}
    layouts_json = load_json_if_exists(layouts_path) or {}

    screens_style_fonts = extract_fonts_from_screens_style(style_json)
    layouts_summary = summarize_layouts(layouts_json)

    per_screen: List[ScreenFontReport] = []

    py_files = sorted(path for path in screens_dir.rglob("*.py") if path.is_file())
    for path in py_files:
        rel = str(path.relative_to(repo_root))
        txt = safe_read_text(path)

        notes: List[str] = []
        try:
            tree = ast.parse(txt)
        except SyntaxError as exc:
            per_screen.append(
                ScreenFontReport(
                    screen_file=rel,
                    truetype_calls=[],
                    lookup_hints=[],
                    notes=[f"SyntaxError parsing file: {exc}"],
                )
            )
            continue

        scanner = FontCallScanner()
        scanner.visit(tree)

        if not scanner.truetype_calls and not scanner.lookup_hints:
            notes.append("No direct font calls detected (may rely on shared helpers/styles).")

        per_screen.append(
            ScreenFontReport(
                screen_file=rel,
                truetype_calls=scanner.truetype_calls,
                lookup_hints=scanner.lookup_hints,
                notes=notes,
            )
        )

    return RepoFontAudit(
        repo_root=str(repo_root),
        screens_style_fonts=screens_style_fonts,
        screens_layouts_summary=layouts_summary,
        per_screen=per_screen,
    )


def print_report(audit: RepoFontAudit) -> None:
    print("=" * 80)
    print("desk_display font audit")
    print("=" * 80)
    print(f"Repo root: {audit.repo_root}")
    print()

    print("-" * 80)
    print("screens_style.json: detected font blocks")
    print("-" * 80)
    if not audit.screens_style_fonts:
        print("No obvious 'fonts' blocks found (or screens_style.json missing).")
    else:
        for path, fonts_block in audit.screens_style_fonts.items():
            print(f"\n[{path}]")
            for key, value in list(fonts_block.items())[:200]:
                print(f"  {key}: {value}")
            if len(fonts_block) > 200:
                print(f"  ... ({len(fonts_block) - 200} more)")
    print()

    print("-" * 80)
    print("screens_layouts.json: summary (heuristics)")
    print("-" * 80)
    if not audit.screens_layouts_summary:
        print("No layout summary available (or screens_layouts.json missing).")
    else:
        print("Top-level keys (first ~200):")
        for key in audit.screens_layouts_summary.get("top_level_keys", []):
            print(f"  - {key}")
        print("\nDisplay target heuristics:")
        for target, info in audit.screens_layouts_summary.get("display_targets", {}).items():
            hits = info.get("heuristic_hits") or []
            print(
                f"  - {target} ({info['expected_size']['w']}x{info['expected_size']['h']}): "
                f"{', '.join(hits) if hits else 'no obvious hits'}"
            )
    print()

    print("=" * 80)
    print("Per-screen font usage")
    print("=" * 80)

    for screen in audit.per_screen:
        print(f"\n## {screen.screen_file}")
        if screen.truetype_calls:
            print("  truetype calls:")
            for call in screen.truetype_calls:
                print(
                    f"    - L{call.lineno}:{call.col}  "
                    f"ImageFont.truetype(font={call.font_arg}, size={call.size_arg})"
                )
        if screen.lookup_hints:
            print("  style/font lookup hints:")
            for hint in screen.lookup_hints[:200]:
                print(f"    - L{hint.lineno}:{hint.col}  {hint.expression}")
            if len(screen.lookup_hints) > 200:
                print(f"    ... ({len(screen.lookup_hints) - 200} more)")
        if screen.notes:
            print("  notes:")
            for note in screen.notes:
                print(f"    - {note}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", dest="json_out", default="", help="Write machine-readable output to this file")
    args = parser.parse_args()

    audit = audit_repo(REPO_ROOT)
    print_report(audit)

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(audit)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {out_path}")


if __name__ == "__main__":
    main()
