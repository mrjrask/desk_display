#!/usr/bin/env python3
"""Font audit for desk_display.

Scans:
- config.py for declared FONT_* constants and their concrete sizes
- screens/**/*.py for ImageFont.truetype calls and font usage references
- screens_style.json for font definitions
- screens_layouts.json for per-display layout variants

Usage:
  python3 scripts/font_audit.py
  python3 scripts/font_audit.py --json out_fonts.json
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import sys
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
    resolved_size: Optional[int]


@dataclass
class FontLookupHint:
    lineno: int
    col: int
    expression: str
    resolved_size: Optional[int]


@dataclass
class FontUsageRef:
    lineno: int
    col: int
    font_ref: str
    resolved_size: Optional[int]


@dataclass
class FontDefinition:
    name: str
    size: Optional[int]
    source: str
    lineno: int


@dataclass
class ScreenFontReport:
    screen_file: str
    truetype_calls: list[TruetypeCall]
    font_references: list[FontUsageRef]
    lookup_hints: list[FontLookupHint]
    resolved_sizes_seen: list[int]
    notes: list[str]


@dataclass
class RepoFontAudit:
    repo_root: str
    known_font_sizes: dict[str, FontDefinition]
    screens_style_fonts: dict[str, Any]
    screens_layouts_summary: dict[str, Any]
    per_screen: list[ScreenFontReport]


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_json_if_exists(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(safe_read_text(path))
    except Exception as exc:
        return {"__error__": f"Failed to parse JSON: {exc}", "__path__": str(path)}


def expr_to_source(node: ast.AST, fallback: str = "") -> str:
    try:
        return ast.unparse(node)  # type: ignore[attr-defined]
    except Exception:
        return fallback


def node_loc(node: ast.AST) -> tuple[int, int]:
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


def looks_like_truetype_call(func_node: ast.AST) -> bool:
    if isinstance(func_node, ast.Attribute):
        return func_node.attr == "truetype"
    if isinstance(func_node, ast.Name):
        return func_node.id == "truetype"
    return False


def _font_ref_from_node(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return expr_to_source(node, "")
    return None


def _extract_int(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    return None


def eval_int_expr(node: ast.AST, constants: dict[str, int]) -> Optional[int]:
    value = _extract_int(node)
    if value is not None:
        return value

    if isinstance(node, ast.Name):
        return constants.get(node.id)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = eval_int_expr(node.operand, constants)
        if inner is None:
            return None
        return inner if isinstance(node.op, ast.UAdd) else -inner

    if isinstance(node, ast.BinOp):
        left = eval_int_expr(node.left, constants)
        right = eval_int_expr(node.right, constants)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv) and right != 0:
            return left // right
        if isinstance(node.op, ast.Mod) and right != 0:
            return left % right
        return None

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = node.func.id
        if fn in {"max", "min"}:
            vals: list[int] = []
            for arg in node.args:
                v = eval_int_expr(arg, constants)
                if v is None:
                    return None
                vals.append(v)
            if not vals:
                return None
            return max(vals) if fn == "max" else min(vals)

    if isinstance(node, ast.IfExp):
        # best-effort: try both branches and keep equal result if deterministic
        body = eval_int_expr(node.body, constants)
        orelse = eval_int_expr(node.orelse, constants)
        if body is not None and body == orelse:
            return body
        return body if body is not None else orelse

    return None


def build_known_font_sizes(repo_root: Path) -> dict[str, FontDefinition]:
    """Extract FONT_* definitions from config.py with concrete sizes when possible."""
    out: dict[str, FontDefinition] = {}
    config_path = repo_root / "config.py"
    if not config_path.exists():
        return out

    tree = ast.parse(safe_read_text(config_path))
    int_constants: dict[str, int] = {}

    for node in tree.body:
        target: Optional[ast.Name] = None
        value: Optional[ast.AST] = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            target = node.target
            value = node.value

        if target is None or value is None:
            continue

        name = target.id
        as_int = eval_int_expr(value, int_constants)
        if as_int is not None:
            int_constants[name] = as_int

        if not name.startswith("FONT_"):
            continue

        size: Optional[int] = None
        source = expr_to_source(value, type(value).__name__)

        if isinstance(value, ast.Call):
            if (isinstance(value.func, ast.Name) and value.func.id == "_load_font") or looks_like_truetype_call(value.func):
                if len(value.args) >= 2:
                    size = eval_int_expr(value.args[1], int_constants)
                for kw in value.keywords or []:
                    if kw.arg == "size" and size is None:
                        size = eval_int_expr(kw.value, int_constants)
            elif isinstance(value.func, ast.Name) and value.func.id == "_load_emoji_font":
                if value.args:
                    size = eval_int_expr(value.args[0], int_constants)
                for kw in value.keywords or []:
                    if kw.arg == "size" and size is None:
                        size = eval_int_expr(kw.value, int_constants)

        if size is None and isinstance(value, ast.Name) and value.id in out:
            size = out[value.id].size

        out[name] = FontDefinition(name=name, size=size, source=source, lineno=getattr(node, "lineno", 0))

    return out


class FontCallScanner(ast.NodeVisitor):
    def __init__(
        self,
        known_font_sizes: dict[str, FontDefinition],
        module_name: str = "",
        runtime_module_fonts: Optional[dict[str, dict[str, int]]] = None,
    ) -> None:
        self.known_font_sizes = known_font_sizes
        self.module_name = module_name
        self.runtime_module_fonts = runtime_module_fonts or {}
        self.local_ints: dict[str, int] = {}
        self.import_aliases: dict[str, str] = {}
        self.truetype_calls: list[TruetypeCall] = []
        self.font_references: list[FontUsageRef] = []
        self.lookup_hints: list[FontLookupHint] = []

    def _resolve_font_size(self, node: ast.AST) -> Optional[int]:
        ref = _font_ref_from_node(node)
        if not ref:
            return None

        key = ref.split(".")[-1]
        if key in self.known_font_sizes:
            return self.known_font_sizes[key].size

        if "." not in ref:
            return self.runtime_module_fonts.get(self.module_name, {}).get(ref)

        root, _, attr = ref.partition(".")
        module_target = self.import_aliases.get(root)
        if module_target:
            return self.runtime_module_fonts.get(module_target, {}).get(attr)
        return None

    def _record_font_ref(self, node: ast.AST) -> None:
        ref = _font_ref_from_node(node)
        if not ref:
            return

        # Avoid noisy locals like plain "font" unless they are known constants.
        if ref not in self.known_font_sizes and not re.search(r"(^|\.)FONT_[A-Z0-9_]+$", ref):
            return

        size = self._resolve_font_size(node)
        lineno, col = node_loc(node)
        self.font_references.append(FontUsageRef(lineno, col, ref, size))

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[-1]
            self.import_aliases[local] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        if not node.module:
            return self.generic_visit(node)
        for alias in node.names:
            local = alias.asname or alias.name
            self.import_aliases[local] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            val = eval_int_expr(node.value, self.local_ints)
            if val is not None:
                self.local_ints[node.targets[0].id] = val
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if isinstance(node.target, ast.Name) and node.value is not None:
            val = eval_int_expr(node.value, self.local_ints)
            if val is not None:
                self.local_ints[node.target.id] = val
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if looks_like_truetype_call(node.func):
            lineno, col = node_loc(node)
            font_arg_node = node.args[0] if len(node.args) >= 1 else None
            size_arg_node = node.args[1] if len(node.args) >= 2 else None

            font_arg = stringify_arg(font_arg_node)
            size_arg = stringify_arg(size_arg_node)
            resolved_size = eval_int_expr(size_arg_node, self.local_ints) if size_arg_node is not None else None

            for kw in node.keywords or []:
                if kw.arg == "font" and not font_arg:
                    font_arg_node = kw.value
                    font_arg = stringify_arg(kw.value)
                if kw.arg == "size" and not size_arg:
                    size_arg_node = kw.value
                    size_arg = stringify_arg(kw.value)
                    resolved_size = eval_int_expr(kw.value, self.local_ints)

            self.truetype_calls.append(
                TruetypeCall(
                    lineno=lineno,
                    col=col,
                    font_arg=font_arg,
                    size_arg=size_arg,
                    resolved_size=resolved_size,
                )
            )

            if font_arg_node is not None:
                self._record_font_ref(font_arg_node)

        for kw in node.keywords or []:
            if kw.arg == "font":
                self._record_font_ref(kw.value)

        call_src = expr_to_source(node, fallback="")
        if isinstance(node.func, ast.Name) and node.func.id == "_text_size" and len(node.args) >= 2:
            lineno, col = node_loc(node)
            resolved_size = self._resolve_font_size(node.args[1])
            self.lookup_hints.append(
                FontLookupHint(lineno, col, call_src or "_text_size(...)", resolved_size)
            )

        if call_src:
            if re.search(r"\bget_font\s*\(", call_src):
                lineno, col = node_loc(node)
                self.lookup_hints.append(FontLookupHint(lineno, col, call_src, None))
            if re.search(r"\bfonts?\s*\[", call_src) or re.search(r"\bstyle\s*\[", call_src):
                lineno, col = node_loc(node)
                if "font" in call_src.lower():
                    self.lookup_hints.append(FontLookupHint(lineno, col, call_src, None))

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        src = expr_to_source(node, fallback="")
        if src and "font" in src.lower():
            lineno, col = node_loc(node)
            self.lookup_hints.append(FontLookupHint(lineno, col, src, None))
        self.generic_visit(node)


def collect_runtime_module_fonts(repo_root: Path) -> dict[str, dict[str, int]]:
    modules: dict[str, dict[str, int]] = {}
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    for path in sorted((repo_root / "screens").rglob("*.py")):
        rel = str(path.relative_to(repo_root))
        module_name = rel[:-3].replace("/", ".")
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue

        font_map: dict[str, int] = {}
        for name, value in vars(mod).items():
            if "FONT" not in name.upper():
                continue
            size = getattr(value, "size", None)
            if isinstance(size, int):
                font_map[name] = size
        if font_map:
            modules[module_name] = font_map

    return modules


def extract_fonts_from_screens_style(style_json: Any) -> dict[str, Any]:
    if not isinstance(style_json, dict):
        return {}

    fonts_found: dict[str, Any] = {}

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


def summarize_layouts(layout_json: Any) -> dict[str, Any]:
    if not isinstance(layout_json, dict):
        return {}

    summary: dict[str, Any] = {
        "top_level_keys": sorted(list(layout_json.keys()))[:200],
        "display_targets": {},
    }

    blob = json.dumps(layout_json)[:2_000_000]
    for name, (width, height) in DISPLAY_TARGETS:
        hits: list[str] = []
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

    style_json = load_json_if_exists(repo_root / "screens_style.json") or {}
    layouts_json = load_json_if_exists(repo_root / "screens_layouts.json") or {}

    known_font_sizes = build_known_font_sizes(repo_root)
    screens_style_fonts = extract_fonts_from_screens_style(style_json)
    layouts_summary = summarize_layouts(layouts_json)

    runtime_module_fonts = collect_runtime_module_fonts(repo_root)
    per_screen: list[ScreenFontReport] = []
    py_files = sorted(path for path in screens_dir.rglob("*.py") if path.is_file())

    for path in py_files:
        rel = str(path.relative_to(repo_root))
        txt = safe_read_text(path)
        notes: list[str] = []

        try:
            tree = ast.parse(txt)
        except SyntaxError as exc:
            per_screen.append(
                ScreenFontReport(
                    screen_file=rel,
                    truetype_calls=[],
                    font_references=[],
                    lookup_hints=[],
                    resolved_sizes_seen=[],
                    notes=[f"SyntaxError parsing file: {exc}"],
                )
            )
            continue

        module_name = rel[:-3].replace("/", ".")
        scanner = FontCallScanner(known_font_sizes, module_name, runtime_module_fonts)
        scanner.visit(tree)

        resolved_sizes = sorted(
            {
                c.resolved_size
                for c in scanner.truetype_calls
                if c.resolved_size is not None
            }
            | {
                f.resolved_size
                for f in scanner.font_references
                if f.resolved_size is not None
            }
        )

        if not scanner.truetype_calls and not scanner.font_references and not scanner.lookup_hints:
            notes.append("No direct font calls detected (may rely on shared helpers/styles).")

        per_screen.append(
            ScreenFontReport(
                screen_file=rel,
                truetype_calls=scanner.truetype_calls,
                font_references=scanner.font_references,
                lookup_hints=scanner.lookup_hints,
                resolved_sizes_seen=resolved_sizes,
                notes=notes,
            )
        )

    return RepoFontAudit(
        repo_root=str(repo_root),
        known_font_sizes=known_font_sizes,
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
    print("Known FONT_* definitions from config.py")
    print("-" * 80)
    if not audit.known_font_sizes:
        print("No FONT_* definitions discovered in config.py.")
    else:
        known = sorted(audit.known_font_sizes.values(), key=lambda item: (item.size is None, item.name))
        for entry in known[:400]:
            size_label = str(entry.size) if entry.size is not None else "?"
            print(f"  - {entry.name}: size={size_label} (L{entry.lineno})")
        if len(known) > 400:
            print(f"  ... ({len(known) - 400} more)")
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

        if screen.resolved_sizes_seen:
            print(f"  resolved font sizes seen: {', '.join(str(s) for s in screen.resolved_sizes_seen)}")

        if screen.truetype_calls:
            print("  truetype calls:")
            for call in screen.truetype_calls:
                size_info = f" -> resolved={call.resolved_size}" if call.resolved_size is not None else ""
                print(
                    f"    - L{call.lineno}:{call.col}  "
                    f"ImageFont.truetype(font={call.font_arg}, size={call.size_arg}){size_info}"
                )

        if screen.font_references:
            print("  font references:")
            for ref in screen.font_references[:200]:
                size_info = f" (size={ref.resolved_size})" if ref.resolved_size is not None else ""
                print(f"    - L{ref.lineno}:{ref.col}  {ref.font_ref}{size_info}")
            if len(screen.font_references) > 200:
                print(f"    ... ({len(screen.font_references) - 200} more)")

        if screen.lookup_hints:
            print("  style/font lookup hints:")
            for hint in screen.lookup_hints[:200]:
                size_info = f" (size={hint.resolved_size})" if hint.resolved_size is not None else ""
                print(f"    - L{hint.lineno}:{hint.col}  {hint.expression}{size_info}")
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
        out_path.write_text(json.dumps(asdict(audit), indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {out_path}")


if __name__ == "__main__":
    main()
