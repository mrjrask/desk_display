"""Convert an existing Desk Display ``.env`` into a server or client configuration.

The conversion edits the file line by line instead of regenerating it, so the
operator's comments, ordering and values that still apply survive:

* a setting the target role does not read is removed (a client also drops
  unknown names, since only the server may hold provider credentials);
* a repeated setting keeps its last line, which is the one dotenv loaders use;
* ``DESK_DISPLAY_ROLE`` is set, and the role's required settings and its
  role-only defaults are appended under one marked block.

Every removal and addition is reported, and nothing is written unless the
result passes the same :func:`deployment_config.validate` checks the server and
client run at startup.  Rendering for humans always goes through
:func:`Conversion.render_report` / :func:`Conversion.render_diff`, which never
include a secret value.
"""

from __future__ import annotations

import contextlib
import difflib
import os
import re
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import deployment_config as dc

ADDED_HEADER = "# Added by scripts/convert_env.py for the {role} role."
TARGET_ROLES = (dc.Role.SERVER, dc.Role.CLIENT)

# Values an operator must supply; a placeholder is written when it is missing.
REQUIRED: dict[dc.Role, tuple[str, ...]] = {
    dc.Role.SERVER: (),
    dc.Role.CLIENT: (
        "DESK_DISPLAY_SERVER_URL",
        "DESK_DISPLAY_CLIENT_ID",
        "DESK_DISPLAY_CLIENT_TOKEN",
        "DESK_DISPLAY_PROFILE",
    ),
}
_REQUIRED_HINTS = {
    "DESK_DISPLAY_SERVER_URL": "the render server's base URL, e.g. https://render-server.lan:8765",
    "DESK_DISPLAY_CLIENT_ID": "a stable, unique ID for this display",
    "DESK_DISPLAY_CLIENT_TOKEN": "the credential from the server's Add a display form",
    "DESK_DISPLAY_PROFILE": "this display's profile, e.g. hyperpixel4_square",
}
# Unknown names that look like credentials are redacted in every report.
_SECRET_NAME_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.IGNORECASE)
_ASSIGN_RE = re.compile(r"^\s*(#\s*)?(export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


class ConversionError(RuntimeError):
    """The file cannot be converted as asked."""


@dataclass(frozen=True)
class Change:
    action: str  # "removed", "added", "updated"
    name: str
    reason: str


@dataclass
class Conversion:
    path: Path
    role: dc.Role
    original: str
    text: str
    changes: list[Change] = field(default_factory=list)
    report: dc.ValidationReport | None = None
    secret_names: frozenset[str] = frozenset()
    secret_values: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.text != self.original

    @property
    def env(self) -> dict[str, str]:
        return _parse_text(self.text)

    def redact(self, text: str) -> str:
        lines = []
        for line in text.split("\n"):
            # Diff lines carry a one-character prefix before the assignment.
            for offset in (0, 1):
                match = _ASSIGN_RE.match(line[offset:])
                if match:
                    if (
                        _is_secret_name(match.group(3), self.secret_names)
                        and match.group(4).strip()
                    ):
                        line = line[: offset + match.start(4)] + dc.REDACTED
                    break
            lines.append(line)
        text = "\n".join(lines)
        for value in self.secret_values:
            text = text.replace(value, dc.REDACTED)
        return text

    def render_diff(self) -> str:
        diff = difflib.unified_diff(
            self.original.splitlines(keepends=True),
            self.text.splitlines(keepends=True),
            fromfile=f"{self.path} (current)",
            tofile=f"{self.path} ({self.role.value})",
        )
        return self.redact("".join(diff))

    def render_report(self) -> str:
        lines = [f"{c.action}: {c.name} ({c.reason})" for c in self.changes]
        if self.report is not None:
            lines += [
                f"{issue.level}: {issue.name}: {issue.message}" for issue in self.report.issues
            ]
            lines.append(
                f"{self.role.value}: {len(self.report.errors)} error(s), "
                f"{len(self.report.warnings)} warning(s)"
            )
        return self.redact("\n".join(lines))


def _is_secret_name(name: str, extra: frozenset[str] = frozenset()) -> bool:
    if name in dc.SECRET_SETTING_NAMES or name in extra:
        return True
    setting = dc.SETTINGS_BY_NAME.get(name)
    if setting is not None:
        return setting.provider
    return bool(_SECRET_NAME_RE.search(name))


def _parse_text(text: str) -> dict[str, str]:
    return dc.parse_env_text(text)


def _line_value(raw: str) -> str:
    return _parse_text("X=" + raw).get("X", "")


def _format_value(value: str) -> str:
    if value and re.search(r"[\"'\s#]", value):
        quote = "'" if '"' in value else '"'
        return f"{quote}{value}{quote}"
    return value


def convert(
    path: str | os.PathLike[str],
    role: dc.Role,
    supplied: Mapping[str, str] | None = None,
    *,
    keep_unknown: bool | None = None,
    check_files: bool = True,
) -> Conversion:
    """Return the conversion of *path* to *role* without touching the file.

    *supplied* gives values (for example a client's provisioned credential)
    that override or fill the file's own.  Unknown names are kept on a server
    and removed from a client unless *keep_unknown* says otherwise.
    """

    if role not in TARGET_ROLES:
        raise ConversionError("convert to the server or client role")
    path = Path(path)
    try:
        original = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConversionError(f"{path} does not exist") from None
    supplied = {k: v for k, v in (supplied or {}).items() if v is not None}
    for name in supplied:
        setting = dc.SETTINGS_BY_NAME.get(name)
        if setting is None or role not in setting.roles:
            raise ConversionError(f"{name} is not a {role.value} setting")
    if keep_unknown is None:
        keep_unknown = role is dc.Role.SERVER

    lines = original.split("\n")
    trailing_newline = original.endswith("\n")
    if trailing_newline:
        lines = lines[:-1]

    # The line dotenv loaders honour for each name is the last one.
    last_index: dict[str, int] = {}
    for index, line in enumerate(lines):
        match = _ASSIGN_RE.match(line)
        if match and not match.group(1):
            last_index[match.group(3)] = index

    changes: list[Change] = []
    kept: list[str] = []
    seen: set[str] = set()
    extra_secret_names: set[str] = set()
    for index, line in enumerate(lines):
        match = _ASSIGN_RE.match(line)
        if not match:
            kept.append(line)
            continue
        commented, name = bool(match.group(1)), match.group(3)
        setting = dc.SETTINGS_BY_NAME.get(name)
        if _is_secret_name(name):
            extra_secret_names.add(name)
        belongs = setting is not None and role in setting.roles
        if commented:
            # A commented-out assignment is documentation, unless it names a
            # setting this role must not carry (a disabled credential on a client).
            if (setting is not None and not belongs) or (setting is None and not keep_unknown):
                changes.append(
                    Change(
                        "removed", name, "commented-out line for a setting this role does not use"
                    )
                )
                continue
            kept.append(line)
            continue
        if setting is None and not keep_unknown:
            changes.append(Change("removed", name, "not a Desk Display setting"))
            continue
        if setting is not None and not belongs:
            owner = ", ".join(sorted(r.value for r in setting.roles))
            changes.append(
                Change("removed", name, f"not used by the {role.value} role (used by: {owner})")
            )
            continue
        if index != last_index[name]:
            changes.append(Change("removed", name, "duplicate; a later line sets it"))
            continue
        seen.add(name)
        if name == dc.ROLE_ENV and _line_value(match.group(4)).lower() != role.value:
            changes.append(Change("updated", name, f"set to {role.value}"))
            line = f"{dc.ROLE_ENV}={role.value}"
        elif name in supplied and _line_value(match.group(4)) != supplied[name]:
            changes.append(Change("updated", name, "value supplied on the command line"))
            line = f"{line[: match.start(3)]}{name}={_format_value(supplied[name])}"
        kept.append(line)

    additions: list[str] = []
    if dc.ROLE_ENV not in seen:
        additions.append(f"{dc.ROLE_ENV}={role.value}")
        changes.append(Change("added", dc.ROLE_ENV, f"selects the {role.value} role"))
    for name in REQUIRED[role]:
        if name in seen:
            continue
        seen.add(name)
        value = supplied.get(name, "")
        if not value:
            additions.append(f"# Required: {_REQUIRED_HINTS[name]}.")
        additions.append(f"{name}={_format_value(value)}")
        changes.append(
            Change("added", name, "required" + ("" if value else "; fill in the placeholder"))
        )
    for name, value in supplied.items():
        if name not in seen:
            seen.add(name)
            additions.append(f"{name}={_format_value(value)}")
            changes.append(Change("added", name, "value supplied on the command line"))
    for setting in dc.settings_for_role(role):
        if dc.Role.STANDALONE in setting.roles or not setting.default or setting.name in seen:
            continue
        seen.add(setting.name)
        additions.append(f"{setting.name}={setting.default}")
        changes.append(Change("added", setting.name, f"{role.value} default"))

    if additions:
        while kept and not kept[-1].strip():
            kept.pop()
        if kept:
            kept.append("")
        kept.append(ADDED_HEADER.format(role=role.value))
        kept.extend(additions)
    text = "\n".join(kept) + ("\n" if kept else "")

    env = _parse_text(text)
    report = dc.validate(role, env, check_files=check_files)
    secret_names = frozenset(extra_secret_names)
    before = _parse_text(original)
    values = {
        str(v).strip()
        for source in (before, env, supplied)
        for k, v in source.items()
        if _is_secret_name(k, secret_names) and len(str(v).strip()) >= 8
    }
    return Conversion(
        path=path,
        role=role,
        original=original,
        text=text,
        changes=changes,
        report=report,
        secret_names=secret_names,
        secret_values=tuple(sorted(values, key=len, reverse=True)),
    )


def backup_path(path: Path, now: float | None = None) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time() if now is None else now))
    candidate = path.with_name(f"{path.name}.bak-{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak-{stamp}-{counter}")
        counter += 1
    return candidate


def write(conversion: Conversion, *, backup: bool = True) -> Path | None:
    """Back up the original and atomically replace it; return the backup path.

    Returns ``None`` without touching anything when there is nothing to change.
    """

    if not conversion.changed:
        return None
    path = conversion.path
    if path.read_text(encoding="utf-8") != conversion.original:
        raise ConversionError(f"{path} changed since it was read; run the conversion again")
    mode = path.stat().st_mode & 0o777
    saved = None
    if backup:
        saved = backup_path(path)
        shutil.copy2(path, saved)
        os.chmod(saved, 0o600)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(conversion.text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return saved
