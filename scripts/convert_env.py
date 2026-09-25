#!/usr/bin/env python3
"""Convert an existing Desk Display .env into a server or client configuration.

Settings the target role does not read are removed, the role's required and
role-only settings are added, and comments and still-valid values are kept.
Every removal and addition is reported; secret values are never printed.
The original is backed up before an atomic write, and a rerun changes nothing.

Usage:
    python3 scripts/convert_env.py --role server --dry-run
    python3 scripts/convert_env.py --role server
    python3 scripts/convert_env.py --role client --credentials office.env.client
    python3 scripts/convert_env.py --role client --server-url https://render.lan:8765 \\
        --client-id office --profile hyperpixel4_square --token-file token.txt

A client's credential comes from the server's "Add a display" form, as a
downloaded .env.client (--credentials) or the bare token (--token-file). It is
never taken on the command line, where it would land in shell history.
The result must pass the role's startup checks before it is written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import deployment_config as dc  # noqa: E402
import env_conversion  # noqa: E402

_CREDENTIAL_KEYS = (
    "DESK_DISPLAY_SERVER_URL",
    "DESK_DISPLAY_CLIENT_ID",
    "DESK_DISPLAY_CLIENT_TOKEN",
    "DESK_DISPLAY_PROFILE",
)


def _supplied(args: argparse.Namespace) -> dict[str, str]:
    values: dict[str, str] = {}
    if args.credentials:
        provisioned = dc.parse_env_file(args.credentials)
        values.update({k: provisioned[k] for k in _CREDENTIAL_KEYS if provisioned.get(k)})
    if args.token_file:
        token = args.token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise env_conversion.ConversionError(f"{args.token_file} is empty")
        values["DESK_DISPLAY_CLIENT_TOKEN"] = token
    for name, value in (
        ("DESK_DISPLAY_SERVER_URL", args.server_url),
        ("DESK_DISPLAY_CLIENT_ID", args.client_id),
        ("DESK_DISPLAY_PROFILE", args.profile),
    ):
        if value:
            values[name] = value
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--role", required=True, choices=[r.value for r in env_conversion.TARGET_ROLES]
    )
    parser.add_argument(
        "--env-file", type=Path, default=REPO_ROOT / ".env", help="file to convert (default: .env)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the change with secrets redacted; write nothing",
    )
    parser.add_argument(
        "--credentials", type=Path, help="client: a provisioned .env.client from the server"
    )
    parser.add_argument(
        "--token-file", type=Path, help="client: a file holding only the client credential"
    )
    parser.add_argument("--server-url", help="client: the render server's base URL")
    parser.add_argument("--client-id", help="client: this display's ID")
    parser.add_argument("--profile", help="client: this display's profile")
    parser.add_argument(
        "--keep-unknown",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="keep lines that are not Desk Display settings (default: server yes, client no)",
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="write even when the result fails the startup checks (placeholders left to fill)",
    )
    args = parser.parse_args(argv)
    role = dc.Role(args.role)

    try:
        supplied = _supplied(args)
        if role is dc.Role.SERVER and supplied:
            raise env_conversion.ConversionError("client options apply only with --role client")
        conversion = env_conversion.convert(
            args.env_file, role, supplied, keep_unknown=args.keep_unknown
        )
    except (env_conversion.ConversionError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not conversion.changed:
        print(f"{args.env_file} is already a {role.value} configuration; nothing to change.")
        print(conversion.render_report())
        return 0 if conversion.report.ok else 1
    if args.dry_run:
        print(conversion.render_diff(), end="")
    print(conversion.render_report())
    if args.dry_run:
        print("Dry run; nothing was written.")
        return 0 if conversion.report.ok else 1
    if not conversion.report.ok and not args.allow_invalid:
        print(
            f"Not written: the {role.value} startup checks above fail. Supply the missing values, "
            "or rerun with --allow-invalid to write placeholders and fill them in by hand."
        )
        return 1
    try:
        saved = env_conversion.write(conversion)
    except (env_conversion.ConversionError, OSError) as exc:
        print(f"error: {exc}; {args.env_file} was left unchanged", file=sys.stderr)
        return 2
    print(f"Wrote {args.env_file} for the {role.value} role; the original is at {saved}.")
    return 0 if conversion.report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
