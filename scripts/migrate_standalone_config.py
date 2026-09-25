#!/usr/bin/env python3
"""Move a standalone Desk Display rotation onto the render server.

Reads the active standalone screen configuration (never changing it) and
turns it into a shared playlist on the server, optionally assigning it to
clients. Without --apply it only previews: every renamed, dropped or
remote-limited screen and every assignment conflict is listed first.

Usage:
    python3 scripts/migrate_standalone_config.py                     # preview
    python3 scripts/migrate_standalone_config.py --assign office,den # preview with clients
    python3 scripts/migrate_standalone_config.py --assign office --apply
    python3 scripts/migrate_standalone_config.py --export backup.json
    python3 scripts/migrate_standalone_config.py --rollback .runtime/server/migrations/standalone-....json

Rerunning is safe: an already-migrated configuration is reused, not
duplicated, and a client already playing another playlist is only
reassigned with --force-assign.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remote_display import migration  # noqa: E402
from remote_display.playlist_store import PlaylistStore, PlaylistStoreError, store_path  # noqa: E402


def _client_capabilities(clients: list[str]) -> dict[str, dict]:
    """Capabilities the server last saw for *clients*, for the preview's warnings."""

    from remote_display.playlist_store import registry_snapshot_path
    from remote_display.registry import read_snapshot

    known = read_snapshot(registry_snapshot_path())["clients"]
    return {c: known[c]["capabilities"] for c in clients if isinstance(known.get(c), dict)
            and known[c].get("capabilities")}


def _print_plan(plan: migration.Plan) -> None:
    report = plan.report()
    print(f"Source: {report['source']['config_path']} ({report['source']['kind']})")
    print(f"Playlist: {report['name']} [{report['revision']}], {len(report['screens'])} screens")
    if report["existing_playlist_id"]:
        print(f"Already on the server as {report['existing_playlist_id']}: it will be reused.")
    for client, step in report["assignments"].items():
        print(f"Client {client}: {step['action']}" + (f" (currently {step['current']})" if step["current"] else ""))
    if not report["changes"]:
        print("No semantic changes.")
    for change in report["changes"]:
        print(f"  [{change['severity']}] {change['message']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--store", type=Path, help="playlist store (default: DESK_DISPLAY_PLAYLIST_STORE_PATH)")
    parser.add_argument("--defaults", choices=sorted(migration.DEFAULT_BUNDLES),
                        help="default bundle to migrate when no rotation config exists")
    parser.add_argument("--name", help="playlist name")
    parser.add_argument("--assign", default="", help="comma-separated client IDs to assign it to")
    parser.add_argument("--force-assign", action="store_true", help="reassign clients that play another playlist")
    parser.add_argument("--install-style", action="store_true",
                        help="also write the migrated style and layouts as the server's (backups kept)")
    parser.add_argument("--apply", action="store_true", help="make the changes (default: preview only)")
    parser.add_argument("--export", type=Path, help="write the migration bundle (a backup) here, change nothing")
    parser.add_argument("--bundle-dir", type=Path, default=migration.DEFAULT_BUNDLE_DIR)
    parser.add_argument("--rollback", type=Path, help="undo an applied migration bundle")
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    args = parser.parse_args(argv)

    store = PlaylistStore(args.store or store_path())
    try:
        if args.rollback:
            result = migration.rollback(args.rollback, store)
            print(json.dumps(result, indent=2) if args.json else
                  f"Rolled back: restored {len(result['assignments'])} assignment(s), "
                  f"{'deleted' if result['deleted_playlist'] else 'kept'} the playlist, "
                  f"restored {len(result['files'])} file(s).")
            return 0
        clients = [c.strip() for c in args.assign.split(",") if c.strip()]
        plan = migration.plan(migration.load_source(defaults=args.defaults), store, clients=clients,
                              name=args.name, client_capabilities=_client_capabilities(clients))
        if args.json and not args.apply:
            print(json.dumps(plan.report(), indent=2))
        else:
            _print_plan(plan)
        if args.export:
            print(f"Wrote {migration.export_bundle(plan, args.export)}")
        if not args.apply:
            if not args.export:
                print("Preview only; rerun with --apply to make these changes.")
            return 1 if plan.blocked else 0
        result = migration.apply(plan, store, bundle_dir=args.bundle_dir, force_assign=args.force_assign,
                                 install_style=args.install_style)
        print(json.dumps(result, indent=2) if args.json else
              f"Applied. Playlist {result['playlist_id']} ({'created' if result['created'] else 'reused'}); "
              f"assigned {', '.join(result['assigned']) or 'no clients'}"
              + (f"; skipped {', '.join(result['skipped'])} (use --force-assign)" if result["skipped"] else "")
              + f". Bundle: {result['bundle']}")
        return 0
    except (migration.MigrationError, PlaylistStoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
