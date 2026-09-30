"""The display locations Jason asked for on 2026-09-30, applied once.

The render server applies them to every display it knows when it starts
after this update (see :func:`apply`).  Hyper is at 41.9037, -87.6357 and
every other display at 42.1373, -87.8446.  A display that already has a
location keeps it, and edits on the Clients page afterwards stick.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from remote_display.locations import Location

SEED_ID = "display-locations-2026-09-30"
DEFAULT_LOCATION = Location(42.1373, -87.8446)
HYPER_LOCATION = Location(41.9037, -87.6357)


def location_for(client_id: str) -> Location:
    """Hyper's location for the hyper display (``hyper`` or ``hyper-…``), else the default."""

    name = client_id.lower()
    return HYPER_LOCATION if name == "hyper" or name.startswith("hyper-") else DEFAULT_LOCATION


def known_clients(store: Any, registry_snapshot: dict[str, Any], provisioned: Iterable[str] = (),
                  static: Iterable[str] = ()) -> set[str]:
    """Every display the server knows of, as the Clients page lists them."""

    data = store.snapshot()
    return {*registry_snapshot.get("clients", {}), *provisioned, *static,
            *data["assignments"], *data["clients"]}


def apply(store: Any, client_ids: Iterable[str]) -> list[str]:
    """Apply the seed to *client_ids* once; returns the displays that changed."""

    return store.apply_location_seed(SEED_ID, {cid: location_for(cid) for cid in client_ids},
                                     actor="location-seed")


__all__ = ["DEFAULT_LOCATION", "HYPER_LOCATION", "SEED_ID", "apply", "known_clients", "location_for"]
