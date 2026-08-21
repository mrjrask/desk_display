#!/usr/bin/env python3
"""Background collector for the "adsb stats" screen.

Polls each receiver configured via ``ADSB_DEVICE_1_HOST``/``ADSB_DEVICE_2_HOST``
on an interval and writes sightings/status into the local SQLite database that
``screens/draw_adsb_stats.py`` reads from (see ``services/adsb.py``). This is
meant to run as its own long-lived process (systemd unit installed by
``Installers/install_adsb_collector_service.sh``), fully decoupled from the
display process: a receiver outage only affects this loop, never rendering.

Usage:
    python scripts/adsb_collector.py            # run forever
    python scripts/adsb_collector.py --once     # poll a single cycle and exit
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from services import aircraft_type_db
from services.adsb import AdsbDevice, AdsbStore, poll_device, today_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("adsb_collector")


def configured_devices() -> list[AdsbDevice]:
    return [AdsbDevice(host=d["host"], label=d["label"]) for d in config.ADSB_DEVICES]


def run_once(store: AdsbStore, devices: list[AdsbDevice]) -> None:
    """Poll every configured device once and persist the results."""

    now = time.time()
    day = today_key(config.CENTRAL_TIME, now=now)
    for device in devices:
        result = poll_device(
            device,
            home_lat=config.ADSB_HOME_LATITUDE,
            home_lon=config.ADSB_HOME_LONGITUDE,
            unit=config.ADSB_DISTANCE_UNIT,
            timeout=config.ADSB_REQUEST_TIMEOUT_SECONDS,
        )
        if not result.ok:
            logger.warning("%s (%s): %s", device.label, device.host, result.error)
        else:
            logger.debug("%s: %d aircraft this poll", device.label, len(result.sightings))
        store.record_poll(result, now=now, day=day)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once", action="store_true", help="Poll a single cycle and exit instead of looping."
    )
    args = parser.parse_args(argv)

    devices = configured_devices()
    if not devices:
        logger.error(
            "No ADS-B receivers configured. Set ADSB_DEVICE_1_HOST (and optionally "
            "ADSB_DEVICE_2_HOST) in .env, then restart this service."
        )
        return 1

    store = AdsbStore()
    logger.info(
        "Polling %d receiver(s) every %.0fs -> %s",
        len(devices),
        config.ADSB_POLL_INTERVAL_SECONDS,
        store.db_path,
    )

    try:
        aircraft_type_db.refresh()
    except Exception:  # pragma: no cover - defensive; startup must not fail on this
        logger.exception("ADS-B: initial aircraft type DB refresh failed")

    last_prune_day: Optional[dt.date] = None
    last_type_db_check_day: Optional[dt.date] = None
    try:
        while True:
            try:
                run_once(store, devices)
                today = dt.datetime.now(tz=config.CENTRAL_TIME).date()
                if today != last_prune_day:
                    removed = store.prune(config.ADSB_RETENTION_DAYS, now_day=today)
                    if removed:
                        logger.info(
                            "Pruned %d sighting row(s) older than %d day(s).",
                            removed,
                            config.ADSB_RETENTION_DAYS,
                        )
                    last_prune_day = today
                if today != last_type_db_check_day:
                    aircraft_type_db.refresh()
                    last_type_db_check_day = today
            except Exception:  # pragma: no cover - defensive loop guard
                logger.exception("Unhandled error during ADS-B poll cycle")

            if args.once:
                return 0
            time.sleep(config.ADSB_POLL_INTERVAL_SECONDS)
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
