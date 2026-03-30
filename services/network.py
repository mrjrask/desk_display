# network.py

import datetime
import threading
import time
import subprocess
import socket
import logging
from typing import Optional

from config import (
    WIFI_CHECK_INTERVAL,
    WIFI_OFF_DURATION,
    FONT_DATE_SPORTS,
    FONT_DATE,
    FONT_TIME,
    FONT_TITLE_SPORTS,
)
from config import get_current_ssid  # your helper in config.py
from utils import clear_display, draw_text_centered, split_time_period
from PIL import Image, ImageDraw

class ConnectivityMonitor:
    """
    Background thread that keeps track of:
      - no_wifi
      - no_internet
      - online
    and automatically toggles the radio on extended outages.
    """
    def __init__(self, display):
        self.display = display
        self.state   = None
        self.lock    = threading.Lock()
        self.last_connected_at: Optional[datetime.datetime] = None
        logging.info("🔌 Starting Wi-Fi monitor…")
        threading.Thread(target=self._loop, daemon=True).start()

    def _check_internet(self):
        try:
            # quick TCP connect to one of our domains
            sock = socket.create_connection(("weatherkit.apple.com", 443), timeout=3)
            sock.close()
            return True
        except:
            return False

    def _loop(self):
        while True:
            ssid = get_current_ssid()
            if not ssid:
                new = "no_wifi"
            elif not self._check_internet():
                new = "no_internet"
            else:
                new = "online"

            with self.lock:
                if new == "online":
                    self.last_connected_at = datetime.datetime.now()
                if new != self.state:
                    self.state = new
                    if new == "no_wifi":
                        logging.warning("❌ No Wi-Fi connection detected.")
                    elif new == "no_internet":
                        logging.warning(f"❌ Wi-Fi ({ssid}) but no Internet.")
                        # cycle radio
                        try:
                            subprocess.run(
                                ["nmcli", "radio", "wifi", "off"],
                                check=False,
                                timeout=10,
                            )
                            time.sleep(WIFI_OFF_DURATION)
                            subprocess.run(
                                ["nmcli", "radio", "wifi", "on"],
                                check=False,
                                timeout=10,
                            )
                        except subprocess.TimeoutExpired:
                            logging.warning("Wi-Fi radio toggle timed out; will retry later.")
                        logging.info("🔌 Wi-Fi re-enabled; retrying…")
                    else:
                        logging.info(f"✅ Wi-Fi ({ssid}) and Internet OK.")
            time.sleep(WIFI_CHECK_INTERVAL)

    def get_state(self):
        with self.lock:
            return self.state

    def get_last_connected_at(self) -> Optional[datetime.datetime]:
        with self.lock:
            return self.last_connected_at


def _format_last_connected(last_connected_at: Optional[datetime.datetime]) -> Optional[str]:
    if not last_connected_at:
        return None
    date_str = last_connected_at.strftime("%a %-m/%-d")
    time_str, ampm = split_time_period(last_connected_at.time())
    return f"Last connected: {date_str} {time_str} {ampm}"


def show_no_wifi_screen(display, last_connected_at: Optional[datetime.datetime] = None):
    """
    Display a static 'No Wi-Fi' + date/time status.
    """
    clear_display(display)
    img = Image.new("RGB", (display.width, display.height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Status line
    last_connected_text = _format_last_connected(last_connected_at)
    draw_text_centered(
        draw,
        "No Wi-Fi.",
        FONT_TITLE_SPORTS,
        y_offset=-28 if last_connected_text else -16,
    )

    # Date line
    now = time.localtime()
    date_str = time.strftime("%a %-m/%-d", now)
    draw_text_centered(draw, date_str, FONT_DATE_SPORTS, y_offset=-6 if last_connected_text else 0)

    # Time line
    t, ampm = split_time_period(datetime.datetime.now().time())
    draw_text_centered(draw, f"{t} {ampm}", FONT_TIME, y_offset=18 if last_connected_text else 24)

    if last_connected_text:
        draw_text_centered(draw, last_connected_text, FONT_DATE, y_offset=48)

    display.image(img)
    display.show()


def show_wifi_no_internet_screen(
    display,
    ssid,
    last_connected_at: Optional[datetime.datetime] = None,
):
    """
    Display 'Wi-Fi connected.' / SSID / 'No Internet.'
    """
    clear_display(display)
    img = Image.new("RGB", (display.width, display.height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    last_connected_text = _format_last_connected(last_connected_at)
    draw_text_centered(
        draw,
        "Wi-Fi connected.",
        FONT_TITLE_SPORTS,
        y_offset=-30 if last_connected_text else -24,
    )
    draw_text_centered(draw, ssid, FONT_DATE_SPORTS, y_offset=-8 if last_connected_text else 0)
    draw_text_centered(
        draw,
        "No Internet.",
        FONT_DATE_SPORTS,
        y_offset=14 if last_connected_text else 24,
    )
    if last_connected_text:
        draw_text_centered(draw, last_connected_text, FONT_DATE, y_offset=40)

    display.image(img)
    display.show()
