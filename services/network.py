# network.py

import datetime
import threading
import time
import subprocess
import socket
import logging

from config import (
    WIFI_CHECK_INTERVAL,
    WIFI_OFF_DURATION,
    FONT_DATE_SPORTS,
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
        logging.info("🔌 Starting Wi-Fi monitor…")
        threading.Thread(target=self._loop, daemon=True).start()

    def _check_internet(self):
        try:
            # quick TCP connect to one of our domains
            sock = socket.create_connection(("api.openweathermap.org", 443), timeout=3)
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
                if new != self.state:
                    self.state = new
                    if new == "no_wifi":
                        logging.warning("❌ No Wi-Fi connection detected.")
                    elif new == "no_internet":
                        logging.warning(f"❌ Wi-Fi ({ssid}) but no Internet.")
                        # cycle radio
                        subprocess.call(["nmcli","radio","wifi","off"])
                        time.sleep(WIFI_OFF_DURATION)
                        subprocess.call(["nmcli","radio","wifi","on"])
                        logging.info("🔌 Wi-Fi re-enabled; retrying…")
                    else:
                        logging.info(f"✅ Wi-Fi ({ssid}) and Internet OK.")
            time.sleep(WIFI_CHECK_INTERVAL)

    def get_state(self):
        with self.lock:
            return self.state


def show_no_wifi_screen(display):
    """
    Display a static 'No Wi-Fi' + date/time status.
    """
    clear_display(display)
    img = Image.new("RGB", (display.width, display.height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Status line
    draw_text_centered(draw, "No Wi-Fi.", FONT_TITLE_SPORTS, y_offset=-16)

    # Date line
    now = time.localtime()
    date_str = time.strftime("%a %-m/%-d", now)
    draw_text_centered(draw, date_str, FONT_DATE_SPORTS, y_offset=0)

    # Time line
    t, ampm = split_time_period(datetime.datetime.now().time())
    draw_text_centered(draw, f"{t} {ampm}", FONT_TIME, y_offset=24)

    display.image(img)
    display.show()


def show_wifi_no_internet_screen(display, ssid):
    """
    Display 'Wi-Fi connected.' / SSID / 'No Internet.'
    """
    clear_display(display)
    img = Image.new("RGB", (display.width, display.height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw_text_centered(draw, "Wi-Fi connected.", FONT_TITLE_SPORTS, y_offset=-24)
    draw_text_centered(draw, ssid,                FONT_DATE_SPORTS,  y_offset=0)
    draw_text_centered(draw, "No Internet.",      FONT_DATE_SPORTS,  y_offset=24)

    display.image(img)
    display.show()
