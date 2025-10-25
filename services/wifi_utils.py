# wifi_utils.py

import threading
import time
import logging
import subprocess
import requests

CHECK_INTERVAL   = 15    # seconds between checks
INTERNET_TIMEOUT = 3     # seconds for each HTTP HEAD

PRIMARY_CHECK_URL   = "https://api.openweathermap.org"
SECONDARY_CHECK_URL = "https://www.google.com"

# Shared state
wifi_status = "no_wifi"    # one of "no_wifi", "no_internet", "ok"
current_ssid = None
_STATE_LOCK = threading.Lock()


def _get_wireless_interfaces():
    """
    Run 'iw dev' and return a list of wireless interface names.
    """
    try:
        out = subprocess.check_output(["iw", "dev"], stderr=subprocess.DEVNULL).decode()
        # lines like: "Interface wlan0"
        return [line.split()[1] for line in out.splitlines() if line.strip().startswith("Interface")]
    except Exception as e:
        logging.debug(f"iw dev failed: {e}")
        return []


def _get_ssid():
    """
    Try, in order:
      1) nmcli
      2) iw dev <iface> link   (for each wireless iface)
      3) iwgetid -r
    """
    # 1) nmcli
    try:
        out = subprocess.check_output(
            ["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"],
            stderr=subprocess.DEVNULL
        ).decode().splitlines()
        for line in out:
            active, ssid = line.split(":", 1)
            if active == "yes" and ssid:
                logging.debug(f"_get_ssid: nmcli reports SSID='{ssid}'")
                return ssid
    except Exception as e:
        logging.debug(f"_get_ssid: nmcli failed: {e}")

    # 2) iw dev ... link
    ifaces = _get_wireless_interfaces()
    for iface in ifaces:
        try:
            out = subprocess.check_output(
                ["iw", "dev", iface, "link"],
                stderr=subprocess.DEVNULL
            ).decode().splitlines()
            for ln in out:
                ln = ln.strip()
                if ln.startswith("SSID:"):
                    ssid = ln.split("SSID:")[1].strip()
                    logging.debug(f"_get_ssid: iw dev {iface} link → SSID='{ssid}'")
                    return ssid
        except Exception as e:
            logging.debug(f"_get_ssid: iw link on {iface} failed: {e}")

    # 3) iwgetid
    try:
        out = subprocess.check_output(["iwgetid", "-r"], stderr=subprocess.DEVNULL).decode().strip()
        if out:
            logging.debug(f"_get_ssid: iwgetid reports SSID='{out}'")
            return out
    except Exception as e:
        logging.debug(f"_get_ssid: iwgetid failed: {e}")

    logging.debug("_get_ssid: no SSID found")
    return None


def _check_internet():
    """
    HTTP HEAD against our primary API, fallback to Google.
    """
    for url in (PRIMARY_CHECK_URL, SECONDARY_CHECK_URL):
        try:
            requests.head(url, timeout=INTERNET_TIMEOUT)
            logging.debug(f"_check_internet: {url} reachable")
            return True
        except Exception as e:
            logging.debug(f"_check_internet: {url} failed: {e}")
    return False


def _monitor_loop():
    """
    Continuously update `wifi_status` to one of:
      – "ok"          (internet reachable)
      – "no_internet" (we’re on WLAN but no Internet)
      – "no_wifi"     (not associated to any SSID)
    """
    global wifi_status, current_ssid
    logging.info("🔌 Wi-Fi monitor thread started")
    last = None

    while True:
        # see if we can reach the Internet
        internet = _check_internet()
        # then try to detect SSID
        ssid = _get_ssid()

        if internet:
            state = "ok"
        elif ssid:
            state = "no_internet"
        else:
            state = "no_wifi"

        with _STATE_LOCK:
            wifi_status = state
            current_ssid = ssid

        if state != last:
            if state == "no_wifi":
                logging.warning("❌ No Wi-Fi connection detected.")
            elif state == "no_internet":
                logging.warning(f"⚠️  Wi-Fi '{ssid}' up but no Internet.")
            else:
                logging.info(f"✅ Wi-Fi '{ssid}' and Internet OK.")
            last = state

        time.sleep(CHECK_INTERVAL)


def start_monitor():
    """
    Start the background Wi-Fi monitor.
    """
    t = threading.Thread(target=_monitor_loop, daemon=True)
    t.start()


def get_wifi_state():
    """Return a tuple of (wifi_status, current_ssid) while holding the shared lock."""
    with _STATE_LOCK:
        return wifi_status, current_ssid
