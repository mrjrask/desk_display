# Vendored sensor drivers

Desk Display vendors a small set of optional Pimoroni-compatible sensor drivers so Raspberry Pi installs can use editable local packages without depending on upstream repositories being reachable at install time.

These packages are installed only by `requirements/sensors-pimoroni.txt` when `INSIDE_SENSOR` is configured for a Pimoroni BME280, BME680, or BME68x/BME688 sensor.

## Included packages

| Path | Package | Source | Vendored version | Why it is vendored |
| --- | --- | --- | --- | --- |
| `vendor/pimoroni-bme280` | `pimoroni-bme280` | <https://github.com/pimoroni/bme280-python> | `1.0.0` | Optional BME280 temperature, pressure, and humidity support. |
| `vendor/pimoroni-bme680` | `bme680` | <https://github.com/pimoroni/bme680-python> | `2.0.0` | Optional BME680 temperature, pressure, humidity, and gas-resistance support. |
| `vendor/bme68x` | `bme68x` | <https://github.com/pi3g/bme68x-python-library> | `1.0.4` | Optional BME68x/BME688 helper support for direct SMBus reads. |

## Trimming policy

Vendored packages should contain only files needed for Desk Display runtime and editable installation:

- runtime package or extension source code;
- license files;
- packaging metadata required by `pip install -e`;
- concise README notes needed to identify upstream usage and install context.

Do not keep upstream examples, tests, CI configuration, development-only requirements, Makefiles, check scripts, installer scripts, uninstaller scripts, caches, build artifacts, or generated distributions unless Desk Display directly uses them.

When refreshing a vendored package, start from the upstream release or commit, re-apply this trimming policy, confirm `requirements/sensors-pimoroni.txt` still points at the editable package paths, and update the table above with the source version.
