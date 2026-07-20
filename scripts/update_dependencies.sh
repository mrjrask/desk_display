#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
COMMON_SCRIPT="$PROJECT_DIR/scripts/helpers/common.sh"
PYTHON_BIN="${PYTHON:-python3}"
REQUIREMENTS_FILE_OVERRIDE="${REQUIREMENTS_FILE:-}"
OUTPUT_MODE="${DESK_DISPLAY_OUTPUT:-}"
INCLUDE_VENDOR_REQUIREMENTS="${INCLUDE_VENDOR_REQUIREMENTS:-0}"
INCLUDE_PLATFORM_SPECIFIC_REQUIREMENTS="${INCLUDE_PLATFORM_SPECIFIC_REQUIREMENTS:-0}"
ADAFRUIT_SENSOR_REQUIREMENTS_FILE="${ADAFRUIT_SENSOR_REQUIREMENTS_FILE:-requirements/sensors-adafruit.txt}"
PIMORONI_SENSOR_REQUIREMENTS_FILE="${PIMORONI_SENSOR_REQUIREMENTS_FILE:-requirements/sensors-pimoroni.txt}"

if [[ ! -f "$COMMON_SCRIPT" ]]; then
  echo "[ERROR] Missing helper script: $COMMON_SCRIPT" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$COMMON_SCRIPT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --requirements)
      REQUIREMENTS_FILE_OVERRIDE="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_MODE="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --include-vendor)
      INCLUDE_VENDOR_REQUIREMENTS=1
      shift
      ;;
    --include-platform-specific)
      INCLUDE_PLATFORM_SPECIFIC_REQUIREMENTS=1
      shift
      ;;
    *)
      echo "Usage: $0 [--requirements <file>] [--output <displayhatmini|minipitft|kernel|framebuffer>] [--python <python-bin>] [--include-vendor] [--include-platform-specific]" >&2
      exit 1
      ;;
  esac
done

pick_requirements_file() {
  if [[ -n "$REQUIREMENTS_FILE_OVERRIDE" ]]; then
    echo "$REQUIREMENTS_FILE_OVERRIDE"
    return 0
  fi

  local normalized_output_mode
  normalized_output_mode=$(printf '%s' "$OUTPUT_MODE" | tr '[:upper:]' '[:lower:]')

  case "$normalized_output_mode" in
    kernel)
      echo "requirements/kernel.txt"
      ;;
    framebuffer)
      echo "requirements/framebuffer.txt"
      ;;
    minipitft)
      echo "requirements/minipitft.txt"
      ;;
    *)
      echo "requirements/displayhatmini.txt"
      ;;
  esac
}

REQUIREMENTS_FILE=$(pick_requirements_file)

normalize_sensor_name() {
  printf '%s' "$1" | tr '[:upper:]- ' '[:lower:]__'
}

read_env_file_value() {
  local env_path="$1"
  local key="$2"

  if [[ ! -f "$env_path" ]]; then
    return 1
  fi

  awk -v key="$key" '
    /^[[:space:]]*(#|$)/ { next }
    {
      line=$0
      sub(/^[[:space:]]*export[[:space:]]+/, "", line)
      if (line !~ "^[[:space:]]*" key "[[:space:]]*=") {
        next
      }
      sub("^[[:space:]]*" key "[[:space:]]*=[[:space:]]*", "", line)
      sub(/[[:space:]]+#.*$/, "", line)
      sub(/^[[:space:]]+|[[:space:]]+$/, "", line)
      if ((line ~ /^".*"$/) || (line ~ /^\047.*\047$/)) {
        line=substr(line, 2, length(line)-2)
      }
      print line
      exit 0
    }
  ' "$env_path"
}

configured_inside_sensor() {
  if [[ -n "${INSIDE_SENSOR:-}" ]]; then
    printf '%s' "$INSIDE_SENSOR"
    return 0
  fi
  if [[ -n "${INDOOR_SENSOR:-}" ]]; then
    printf '%s' "$INDOOR_SENSOR"
    return 0
  fi

  local env_path="$PROJECT_DIR/.env"
  local value
  value=$(read_env_file_value "$env_path" "INSIDE_SENSOR" || true)
  if [[ -n "$value" ]]; then
    printf '%s' "$value"
    return 0
  fi
  value=$(read_env_file_value "$env_path" "INDOOR_SENSOR" || true)
  if [[ -n "$value" ]]; then
    printf '%s' "$value"
    return 0
  fi

  return 1
}

should_install_adafruit_sensor_requirements() {
  local sensor
  sensor=$(configured_inside_sensor || true)
  if [[ -z "$sensor" ]]; then
    return 1
  fi

  case "$(normalize_sensor_name "$sensor")" in
    adafruit_bme280|adafruit_bme680|adafruit_sht4x|adafruit_sht41)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

should_install_pimoroni_sensor_requirements() {
  local sensor
  sensor=$(configured_inside_sensor || true)
  if [[ -z "$sensor" ]]; then
    return 1
  fi

  case "$(normalize_sensor_name "$sensor")" in
    pimoroni_bme280|pimoroni_bme680|pimoroni_bme68x|pimoroni_bme688)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

cleanup_stale_egg_info() {
  local vendor_dir="$PROJECT_DIR/vendor"
  local had_permission_errors=0

  if [[ ! -d "$vendor_dir" ]]; then
    return 0
  fi

  while IFS= read -r -d '' egg_info_dir; do
    warn "Removing stale metadata directory: ${egg_info_dir#$PROJECT_DIR/}"
    if ! rm -rf "$egg_info_dir" 2>/dev/null; then
      had_permission_errors=1
      warn "Unable to remove ${egg_info_dir#$PROJECT_DIR/}; this usually means ownership/permissions are incorrect."
    fi
  done < <(find "$vendor_dir" -mindepth 2 -maxdepth 2 -type d -name '*.egg-info' -print0)

  if [[ "$had_permission_errors" -ne 0 ]]; then
    echo "[ERROR] One or more *.egg-info directories under vendor/ could not be removed." >&2
    echo "[ERROR] Editable installs will fail until permissions are fixed." >&2
    echo "[ERROR] Try: sudo chown -R \"$(id -un):$(id -gn)\" \"$PROJECT_DIR/vendor\" && rerun this script." >&2
    return 1
  fi
}

VENV_DIR="$PROJECT_DIR/venv"
EXISTING_VENV=$(detect_existing_venv "$PROJECT_DIR" || true)
if [[ -n "$EXISTING_VENV" ]]; then
  VENV_DIR="$EXISTING_VENV"
  log "Found existing virtual environment at $VENV_DIR"
fi

if [[ ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
  if [[ -d "$VENV_DIR" ]]; then
    warn "$VENV_DIR exists but does not look like a virtual environment. Recreating."
  fi
  log "Creating virtual environment with $PYTHON_BIN at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  log "Virtual environment already exists at $VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

check_venv_permissions() {
  local current_user current_group
  current_user=$(id -un)
  current_group=$(id -gn)

  # pip upgrades can fail mid-uninstall when these files are not writable
  # (for example after previously running the script with sudo).
  local required_paths=(
    "$VENV_DIR/bin"
    "$VENV_DIR/bin/pip"
    "$VENV_DIR/lib"
  )

  local path
  for path in "${required_paths[@]}"; do
    if [[ -e "$path" && ! -w "$path" ]]; then
      echo "[ERROR] Virtual environment is not writable: $path" >&2
      echo "[ERROR] This usually means the venv is owned by another user (often root)." >&2
      echo "[ERROR] Fix ownership, then rerun:" >&2
      echo "[ERROR]   sudo chown -R \"$current_user:$current_group\" \"$VENV_DIR\"" >&2
      return 1
    fi
  done
}

check_venv_permissions

log "Upgrading pip"
pip install --upgrade pip

build_install_requirements_file() {
  local source_requirements="$1"
  local install_requirements="$2"
  local host_os
  host_os=$(uname -s)

  SOURCE_REQUIREMENTS="$source_requirements" \
  INSTALL_REQUIREMENTS="$install_requirements" \
  INCLUDE_VENDOR_REQUIREMENTS="$INCLUDE_VENDOR_REQUIREMENTS" \
  INCLUDE_PLATFORM_SPECIFIC_REQUIREMENTS="$INCLUDE_PLATFORM_SPECIFIC_REQUIREMENTS" \
  HOST_OS="$host_os" \
  python3 - <<'PYREQ'
import os
from pathlib import Path

source = Path(os.environ["SOURCE_REQUIREMENTS"]).resolve()
output = Path(os.environ["INSTALL_REQUIREMENTS"])
include_vendor = os.environ.get("INCLUDE_VENDOR_REQUIREMENTS") == "1"
include_platform = os.environ.get("INCLUDE_PLATFORM_SPECIFIC_REQUIREMENTS") == "1"
host_os = os.environ.get("HOST_OS", "")
linux_only = {"spidev", "smbus", "lgpio", "rpi.gpio", "gpiozero", "displayhatmini"}
seen = set()
lines = []


def requirement_name(line):
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "-")) or stripped.startswith((".", "/", "git+", "http://", "https://")):
        return ""
    name = stripped.split(";", 1)[0].strip()
    for marker in ("===", ">=", "<=", "!=", "~=", "==", ">", "<"):
        name = name.split(marker, 1)[0]
    name = name.split("[", 1)[0]
    return name.strip().lower()


def include_target(line, base_dir):
    parts = line.split()
    if len(parts) >= 2 and parts[0] in {"-r", "--requirement"}:
        return (base_dir / parts[1]).resolve()
    if line.startswith("--requirement="):
        return (base_dir / line.split("=", 1)[1]).resolve()
    return None


def expand(path):
    path = path.resolve()
    if path in seen:
        return
    seen.add(path)
    base_dir = path.parent
    lines.append(f"# Expanded from {path}")
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        target = include_target(stripped, base_dir)
        if target is not None:
            expand(target)
            continue
        if not include_vendor and (stripped.startswith("-e ./vendor/") or stripped.startswith("./vendor/") or stripped.startswith("-e ../vendor/") or stripped.startswith("../vendor/") or "/vendor/" in stripped):
            continue
        if host_os == "Darwin" and not include_platform and requirement_name(stripped) in linux_only:
            continue
        lines.append(raw)


expand(source)
output.write_text("\n".join(lines) + "\n")
PYREQ

  if [[ "$INCLUDE_VENDOR_REQUIREMENTS" != "1" ]]; then
    log "Skipping local vendor requirements (use --include-vendor to include them)"
  fi
  if [[ "$host_os" == "Darwin" && "$INCLUDE_PLATFORM_SPECIFIC_REQUIREMENTS" != "1" ]]; then
    log "macOS detected; skipping Linux-only hardware dependencies (use --include-platform-specific to include them)"
  fi
}

if [[ -f "$PROJECT_DIR/$REQUIREMENTS_FILE" ]]; then
  log "Installing Python dependencies from $REQUIREMENTS_FILE"
  if [[ "$INCLUDE_VENDOR_REQUIREMENTS" == "1" ]]; then
    cleanup_stale_egg_info
  fi
  INSTALL_REQUIREMENTS_FILE=$(mktemp)
  trap 'rm -f "$INSTALL_REQUIREMENTS_FILE"' EXIT
  build_install_requirements_file "$PROJECT_DIR/$REQUIREMENTS_FILE" "$INSTALL_REQUIREMENTS_FILE"
  pushd "$PROJECT_DIR" >/dev/null
  pip install -r "$INSTALL_REQUIREMENTS_FILE"
  popd >/dev/null
else
  warn "$REQUIREMENTS_FILE not found; skipping pip install."
fi

if should_install_adafruit_sensor_requirements; then
  if [[ -f "$PROJECT_DIR/$ADAFRUIT_SENSOR_REQUIREMENTS_FILE" ]]; then
    log "Installing optional Adafruit sensor dependencies from $ADAFRUIT_SENSOR_REQUIREMENTS_FILE"
    pushd "$PROJECT_DIR" >/dev/null
    pip install -r "$ADAFRUIT_SENSOR_REQUIREMENTS_FILE"
    popd >/dev/null
  else
    warn "$ADAFRUIT_SENSOR_REQUIREMENTS_FILE not found; skipping optional Adafruit sensor dependencies."
  fi
else
  log "Skipping optional Adafruit sensor dependencies (set INSIDE_SENSOR to adafruit_bme280, adafruit_bme680, or adafruit_sht4x to install them)."
fi

if should_install_pimoroni_sensor_requirements; then
  if [[ -f "$PROJECT_DIR/$PIMORONI_SENSOR_REQUIREMENTS_FILE" ]]; then
    log "Installing optional Pimoroni sensor dependencies from $PIMORONI_SENSOR_REQUIREMENTS_FILE"
    cleanup_stale_egg_info
    pushd "$PROJECT_DIR" >/dev/null
    pip install -r "$PIMORONI_SENSOR_REQUIREMENTS_FILE"
    popd >/dev/null
  else
    warn "$PIMORONI_SENSOR_REQUIREMENTS_FILE not found; skipping optional Pimoroni sensor dependencies."
  fi
else
  log "Skipping optional Pimoroni sensor dependencies (set INSIDE_SENSOR to pimoroni_bme280, pimoroni_bme680, or pimoroni_bme68x to install them)."
fi

deactivate

log "Dependency update complete."
