#!/usr/bin/env bash
set -euo pipefail

AIRPLAY_HELPER_LOADED=1

log_info() { printf '[AIRPLAY][INFO] %s\n' "$*"; }
log_warn() { printf '[AIRPLAY][WARN] %s\n' "$*"; }
log_error() { printf '[AIRPLAY][ERROR] %s\n' "$*" >&2; }

init_sudo() {
  if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    SUDO="sudo"
  else
    SUDO=""
  fi
}

resolve_project_dir() {
  local caller_script_dir="$1"
  if [[ -n "${PROJECT_DIR:-}" ]]; then
    printf '%s\n' "$PROJECT_DIR"
    return 0
  fi

  local candidate
  for candidate in \
    "$caller_script_dir/.." \
    "$caller_script_dir/../.." \
    "$(pwd)"; do
    if [[ -f "$candidate/main.py" && -d "$candidate/scripts" ]]; then
      printf '%s\n' "$(cd -- "$candidate" && pwd)"
      return 0
    fi
  done

  printf '%s\n' "$(cd -- "$caller_script_dir/.." && pwd)"
}

load_env_file() {
  local env_file="$1"
  [[ -f "$env_file" ]] || return 0

  local raw_line line key value
  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line#"${raw_line%%[![:space:]]*}"}"
    [[ -z "$line" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue
    [[ "$line" == *=* ]] || continue

    key="${line%%=*}"
    value="${line#*=}"

    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"

    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue

    if [[ ${#value} -ge 2 ]]; then
      if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
        value="${value:1:${#value}-2}"
      fi
    fi

    export "$key=$value"
  done < "$env_file"
}

ensure_executable() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  chmod +x "$path"
}

service_user_home() {
  local user_name="$1"
  local resolved
  resolved=$(getent passwd "$user_name" | cut -d: -f6)
  if [[ -n "$resolved" ]]; then
    printf '%s\n' "$resolved"
  else
    printf '/home/%s\n' "$user_name"
  fi
}

detect_display_resolution() {
  local modes_path mode
  for modes_path in /sys/class/drm/card*-*/modes; do
    [[ -r "$modes_path" ]] || continue
    read -r mode < "$modes_path" || true
    if [[ "$mode" =~ ^[0-9]+x[0-9]+$ ]]; then
      printf '%s\n' "$mode"
      return 0
    fi
  done

  if command -v fbset >/dev/null 2>&1; then
    local fb_mode
    fb_mode=$(fbset -s 2>/dev/null | awk '/geometry/ {print $2"x"$3; exit}')
    if [[ "$fb_mode" =~ ^[0-9]+x[0-9]+$ ]]; then
      printf '%s\n' "$fb_mode"
      return 0
    fi
  fi

  printf '%s\n' "${AIRPLAY_RESOLUTION_DEFAULT:-800x480}"
}

systemctl_safe() {
  if ! command -v systemctl >/dev/null 2>&1; then
    return 1
  fi

  if [[ -n "${SUDO:-}" ]]; then
    $SUDO systemctl "$@"
  else
    systemctl "$@"
  fi
}

user_systemctl_safe() {
  local user_name="$1"
  shift
  if ! command -v systemctl >/dev/null 2>&1; then
    return 1
  fi

  local uid
  uid=$(id -u "$user_name" 2>/dev/null || true)
  [[ -n "$uid" ]] || return 1

  if [[ -n "${SUDO:-}" ]]; then
    $SUDO -u "$user_name" XDG_RUNTIME_DIR="/run/user/$uid" systemctl --user "$@"
  else
    XDG_RUNTIME_DIR="/run/user/$uid" systemctl --user "$@"
  fi
}

write_file_as_root() {
  local dest="$1"
  local content="$2"
  if [[ -n "${SUDO:-}" ]]; then
    printf '%s\n' "$content" | $SUDO tee "$dest" >/dev/null
  else
    printf '%s\n' "$content" > "$dest"
  fi
}
