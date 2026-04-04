#!/usr/bin/env bash
set -euo pipefail

IFACE="${1:-wlan0}"

lease_candidates=(
  "/var/lib/NetworkManager/dnsmasq-${IFACE}.leases"
  "/var/lib/NetworkManager/dnsmasq-shared-${IFACE}.leases"
  "/var/lib/NetworkManager/internal-dnsmasq-${IFACE}.leases"
  "/var/lib/misc/dnsmasq.leases"
  "/var/lib/NetworkManager/*.leases"
  "/run/NetworkManager/*.leases"
  "/run/NetworkManager/dnsmasq*.leases"
)

print_header() {
  echo "=============================="
  echo " Hotspot Clients (${IFACE})"
  echo "=============================="
  echo
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

expand_candidates() {
  local pattern
  for pattern in "${lease_candidates[@]}"; do
    compgen -G "$pattern" || true
  done
}

declare -A lease_ip
declare -A lease_host
declare -A station_signal
declare -A neigh_ip
declare -A seen_macs

read_leases() {
  local found=0
  local lease_file
  while IFS= read -r lease_file; do
    [[ -f "$lease_file" ]] || continue
    found=1
    while read -r _expiry mac ip host _clientid; do
      [[ -n "${mac:-}" && -n "${ip:-}" ]] || continue
      mac="${mac,,}"
      lease_ip["$mac"]="$ip"
      [[ "$host" == "*" ]] || lease_host["$mac"]="$host"
      seen_macs["$mac"]=1
    done <"$lease_file"
  done < <(expand_candidates)

  if [[ "$found" -eq 0 ]]; then
    echo "[WARN] Lease file not found. Checked:"
    local candidate
    for candidate in "${lease_candidates[@]}"; do
      echo "  - ${candidate}"
    done
    echo "[INFO] Continuing with live client detection via 'iw' and 'ip neigh'."
    echo
  fi
}

read_stations() {
  if ! have_cmd iw; then
    echo "[WARN] 'iw' not available; cannot inspect associated stations."
    return
  fi

  local line current_mac=""
  while IFS= read -r line; do
    if [[ "$line" =~ ^Station[[:space:]]+([0-9a-fA-F:]{17}) ]]; then
      current_mac="${BASH_REMATCH[1],,}"
      seen_macs["$current_mac"]=1
      continue
    fi
    if [[ -n "$current_mac" && "$line" =~ signal:[[:space:]]*(-?[0-9]+)[[:space:]]*dBm ]]; then
      station_signal["$current_mac"]="${BASH_REMATCH[1]} dBm"
    fi
  done < <(iw dev "$IFACE" station dump 2>/dev/null || true)
}

read_neighbors() {
  if ! have_cmd ip; then
    return
  fi

  local line ip mac
  while IFS= read -r line; do
    ip="$(awk '{print $1}' <<<"$line")"
    mac="$(awk '{for(i=1;i<=NF;i++) if($i=="lladdr") {print $(i+1); exit}}' <<<"$line")"
    [[ -n "${mac:-}" ]] || continue
    mac="${mac,,}"
    neigh_ip["$mac"]="$ip"
    seen_macs["$mac"]=1
  done < <(ip -4 neigh show dev "$IFACE" 2>/dev/null || true)
}

print_rows() {
  local printed=0
  printf '%-18s %-15s %-24s %s\n' "MAC" "IP" "HOSTNAME" "SIGNAL"
  printf '%-18s %-15s %-24s %s\n' "------------------" "---------------" "------------------------" "------"

  local mac ip host signal
  for mac in "${!seen_macs[@]}"; do
    ip="${lease_ip[$mac]:-${neigh_ip[$mac]:--}}"
    host="${lease_host[$mac]:--}"
    signal="${station_signal[$mac]:--}"
    printf '%-18s %-15s %-24s %s\n' "$mac" "$ip" "$host" "$signal"
    printed=1
  done

  if [[ "$printed" -eq 0 ]]; then
    echo "(No connected clients detected on ${IFACE})"
  fi
}

print_header
read_leases
read_stations
read_neighbors
print_rows
