#!/usr/bin/env bash
# halo forge banner with machine/GPU info and ROCm version

# Load ROCm env quietly if present
[[ -f /etc/profile.d/01-rocm.sh ]] && . /etc/profile.d/01-rocm.sh 2>/dev/null

oem_info() {
  local v="" m="" d lv lm
  for d in /sys/class/dmi/id /sys/devices/virtual/dmi/id; do
    [[ -r "$d/sys_vendor" ]] && v=$(<"$d/sys_vendor")
    [[ -r "$d/product_name" ]] && m=$(<"$d/product_name")
    [[ -n "$v" || -n "$m" ]] && break
  done
  # ARM/SBC fallback
  if [[ -z "$v" && -z "$m" && -r /proc/device-tree/model ]]; then
    tr -d '\0' </proc/device-tree/model
    return
  fi
  lv=$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]')
  lm=$(printf '%s' "$m" | tr '[:upper:]' '[:lower:]')
  if [[ -n "$m" && "$lm" == "$lv "* ]]; then
    printf '%s\n' "$m"
  else
    printf '%s %s\n' "${v:-Unknown}" "${m:-Unknown}"
  fi
}

gpu_name() {
  local name=""
  if command -v rocm-smi >/dev/null 2>&1; then
    name=$(rocm-smi --showproductname --csv 2>/dev/null | tail -n1 | cut -d, -f2)
    [[ -z "$name" ]] && name=$(rocm-smi --showproductname 2>/dev/null | grep -m1 -E 'Product Name|Card series' | sed 's/.*: //')
  fi
  if [[ -z "$name" ]] && command -v rocminfo >/dev/null 2>&1; then
    name=$(rocminfo 2>/dev/null | awk -F': ' '/^[[:space:]]*Name:/{print $2; exit}')
  fi
  if [[ -z "$name" ]] && command -v lspci >/dev/null 2>&1; then
    name=$(lspci -nn 2>/dev/null | grep -Ei 'vga|display|gpu' | grep -i amd | head -n1 | cut -d: -f3-)
  fi
  name=$(printf '%s' "$name" | sed -e 's/^[[:space:]]\+//' -e 's/[[:space:]]\+$//' -e 's/[[:space:]]\{2,\}/ /g')
  printf '%s\n' "${name:-Unknown AMD GPU}"
}

rocm_version() {
  local PY="/opt/venv/bin/python"
  [[ -x "$PY" ]] || PY="python"
  "$PY" - <<'PY' 2>/dev/null || true
try:
    import importlib.metadata as im
    try:
        print(im.version('_rocm_sdk_core'))
    except Exception:
        print(im.version('rocm'))
except Exception:
    print("")
PY
}

MACHINE="$(oem_info)"
GPU="$(gpu_name)"
ROCM_VER="$(rocm_version)"

echo
cat <<'ASCII'
██╗  ██╗ █████╗ ██╗      ██████╗       ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
██║  ██║██╔══██╗██║     ██╔═══██╗      ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
███████║███████║██║     ██║   ██║█████╗█████╗  ██║   ██║██████╔╝██║  ███╗█████╗  
██╔══██║██╔══██║██║     ██║   ██║╚════╝██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  
██║  ██║██║  ██║███████╗╚██████╔╝      ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝       ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝

                    R L V R   T R A I N I N G   F R A M E W O R K                 

ASCII
echo
printf 'AMD Strix Halo — RLVR Training (gfx1151, ROCm via TheRock)\n'
[[ -n "$ROCM_VER" ]] && printf 'ROCm nightly: %s\n' "$ROCM_VER"
echo
printf 'Machine: %s\n' "$MACHINE"
printf 'GPU    : %s\n\n' "$GPU"
printf 'Docs   : https://halo-forge.pages.dev\n'
printf 'Repo   : https://github.com/professor-moody/halo-forge\n\n'
printf 'Quickstart:\n'
printf '  halo-forge test --level smoke    → Validate environment\n'
printf '  halo-forge raft train --help     → RAFT training options\n'
echo

