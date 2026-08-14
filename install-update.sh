#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_SETUP="${SCRIPT_DIR}/setup-host.sh"
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage: install-update.sh [--check]

Checks or aligns the host runtime used by mdexplore and mdExt. mdExt's native
vector PDF export uses the same PySide6 Qt WebEngine runtime as mdexplore.

Options:
  --check    Verify requirements without changing the host
  -h,--help  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check|--dry-run) CHECK_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

[[ -x "${HOST_SETUP}" ]] || {
  printf 'ERROR: host setup script is missing or not executable: %s\n' "${HOST_SETUP}" >&2
  exit 2
}

classify_host_platform() {
  if [[ ! -r /etc/os-release ]]; then
    printf '[install-update] uncertain: /etc/os-release is unavailable, so direct host alignment cannot be classified safely.\n' >&2
    return 2
  fi
  # shellcheck disable=SC1091
  source /etc/os-release
  case "${ID:-}:${ID_LIKE:-}" in
    ubuntu:*|debian:*|*:debian*|*:ubuntu*) return 0 ;;
    *)
      printf '[install-update] incompatible: direct host alignment currently supports Debian/Ubuntu, not %s.\n' "${PRETTY_NAME:-${ID:-unknown}}" >&2
      return 3
      ;;
  esac
}

platform_status=0
classify_host_platform || platform_status=$?
if [[ "${platform_status}" -ne 0 ]]; then
  exit "${platform_status}"
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  if "${HOST_SETUP}" --check-only; then
    printf '[install-update] compatible: host and project .venv satisfy mdexplore/mdExt requirements.\n'
    exit 0
  fi
  printf '[install-update] safely alignable: requirements are missing or incomplete; rerun without --check to repair them.\n' >&2
  exit 1
fi

if "${HOST_SETUP}" --check-only >/dev/null 2>&1; then
  printf '[install-update] Required host and project runtime are already satisfied; no changes needed.\n'
  exit 0
fi

printf '%s\n' 'The host is missing one or more mdexplore/mdExt requirements.'
printf '%s\n' 'The setup may install Debian/Ubuntu packages with sudo and will create/update the project .venv.'
read -r -p 'Proceed with host alignment? [y/N] ' reply
case "${reply}" in
  y|Y|yes|YES) ;;
  *) printf '[install-update] No changes made.\n'; exit 1 ;;
esac

"${HOST_SETUP}"
printf '[install-update] Host alignment complete.\n'
