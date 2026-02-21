#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
APP_FILE="${SCRIPT_DIR}/mdexplore.py"
REQ_HASH_FILE="${VENV_DIR}/.requirements.sha256"

usage() {
  echo "Usage: $(basename "$0") [PATH]"
  echo "If PATH is omitted, current directory is used."
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 1
fi

TARGET_PATH="${1:-.}"
if [[ ! -d "${TARGET_PATH}" ]]; then
  echo "Path is not a directory: ${TARGET_PATH}" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found in PATH." >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "No GUI display detected (DISPLAY/WAYLAND_DISPLAY unset)." >&2
  echo "Run this from a desktop session with GUI access." >&2
  exit 1
fi

needs_install=0
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}..."
  python3 -m venv "${VENV_DIR}"
  needs_install=1
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Virtual environment python not found: ${VENV_PYTHON}" >&2
  exit 1
fi

current_hash="$(sha256sum "${REQUIREMENTS_FILE}" | awk '{print $1}')"
stored_hash=""
if [[ -f "${REQ_HASH_FILE}" ]]; then
  stored_hash="$(cat "${REQ_HASH_FILE}")"
fi

if [[ "${needs_install}" -eq 1 || "${current_hash}" != "${stored_hash}" ]]; then
  echo "Installing Python dependencies..."
  "${VENV_PYTHON}" -m pip install --disable-pip-version-check -r "${REQUIREMENTS_FILE}"
  printf '%s\n' "${current_hash}" > "${REQ_HASH_FILE}"
else
  echo "Dependencies already up to date."
fi

echo "Launching mdexplore for: ${TARGET_PATH}"
"${VENV_PYTHON}" "${APP_FILE}" "${TARGET_PATH}"
