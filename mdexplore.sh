#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
APP_FILE="${SCRIPT_DIR}/mdexplore.py"
REQ_HASH_FILE="${VENV_DIR}/.requirements.sha256"
LOG_DIR="${XDG_CACHE_HOME:-${HOME}/.cache}/mdexplore"
LOG_FILE="${LOG_DIR}/launcher.log"
MAX_LOG_LINES=1000
NON_INTERACTIVE=0

trim_log_file() {
  local file_path="$1"
  local max_lines="$2"
  local tmp_file=""
  if [[ ! -f "${file_path}" ]]; then
    return 0
  fi

  local line_count
  line_count="$(wc -l < "${file_path}")"
  if [[ "${line_count}" -le "${max_lines}" ]]; then
    return 0
  fi

  tmp_file="${file_path}.tmp.$$"
  if tail -n "${max_lines}" "${file_path}" > "${tmp_file}"; then
    mv "${tmp_file}" "${file_path}"
  else
    rm -f "${tmp_file}"
  fi
}

trim_log_file_inplace() {
  local file_path="$1"
  local max_lines="$2"
  local tmp_file=""
  if [[ ! -f "${file_path}" ]]; then
    return 0
  fi

  local line_count
  line_count="$(wc -l < "${file_path}")"
  if [[ "${line_count}" -le "${max_lines}" ]]; then
    return 0
  fi

  tmp_file="${file_path}.tmp.$$"
  if tail -n "${max_lines}" "${file_path}" > "${tmp_file}"; then
    : > "${file_path}"
    cat "${tmp_file}" >> "${file_path}"
    rm -f "${tmp_file}"
  else
    rm -f "${tmp_file}"
  fi
}

if [[ ! -t 1 ]]; then
  NON_INTERACTIVE=1
  mkdir -p "${LOG_DIR}" 2>/dev/null || true
  trim_log_file "${LOG_FILE}" "${MAX_LOG_LINES}"
  exec >> "${LOG_FILE}" 2>&1
fi

notify_failure() {
  local message="$1"
  if [[ "${NON_INTERACTIVE}" -eq 1 ]] && command -v notify-send >/dev/null 2>&1; then
    notify-send "mdexplore launcher failed" "${message}" || true
  fi
}

on_error() {
  local exit_code="$1"
  local line_no="$2"
  local message="Exit ${exit_code} at line ${line_no}. See ${LOG_FILE}"
  echo "ERROR: ${message}"
  if [[ "${NON_INTERACTIVE}" -eq 1 ]]; then
    trim_log_file_inplace "${LOG_FILE}" "${MAX_LOG_LINES}"
  fi
  notify_failure "${message}"
}

trap 'on_error "$?" "$LINENO"' ERR
echo "[$(date +'%Y-%m-%d %H:%M:%S')] mdexplore.sh start args: $*"
if [[ "${NON_INTERACTIVE}" -eq 1 ]]; then
  trim_log_file_inplace "${LOG_FILE}" "${MAX_LOG_LINES}"
fi

usage() {
  echo "Usage: $(basename "$0") [PATH]"
  echo "If PATH is omitted, mdexplore uses ~/.mdexplore.cfg or HOME."
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found in PATH." >&2
  exit 1
fi

decode_file_uri() {
  local uri="$1"
  python3 - "$uri" <<'PY'
import sys
from urllib.parse import unquote, urlparse

uri = sys.argv[1]
parsed = urlparse(uri)
if parsed.scheme != "file":
    print(uri)
    raise SystemExit(0)

path = unquote(parsed.path or "")
if parsed.netloc and parsed.netloc not in ("", "localhost"):
    path = f"//{parsed.netloc}{path}"
print(path)
PY
}

TARGET_PATH=""
if [[ $# -ge 1 ]]; then
  for RAW_PATH in "$@"; do
    [[ -z "${RAW_PATH}" ]] && continue
    TARGET_PATH=""
    URI_INPUT=0
    case "${RAW_PATH}" in
      ""|"%u"|"%U"|"%f"|"%F")
        continue
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      -*)
        # Some desktop launchers may add non-path args. Ignore them.
        echo "Ignoring non-path argument: ${RAW_PATH}"
        continue
        ;;
      file://*)
        URI_INPUT=1
        TARGET_PATH="$(decode_file_uri "${RAW_PATH}")"
        ;;
      *)
        TARGET_PATH="${RAW_PATH}"
        ;;
    esac

    if [[ -f "${TARGET_PATH}" ]]; then
      TARGET_PATH="$(dirname "${TARGET_PATH}")"
    fi

    if [[ -d "${TARGET_PATH}" ]]; then
      break
    fi

    if [[ "${URI_INPUT}" -eq 1 ]]; then
      # Desktop URI launches can occasionally pass odd values. Fall back
      # to default root instead of aborting a dock/menu launch.
      echo "Ignoring invalid URI-derived path: ${TARGET_PATH}"
      TARGET_PATH=""
      continue
    fi

    if [[ $# -eq 1 ]]; then
      echo "Path is not a directory: ${TARGET_PATH}" >&2
      exit 1
    fi

    echo "Ignoring invalid path argument: ${TARGET_PATH}"
    TARGET_PATH=""
  done
fi

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "No GUI display detected (DISPLAY/WAYLAND_DISPLAY unset)." >&2
  echo "Run this from a desktop session with GUI access." >&2
  if [[ "${NON_INTERACTIVE}" -eq 1 ]]; then
    trim_log_file_inplace "${LOG_FILE}" "${MAX_LOG_LINES}"
  fi
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

runtime_import_check() {
  "${VENV_PYTHON}" - <<'PY'
import importlib

required = [
    "markdown_it",
    "linkify_it",
    "PySide6.QtWebEngineWidgets",
]

missing = []
for module_name in required:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append(f"{module_name}: {exc}")

if missing:
    print("Runtime dependency check failed:")
    for item in missing:
        print(f"  - {item}")
    raise SystemExit(1)
PY
}

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

if ! runtime_import_check; then
  echo "Detected incomplete Python runtime. Reinstalling dependencies..."
  "${VENV_PYTHON}" -m pip install --disable-pip-version-check --upgrade --force-reinstall -r "${REQUIREMENTS_FILE}"
  printf '%s\n' "${current_hash}" > "${REQ_HASH_FILE}"
  runtime_import_check
fi

if [[ "${NON_INTERACTIVE}" -eq 1 ]]; then
  trim_log_file_inplace "${LOG_FILE}" "${MAX_LOG_LINES}"
fi

if [[ -n "${TARGET_PATH}" ]]; then
  echo "Launching mdexplore for: ${TARGET_PATH}"
  # Do not override argv[0] for python; it can break venv detection and
  # cause imports to resolve against system packages.
  exec "${VENV_PYTHON}" "${APP_FILE}" "${TARGET_PATH}"
else
  echo "Launching mdexplore using configured default root (or home)"
  exec "${VENV_PYTHON}" "${APP_FILE}"
fi
