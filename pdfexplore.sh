#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
REQ_HASH_FILE="${VENV_DIR}/.requirements.sha256"
LOG_DIR="${XDG_CACHE_HOME:-${HOME}/.cache}/pdfexplore"
LOG_FILE="${LOG_DIR}/launcher.log"
MAX_LOG_LINES=1000
NON_INTERACTIVE=0
DEBUG_MODE=false

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
    notify-send "pdfexplore launcher failed" "${message}" || true
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

usage() {
  echo "Usage: $(basename "$0") [PATH]"
  echo "If PATH is omitted, pdfexplore uses ~/.pdfexplore.cfg or HOME."
  echo "Debug logging can be toggled by editing DEBUG_MODE in this script."
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

configure_qt_graphics_fallback() {
  if [[ -z "${QT_QUICK_BACKEND:-}" ]]; then
    export QT_QUICK_BACKEND="software"
  fi
  if [[ -z "${QT_OPENGL:-}" ]]; then
    export QT_OPENGL="software"
  fi
  local chromium_flags="${QTWEBENGINE_CHROMIUM_FLAGS:-}"
  if [[ " ${chromium_flags} " != *" --disable-gpu "* ]]; then
    chromium_flags="${chromium_flags} --disable-gpu"
    chromium_flags="${chromium_flags#"${chromium_flags%%[![:space:]]*}"}"
    chromium_flags="${chromium_flags%"${chromium_flags##*[![:space:]]}"}"
    export QTWEBENGINE_CHROMIUM_FLAGS="${chromium_flags}"
  fi
}

TARGET_PATH=""
APP_ARGS=()
while [[ $# -gt 0 ]]; do
  RAW_ARG="$1"
  shift
  [[ -z "${RAW_ARG}" ]] && continue
  case "${RAW_ARG}" in
    ""|"%u"|"%U"|"%f"|"%F")
      continue
      ;;
    -h|--help)
      usage
      exit 0
      ;;
  esac

  URI_INPUT=0
  CANDIDATE_PATH=""
  case "${RAW_ARG}" in
    -*)
      echo "Ignoring non-path argument: ${RAW_ARG}"
      continue
      ;;
    file://*)
      URI_INPUT=1
      CANDIDATE_PATH="$(decode_file_uri "${RAW_ARG}")"
      ;;
    *)
      CANDIDATE_PATH="${RAW_ARG}"
      ;;
  esac

  if [[ -f "${CANDIDATE_PATH}" ]]; then
    TARGET_PATH="${CANDIDATE_PATH}"
    continue
  fi
  if [[ -d "${CANDIDATE_PATH}" ]]; then
    TARGET_PATH="${CANDIDATE_PATH}"
    continue
  fi
  if [[ "${URI_INPUT}" -eq 1 ]]; then
    echo "Ignoring invalid URI-derived path: ${CANDIDATE_PATH}"
    continue
  fi
  if [[ -n "${CANDIDATE_PATH}" ]]; then
    echo "Ignoring invalid path argument: ${CANDIDATE_PATH}"
  fi
done

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}..."
  python3 -m venv "${VENV_DIR}"
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
    "PySide6.QtWebEngineWidgets",
    "pypdf",
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

if [[ "${current_hash}" != "${stored_hash}" ]]; then
  echo "Installing Python dependencies..."
  "${VENV_PYTHON}" -m pip install --disable-pip-version-check -r "${REQUIREMENTS_FILE}"
  printf '%s\n' "${current_hash}" > "${REQ_HASH_FILE}"
fi

if ! runtime_import_check; then
  echo "Detected incomplete Python runtime. Reinstalling dependencies..."
  "${VENV_PYTHON}" -m pip install --disable-pip-version-check --upgrade --force-reinstall -r "${REQUIREMENTS_FILE}"
  printf '%s\n' "${current_hash}" > "${REQ_HASH_FILE}"
  runtime_import_check
fi

if [[ "${DEBUG_MODE}" == "true" ]]; then
  APP_ARGS+=("--debug")
fi

# pdfexplore is a read-only viewer backed by QtWebEngine/pdf.js. Default to the
# conservative software path because it is more reliable across mixed desktop
# driver stacks than Chromium's GPU/Vulkan probing.
configure_qt_graphics_fallback

launch_pdfexplore() {
  local -a cmd=("${VENV_PYTHON}" "-m" "pdfexplore")
  if [[ -n "${TARGET_PATH}" ]]; then
    cmd+=("${APP_ARGS[@]}" "${TARGET_PATH}")
  else
    cmd+=("${APP_ARGS[@]}")
  fi
  PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${cmd[@]}"
}

launch_pdfexplore
