#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SETUP="${SCRIPT_DIR}/setup-mdexplore.sh"
CHECK_ONLY=0
SKIP_PROJECT_SETUP=0

APT_PACKAGES=(
  ca-certificates
  curl
  git
  python3
  python3-venv
  python3-pip
  build-essential
  poppler-utils
  tesseract-ocr
  antiword
  catdoc
)

JAVA_PACKAGE_CANDIDATES=(
  default-jre-headless
  default-jre
  openjdk-21-jre-headless
  openjdk-17-jre-headless
)

usage() {
  cat <<'EOF'
Usage: setup-host.sh [options]

Configure a Debian/Ubuntu host for mdexplore, pdfexplore, and hfind.

The script:
  - installs required native/system packages with apt
  - ensures Python virtual-environment support is available
  - installs Java for PlantUML
  - installs Poppler and Tesseract for hfind PDF/image extraction and OCR
  - installs antiword/catdoc tools for legacy Microsoft Office text extraction
  - verifies the important host commands
  - runs setup-mdexplore.sh to create/update the project .venv and runtime assets

Options:
  --check-only          Verify the host without installing or changing anything
  --skip-project-setup  Install/verify host packages but do not run setup-mdexplore.sh
  -h, --help            Show this help text

This script is intended for Debian/Ubuntu systems and may prompt for sudo.
It is safe to rerun.
EOF
}

log() {
  printf '[host-setup] %s\n' "$*"
}

warn() {
  printf '[host-setup] WARNING: %s\n' "$*" >&2
}

die() {
  printf '[host-setup] ERROR: %s\n' "$*" >&2
  exit 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

apt_package_installed() {
  dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q '^install ok installed$'
}

apt_package_available() {
  apt-cache show "$1" >/dev/null 2>&1
}

select_java_package() {
  local package=''
  for package in "${JAVA_PACKAGE_CANDIDATES[@]}"; do
    if apt_package_available "${package}"; then
      printf '%s' "${package}"
      return 0
    fi
  done
  return 1
}

require_debian_host() {
  [[ -r /etc/os-release ]] || die '/etc/os-release is unavailable; expected Debian/Ubuntu.'
  # shellcheck disable=SC1091
  source /etc/os-release
  case "${ID:-}:${ID_LIKE:-}" in
    ubuntu:*|debian:*|*:debian*|*:ubuntu*) ;;
    *) die "Unsupported distribution: ${PRETTY_NAME:-${ID:-unknown}}. This installer currently supports Debian/Ubuntu." ;;
  esac
  have_cmd apt-get || die 'apt-get is required.'
  have_cmd dpkg-query || die 'dpkg-query is required.'
}

sudo_prefix() {
  if [[ "$(id -u)" -eq 0 ]]; then
    printf '%s' ''
    return
  fi
  have_cmd sudo || die 'sudo is required when setup-host.sh is not run as root.'
  printf '%s' 'sudo'
}

install_system_packages() {
  local missing=()
  local package=''
  local java_package=''
  local sudo_cmd=''
  local need_java=0

  for package in "${APT_PACKAGES[@]}"; do
    if ! apt_package_installed "${package}"; then
      missing+=("${package}")
    fi
  done

  if ! have_cmd java; then
    need_java=1
  fi

  if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    if [[ "${#missing[@]}" -gt 0 ]]; then
      printf '[host-setup] Missing apt packages:\n' >&2
      printf '  - %s\n' "${missing[@]}" >&2
    fi
    if [[ "${need_java}" -eq 1 ]]; then
      printf '[host-setup] Missing Java runtime (one supported JRE package will be selected during installation).\n' >&2
    fi
    [[ "${#missing[@]}" -eq 0 && "${need_java}" -eq 0 ]]
    return
  fi

  if [[ "${#missing[@]}" -eq 0 && "${need_java}" -eq 0 ]]; then
    log 'Required apt packages are already installed.'
    return
  fi

  sudo_cmd="$(sudo_prefix)"
  log 'Refreshing apt package metadata.'
  if [[ -n "${sudo_cmd}" ]]; then
    ${sudo_cmd} apt-get update
  else
    apt-get update
  fi

  if [[ "${need_java}" -eq 1 ]]; then
    java_package="$(select_java_package || true)"
    [[ -n "${java_package}" ]] || die "No supported Java runtime package is available. Tried: ${JAVA_PACKAGE_CANDIDATES[*]}"
    missing+=("${java_package}")
  fi

  log "Installing missing apt packages: ${missing[*]}"
  if [[ -n "${sudo_cmd}" ]]; then
    ${sudo_cmd} apt-get install -y --no-install-recommends "${missing[@]}"
  else
    apt-get install -y --no-install-recommends "${missing[@]}"
  fi
}

verify_command() {
  local command_name="$1"
  local purpose="$2"
  if have_cmd "${command_name}"; then
    log "OK: ${command_name} ($(command -v "${command_name}")) - ${purpose}"
  else
    printf '[host-setup] MISSING: %s - %s\n' "${command_name}" "${purpose}" >&2
    return 1
  fi
}

verify_host() {
  local failed=0

  verify_command python3 'Python runtime' || failed=1
  verify_command curl 'runtime asset downloads/bootstrap' || failed=1
  verify_command git 'source/bootstrap operations' || failed=1
  verify_command java 'PlantUML rendering' || failed=1
  verify_command pdftotext 'PDF text extraction' || failed=1
  verify_command pdftoppm 'PDF rasterization for OCR' || failed=1
  verify_command tesseract 'OCR for scanned PDFs and explicitly selected images' || failed=1
  verify_command antiword 'legacy Microsoft Word text extraction' || failed=1
  verify_command xls2csv 'legacy Microsoft Excel text extraction' || failed=1
  verify_command catppt 'legacy Microsoft PowerPoint text extraction' || failed=1

  if have_cmd python3; then
    if python3 -m venv --help >/dev/null 2>&1; then
      log 'OK: python3 venv support'
    else
      printf '[host-setup] MISSING: python3 venv support\n' >&2
      failed=1
    fi
  fi

  if [[ "${failed}" -ne 0 ]]; then
    return 1
  fi
  log 'Host command verification passed.'
}

run_project_setup() {
  [[ -x "${PROJECT_SETUP}" ]] || die "Project bootstrap script is not executable: ${PROJECT_SETUP}"
  log 'Running mdexplore project bootstrap.'
  "${PROJECT_SETUP}"

  local venv_python="${SCRIPT_DIR}/.venv/bin/python"
  [[ -x "${venv_python}" ]] || die "Project virtual-environment Python is missing: ${venv_python}"

  "${venv_python}" - <<'PY'
import importlib

required = [
    "markdown_it",
    "mdit_py_plugins.dollarmath",
    "linkify_it",
    "PySide6.QtWebEngineWidgets",
    "pypdf",
    "reportlab.pdfgen.canvas",
]
for name in required:
    importlib.import_module(name)
print("[host-setup] OK: project Python runtime imports")
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      ;;
    --skip-project-setup)
      SKIP_PROJECT_SETUP=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
  shift
done

require_debian_host

package_status=0
install_system_packages || package_status=$?

verify_status=0
verify_host || verify_status=$?

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  if [[ "${package_status}" -ne 0 || "${verify_status}" -ne 0 ]]; then
    die 'Host configuration check failed. Run setup-host.sh without --check-only to install missing requirements.'
  fi
  if [[ "${SKIP_PROJECT_SETUP}" -eq 0 ]]; then
    if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
      log 'Project .venv exists.'
    else
      warn 'Project .venv does not exist yet. Run setup-host.sh without --check-only to complete project bootstrap.'
      exit 1
    fi
  fi
  log 'Host configuration check passed.'
  exit 0
fi

[[ "${package_status}" -eq 0 ]] || die 'System package installation failed.'
[[ "${verify_status}" -eq 0 ]] || die 'Host verification failed after package installation.'

if [[ "${SKIP_PROJECT_SETUP}" -eq 0 ]]; then
  run_project_setup
else
  log 'Skipping project bootstrap by request.'
fi

log 'Host setup complete.'
log "hfind OCR prerequisites: $(command -v pdftoppm), $(command -v tesseract)"
log "hfind legacy Office prerequisites: $(command -v antiword), $(command -v xls2csv), $(command -v catppt)"
log "Run PDF OCR using: ${SCRIPT_DIR}/hfind.sh -c --ocr-pdf QUERY 'PATH/*.pdf'"
log "Run image OCR using: ${SCRIPT_DIR}/hfind.sh -c -t 'png|jpg|jpeg|tif|tiff' QUERY 'PATH/*'"
