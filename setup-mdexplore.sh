#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
REQ_HASH_FILE="${VENV_DIR}/.requirements.sha256"

VENDOR_DIR="${SCRIPT_DIR}/vendor"
MATHJAX_DIR="${VENDOR_DIR}/mathjax/es5"
MERMAID_DIR="${VENDOR_DIR}/mermaid/dist"
PLANTUML_DIR="${VENDOR_DIR}/plantuml"
RUST_RENDERER_DIR="${VENDOR_DIR}/mermaid-rs-renderer"
RUST_RENDERER_BIN="${RUST_RENDERER_DIR}/target/release/mmdr"

MATHJAX_TEX_SVG_URL="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
MATHJAX_TEX_CHTML_URL="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
MERMAID_JS_URL="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"
PLANTUML_JAR_URL="https://github.com/plantuml/plantuml/releases/latest/download/plantuml.jar"
RUST_RENDERER_REPO_URL="${MDEXPLORE_SETUP_MERMAID_RS_REPO:-https://github.com/1jehuang/mermaid-rs-renderer.git}"
RUSTUP_INIT_URL="https://sh.rustup.rs"

SKIP_PYTHON=0
SKIP_ASSETS=0
SKIP_RUST=0
REBUILD_RUST=0

usage() {
  cat <<'EOF'
Usage: setup-mdexplore.sh [options]

Bootstraps the local mdexplore checkout by:
  - creating/updating .venv
  - installing Python requirements
  - downloading local MathJax, Mermaid, and PlantUML assets if missing
  - cloning/building the Rust Mermaid renderer (mmdr) under vendor/ if missing

Options:
  --skip-python    Skip .venv creation and pip install
  --skip-assets    Skip local asset download checks
  --skip-rust      Skip Mermaid Rust renderer bootstrap/build
  --rebuild-rust   Force cargo rebuild even if mmdr already exists
  -h, --help       Show this help text
EOF
}

log() {
  printf '[setup] %s\n' "$*"
}

die() {
  printf '[setup] ERROR: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || die "Required command not found in PATH: ${cmd}"
}

download_file() {
  local url="$1"
  local dest="$2"
  local tmp="${dest}.tmp.$$"

  need_cmd curl
  mkdir -p "$(dirname "${dest}")"
  log "Downloading $(basename "${dest}")"
  curl --fail --location --show-error --silent "${url}" -o "${tmp}"
  mv "${tmp}" "${dest}"
}

runtime_import_check() {
  local venv_python="$1"
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

ensure_venv() {
  local venv_python=""
  local current_hash=""
  local stored_hash=""

  need_cmd python3
  if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    die "requirements.txt not found: ${REQUIREMENTS_FILE}"
  fi

  if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating virtual environment at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}" || die "Failed to create virtual environment. Ensure python3-venv is installed."
  fi

  venv_python="${VENV_DIR}/bin/python"
  [[ -x "${venv_python}" ]] || die "Virtual environment python not found: ${venv_python}"

  current_hash="$(sha256sum "${REQUIREMENTS_FILE}" | awk '{print $1}')"
  if [[ -f "${REQ_HASH_FILE}" ]]; then
    stored_hash="$(cat "${REQ_HASH_FILE}")"
  fi

  if [[ "${current_hash}" != "${stored_hash}" ]]; then
    log "Installing Python dependencies"
    "${venv_python}" -m pip install --disable-pip-version-check -r "${REQUIREMENTS_FILE}"
    printf '%s\n' "${current_hash}" > "${REQ_HASH_FILE}"
  else
    log "Python dependencies already up to date"
  fi

  if ! runtime_import_check "${venv_python}"; then
    log "Detected incomplete Python runtime; reinstalling requirements"
    "${venv_python}" -m pip install --disable-pip-version-check --upgrade --force-reinstall -r "${REQUIREMENTS_FILE}"
    printf '%s\n' "${current_hash}" > "${REQ_HASH_FILE}"
    runtime_import_check "${venv_python}"
  fi
}

ensure_local_assets() {
  if [[ ! -f "${MATHJAX_DIR}/tex-svg.js" ]]; then
    download_file "${MATHJAX_TEX_SVG_URL}" "${MATHJAX_DIR}/tex-svg.js"
  else
    log "MathJax tex-svg.js already present"
  fi

  if [[ ! -f "${MATHJAX_DIR}/tex-mml-chtml.js" ]]; then
    download_file "${MATHJAX_TEX_CHTML_URL}" "${MATHJAX_DIR}/tex-mml-chtml.js"
  else
    log "MathJax tex-mml-chtml.js already present"
  fi

  if [[ ! -f "${MERMAID_DIR}/mermaid.min.js" ]]; then
    download_file "${MERMAID_JS_URL}" "${MERMAID_DIR}/mermaid.min.js"
  else
    log "Mermaid bundle already present"
  fi

  if [[ ! -f "${PLANTUML_DIR}/plantuml.jar" ]]; then
    download_file "${PLANTUML_JAR_URL}" "${PLANTUML_DIR}/plantuml.jar"
  else
    log "PlantUML jar already present"
  fi
}

ensure_cargo() {
  local cargo_bin=""

  if command -v cargo >/dev/null 2>&1; then
    return 0
  fi

  need_cmd curl

  if [[ ! -x "${HOME}/.cargo/bin/rustup" ]]; then
    log "Installing Rust toolchain with rustup"
    curl --proto '=https' --tlsv1.2 --fail --location --show-error --silent "${RUSTUP_INIT_URL}" | sh -s -- -y --profile minimal
  fi

  if [[ -f "${HOME}/.cargo/env" ]]; then
    # Load cargo/rustc into this process so the setup can continue immediately.
    # shellcheck disable=SC1090
    source "${HOME}/.cargo/env"
  else
    export PATH="${HOME}/.cargo/bin:${PATH}"
  fi

  cargo_bin="$(command -v cargo || true)"
  [[ -n "${cargo_bin}" ]] || die "cargo is still unavailable after rustup installation"
}

ensure_rust_renderer_source() {
  if [[ -d "${RUST_RENDERER_DIR}" && -f "${RUST_RENDERER_DIR}/Cargo.toml" ]]; then
    log "Using existing Rust Mermaid renderer source at ${RUST_RENDERER_DIR}"
    return 0
  fi

  need_cmd git
  mkdir -p "${VENDOR_DIR}"
  rm -rf "${RUST_RENDERER_DIR}"
  log "Cloning Mermaid Rust renderer into ${RUST_RENDERER_DIR}"
  git clone --depth 1 "${RUST_RENDERER_REPO_URL}" "${RUST_RENDERER_DIR}"
}

ensure_rust_renderer_build() {
  local needs_build=0

  ensure_cargo
  ensure_rust_renderer_source

  if [[ "${REBUILD_RUST}" -eq 1 ]]; then
    needs_build=1
  elif [[ ! -x "${RUST_RENDERER_BIN}" ]]; then
    needs_build=1
  elif [[ "${RUST_RENDERER_DIR}/Cargo.toml" -nt "${RUST_RENDERER_BIN}" || "${RUST_RENDERER_DIR}/Cargo.lock" -nt "${RUST_RENDERER_BIN}" ]]; then
    needs_build=1
  fi

  if [[ "${needs_build}" -eq 1 ]]; then
    log "Building Mermaid Rust renderer (mmdr)"
    (
      cd "${RUST_RENDERER_DIR}"
      cargo build --release --locked
    )
  else
    log "Mermaid Rust renderer already built"
  fi

  [[ -x "${RUST_RENDERER_BIN}" ]] || die "mmdr build did not produce ${RUST_RENDERER_BIN}"
}

warn_optional_runtime() {
  if ! command -v java >/dev/null 2>&1; then
    log "WARNING: java not found in PATH. PlantUML preview/export will not work until Java is installed."
  fi
  if ! command -v code >/dev/null 2>&1; then
    log "WARNING: VS Code 'code' launcher not found. The Edit button will be unavailable."
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-python)
      SKIP_PYTHON=1
      ;;
    --skip-assets)
      SKIP_ASSETS=1
      ;;
    --skip-rust)
      SKIP_RUST=1
      ;;
    --rebuild-rust)
      REBUILD_RUST=1
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

if [[ "${SKIP_PYTHON}" -eq 0 ]]; then
  ensure_venv
else
  log "Skipping Python environment setup"
fi

if [[ "${SKIP_ASSETS}" -eq 0 ]]; then
  ensure_local_assets
else
  log "Skipping local asset download checks"
fi

if [[ "${SKIP_RUST}" -eq 0 ]]; then
  ensure_rust_renderer_build
else
  log "Skipping Mermaid Rust renderer bootstrap"
fi

warn_optional_runtime

log "Setup complete"
log "Run the app with: ${SCRIPT_DIR}/mdexplore.sh"
if [[ -x "${RUST_RENDERER_BIN}" ]]; then
  log "Rust Mermaid renderer: ${RUST_RENDERER_BIN}"
fi

