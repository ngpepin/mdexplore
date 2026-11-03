#!/usr/bin/env bash
# Idempotent dependency setup and compilation for the mdExt VS Code extension.
set -Eeuo pipefail

SOURCE="${BASH_SOURCE[0]}"
while [[ -L "$SOURCE" ]]; do
  SOURCE_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
  TARGET="$(readlink "$SOURCE")"
  if [[ "$TARGET" = /* ]]; then SOURCE="$TARGET"; else SOURCE="$SOURCE_DIR/$TARGET"; fi
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
cd "$SCRIPT_DIR"

CHECK_ONLY=false
FORCE=false
PACKAGE=false
RUN_TESTS=true
usage() {
  cat <<'EOF'
Usage: ./install-update.sh [--check] [--force] [--package] [--no-test]

Install missing/outdated locked npm dependencies, test changed sources, and
compile mdExt. Repeated runs skip work when inputs and outputs are unchanged.
  --check, --dry-run  Report build readiness without writing or installing
  --force             Run tests, bump the patch version, rebuild, and create a new .vsix
  --package           Create/update the current-version .vsix without bumping its version
  --no-test           Skip unit tests when recompiling
  -h, --help          Show this help without modifying anything

The script does not install or replace the extension in VS Code.
EOF
}
while (($#)); do
  case "$1" in
    --check|--dry-run) CHECK_ONLY=true ;;
    --force) FORCE=true ;;
    --package) PACKAGE=true ;;
    --no-test) RUN_TESTS=false ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

# A forced build is a new patch release, not merely a recompilation.
# --package alone retains the existing version.
if "$FORCE"; then
  PACKAGE=true
fi

# Exit-code convention for --check: 0 compatible, 2 safely alignable,
# 3 uncertain/incompatible toolchain, 4 requires a separate runtime.
classification() {
  printf '%s: %s\n' "$1" "$2"
  return "$3"
}
if [[ ! -f package-lock.json || ! -f package.json ]]; then
  if "$CHECK_ONLY"; then classification host_compatibility_uncertain 'package.json or package-lock.json is missing' 3; fi
  printf 'ERROR: npm manifest and lockfile are required.\n' >&2
  exit 1
fi
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  if "$CHECK_ONLY"; then classification host_compatibility_uncertain 'Node.js and npm must be installed' 3; fi
  printf 'ERROR: install a compatible Node.js and npm before compiling mdExt.\n' >&2
  exit 1
fi
if ! node -e 'process.exit(+process.versions.node.split(".")[0] >= 20 ? 0 : 1)'; then
  if "$CHECK_ONLY"; then classification host_compatibility_uncertain 'Node.js 20 or newer is required for the build tools' 3; fi
  printf 'ERROR: mdExt build tools require Node.js 20 or newer.\n' >&2
  exit 1
fi
if ! command -v sha256sum >/dev/null 2>&1; then
  if "$CHECK_ONLY"; then classification host_compatibility_uncertain 'sha256sum is required' 3; fi
  printf 'ERROR: sha256sum is required.\n' >&2
  exit 1
fi

STATE_DIR="$SCRIPT_DIR/.tmp"
DEPS_STAMP="$SCRIPT_DIR/node_modules/.mdext-lock.sha256"
BUILD_STAMP="$STATE_DIR/build-inputs.sha256"
OUTPUT_STAMP="$STATE_DIR/build-output.sha256"
PACKAGE_STAMP="$STATE_DIR/package-inputs.sha256"

# Hash file names and bytes, so a restored timestamp or same-size edit still
# invalidates the cache. File ordering is stable across invocations.
fingerprint() {
  local file
  for file in "$@"; do
    if [[ ! -f "$file" ]]; then
      printf 'MISSING %s\n' "$file"
    else
      printf '%s %s\n' "$file" "$(sha256sum "$file" | cut -d ' ' -f1)"
    fi
  done | sha256sum | cut -d ' ' -f1
}
read_stamp() {
  if [[ -f "$1" ]]; then cat "$1"; else printf 'missing'; fi
}

LOCK_HASH="$(fingerprint package.json package-lock.json)"
# npm ls validates installed versions even if this script did not install them.
deps_ready=false
if [[ -x node_modules/.bin/esbuild && -x node_modules/.bin/vsce ]] && npm ls --offline --depth=0 >/dev/null 2>&1; then
  deps_ready=true
fi

mapfile -d '' BUILD_FILES < <(find src -type f -name '*.js' -print0 | LC_ALL=C sort -z)
BUILD_FILES+=(package.json package-lock.json ../assets/js/preview/form_fields.js ../assets/js/preview/dil_widgets.js ../assets/css/preview/dil_widgets.css ../MARKDOWN-FORMS.md)
BUILD_HASH="$(fingerprint "${BUILD_FILES[@]}")"
output_hash() { fingerprint dist/extension.js media/formFields.js media/dilWidgets.js media/dilWidgets.css MARKDOWN-FORMS.md; }
build_ready=false
if [[ "$deps_ready" == true && -f dist/extension.js && "$(read_stamp "$BUILD_STAMP")" == "$BUILD_HASH" && "$(read_stamp "$OUTPUT_STAMP")" == "$(output_hash)" ]]; then
  build_ready=true
fi

VSIX_FILE="mdext-$(node -p 'require("./package.json").version').vsix"
package_files=(package.json README.md LICENSE MARKDOWN-FORMS.md)
for directory in dist media python bin vendor; do
  if [[ -d "$directory" ]]; then
    while IFS= read -r -d '' file; do package_files+=("$file"); done < <(find "$directory" -type f -print0 | LC_ALL=C sort -z)
  fi
done
package_hash() { fingerprint "${package_files[@]}"; }
package_ready=false
if [[ "$build_ready" == true && -s "$VSIX_FILE" && "$(read_stamp "$PACKAGE_STAMP")" == "$(package_hash)" ]]; then
  package_ready=true
fi

if "$CHECK_ONLY"; then
  if "$FORCE"; then
    classification host_safely_alignable 'a forced run would create a new patch-versioned VSIX' 2
    exit $?
  fi
  if [[ "$deps_ready" == true && "$build_ready" == true && ( "$PACKAGE" == false || "$package_ready" == true ) ]]; then
    classification host_compatible 'npm dependencies and requested build artifacts are up to date' 0
  else
    classification host_safely_alignable 'npm dependencies or requested build artifacts need installing/rebuilding' 2
  fi
  exit $?
fi

if [[ "$deps_ready" != true || ( -f "$DEPS_STAMP" && "$(read_stamp "$DEPS_STAMP")" != "$LOCK_HASH" ) ]]; then
  printf '[mdExt] Restoring dependencies from package-lock.json (npm ci)...\n'
  npm ci --no-audit --no-fund
  deps_ready=true
  build_ready=false
fi
printf '%s\n' "$LOCK_HASH" > "$DEPS_STAMP"

if [[ "$build_ready" != true || "$FORCE" == true ]]; then
  if "$RUN_TESTS"; then
    printf '[mdExt] Running regression tests...\n'
    npm test
  fi
  if "$FORCE"; then
    printf '[mdExt] Incrementing patch version for the new VSIX...\n'
    npm version patch --no-git-tag-version --ignore-scripts
    # Version changes update both manifests, so refresh the cache hashes and
    # destination filename before compiling and packaging the new release.
    LOCK_HASH="$(fingerprint package.json package-lock.json)"
    BUILD_HASH="$(fingerprint "${BUILD_FILES[@]}")"
    printf '%s\n' "$LOCK_HASH" > "$DEPS_STAMP"
    VSIX_FILE="mdext-$(node -p 'require("./package.json").version').vsix"
  fi
  printf '[mdExt] Compiling extension...\n'
  npm run build
  node --check dist/extension.js
  mkdir -p "$STATE_DIR"
  printf '%s\n' "$BUILD_HASH" > "$BUILD_STAMP"
  output_hash > "$OUTPUT_STAMP"
  package_ready=false
else
  printf '[mdExt] Build is current; skipping compilation.\n'
fi

if "$PACKAGE"; then
  # Package from the already-compiled bundle; npm run package would rebuild it.
  if [[ "$package_ready" != true || "$FORCE" == true ]]; then
    printf '[mdExt] Packaging %s...\n' "$VSIX_FILE"
    ./node_modules/.bin/vsce package --no-dependencies --allow-missing-repository --no-rewrite-relative-links
    package_files=(package.json README.md LICENSE MARKDOWN-FORMS.md)
    for directory in dist media python bin vendor; do
      if [[ -d "$directory" ]]; then
        while IFS= read -r -d '' file; do package_files+=("$file"); done < <(find "$directory" -type f -print0 | LC_ALL=C sort -z)
      fi
    done
    package_hash > "$PACKAGE_STAMP"
  else
    printf '[mdExt] %s is current; skipping packaging.\n' "$VSIX_FILE"
  fi
fi
printf '[mdExt] Ready: %s\n' "$SCRIPT_DIR/dist/extension.js"
