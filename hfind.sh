#!/usr/bin/env bash
set -euo pipefail

SOURCE_PATH="${BASH_SOURCE[0]}"
while [[ -L "${SOURCE_PATH}" ]]; do
	LINK_DIR="$(cd "$(dirname "${SOURCE_PATH}")" && pwd)"
	SOURCE_PATH="$(readlink "${SOURCE_PATH}")"
	[[ "${SOURCE_PATH}" != /* ]] && SOURCE_PATH="${LINK_DIR}/${SOURCE_PATH}"
done
SCRIPT_DIR="$(cd "$(dirname "${SOURCE_PATH}")" && pwd)"

# Keep a high maximum worker count; hfind dynamically reduces active work when
# its CPU controller observes usage above HFIND_CPU_LIMIT (90% by default).
export HFIND_SEARCH_THREADS="${HFIND_SEARCH_THREADS:-35}"

exec python3 "$SCRIPT_DIR/hfind.py" "$@"
