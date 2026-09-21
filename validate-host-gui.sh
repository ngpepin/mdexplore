#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE="${BASH_SOURCE[0]}"
while [[ -h "${SOURCE}" ]]; do
  SOURCE_DIR="$(cd -P "$(dirname "${SOURCE}")" && pwd)"
  TARGET="$(readlink "${SOURCE}")"
  if [[ "${TARGET}" == /* ]]; then
    SOURCE="${TARGET}"
  else
    SOURCE="${SOURCE_DIR}/${TARGET}"
  fi
done
PROJECT_ROOT="$(cd -P "$(dirname "${SOURCE}")" && pwd)"
cd "${PROJECT_ROOT}"

usage() {
  cat <<'EOF'
Usage: validate-host-gui.sh

Runs the focused mdexplore/pdfexplore directory-sort GUI validation on the
actual host desktop session. It is intentionally host-only and refuses to run
inside the gateway container.

Requirements:
  - a graphical desktop session (DISPLAY or WAYLAND_DISPLAY)
  - project .venv created by install-update.sh/setup-mdexplore.sh
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  "") ;;
  *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
esac

if [[ "${CONTAINER_AGENT_EXECUTION_ENVIRONMENT:-}" == "container" ]] || [[ -n "${CONTAINER_AGENT_RUNTIME_ID:-}" ]]; then
  printf 'ERROR: this validation must run on the host desktop, not in the gateway container.\n' >&2
  exit 3
fi

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  printf 'ERROR: no host GUI display detected (DISPLAY/WAYLAND_DISPLAY unset).\n' >&2
  exit 4
fi

VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  printf 'ERROR: host project virtual environment is missing: %s\n' "${VENV_PYTHON}" >&2
  printf 'Run ./install-update.sh first.\n' >&2
  exit 5
fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${VENV_PYTHON}" -u - <<'PY'
import json
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from mdexplore_app.tree import MarkdownDirectorySortProxyModel, MarkdownTreeView
from pdfexplore.tree import PdfDirectorySortProxyModel, PdfTreeView

app = QApplication.instance() or QApplication([])
ROLE_DIR = int(Qt.ItemDataRole.UserRole) + 1
ROLE_PATH = int(Qt.ItemDataRole.UserRole) + 2

class Source(QStandardItemModel):
    def __init__(self, root_path: Path):
        super().__init__()
        self.root_path = Path(root_path)

    def isDir(self, index):
        return bool(index.isValid() and index.data(ROLE_DIR))

    def fileName(self, index):
        return str(index.data(Qt.ItemDataRole.DisplayRole) or "")

    def filePath(self, index):
        if not index.isValid():
            return str(self.root_path)
        return str(index.data(ROLE_PATH) or "")


def make_item(name: str, is_dir: bool, path: Path):
    item = QStandardItem(name)
    item.setData(bool(is_dir), ROLE_DIR)
    item.setData(str(path), ROLE_PATH)
    return item


def exercise(label, proxy_cls, view_cls, sidecar):
    with tempfile.TemporaryDirectory(prefix=f"{label}-host-gui-") as td:
        root = Path(td)
        nested_path = root / "nested"
        nested_path.mkdir()

        source = Source(root)
        nested = make_item("nested", True, nested_path)
        source.appendRow(nested)
        source.appendRow(make_item("alpha", False, root / "alpha"))
        source.appendRow(make_item("zeta", False, root / "zeta"))
        nested.appendRow(make_item("alpha", False, nested_path / "alpha"))
        nested.appendRow(make_item("zeta", False, nested_path / "zeta"))

        model = proxy_cls(source)
        model.sort(0, Qt.SortOrder.AscendingOrder)

        def top(name):
            for row in range(model.rowCount()):
                idx = model.index(row, 0)
                if idx.data(Qt.ItemDataRole.DisplayRole) == name:
                    return idx
            raise AssertionError(name)

        def child_files(parent):
            return [
                model.index(row, 0, parent).data(Qt.ItemDataRole.DisplayRole)
                for row in range(model.rowCount(parent))
                if not model.isDir(model.index(row, 0, parent))
            ]

        nested_idx = top("nested")
        assert child_files(nested_idx) == ["alpha", "zeta"]
        assert model.directory_sort_descending(nested_path) is False
        assert not (nested_path / sidecar).exists()

        view = view_cls()
        view.resize(560, 320)
        view.setModel(model)
        view.show()
        view.expand(nested_idx)
        app.processEvents()

        nested_idx = top("nested")
        view.scrollTo(nested_idx)
        app.processEvents()
        rect = view.sort_icon_rect(nested_idx)
        assert rect.isValid(), (label, "invalid sort icon rect")
        right_gap = view.viewport().width() - 1 - rect.right()
        assert 0 <= right_gap <= 24, (label, "sort icon not at far right", rect, right_gap)

        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
        app.processEvents()

        nested_idx = top("nested")
        assert model.directory_sort_descending(nested_path) is True
        assert child_files(nested_idx) == ["zeta", "alpha"]
        assert json.loads((nested_path / sidecar).read_text(encoding="utf-8")) == {"order": "descending"}

        view.close()
        app.processEvents()
        print(f"{label}: host GUI icon click/direct-child sort/persistence PASS", flush=True)


exercise("mdexplore", MarkdownDirectorySortProxyModel, MarkdownTreeView, ".mdexplore-sort.json")
exercise("pdfexplore", PdfDirectorySortProxyModel, PdfTreeView, ".pdfexplore-sort.json")
print("HOST_GUI_VALIDATION=PASS", flush=True)
PY
