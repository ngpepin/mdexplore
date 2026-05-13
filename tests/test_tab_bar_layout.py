from __future__ import annotations

import unittest

from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from mdexplore_app.tabs import ViewTabBar


class ViewTabBarLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.host = QWidget()
        layout = QVBoxLayout(self.host)
        layout.setContentsMargins(0, 0, 0, 0)
        self.bar = ViewTabBar()
        self.bar.setDocumentMode(True)
        self.bar.setMovable(False)
        self.bar.setDrawBase(False)
        self.bar.setExpanding(False)
        self.bar.setUsesScrollButtons(True)
        self.bar.setTabsClosable(True)
        layout.addWidget(self.bar)

        labels = ["3910", "ATS keywords", "Upwork Strategy", "Upwork Pro"]
        for idx, text in enumerate(labels):
            tab_index = self.bar.addTab(text)
            self.bar.setTabData(
                tab_index,
                {
                    "view_id": idx + 1,
                    "sequence": idx + 1,
                    "color_slot": idx,
                    "progress": 0.5,
                    "custom_label": None if idx == 0 else text,
                    "custom_label_anchor_scroll_y": 0.0,
                    "custom_label_anchor_top_line": 1,
                },
            )
        self.bar.setCurrentIndex(2)
        self.host.resize(1100, 64)
        self.host.show()
        QApplication.processEvents()

    def tearDown(self) -> None:
        self.host.close()
        QApplication.processEvents()

    def test_custom_labeled_tab_text_rect_fits_full_label(self) -> None:
        tab_index = 3
        rect = self.bar.tabRect(tab_index).adjusted(2, 2, -2, -1)
        text_rect = self.bar._label_text_rect(
            tab_index, rect, self.bar._reset_icon_rect(tab_index)
        )
        text_width = self.bar.fontMetrics().horizontalAdvance(self.bar.tabText(tab_index))
        self.assertGreaterEqual(
            text_rect.width(),
            text_width,
            "Custom-labeled tab text budget should fit the full label",
        )

    def test_stale_close_button_geometry_uses_right_edge_fallback(self) -> None:
        tab_index = 3
        rect = self.bar.tabRect(tab_index).adjusted(2, 2, -2, -1)
        button = self.bar.tabButton(tab_index, ViewTabBar.ButtonPosition.RightSide)
        self.assertIsNotNone(button)
        button = button
        button.setGeometry(rect.left() + 24, rect.top() + 3, 20, 20)

        close_rect = self.bar._effective_close_button_rect(tab_index, rect)
        text_rect = self.bar._label_text_rect(
            tab_index, rect, self.bar._reset_icon_rect(tab_index)
        )
        text_width = self.bar.fontMetrics().horizontalAdvance(self.bar.tabText(tab_index))

        self.assertGreaterEqual(
            close_rect.left(),
            rect.center().x(),
            "Fallback close-button slot should remain on the right side of the tab",
        )
        self.assertGreaterEqual(
            text_rect.width(),
            text_width,
            "Stale close-button geometry should not shrink the tab label budget",
        )


if __name__ == "__main__":
    unittest.main()
