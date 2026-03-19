"""Tab bar helpers for the multi-view preview strip."""

from __future__ import annotations

import math

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QStyle, QTabBar

from .icons import load_png_icon_two_tone, load_svg_icon_two_tone


class ViewTabBar(QTabBar):
    """Custom tab bar that paints dark-theme-friendly pastel tab backgrounds."""

    homeRequested = Signal(int)
    beginningResetRequested = Signal(int)

    PASTEL_SEQUENCE = [
        "#8fb8ff",
        "#9fd8c9",
        "#d7b8ff",
        "#f6c89f",
        "#f4b8c9",
        "#c8d8a0",
        "#b5d5f4",
        "#e8c6a7",
    ]
    WIDTH_TEMPLATE_TEXT = "999999"
    MAX_LABEL_CHARS = 48
    WIDTH_SIDE_PADDING = 10
    TAB_TEXT_BREATHING_ROOM = 14
    CUSTOM_LABEL_BREATHING_ROOM = 12
    POSITION_BAR_WIDTH = 26
    POSITION_BAR_HEIGHT = 8
    POSITION_BAR_TEXT_GAP = 7
    POSITION_BAR_SEGMENTS = 8
    HOME_ICON_GAP = 6
    RESET_ICON_GAP = 3
    LABEL_TO_RESET_ICON_GAP = 8
    TEXT_RIGHT_PADDING = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._home_icon = load_png_icon_two_tone(
            "home3.png",
            QColor("#425066"),
            QColor("#f8fafc"),
            size=64,
        )
        if self._home_icon.isNull():
            self._home_icon = load_svg_icon_two_tone(
                "home3.svg",
                QColor("#425066"),
                QColor("#f8fafc"),
                size=64,
            )
        if self._home_icon.isNull():
            self._home_icon = self.style().standardIcon(
                QStyle.StandardPixmap.SP_DirHomeIcon
            )
        self._reset_icon = load_png_icon_two_tone(
            "refresh.png",
            QColor("#425066"),
            QColor("#f8fafc"),
            size=64,
        )
        if self._reset_icon.isNull():
            self._reset_icon = load_svg_icon_two_tone(
                "refresh.svg",
                QColor("#425066"),
                QColor("#f8fafc"),
                size=64,
            )
        if self._reset_icon.isNull():
            self._reset_icon = self.style().standardIcon(
                QStyle.StandardPixmap.SP_BrowserReload
            )
        # Drag state is tracked here instead of relying on Qt's default drag
        # ghost so the whole tab, not just the close button, appears to move.
        self._drag_candidate_index = -1
        self._drag_start_pos = None
        self._dragging_index = -1
        self._dragging_tab_x = 0
        self._dragging_tab_offset_x = 0

    @staticmethod
    def _event_pos(event):
        """Return event position as QPoint across Qt6 API variants."""
        try:
            return event.position().toPoint()
        except Exception:
            return event.pos()

    def _is_close_button_hit(self, tab_index: int, pos) -> bool:
        """Detect whether a pointer position targets a tab close button."""
        button = self.tabButton(tab_index, QTabBar.ButtonPosition.RightSide)
        return bool(
            button is not None
            and button.isVisible()
            and button.geometry().contains(pos)
        )

    def _home_icon_size_px(self) -> int:
        """Size home icon roughly to the close-button glyph size."""
        close_w = self.style().pixelMetric(
            QStyle.PixelMetric.PM_TabCloseIndicatorWidth, None, self
        )
        close_h = self.style().pixelMetric(
            QStyle.PixelMetric.PM_TabCloseIndicatorHeight, None, self
        )
        icon_size = (
            min(close_w, close_h)
            if close_w > 0 and close_h > 0
            else max(close_w, close_h)
        )
        if icon_size <= 0:
            icon_size = 12
        icon_size = int(round(icon_size * 1.10))
        return max(11, min(20, icon_size))

    def _home_icon_rect(self, tab_index: int) -> QRect:
        """Return home icon hit box for a labeled tab, or empty rect."""
        if (
            tab_index < 0
            or tab_index >= self.count()
            or not self._tab_has_custom_label(tab_index)
        ):
            return QRect()
        rect = self.tabRect(tab_index).adjusted(2, 2, -2, -1)
        if rect.width() <= 2 or rect.height() <= 2:
            return QRect()
        icon_size = self._home_icon_size_px()
        icon_x = rect.left() + self.WIDTH_SIDE_PADDING - 1
        icon_y = rect.center().y() - (icon_size // 2)
        return QRect(icon_x, icon_y, icon_size, icon_size)

    def _reset_icon_rect(self, tab_index: int) -> QRect:
        """Return reset icon hit box for a labeled tab, or empty rect."""
        if (
            tab_index < 0
            or tab_index >= self.count()
            or not self._tab_has_custom_label(tab_index)
        ):
            return QRect()
        rect = self.tabRect(tab_index).adjusted(2, 2, -2, -1)
        if rect.width() <= 2 or rect.height() <= 2:
            return QRect()
        icon_size = self._home_icon_size_px()
        icon_right = rect.right() - self.WIDTH_SIDE_PADDING
        close_button = self.tabButton(tab_index, QTabBar.ButtonPosition.RightSide)
        if close_button is not None and close_button.isVisible():
            icon_right = close_button.geometry().left() - self.RESET_ICON_GAP - 1
        icon_x = icon_right - icon_size + 1
        if icon_x <= rect.left():
            return QRect()
        icon_y = rect.center().y() - (icon_size // 2)
        return QRect(icon_x, icon_y, icon_size, icon_size)

    def _is_reset_icon_hit(self, tab_index: int, pos) -> bool:
        """Detect whether a pointer position targets the reset-beginning icon."""
        rect = self._reset_icon_rect(tab_index)
        return bool(rect.isValid() and rect.contains(pos))

    def _set_all_close_buttons_visible(self, visible: bool) -> None:
        """Show/hide close buttons while custom drag ghost is active."""
        for index in range(self.count()):
            button = self.tabButton(index, QTabBar.ButtonPosition.RightSide)
            if button is not None:
                button.setVisible(visible)

    def _begin_tab_drag(self, tab_index: int, pos) -> None:
        """Start custom full-tab drag ghost for smoother visual feedback."""
        if tab_index < 0 or tab_index >= self.count():
            return
        rect = self.tabRect(tab_index)
        if not rect.isValid():
            return

        self._dragging_index = tab_index
        self._dragging_tab_offset_x = max(0, min(rect.width() - 1, pos.x() - rect.x()))
        self._dragging_tab_x = rect.x()
        self._set_all_close_buttons_visible(False)
        self.update()

    def _target_index_for_x(self, center_x: int) -> int:
        """Map cursor X position to closest insertion tab index."""
        if self.count() <= 0:
            return -1
        for index in range(self.count()):
            if center_x <= self.tabRect(index).center().x():
                return index
        return self.count() - 1

    def _update_tab_drag(self, pos) -> None:
        """Move drag ghost and reorder tabs as cursor crosses boundaries."""
        if self._dragging_index < 0:
            return
        self._dragging_tab_x = pos.x() - self._dragging_tab_offset_x
        target_index = self._target_index_for_x(pos.x())
        if target_index >= 0 and target_index != self._dragging_index:
            self.moveTab(self._dragging_index, target_index)
            self._dragging_index = target_index
        self.update()

    def _end_tab_drag(self) -> None:
        """Finish drag mode and restore normal close button visibility."""
        self._drag_candidate_index = -1
        self._drag_start_pos = None
        self._dragging_index = -1
        self._dragging_tab_x = 0
        self._dragging_tab_offset_x = 0
        self._set_all_close_buttons_visible(True)
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Record potential drag start while preserving regular tab behavior."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self._event_pos(event)
            tab_index = self.tabAt(pos)
            if tab_index >= 0 and self._is_reset_icon_hit(tab_index, pos):
                self.beginningResetRequested.emit(tab_index)
                event.accept()
                return
            if tab_index >= 0 and self._home_icon_rect(tab_index).contains(pos):
                self.homeRequested.emit(tab_index)
                event.accept()
                return
            if tab_index >= 0 and not self._is_close_button_hit(tab_index, pos):
                self._drag_candidate_index = tab_index
                self._drag_start_pos = pos
            else:
                self._drag_candidate_index = -1
                self._drag_start_pos = None
        else:
            self._drag_candidate_index = -1
            self._drag_start_pos = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        """Drive custom tab drag animation and reorder behavior."""
        if (
            self._drag_candidate_index >= 0
            and self._drag_start_pos is not None
            and (event.buttons() & Qt.MouseButton.LeftButton)
        ):
            pos = self._event_pos(event)
            if self._dragging_index < 0:
                if (
                    pos - self._drag_start_pos
                ).manhattanLength() >= QApplication.startDragDistance():
                    self._begin_tab_drag(self._drag_candidate_index, pos)
            if self._dragging_index >= 0:
                self._update_tab_drag(pos)
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        """End custom drag mode on release and continue normal processing."""
        if event.button() == Qt.MouseButton.LeftButton and self._dragging_index >= 0:
            self._end_tab_drag()
            super().mouseReleaseEvent(event)
            return
        self._drag_candidate_index = -1
        self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def _color_slot_for_index(self, tab_index: int) -> int:
        """Resolve palette slot from tab metadata, with sequence fallback."""
        palette_size = len(self.PASTEL_SEQUENCE)
        data = self.tabData(tab_index)
        if isinstance(data, dict):
            raw_slot = data.get("color_slot")
            try:
                slot = int(raw_slot)
            except Exception:
                slot = -1
            if 0 <= slot < palette_size:
                return slot
        sequence = self._sequence_for_index(tab_index)
        return (max(1, sequence) - 1) % palette_size

    def _sequence_for_index(self, tab_index: int) -> int:
        """Extract open-sequence index from tab data with backward compatibility."""
        data = self.tabData(tab_index)
        if isinstance(data, dict):
            raw = data.get("sequence")
            if isinstance(raw, int) and raw > 0:
                return raw
        if isinstance(data, int) and data > 0:
            return data
        return tab_index + 1

    def _base_color_for_index(self, tab_index: int) -> QColor:
        """Return base color for a tab from the configured pastel sequence."""
        color_slot = self._color_slot_for_index(tab_index)
        color_hex = self.PASTEL_SEQUENCE[color_slot]
        return QColor(color_hex)

    def _paint_single_tab(
        self,
        painter: QPainter,
        tab_index: int,
        rect,
        *,
        selected: bool,
        force_opaque: bool,
    ) -> None:
        """Paint one tab using shared logic for static and drag-ghost rendering."""
        base = self._base_color_for_index(tab_index)
        fill = QColor(base)
        if selected:
            fill = fill.lighter(107)
            if force_opaque:
                fill.setAlpha(255)
            else:
                fill.setAlpha(236)
            border = QColor(fill).darker(130)
        else:
            if force_opaque:
                fill.setAlpha(244)
            else:
                fill.setAlpha(172)
            border = QColor(fill).darker(155)

        painter.setPen(QPen(border, 1.1))
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, 6.0, 6.0)

        has_custom_label = self._tab_has_custom_label(tab_index)

        house_offset = 0
        reset_rect = QRect()
        if has_custom_label:
            # A house icon marks tabs that have an explicit labeled "beginning".
            icon_rect = self._home_icon_rect(tab_index)
            if icon_rect.isValid():
                icon_pixmap = self._home_icon.pixmap(icon_rect.size())
                if not icon_pixmap.isNull():
                    painter.drawPixmap(icon_rect, icon_pixmap)
                house_offset = icon_rect.width() + self.HOME_ICON_GAP
            reset_rect = self._reset_icon_rect(tab_index)
            if reset_rect.isValid():
                reset_pixmap = self._reset_icon.pixmap(reset_rect.size())
                if not reset_pixmap.isNull():
                    painter.drawPixmap(reset_rect, reset_pixmap)

        # Draw a compact segmented bargraph at the left to indicate each
        # tab's approximate position within the current document.
        bar_x = rect.left() + self.WIDTH_SIDE_PADDING - 1 + house_offset
        bar_y = rect.center().y() - (self.POSITION_BAR_HEIGHT // 2)
        bar_w = self.POSITION_BAR_WIDTH
        bar_h = self.POSITION_BAR_HEIGHT
        track_fill = QColor("#1f2937" if selected else "#182233")
        if force_opaque:
            track_fill.setAlpha(236 if selected else 214)
        else:
            track_fill.setAlpha(188 if selected else 152)
        track_border = QColor("#314156")
        painter.setPen(QPen(track_border, 0.9))
        painter.setBrush(track_fill)
        painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 2.2, 2.2)

        inner_x = bar_x + 1
        inner_y = bar_y + 1
        inner_w = max(1, bar_w - 2)
        inner_h = max(1, bar_h - 2)
        segments = self.POSITION_BAR_SEGMENTS
        segment_gap = 1
        segment_w = max(1, (inner_w - ((segments - 1) * segment_gap)) // segments)
        used_w = (segment_w * segments) + ((segments - 1) * segment_gap)
        start_x = inner_x + max(0, (inner_w - used_w) // 2)
        progress = self._progress_for_index(tab_index)
        filled_segments = int(round(progress * segments))
        if progress > 0.0 and filled_segments <= 0:
            filled_segments = 1
        filled_segments = max(0, min(segments, filled_segments))
        segment_active = QColor(base).darker(116 if selected else 128)
        segment_inactive = QColor("#425066")
        if force_opaque:
            segment_inactive.setAlpha(148 if selected else 126)
        else:
            segment_inactive.setAlpha(115 if selected else 92)
        painter.setPen(Qt.PenStyle.NoPen)
        for segment_index in range(segments):
            segment_x = start_x + (segment_index * (segment_w + segment_gap))
            segment_color = (
                segment_active if segment_index < filled_segments else segment_inactive
            )
            painter.setBrush(segment_color)
            painter.drawRect(segment_x, inner_y, segment_w, inner_h)

        text_left = bar_x + bar_w + self.POSITION_BAR_TEXT_GAP
        text_rect = rect.adjusted(
            text_left - rect.left(), 0, -self.TEXT_RIGHT_PADDING, 0
        )
        if reset_rect.isValid():
            text_rect.setRight(
                min(
                    text_rect.right(),
                    reset_rect.left() - self.LABEL_TO_RESET_ICON_GAP,
                )
            )
        close_button = self.tabButton(tab_index, QTabBar.ButtonPosition.RightSide)
        if close_button is not None and close_button.isVisible():
            text_rect.setRight(
                min(text_rect.right(), close_button.geometry().left() - 4)
            )

        text_color = QColor("#0b1220" if selected else "#1b2436")
        if not self.isTabEnabled(tab_index):
            text_color.setAlpha(130)
        painter.setPen(text_color)
        painter.drawText(
            text_rect, Qt.AlignmentFlag.AlignCenter, self.tabText(tab_index)
        )

    def _progress_for_index(self, tab_index: int) -> float:
        """Read normalized document-position progress (0..1) from tab metadata."""
        data = self.tabData(tab_index)
        if isinstance(data, dict):
            raw = data.get("progress")
            try:
                value = float(raw)
            except Exception:
                value = 0.0
            if math.isfinite(value):
                return max(0.0, min(1.0, value))
        return 0.0

    def _constant_tab_width(self) -> int:
        """Compute a stable tab width sized for six digits plus close button."""
        text_width = self.fontMetrics().horizontalAdvance(self.WIDTH_TEMPLATE_TEXT)
        close_width = 0
        if self.tabsClosable():
            close_width = (
                self.style().pixelMetric(
                    QStyle.PixelMetric.PM_TabCloseIndicatorWidth, None, self
                )
                + 8
            )
        return (
            text_width
            + (self.WIDTH_SIDE_PADDING * 2)
            + self.POSITION_BAR_WIDTH
            + self.POSITION_BAR_TEXT_GAP
            + close_width
        )

    def _tab_width_for_text(self, text: str) -> int:
        """Return tab width for one label, bounded below by the default width."""
        text_width = self.fontMetrics().horizontalAdvance(
            text or self.WIDTH_TEMPLATE_TEXT
        )
        close_width = 0
        if self.tabsClosable():
            close_width = (
                self.style().pixelMetric(
                    QStyle.PixelMetric.PM_TabCloseIndicatorWidth, None, self
                )
                + 8
            )
        dynamic_width = (
            text_width
            + self.TAB_TEXT_BREATHING_ROOM
            + (self.WIDTH_SIDE_PADDING * 2)
            + self.POSITION_BAR_WIDTH
            + self.POSITION_BAR_TEXT_GAP
            + close_width
        )
        return max(self._constant_tab_width(), dynamic_width)

    def _tab_has_custom_label(self, index: int) -> bool:
        """Return whether a tab currently has a non-empty custom label."""
        data = self.tabData(index)
        if not isinstance(data, dict):
            return False
        raw = data.get("custom_label")
        return isinstance(raw, str) and bool(raw.strip())

    def tabSizeHint(self, index: int) -> QSize:  # noqa: N802
        """Return per-tab width, expanding for custom labels when needed."""
        base = super().tabSizeHint(index)
        width = self._tab_width_for_text(self.tabText(index))
        if self._tab_has_custom_label(index):
            icon_space = self._home_icon_size_px()
            width += icon_space + self.HOME_ICON_GAP
            width += icon_space + self.RESET_ICON_GAP
            width += self.LABEL_TO_RESET_ICON_GAP
            width += self.CUSTOM_LABEL_BREATHING_ROOM
        return QSize(width, base.height())

    def minimumTabSizeHint(self, index: int) -> QSize:  # noqa: N802
        """Match minimum width to the rendered label width."""
        base = super().minimumTabSizeHint(index)
        width = self._tab_width_for_text(self.tabText(index))
        if self._tab_has_custom_label(index):
            icon_space = self._home_icon_size_px()
            width += icon_space + self.HOME_ICON_GAP
            width += icon_space + self.RESET_ICON_GAP
            width += self.LABEL_TO_RESET_ICON_GAP
            width += self.CUSTOM_LABEL_BREATHING_ROOM
        return QSize(width, base.height())

    def paintEvent(self, event) -> None:  # noqa: N802
        """Draw rounded pastel tabs while preserving built-in tab close buttons."""
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        for tab_index in range(self.count()):
            if self._dragging_index >= 0 and tab_index == self._dragging_index:
                # Draw active dragged tab as a floating ghost after static tabs.
                continue
            rect = self.tabRect(tab_index).adjusted(2, 2, -2, -1)
            if rect.width() <= 2 or rect.height() <= 2:
                continue

            selected = tab_index == self.currentIndex()
            self._paint_single_tab(
                painter, tab_index, rect, selected=selected, force_opaque=False
            )

        if self._dragging_index >= 0 and self.count() > 0:
            current_rect = self.tabRect(
                max(0, min(self._dragging_index, self.count() - 1))
            )
            ghost_rect = current_rect.adjusted(2, 2, -2, -1)
            if ghost_rect.width() <= 2 or ghost_rect.height() <= 2:
                painter.end()
                return
            draw_y = max(2, current_rect.y() + 2)
            draw_x = int(self._dragging_tab_x + 2)
            max_x = max(2, self.width() - ghost_rect.width() - 2)
            draw_x = max(2, min(max_x, draw_x))
            ghost_rect.moveLeft(draw_x)
            ghost_rect.moveTop(draw_y)
            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(5, 10, 18, 108))
            painter.drawRoundedRect(ghost_rect.adjusted(1, 1, 3, 3), 6.0, 6.0)
            selected = self._dragging_index == self.currentIndex()
            self._paint_single_tab(
                painter,
                self._dragging_index,
                ghost_rect,
                selected=selected,
                force_opaque=True,
            )
            painter.restore()

        painter.end()
