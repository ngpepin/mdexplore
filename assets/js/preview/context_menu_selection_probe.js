(() => {
  const sel = window.getSelection();
  const root = document.querySelector("main") || document.body;
  const hintedText = __SELECTED_HINT__;
  const skipTags = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEXTAREA"]);
  function splitTextPieces(value) {
    const source = typeof value === "string" ? value : "";
    const pieces = [];
    const whitespaceRe = /\s+/g;
    let cursor = 0;
    let match = null;
    while ((match = whitespaceRe.exec(source)) !== null) {
      if (match.index > cursor) {
        pieces.push({
          text: source.slice(cursor, match.index),
          countable: true,
        });
      }
      const raw = match[0];
      pieces.push({
        text: raw,
        countable: !/[\r\n\t]/.test(raw),
      });
      cursor = match.index + raw.length;
    }
    if (cursor < source.length) {
      pieces.push({
        text: source.slice(cursor),
        countable: true,
      });
    }
    return pieces.filter((piece) => piece.text.length > 0);
  }
  function countableText(value) {
    let out = "";
    for (const piece of splitTextPieces(value)) {
      if (piece.countable) out += piece.text;
    }
    return out;
  }
  function countableLength(value) {
    let total = 0;
    for (const piece of splitTextPieces(value)) {
      if (piece.countable) total += piece.text.length;
    }
    return total;
  }

  function shouldSkipTextNode(node) {
    if (!node || typeof node.nodeValue !== "string") return true;
    return countableLength(node.nodeValue) <= 0;
  }

  function lineInfo(node) {
    if (!node) return null;
    if (node.nodeType === Node.TEXT_NODE) node = node.parentElement;
    if (!(node instanceof Element)) return null;
    const el = node.closest('[data-md-line-start][data-md-line-end]');
    if (!el) return null;
    const start = parseInt(el.getAttribute("data-md-line-start"), 10);
    const end = parseInt(el.getAttribute("data-md-line-end"), 10);
    if (Number.isNaN(start) || Number.isNaN(end)) return null;
    return { start, end };
  }

  function normalizeRange(startInfo, endInfo) {
    let start = startInfo ? startInfo.start : endInfo.start;
    let end = endInfo ? endInfo.end : startInfo.end;
    if (start > end) {
      const tmp = start;
      start = end;
      end = tmp;
    }
    if (end <= start) end = start + 1;
    return { start, end };
  }

  function selectionOffsets(range) {
    if (!(range instanceof Range) || !root) return null;
    try {
      function textLengthToBoundary(container, offset) {
        const probe = document.createRange();
        probe.selectNodeContents(root);
        probe.setEnd(container, offset);
        const fragment = probe.cloneContents();
        const walker = document.createTreeWalker(
          fragment,
          NodeFilter.SHOW_TEXT,
          {
            acceptNode(node) {
              if (shouldSkipTextNode(node)) return NodeFilter.FILTER_REJECT;
              const parent = node.parentElement;
              if (!parent) return NodeFilter.FILTER_REJECT;
              if (skipTags.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
              return NodeFilter.FILTER_ACCEPT;
            },
          },
        );
        let total = 0;
        while (walker.nextNode()) {
          total += countableLength(walker.currentNode.nodeValue || "");
        }
        return total;
      }

      let start = textLengthToBoundary(range.startContainer, range.startOffset);
      let end = textLengthToBoundary(range.endContainer, range.endOffset);
      if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
      if (start > end) {
        const tmp = start;
        start = end;
        end = tmp;
      }
      if (end <= start) end = start + 1;
      return { start, end };
    } catch (_err) {
      return null;
    }
  }

  function clickTextOffset(clickX, clickY) {
    if (!root) return null;
    try {
      let pointRange = null;
      if (document.caretRangeFromPoint) {
        pointRange = document.caretRangeFromPoint(clickX, clickY);
      } else if (document.caretPositionFromPoint) {
        const pos = document.caretPositionFromPoint(clickX, clickY);
        if (pos && pos.offsetNode) {
          pointRange = document.createRange();
          pointRange.setStart(pos.offsetNode, pos.offset || 0);
          pointRange.collapse(true);
        }
      }
      if (!(pointRange instanceof Range)) return null;
      const probe = document.createRange();
      probe.selectNodeContents(root);
      probe.setEnd(pointRange.startContainer, pointRange.startOffset);
      const fragment = probe.cloneContents();
      const walker = document.createTreeWalker(
        fragment,
        NodeFilter.SHOW_TEXT,
        {
          acceptNode(node) {
            if (shouldSkipTextNode(node)) return NodeFilter.FILTER_REJECT;
            const parent = node.parentElement;
            if (!parent) return NodeFilter.FILTER_REJECT;
            if (skipTags.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
            return NodeFilter.FILTER_ACCEPT;
          },
        },
      );
      let offset = 0;
      while (walker.nextNode()) {
        offset += countableLength(walker.currentNode.nodeValue || "");
      }
      return Number.isFinite(offset) && offset >= 0 ? offset : null;
    } catch (_err) {
      return null;
    }
  }

  function rootTextContent() {
    if (!root) return "";
    const walker = document.createTreeWalker(
      root,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          if (shouldSkipTextNode(node)) return NodeFilter.FILTER_REJECT;
          const parent = node.parentElement;
          if (!parent) return NodeFilter.FILTER_REJECT;
          if (skipTags.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        },
      },
    );
    let out = "";
    while (walker.nextNode()) {
      out += countableText(walker.currentNode.nodeValue || "");
    }
    return out;
  }

  function nearestTextOffsets(selected, clickX, clickY) {
    if (!selected || !selected.length) return null;
    const haystack = rootTextContent();
    if (!haystack.length) return null;
    const candidates = [];
    const collapsed = selected.replace(/\s+/g, " ").trim();
    if (collapsed) candidates.push(collapsed);
    const trimmed = selected.trim();
    if (trimmed && !candidates.includes(trimmed)) candidates.push(trimmed);
    const noCR = selected.replace(/\r/g, "");
    if (noCR && !candidates.includes(noCR)) candidates.push(noCR);
    if (selected && !candidates.includes(selected)) candidates.push(selected);

    const clickOffset = clickTextOffset(clickX, clickY);
    let best = null;
    for (const candidate of candidates) {
      let idx = haystack.indexOf(candidate);
      while (idx >= 0) {
        const score = clickOffset === null ? 0 : Math.abs(idx - clickOffset);
        if (!best || score < best.score) {
          best = { start: idx, end: idx + candidate.length, score };
        }
        idx = haystack.indexOf(candidate, idx + Math.max(1, candidate.length));
      }
    }
    if (!best) return null;
    return { start: best.start, end: best.end };
  }

  function elementFromClick(x, y) {
    const dpr = window.devicePixelRatio || 1;
    const sx =
      window.scrollX ||
      window.pageXOffset ||
      document.documentElement.scrollLeft ||
      document.body.scrollLeft ||
      0;
    const sy =
      window.scrollY ||
      window.pageYOffset ||
      document.documentElement.scrollTop ||
      document.body.scrollTop ||
      0;
    const candidates = [
      [x, y],
      [x - sx, y - sy],
      [x + sx, y + sy],
      [x * dpr, y * dpr],
      [(x - sx) * dpr, (y - sy) * dpr],
      [x / dpr, y / dpr],
      [(x - sx) / dpr, (y - sy) / dpr],
    ];
    for (const pair of candidates) {
      if (!Number.isFinite(pair[0]) || !Number.isFinite(pair[1])) {
        continue;
      }
      const el = document.elementFromPoint(pair[0], pair[1]);
      if (el) return el;
    }
    return null;
  }

  function normalizeHighlightEntries(raw) {
    if (!Array.isArray(raw)) return [];
    const prepared = [];
    for (const item of raw) {
      if (!item || typeof item !== "object") continue;
      const id = typeof item.id === "string" ? item.id.trim() : "";
      const start = Number(item.start);
      const end = Number(item.end);
      if (!id || !Number.isFinite(start) || !Number.isFinite(end)) continue;
      if (end <= start || start < 0) continue;
      prepared.push({ id, start: Math.floor(start), end: Math.floor(end) });
    }
    if (!prepared.length) return [];
    prepared.sort((a, b) => (a.start - b.start) || (a.end - b.end));
    const merged = [];
    for (const item of prepared) {
      if (!merged.length) {
        merged.push(item);
        continue;
      }
      const prev = merged[merged.length - 1];
      if (item.start <= prev.end) {
        prev.end = Math.max(prev.end, item.end);
        continue;
      }
      merged.push(item);
    }
    return merged;
  }

  function overlapInfo(rangeStart, rangeEnd, entries) {
    const ids = new Set();
    const overlaps = [];
    for (const item of entries) {
      const start = Math.max(rangeStart, item.start);
      const end = Math.min(rangeEnd, item.end);
      if (end <= start) continue;
      overlaps.push({ start, end });
      ids.add(item.id);
    }
    if (!overlaps.length) {
      return { highlightedLength: 0, ids: [] };
    }
    overlaps.sort((a, b) => (a.start - b.start) || (a.end - b.end));
    const merged = [overlaps[0]];
    for (let i = 1; i < overlaps.length; i += 1) {
      const item = overlaps[i];
      const prev = merged[merged.length - 1];
      if (item.start <= prev.end) {
        prev.end = Math.max(prev.end, item.end);
      } else {
        merged.push(item);
      }
    }
    let highlightedLength = 0;
    for (const item of merged) {
      highlightedLength += Math.max(0, item.end - item.start);
    }
    return { highlightedLength, ids: Array.from(ids) };
  }

  function collectIntersectingRange(range) {
    if (!(range instanceof Range)) return null;
    let rootNode = range.commonAncestorContainer;
    if (rootNode && rootNode.nodeType === Node.TEXT_NODE) {
      rootNode = rootNode.parentElement;
    }
    const scope =
      rootNode instanceof Element
        ? rootNode
        : document.body || document.documentElement || document;
    const tagged = scope.querySelectorAll
      ? scope.querySelectorAll('[data-md-line-start][data-md-line-end]')
      : document.querySelectorAll('[data-md-line-start][data-md-line-end]');
    let minStart = null;
    let maxEnd = null;
    for (const el of tagged) {
      try {
        if (!range.intersectsNode(el)) continue;
      } catch (_err) {
        continue;
      }
      const start = parseInt(el.getAttribute("data-md-line-start") || "", 10);
      const end = parseInt(el.getAttribute("data-md-line-end") || "", 10);
      if (Number.isNaN(start) || Number.isNaN(end)) continue;
      minStart = minStart === null ? start : Math.min(minStart, start);
      maxEnd = maxEnd === null ? end : Math.max(maxEnd, end);
    }
    if (minStart === null || maxEnd === null) return null;
    return { start: minStart, end: Math.max(minStart + 1, maxEnd) };
  }

  let selectedText = sel && sel.toString ? sel.toString() : "";
  if (!selectedText.trim() && hintedText) {
    selectedText = hintedText;
  }
  const hasSelection = !!(selectedText && selectedText.trim());
  const highlightEntries = normalizeHighlightEntries(
    window.__mdexplorePersistentHighlights || []
  );
  const fallbackClickX =
    typeof window.__mdexploreLastContextClientX === "number" &&
    Number.isFinite(window.__mdexploreLastContextClientX)
      ? window.__mdexploreLastContextClientX
      : null;
  const fallbackClickY =
    typeof window.__mdexploreLastContextClientY === "number" &&
    Number.isFinite(window.__mdexploreLastContextClientY)
      ? window.__mdexploreLastContextClientY
      : null;
  const clickedNode =
    elementFromClick(__CLICK_X__, __CLICK_Y__) ||
    (Number.isFinite(fallbackClickX) && Number.isFinite(fallbackClickY)
      ? elementFromClick(fallbackClickX, fallbackClickY)
      : null);
  const clickedHighlight =
    clickedNode && clickedNode.closest
      ? clickedNode.closest('span[data-mdexplore-persistent-highlight="1"]')
      : null;
  const fallbackClickedHighlightId =
    typeof window.__mdexploreLastPersistentHighlightId === "string"
      ? window.__mdexploreLastPersistentHighlightId.trim()
      : "";
  const clickedHighlightId = clickedHighlight
    ? String(
        clickedHighlight.getAttribute("data-mdexplore-persistent-highlight-id") || ""
      )
    : fallbackClickedHighlightId;
  let clickedOffset = clickTextOffset(__CLICK_X__, __CLICK_Y__);
  if (
    clickedOffset === null &&
    Number.isFinite(fallbackClickX) &&
    Number.isFinite(fallbackClickY)
  ) {
    clickedOffset = clickTextOffset(fallbackClickX, fallbackClickY);
  }
  if (
    clickedOffset === null &&
    typeof window.__mdexploreLastPersistentHighlightOffset === "number" &&
    Number.isFinite(window.__mdexploreLastPersistentHighlightOffset)
  ) {
    clickedOffset = Math.max(
      0,
      Math.floor(window.__mdexploreLastPersistentHighlightOffset)
    );
  }
  let selectionOffsetStart = null;
  let selectionOffsetEnd = null;
  let selectionHasHighlightedPart = false;
  let selectionHasUnhighlightedPart = false;
  let selectedHighlightIds = [];

  // Preferred path: map the active text selection to source line metadata.
  if (sel && sel.rangeCount > 0 && !sel.isCollapsed) {
    const range = sel.getRangeAt(0);
    const offsets = selectionOffsets(range);
    if (offsets) {
      selectionOffsetStart = offsets.start;
      selectionOffsetEnd = offsets.end;
      const overlap = overlapInfo(offsets.start, offsets.end, highlightEntries);
      selectedHighlightIds = overlap.ids;
      selectionHasHighlightedPart = overlap.highlightedLength > 0;
      selectionHasUnhighlightedPart =
        overlap.highlightedLength < Math.max(1, offsets.end - offsets.start);
    } else {
      selectionHasUnhighlightedPart = true;
    }
    const intersecting = collectIntersectingRange(range);
    if (intersecting) {
      return {
        hasSelection: true,
        selectedText,
        ...intersecting,
        via: "selection-intersects",
        selectionOffsetStart,
        selectionOffsetEnd,
        selectionHasHighlightedPart,
        selectionHasUnhighlightedPart,
        selectedHighlightIds,
        clickedHighlightId,
        clickedOffset,
      };
    }
    const startInfo = lineInfo(range.startContainer);
    const endInfo = lineInfo(range.endContainer);
    if (startInfo || endInfo) {
      return {
        hasSelection: true,
        selectedText,
        ...normalizeRange(startInfo, endInfo),
        via: "selection",
        selectionOffsetStart,
        selectionOffsetEnd,
        selectionHasHighlightedPart,
        selectionHasUnhighlightedPart,
        selectedHighlightIds,
        clickedHighlightId,
        clickedOffset,
      };
    }
  }

  // Robust fallback: selection may collapse before the context menu action
  // is handled. Recover offsets using selected text near the click location.
  if (
    (selectionOffsetStart === null || selectionOffsetEnd === null) &&
    selectedText
  ) {
    const guessed = nearestTextOffsets(selectedText, __CLICK_X__, __CLICK_Y__);
    if (guessed && guessed.end > guessed.start) {
      selectionOffsetStart = guessed.start;
      selectionOffsetEnd = guessed.end;
      const overlap = overlapInfo(
        guessed.start,
        guessed.end,
        highlightEntries
      );
      selectedHighlightIds = overlap.ids;
      selectionHasHighlightedPart = overlap.highlightedLength > 0;
      selectionHasUnhighlightedPart =
        overlap.highlightedLength < Math.max(1, guessed.end - guessed.start);
    }
  }

  // Fallback: map from right-clicked block location.
  const clicked = elementFromClick(__CLICK_X__, __CLICK_Y__);
  const clickedInfo = lineInfo(clicked);
  if (clickedInfo) {
    return {
      hasSelection,
      selectedText,
      start: clickedInfo.start,
      end: clickedInfo.end,
      via: "click",
      selectionOffsetStart,
      selectionOffsetEnd,
      selectionHasHighlightedPart,
      selectionHasUnhighlightedPart,
      selectedHighlightIds,
      clickedHighlightId,
      clickedOffset,
    };
  }

  return {
    hasSelection,
    selectedText,
    selectionOffsetStart,
    selectionOffsetEnd,
    selectionHasHighlightedPart,
    selectionHasUnhighlightedPart: hasSelection ? true : false,
    selectedHighlightIds,
    clickedHighlightId,
    clickedOffset,
  };
})();
