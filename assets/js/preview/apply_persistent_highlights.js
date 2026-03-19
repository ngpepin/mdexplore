(() => {
  const incoming = __PAYLOAD__;
  const highlightColor = __COLOR__;
  const importantHighlightColor = __IMPORTANT_COLOR__;
  const importantHighlightTextColor = __IMPORTANT_TEXT_COLOR__;
  const highlightMarkerColor = __MARKER_COLOR__;
  const importantHighlightMarkerColor = __IMPORTANT_MARKER_COLOR__;
  window.__mdexplorePersistentHighlightMarkerColor = highlightMarkerColor;
  window.__mdexplorePersistentHighlightImportantMarkerColor =
    importantHighlightMarkerColor;
  const root = document.querySelector("main") || document.body;
  if (!root) {
    window.__mdexplorePersistentHighlights = [];
    return { applied: 0, entries: 0 };
  }

  function normalizeEventTarget(rawTarget) {
    if (rawTarget instanceof Element) {
      return rawTarget;
    }
    if (
      rawTarget &&
      rawTarget.nodeType === Node.TEXT_NODE &&
      rawTarget.parentElement instanceof Element
    ) {
      return rawTarget.parentElement;
    }
    return null;
  }

  function textOffsetFromPoint(clickX, clickY) {
    if (!Number.isFinite(clickX) || !Number.isFinite(clickY) || !root) {
      return null;
    }
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
      if (!(pointRange instanceof Range)) {
        return null;
      }
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
        offset += (walker.currentNode.nodeValue || "").length;
      }
      return Number.isFinite(offset) && offset >= 0 ? Math.floor(offset) : null;
    } catch (_err) {
      return null;
    }
  }

  if (!window.__mdexplorePersistentHighlightContextHooked) {
    const updateLastClickedHighlight = (event) => {
      let clickedHighlightId = "";
      let clickedOffset = null;
      try {
        const eventType = event && typeof event.type === "string" ? event.type : "";
        const clickX =
          event && Number.isFinite(event.clientX) ? event.clientX : null;
        const clickY =
          event && Number.isFinite(event.clientY) ? event.clientY : null;
        if (Number.isFinite(clickX) && Number.isFinite(clickY)) {
          window.__mdexploreLastContextClientX = clickX;
          window.__mdexploreLastContextClientY = clickY;
        }

        let target = normalizeEventTarget(event ? event.target : null);
        if (!target && Number.isFinite(clickX) && Number.isFinite(clickY)) {
          target = normalizeEventTarget(document.elementFromPoint(clickX, clickY));
        }
        const mark =
          target && target.closest
            ? target.closest('span[data-mdexplore-persistent-highlight="1"]')
            : null;
        clickedHighlightId = mark
          ? String(mark.getAttribute("data-mdexplore-persistent-highlight-id") || "").trim()
          : "";

        // Heavy text-offset probing is intentionally limited to context-menu
        // events to keep normal preview interactions responsive.
        if (
          eventType === "contextmenu" &&
          !clickedHighlightId &&
          Number.isFinite(clickX) &&
          Number.isFinite(clickY)
        ) {
          clickedOffset = textOffsetFromPoint(clickX, clickY);
        }

        if (
          eventType === "contextmenu" &&
          !clickedHighlightId &&
          Number.isFinite(clickedOffset)
        ) {
          const entries = Array.isArray(window.__mdexplorePersistentHighlights)
            ? window.__mdexplorePersistentHighlights
            : [];
          for (const item of entries) {
            if (!item || typeof item !== "object") continue;
            const id = typeof item.id === "string" ? item.id.trim() : "";
            const start = Number(item.start);
            const end = Number(item.end);
            if (!id || !Number.isFinite(start) || !Number.isFinite(end)) continue;
            if (Math.floor(start) <= clickedOffset && clickedOffset <= Math.floor(end)) {
              clickedHighlightId = id;
              break;
            }
          }
        }
      } catch (_err) {
        clickedHighlightId = "";
        clickedOffset = null;
      }
      window.__mdexploreLastPersistentHighlightId = clickedHighlightId;
      window.__mdexploreLastPersistentHighlightOffset = Number.isFinite(clickedOffset)
        ? Math.floor(clickedOffset)
        : null;
    };
    document.addEventListener("contextmenu", updateLastClickedHighlight, true);
    document.addEventListener("mousedown", updateLastClickedHighlight, true);
    window.__mdexplorePersistentHighlightContextHooked = true;
  }
  if (typeof window.__mdexploreLastPersistentHighlightId !== "string") {
    window.__mdexploreLastPersistentHighlightId = "";
  }
  if (
    typeof window.__mdexploreLastPersistentHighlightOffset !== "number" ||
    !Number.isFinite(window.__mdexploreLastPersistentHighlightOffset)
  ) {
    window.__mdexploreLastPersistentHighlightOffset = null;
  }
  if (
    typeof window.__mdexploreLastContextClientX !== "number" ||
    !Number.isFinite(window.__mdexploreLastContextClientX)
  ) {
    window.__mdexploreLastContextClientX = null;
  }
  if (
    typeof window.__mdexploreLastContextClientY !== "number" ||
    !Number.isFinite(window.__mdexploreLastContextClientY)
  ) {
    window.__mdexploreLastContextClientY = null;
  }

  function shouldSkipTextNode(node) {
    if (!node || typeof node.nodeValue !== "string") return true;
    const value = node.nodeValue;
    if (!value.length) return true;
    // Ignore formatting-only whitespace that contains newlines/tabs so
    // highlights do not materialize as visual linefeed artifacts.
    if (!/[^\s]/.test(value) && /[\r\n\t]/.test(value)) return true;
    return false;
  }

  for (const mark of Array.from(root.querySelectorAll('span[data-mdexplore-persistent-highlight="1"]'))) {
    const parent = mark.parentNode;
    if (!parent) continue;
    parent.replaceChild(document.createTextNode(mark.textContent || ""), mark);
    parent.normalize();
  }

  function normalizeEntries(raw) {
    if (!Array.isArray(raw)) return [];
    const prepared = [];
    for (const item of raw) {
      if (!item || typeof item !== "object") continue;
      const id = typeof item.id === "string" ? item.id.trim() : "";
      const start = Number(item.start);
      const end = Number(item.end);
      const kind =
        String(item.kind || "__NORMAL_KIND__").trim().toLowerCase() === "__IMPORTANT_KIND__"
          ? "__IMPORTANT_KIND__"
          : "__NORMAL_KIND__";
      if (!id || !Number.isFinite(start) || !Number.isFinite(end)) continue;
      if (end <= start || start < 0) continue;
      prepared.push({ id, start: Math.floor(start), end: Math.floor(end), kind });
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
      if (item.kind === prev.kind && item.start <= prev.end) {
        prev.end = Math.max(prev.end, item.end);
      } else {
        merged.push(item);
      }
    }
    return merged;
  }

  const entries = normalizeEntries(incoming);
  window.__mdexplorePersistentHighlights = entries;
  if (!entries.length) {
    if (typeof window.__mdexploreRefreshPersistentHighlightMarkers === "function") {
      window.__mdexploreRefreshPersistentHighlightMarkers();
    }
    return { applied: 0, entries: 0 };
  }

  const skipTags = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEXTAREA"]);
  const walker = document.createTreeWalker(
    root,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        if (shouldSkipTextNode(node)) {
          return NodeFilter.FILTER_REJECT;
        }
        const parent = node.parentElement;
        if (!parent) {
          return NodeFilter.FILTER_REJECT;
        }
        if (skipTags.has(parent.tagName)) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    },
  );

  const segments = [];
  let totalLength = 0;
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const text = node.nodeValue || "";
    const start = totalLength;
    const end = start + text.length;
    segments.push({ node, text, start, end });
    totalLength = end;
  }

  if (!segments.length) {
    return {
      applied: 0,
      entries: entries.length,
      segments: 0,
      totalLength: 0,
    };
  }

  let applied = 0;
  let entryIndex = 0;
  for (const segment of segments) {
    while (entryIndex < entries.length && entries[entryIndex].end <= segment.start) {
      entryIndex += 1;
    }
    let localRanges = [];
    for (let idx = entryIndex; idx < entries.length; idx += 1) {
      const entry = entries[idx];
      if (entry.start >= segment.end) {
        break;
      }
      const overlapStart = Math.max(entry.start, segment.start);
      const overlapEnd = Math.min(entry.end, segment.end);
      if (overlapEnd > overlapStart) {
        localRanges.push({
          start: overlapStart - segment.start,
          end: overlapEnd - segment.start,
          id: entry.id,
          kind: entry.kind,
        });
      }
    }
    if (!localRanges.length) {
      continue;
    }

    localRanges.sort((a, b) => a.start - b.start);
    const fragment = document.createDocumentFragment();
    let cursor = 0;
    for (const range of localRanges) {
      if (range.start > cursor) {
        fragment.appendChild(
          document.createTextNode(segment.text.slice(cursor, range.start))
        );
      }
      const mark = document.createElement("span");
      mark.setAttribute("data-mdexplore-persistent-highlight", "1");
      mark.setAttribute("data-mdexplore-persistent-highlight-id", range.id);
      mark.setAttribute(
        "data-mdexplore-persistent-highlight-kind",
        range.kind === "__IMPORTANT_KIND__" ? "__IMPORTANT_KIND__" : "__NORMAL_KIND__"
      );
      const isImportant = range.kind === "__IMPORTANT_KIND__";
      mark.style.backgroundColor = isImportant
        ? importantHighlightColor
        : highlightColor;
      mark.style.color = isImportant ? importantHighlightTextColor : "";
      mark.style.borderRadius = "2px";
      mark.style.padding = "0 1px";
      mark.style.boxDecorationBreak = "clone";
      mark.style.webkitBoxDecorationBreak = "clone";
      mark.textContent = segment.text.slice(range.start, range.end);
      fragment.appendChild(mark);
      cursor = range.end;
      applied += 1;
    }
    if (cursor < segment.text.length) {
      fragment.appendChild(document.createTextNode(segment.text.slice(cursor)));
    }
    const parent = segment.node.parentNode;
    if (parent) {
      parent.replaceChild(fragment, segment.node);
    }
  }

  if (typeof window.__mdexploreRefreshPersistentHighlightMarkers === "function") {
    window.__mdexploreRefreshPersistentHighlightMarkers();
  }
  return {
    applied,
    entries: entries.length,
    segments: segments.length,
    totalLength,
  };
})();
