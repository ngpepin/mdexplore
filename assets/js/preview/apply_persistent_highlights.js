(() => {
  const incoming = __PAYLOAD__;
  const highlightColor = __COLOR__;
  const importantHighlightColor = __IMPORTANT_COLOR__;
  const importantHighlightTextColor = __IMPORTANT_TEXT_COLOR__;
  const highlightMarkerColor = __MARKER_COLOR__;
  const importantHighlightMarkerColor = __IMPORTANT_MARKER_COLOR__;
  const previewOffsetSpace = "__OFFSET_SPACE_PREVIEW__";
  const sourceOffsetSpace = "__OFFSET_SPACE_SOURCE__";
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

  function splitTextPieces(value) {
    const source = typeof value === "string" ? value : "";
    const pieces = [];
    const whitespaceRe = /\s+/g;
    let cursor = 0;
    let match = null;
    while ((match = whitespaceRe.exec(source)) !== null) {
      if (match.index > cursor) {
        pieces.push({
          rawStart: cursor,
          rawEnd: match.index,
          text: source.slice(cursor, match.index),
          countable: true,
        });
      }
      const raw = match[0];
      pieces.push({
        rawStart: match.index,
        rawEnd: match.index + raw.length,
        text: raw,
        countable: !/[\r\n\t]/.test(raw),
      });
      cursor = match.index + raw.length;
    }
    if (cursor < source.length) {
      pieces.push({
        rawStart: cursor,
        rawEnd: source.length,
        text: source.slice(cursor),
        countable: true,
      });
    }
    return pieces.filter((piece) => piece.rawEnd > piece.rawStart);
  }

  function countableLength(value) {
    let total = 0;
    for (const piece of splitTextPieces(value)) {
      if (piece.countable) total += piece.text.length;
    }
    return total;
  }

  function normalizeSearchText(value) {
    return String(value || "").replace(/\s+/g, " ").trim();
  }

  function buildCompactSearchIndex(value) {
    const source = typeof value === "string" ? value : "";
    let compact = "";
    const map = [];
    for (let index = 0; index < source.length; index += 1) {
      const ch = source[index];
      if (/\s/.test(ch)) continue;
      compact += ch;
      map.push(index);
    }
    return { text: compact, map };
  }

  function buildAnchorCandidates(value) {
    const normalized = normalizeSearchText(value);
    if (!normalized) return [];
    const variants = [];
    const pushCandidate = (candidate) => {
      const normalizedCandidate = normalizeSearchText(candidate);
      if (normalizedCandidate.length < 12) return;
      if (!variants.includes(normalizedCandidate)) {
        variants.push(normalizedCandidate);
      }
    };
    pushCandidate(normalized);
    const stripped = normalized.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "").trim();
    pushCandidate(stripped);
    const boundaries = [0];
    for (let index = 0; index < normalized.length; index += 1) {
      if (normalized[index] === " " && index + 1 < normalized.length) {
        boundaries.push(index + 1);
      }
    }
    for (const start of boundaries.slice(0, 16)) {
      pushCandidate(normalized.slice(start, start + 220));
    }
    if (normalized.length > 220) {
      pushCandidate(normalized.slice(-220));
    }
    return variants;
  }

  function buildPreviewAnchorCandidates(value) {
    const normalized = normalizeSearchText(value);
    if (!normalized) return [];
    const candidates = [];
    const pushCandidate = (candidate) => {
      const text = normalizeSearchText(candidate);
      if (text.length < 3) return;
      if (!candidates.includes(text)) {
        candidates.push(text);
      }
    };
    pushCandidate(normalized);
    pushCandidate(normalized.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ""));
    return candidates;
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
        offset += countableLength(walker.currentNode.nodeValue || "");
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
    return countableLength(node.nodeValue) <= 0;
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
    const normalizeCandidateArray = (value) => {
      if (!Array.isArray(value)) return [];
      const unique = [];
      for (const item of value) {
        const normalizedCandidate = normalizeSearchText(item);
        if (normalizedCandidate.length < 3) continue;
        if (!unique.includes(normalizedCandidate)) {
          unique.push(normalizedCandidate);
        }
      }
      return unique;
    };
    for (const item of raw) {
      if (!item || typeof item !== "object") continue;
      const id = typeof item.id === "string" ? item.id.trim() : "";
      const start = Number(item.start);
      const end = Number(item.end);
      const kind =
        String(item.kind || "__NORMAL_KIND__").trim().toLowerCase() === "__IMPORTANT_KIND__"
          ? "__IMPORTANT_KIND__"
          : "__NORMAL_KIND__";
      const offsetSpace = String(
        item.offset_space || item.offsetSpace || ""
      ).trim().toLowerCase();
      const legacyAnchorText = normalizeSearchText(
        item.legacy_anchor_text || item.legacyAnchorText || ""
      );
      const legacyDirectCandidates = normalizeCandidateArray(
        item.legacy_direct_candidates || item.legacyDirectCandidates
      );
      const legacyContextCandidates = normalizeCandidateArray(
        item.legacy_context_candidates || item.legacyContextCandidates
      );
      const previewAnchorText = normalizeSearchText(
        item.anchor_text || item.anchorText || item.previewAnchorText || ""
      );
      if (!id || !Number.isFinite(start) || !Number.isFinite(end)) continue;
      if (end <= start || start < 0) continue;
      prepared.push({
        id,
        start: Math.floor(start),
        end: Math.floor(end),
        kind,
        offsetSpace:
          offsetSpace === previewOffsetSpace || offsetSpace === sourceOffsetSpace
            ? offsetSpace
            : "",
        previewAnchorText,
        legacyAnchorText,
        legacyDirectCandidates,
        legacyContextCandidates,
      });
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

  const nodeRecords = [];
  let logicalText = "";
  let totalLength = 0;
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const text = node.nodeValue || "";
    const pieces = splitTextPieces(text).map((piece) => {
      const record = {
        ...piece,
        start: totalLength,
        end: totalLength,
      };
      if (piece.countable) {
        record.end = totalLength + piece.text.length;
        totalLength = record.end;
        logicalText += piece.text;
      }
      return record;
    });
    nodeRecords.push({ node, pieces });
  }

  function resolveLegacyEntry(entry, compactIndex) {
    if (!entry || typeof entry !== "object") return entry;
    const candidateGroups = [];
    if (entry.offsetSpace === previewOffsetSpace) {
      const previewCandidates = buildPreviewAnchorCandidates(entry.previewAnchorText);
      if (!previewCandidates.length) {
        return entry;
      }
      candidateGroups.push(previewCandidates);
    } else {
      candidateGroups.push(
        Array.isArray(entry.legacyDirectCandidates) ? entry.legacyDirectCandidates : []
      );
      candidateGroups.push(
        Array.isArray(entry.legacyContextCandidates) && entry.legacyContextCandidates.length
          ? entry.legacyContextCandidates
          : buildAnchorCandidates(entry.legacyAnchorText)
      );
    }
    const compactCurrentText = String(logicalText.slice(entry.start, entry.end) || "").replace(
      /\s+/g,
      ""
    );
    const currentOffsetsAlreadyMatch =
      entry.offsetSpace !== sourceOffsetSpace &&
      !!compactCurrentText &&
      Array.isArray(candidateGroups[0]) &&
      candidateGroups[0].some((candidate) => {
        const compactCandidate = candidate.replace(/\s+/g, "");
        return !!compactCandidate && compactCurrentText === compactCandidate;
      });
    if (currentOffsetsAlreadyMatch) {
      return {
        ...entry,
        offsetSpace: previewOffsetSpace,
      };
    }
    let best = null;
    const entryLength = Math.max(1, Math.floor(entry.end - entry.start));
    for (let groupIndex = 0; groupIndex < candidateGroups.length; groupIndex += 1) {
      const candidates = candidateGroups[groupIndex];
      let groupBest = null;
      for (const candidate of candidates) {
        const compactCandidate = candidate.replace(/\s+/g, "");
        if (compactCandidate.length < (groupIndex === 0 ? 3 : 8)) continue;
        let matchIndex = compactIndex.text.indexOf(compactCandidate);
        while (matchIndex >= 0) {
          const rawStart = compactIndex.map[matchIndex];
          const rawEnd =
            compactIndex.map[matchIndex + compactCandidate.length - 1] + 1;
          const score = [
            Math.abs(candidate.length - entryLength),
            -candidate.length,
            Math.abs(rawStart - entry.start),
            rawStart,
          ];
          if (
            !groupBest ||
            score[0] < groupBest.score[0] ||
            (score[0] === groupBest.score[0] && score[1] < groupBest.score[1]) ||
            (score[0] === groupBest.score[0] &&
              score[1] === groupBest.score[1] &&
              score[2] < groupBest.score[2]) ||
            (score[0] === groupBest.score[0] &&
              score[1] === groupBest.score[1] &&
              score[2] === groupBest.score[2] &&
              score[3] < groupBest.score[3])
          ) {
            groupBest = { rawStart, rawEnd, score };
          }
          matchIndex = compactIndex.text.indexOf(compactCandidate, matchIndex + 1);
        }
      }
      if (groupBest) {
        best = groupBest;
        break;
      }
    }
    if (!best) return entry;
    return {
      ...entry,
      start: Math.floor(best.rawStart),
      end: Math.floor(best.rawEnd),
      offsetSpace: previewOffsetSpace,
    };
  }

  const compactIndex = buildCompactSearchIndex(logicalText);
  const resolvedEntries = normalizeEntries(
    entries.map((entry) => resolveLegacyEntry(entry, compactIndex))
  );
  window.__mdexplorePersistentHighlights = resolvedEntries;
  if (!resolvedEntries.length) {
    if (typeof window.__mdexploreRefreshPersistentHighlightMarkers === "function") {
      window.__mdexploreRefreshPersistentHighlightMarkers();
    }
    return { applied: 0, entries: 0 };
  }

  if (!nodeRecords.length) {
    return {
      applied: 0,
      entries: resolvedEntries.length,
      segments: 0,
      totalLength: 0,
    };
  }

  let applied = 0;
  let countableSegmentCount = 0;
  for (const record of nodeRecords) {
    const fragment = document.createDocumentFragment();
    let nodeChanged = false;
    for (const piece of record.pieces) {
      if (!piece.countable) {
        fragment.appendChild(document.createTextNode(piece.text));
        continue;
      }
      countableSegmentCount += 1;
      let localRanges = [];
      for (const entry of resolvedEntries) {
        if (entry.end <= piece.start) {
          continue;
        }
        if (entry.start >= piece.end) {
          break;
        }
        const overlapStart = Math.max(entry.start, piece.start);
        const overlapEnd = Math.min(entry.end, piece.end);
        if (overlapEnd > overlapStart) {
          localRanges.push({
            start: overlapStart - piece.start,
            end: overlapEnd - piece.start,
            id: entry.id,
            kind: entry.kind,
          });
        }
      }
      if (!localRanges.length) {
        fragment.appendChild(document.createTextNode(piece.text));
        continue;
      }

      nodeChanged = true;
      localRanges.sort((a, b) => a.start - b.start);
      let cursor = 0;
      for (const range of localRanges) {
        if (range.start > cursor) {
          fragment.appendChild(
            document.createTextNode(piece.text.slice(cursor, range.start))
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
        mark.textContent = piece.text.slice(range.start, range.end);
        fragment.appendChild(mark);
        cursor = range.end;
        applied += 1;
      }
      if (cursor < piece.text.length) {
        fragment.appendChild(document.createTextNode(piece.text.slice(cursor)));
      }
    }
    const parent = record.node.parentNode;
    if (nodeChanged && parent) {
      parent.replaceChild(fragment, record.node);
    }
  }

  if (typeof window.__mdexploreRefreshPersistentHighlightMarkers === "function") {
    window.__mdexploreRefreshPersistentHighlightMarkers();
  }
  return {
    applied,
    entries: resolvedEntries.length,
    segments: countableSegmentCount,
    totalLength,
    resolvedEntries: resolvedEntries.map((entry) => ({
      id: entry.id,
      start: entry.start,
      end: entry.end,
      kind: entry.kind,
      anchor_text: entry.previewAnchorText || "",
      offset_space:
        entry.offsetSpace === sourceOffsetSpace
          ? sourceOffsetSpace
          : previewOffsetSpace,
    })),
  };
})();
