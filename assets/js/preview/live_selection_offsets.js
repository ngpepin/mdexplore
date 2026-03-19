(() => {
  const root = document.querySelector("main") || document.body;
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
  if (!root) {
    return {
      hasSelection: false,
      selectedText: "",
      selectionOffsetStart: null,
      selectionOffsetEnd: null,
    };
  }

  function collectRootText() {
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
    let text = "";
    while (walker.nextNode()) {
      text += countableText(walker.currentNode.nodeValue || "");
    }
    return text;
  }

  function compactSearchIndex(value) {
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

  function selectionOffsets(range) {
    if (!(range instanceof Range)) return null;
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

      const start = Math.max(
        0,
        Math.floor(textLengthToBoundary(range.startContainer, range.startOffset))
      );
      const end = Math.max(
        0,
        Math.floor(textLengthToBoundary(range.endContainer, range.endOffset))
      );
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
        return null;
      }
      return { start, end };
    } catch (_err) {
      return null;
    }
  }

  const selection = window.getSelection();
  let selectedText =
    selection && typeof selection.toString === "function" ? selection.toString() : "";
  const domSelectedText = selectedText;
  let offsets = null;

  function recoverOffsetsFromText(rootText, targetText) {
    if (!targetText) return null;
    const compactRoot = compactSearchIndex(rootText);
    const candidates = [];
    const collapsed = targetText.replace(/\s+/g, " ").trim();
    if (collapsed) candidates.push(collapsed);
    const trimmed = targetText.trim();
    if (trimmed && !candidates.includes(trimmed)) candidates.push(trimmed);
    const noCR = targetText.replace(/\r/g, "");
    if (noCR && !candidates.includes(noCR)) candidates.push(noCR);
    const compact = targetText.replace(/\s+/g, "");
    if (compact && !candidates.includes(compact)) candidates.push(compact);
    if (targetText && !candidates.includes(targetText)) candidates.push(targetText);

    for (const candidate of candidates) {
      const first = rootText.indexOf(candidate);
      if (first >= 0) {
        return { start: first, end: first + candidate.length };
      }

      const compactCandidate = candidate.replace(/\s+/g, "");
      if (!compactCandidate) continue;
      const compactMatch = compactRoot.text.indexOf(compactCandidate);
      if (compactMatch < 0) continue;
      const mappedStart = compactRoot.map[compactMatch];
      const mappedEnd =
        compactRoot.map[compactMatch + compactCandidate.length - 1] + 1;
      if (!Number.isFinite(mappedStart) || !Number.isFinite(mappedEnd)) continue;
      return { start: mappedStart, end: mappedEnd };
    }

    return null;
  }

  if (selection && selection.rangeCount > 0) {
    offsets = selectionOffsets(selection.getRangeAt(0));
  }
  if (!selectedText.trim()) {
    selectedText = __SELECTED_HINT__;
  }

  const hintedText = __SELECTED_HINT__;
  const normalizedDomSelected = domSelectedText.replace(/\s+/g, " ").trim();
  const normalizedHinted = String(hintedText || "").replace(/\s+/g, " ").trim();

  // Right-click handling can collapse the browser selection. If that happened,
  // recover using a unique selected-text occurrence in the same text stream.
  if ((!offsets || offsets.end <= offsets.start) && selectedText) {
    const rootText = collectRootText();
    offsets = recoverOffsetsFromText(rootText, selectedText);
  }

  // When context-menu interactions shrink the live DOM selection, prefer the
  // fuller hinted selection if it is substantially longer and can be resolved.
  if (normalizedHinted) {
    const hintedLooksLonger =
      !normalizedDomSelected ||
      normalizedHinted.length > Math.max(6, Math.floor(normalizedDomSelected.length * 1.2));
    if (hintedLooksLonger) {
      const rootText = collectRootText();
      const hintedOffsets = recoverOffsetsFromText(rootText, hintedText);
      if (
        hintedOffsets &&
        hintedOffsets.end > hintedOffsets.start &&
        (!offsets || (hintedOffsets.end - hintedOffsets.start) > (offsets.end - offsets.start))
      ) {
        offsets = hintedOffsets;
        selectedText = hintedText;
      }
    }
  }

  return {
    hasSelection: !!(offsets && offsets.end > offsets.start),
    selectedText,
    selectionOffsetStart: offsets ? offsets.start : null,
    selectionOffsetEnd: offsets ? offsets.end : null,
  };
})()
