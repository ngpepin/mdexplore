(() => {
  const root = document.querySelector("main") || document.body;
  const skipTags = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEXTAREA"]);
  function shouldSkipTextNode(node) {
    if (!node || typeof node.nodeValue !== "string") return true;
    const value = node.nodeValue;
    if (!value.length) return true;
    // Ignore formatting-only whitespace that contains newlines/tabs so
    // recovered offsets match user-visible text flow.
    if (!/[^\s]/.test(value) && /[\r\n\t]/.test(value)) return true;
    return false;
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
      text += walker.currentNode.nodeValue || "";
    }
    return text;
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
          total += (walker.currentNode.nodeValue || "").length;
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
  let offsets = null;

  if (selection && selection.rangeCount > 0) {
    offsets = selectionOffsets(selection.getRangeAt(0));
  }
  if (!selectedText.trim()) {
    selectedText = __SELECTED_HINT__;
  }

  // Right-click handling can collapse the browser selection. If that happened,
  // recover using a unique selected-text occurrence in the same text stream.
  if ((!offsets || offsets.end <= offsets.start) && selectedText) {
    const rootText = collectRootText();
    const candidates = [];
    const collapsed = selectedText.replace(/\s+/g, " ").trim();
    if (collapsed) candidates.push(collapsed);
    const trimmed = selectedText.trim();
    if (trimmed && !candidates.includes(trimmed)) candidates.push(trimmed);
    const noCR = selectedText.replace(/\r/g, "");
    if (noCR && !candidates.includes(noCR)) candidates.push(noCR);
    if (selectedText && !candidates.includes(selectedText)) candidates.push(selectedText);

    for (const candidate of candidates) {
      const first = rootText.indexOf(candidate);
      if (first >= 0) {
        offsets = { start: first, end: first + candidate.length };
        break;
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
