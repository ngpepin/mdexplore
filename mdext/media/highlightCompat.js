'use strict';

(() => {
  const PREVIEW_HIGHLIGHT_KIND_NORMAL = 'normal';
  const PREVIEW_HIGHLIGHT_KIND_IMPORTANT = 'important';
  const PREVIEW_HIGHLIGHT_OFFSET_SPACE_PREVIEW = 'preview_text_v2';
  const PREVIEW_HIGHLIGHT_OFFSET_SPACE_SOURCE = 'markdown_source_v1';
  const DEFAULT_COLOR = 'rgba(102, 86, 178, 0.36)';
  const DEFAULT_IMPORTANT_COLOR = 'rgba(225, 214, 255, 0.76)';
  const DEFAULT_IMPORTANT_TEXT_COLOR = '#170534';
  const DEFAULT_MARKER_COLOR = 'rgba(112, 90, 188, 0.92)';
  const DEFAULT_IMPORTANT_MARKER_COLOR = 'rgba(154, 132, 220, 0.96)';

  function previewRoot() {
    return document.getElementById('preview-content') || document.querySelector('main') || document.body;
  }

  function splitTextPieces(value) {
    const source = typeof value === 'string' ? value : '';
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
      if (piece.countable) {
        total += piece.text.length;
      }
    }
    return total;
  }

  function countableText(value) {
    let text = '';
    for (const piece of splitTextPieces(value)) {
      if (piece.countable) {
        text += piece.text;
      }
    }
    return text;
  }

  function normalizeSearchText(value) {
    return String(value || '').replace(/\s+/g, ' ').trim();
  }

  function buildCompactSearchIndex(value) {
    const source = typeof value === 'string' ? value : '';
    let compact = '';
    const map = [];
    for (let index = 0; index < source.length; index += 1) {
      const character = source[index];
      if (/\s/.test(character)) {
        continue;
      }
      compact += character;
      map.push(index);
    }
    return { text: compact, map };
  }

  function buildAnchorCandidates(value) {
    const normalized = normalizeSearchText(value);
    if (!normalized) {
      return [];
    }
    const variants = [];
    const pushCandidate = (candidate) => {
      const normalizedCandidate = normalizeSearchText(candidate);
      if (normalizedCandidate.length < 12) {
        return;
      }
      if (!variants.includes(normalizedCandidate)) {
        variants.push(normalizedCandidate);
      }
    };
    pushCandidate(normalized);
    pushCandidate(normalized.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '').trim());
    const boundaries = [0];
    for (let index = 0; index < normalized.length; index += 1) {
      if (normalized[index] === ' ' && index + 1 < normalized.length) {
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
    if (!normalized) {
      return [];
    }
    const candidates = [];
    const pushCandidate = (candidate) => {
      const text = normalizeSearchText(candidate);
      if (text.length < 3) {
        return;
      }
      if (!candidates.includes(text)) {
        candidates.push(text);
      }
    };
    pushCandidate(normalized);
    pushCandidate(normalized.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ''));
    return candidates;
  }

  function shouldSkipTextNode(node) {
    return !node || typeof node.nodeValue !== 'string' || countableLength(node.nodeValue) <= 0;
  }

  function currentSelectionText() {
    const selection = window.getSelection();
    if (!selection || !selection.rangeCount) {
      return '';
    }
    const range = selection.getRangeAt(0);
    const ancestor = range.commonAncestorContainer.nodeType === Node.TEXT_NODE
      ? range.commonAncestorContainer.parentElement
      : range.commonAncestorContainer;
    const root = previewRoot();
    if (!(ancestor instanceof Node) || !root || !root.contains(ancestor)) {
      return '';
    }
    return String(selection.toString() || '');
  }

  function getLiveSelectionOffsets(selectedHint = '') {
    const root = previewRoot();
    const skipTags = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEXTAREA']);
    if (!root) {
      return {
        hasSelection: false,
        selectedText: '',
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
            if (!parent || skipTags.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
            return NodeFilter.FILTER_ACCEPT;
          },
        },
      );
      let text = '';
      while (walker.nextNode()) {
        text += countableText(walker.currentNode.nodeValue || '');
      }
      return text;
    }

    function selectionOffsets(range) {
      if (!(range instanceof Range)) {
        return null;
      }
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
                if (!parent || skipTags.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
                return NodeFilter.FILTER_ACCEPT;
              },
            },
          );
          let total = 0;
          while (walker.nextNode()) {
            total += countableLength(walker.currentNode.nodeValue || '');
          }
          return total;
        }

        const start = Math.max(0, Math.floor(textLengthToBoundary(range.startContainer, range.startOffset)));
        const end = Math.max(0, Math.floor(textLengthToBoundary(range.endContainer, range.endOffset)));
        if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
          return null;
        }
        return { start, end };
      } catch {
        return null;
      }
    }

    function recoverOffsetsFromText(rootText, targetText) {
      if (!targetText) {
        return null;
      }
      const compactRoot = buildCompactSearchIndex(rootText);
      const candidates = [];
      const collapsed = targetText.replace(/\s+/g, ' ').trim();
      if (collapsed) candidates.push(collapsed);
      const trimmed = targetText.trim();
      if (trimmed && !candidates.includes(trimmed)) candidates.push(trimmed);
      const noCR = targetText.replace(/\r/g, '');
      if (noCR && !candidates.includes(noCR)) candidates.push(noCR);
      const compact = targetText.replace(/\s+/g, '');
      if (compact && !candidates.includes(compact)) candidates.push(compact);
      if (targetText && !candidates.includes(targetText)) candidates.push(targetText);

      for (const candidate of candidates) {
        const first = rootText.indexOf(candidate);
        if (first >= 0) {
          return { start: first, end: first + candidate.length };
        }
        const compactCandidate = candidate.replace(/\s+/g, '');
        if (!compactCandidate) {
          continue;
        }
        const compactMatch = compactRoot.text.indexOf(compactCandidate);
        if (compactMatch < 0) {
          continue;
        }
        const mappedStart = compactRoot.map[compactMatch];
        const mappedEnd = compactRoot.map[compactMatch + compactCandidate.length - 1] + 1;
        if (!Number.isFinite(mappedStart) || !Number.isFinite(mappedEnd)) {
          continue;
        }
        return { start: mappedStart, end: mappedEnd };
      }

      return null;
    }

    const selection = window.getSelection();
    let selectedText = selection && typeof selection.toString === 'function' ? selection.toString() : '';
    const domSelectedText = selectedText;
    let offsets = null;
    if (selection && selection.rangeCount > 0) {
      offsets = selectionOffsets(selection.getRangeAt(0));
    }
    if (!selectedText.trim()) {
      selectedText = selectedHint;
    }

    const normalizedDomSelected = domSelectedText.replace(/\s+/g, ' ').trim();
    const normalizedHinted = String(selectedHint || '').replace(/\s+/g, ' ').trim();

    if ((!offsets || offsets.end <= offsets.start) && selectedText) {
      offsets = recoverOffsetsFromText(collectRootText(), selectedText);
    }

    if (normalizedHinted) {
      const hintedLooksLonger = !normalizedDomSelected
        || normalizedHinted.length > Math.max(6, Math.floor(normalizedDomSelected.length * 1.2));
      if (hintedLooksLonger) {
        const hintedOffsets = recoverOffsetsFromText(collectRootText(), selectedHint);
        if (
          hintedOffsets
          && hintedOffsets.end > hintedOffsets.start
          && (!offsets || (hintedOffsets.end - hintedOffsets.start) > (offsets.end - offsets.start))
        ) {
          offsets = hintedOffsets;
          selectedText = selectedHint;
        }
      }
    }

    return {
      hasSelection: !!(offsets && offsets.end > offsets.start),
      selectedText,
      selectionOffsetStart: offsets ? offsets.start : null,
      selectionOffsetEnd: offsets ? offsets.end : null,
    };
  }

  function installMarkerOverlays() {
    if (window.__mdextPreviewMarkersInstalled) {
      return;
    }
    window.__mdextPreviewMarkersInstalled = true;

    const hitOverlay = document.createElement('div');
    hitOverlay.className = 'mdexplore-scroll-hit-overlay';
    document.body.appendChild(hitOverlay);
    const highlightOverlay = document.createElement('div');
    highlightOverlay.className = 'mdexplore-scroll-highlight-overlay';
    document.body.appendChild(highlightOverlay);

    const state = {
      searchRefreshTimer: null,
      highlightRefreshTimer: null,
    };

    const viewportScrollableHeight = () => Math.max(
      0,
      (document.documentElement ? document.documentElement.scrollHeight : 0) - window.innerHeight,
    );

    const scrollbarWidth = () => {
      const documentElement = document.documentElement;
      return Math.max(0, window.innerWidth - (documentElement ? documentElement.clientWidth : window.innerWidth));
    };

    const syncOverlayHorizontalPosition = () => {
      const docClientWidth = document.documentElement ? document.documentElement.clientWidth : window.innerWidth;
      const overlayWidth = Math.max(4, hitOverlay.offsetWidth || 6);
      const overlayLeft = Math.max(0, docClientWidth - overlayWidth);
      hitOverlay.style.left = `${Math.round(overlayLeft)}px`;
      highlightOverlay.style.left = '0px';
    };

    const jumpToTarget = (target) => {
      if (!target || typeof target.getBoundingClientRect !== 'function') {
        return;
      }
      const rect = target.getBoundingClientRect();
      const absoluteTop = window.scrollY + rect.top;
      const targetCenter = absoluteTop + Math.max(0, rect.height * 0.5);
      const desiredTop = Math.max(0, targetCenter - (window.innerHeight * 0.5));
      window.scrollTo({ top: desiredTop, behavior: 'auto' });
    };

    const refreshSearchHitMarkers = () => {
      if (!hitOverlay.isConnected) {
        return;
      }
      hitOverlay.replaceChildren();
      const marks = Array.from(document.querySelectorAll('[data-mdexplore-search-mark="1"]'));
      const scrollHeight = Math.max(
        1,
        document.documentElement ? document.documentElement.scrollHeight : 0,
        document.body ? document.body.scrollHeight : 0,
      );
      const scrollableHeight = Math.max(1, scrollHeight - window.innerHeight);
      if (!marks.length || scrollHeight <= window.innerHeight) {
        hitOverlay.classList.remove('mdexplore-visible');
        syncOverlayHorizontalPosition();
        return;
      }

      const trackHeight = window.innerHeight;
      const markerPositions = [];
      for (const mark of marks) {
        const rect = mark.getBoundingClientRect();
        if (!rect || rect.height <= 0) {
          continue;
        }
        const absoluteTop = window.scrollY + rect.top;
        const absoluteBottom = absoluteTop + rect.height;
        const topPx = Math.max(0, Math.min(trackHeight - 4, (absoluteTop / scrollableHeight) * trackHeight));
        const bottomPx = Math.max(topPx + 3, Math.min(trackHeight, (absoluteBottom / scrollableHeight) * trackHeight));
        const centerPx = Math.max(topPx, Math.min(bottomPx, ((absoluteTop + absoluteBottom) * 0.5 / scrollableHeight) * trackHeight));
        markerPositions.push({
          top: topPx,
          bottom: bottomPx,
          target: mark,
          center: centerPx,
        });
      }

      if (!markerPositions.length) {
        hitOverlay.classList.remove('mdexplore-visible');
        syncOverlayHorizontalPosition();
        return;
      }

      markerPositions.sort((left, right) => left.top - right.top);
      const merged = [];
      for (const item of markerPositions) {
        const top = Math.round(item.top);
        const bottom = Math.round(item.bottom);
        const previous = merged.length ? merged[merged.length - 1] : null;
        if (previous && top <= previous.bottom + 2) {
          previous.bottom = Math.max(previous.bottom, bottom);
          previous.targets.push({ element: item.target, center: item.center });
          continue;
        }
        merged.push({
          top,
          bottom,
          targets: [{ element: item.target, center: item.center }],
        });
      }

      for (const item of merged) {
        const marker = document.createElement('div');
        marker.className = 'mdexplore-scroll-hit-marker';
        marker.style.top = `${item.top}px`;
        marker.style.height = `${Math.max(5, item.bottom - item.top)}px`;
        const activateMarker = (event) => {
          if (typeof event.button === 'number' && event.button !== 0) {
            return;
          }
          event.preventDefault();
          event.stopPropagation();
          if (typeof event.stopImmediatePropagation === 'function') {
            event.stopImmediatePropagation();
          }
          const clickY = Number.isFinite(event.clientY) ? event.clientY : (item.top + item.bottom) * 0.5;
          const targetInfo = Array.isArray(item.targets) && item.targets.length
            ? item.targets.reduce((best, candidate) => {
                if (!best) return candidate;
                return Math.abs(candidate.center - clickY) < Math.abs(best.center - clickY) ? candidate : best;
              }, null)
            : null;
          const target = targetInfo && targetInfo.element ? targetInfo.element : null;
          if (target) {
            jumpToTarget(target);
          }
        };
        marker.addEventListener('mousedown', activateMarker);
        marker.addEventListener('pointerdown', activateMarker);
        hitOverlay.appendChild(marker);
      }
      syncOverlayHorizontalPosition();
      hitOverlay.classList.add('mdexplore-visible');
    };

    const hasPersistentHighlights = () => {
      const entries = window.__mdexplorePersistentHighlights;
      return Array.isArray(entries) && entries.length > 0;
    };

    const refreshPersistentHighlightMarkers = () => {
      if (!highlightOverlay.isConnected) {
        return;
      }
      const highlightMarkerColor = String(window.__mdexplorePersistentHighlightMarkerColor || '').trim() || DEFAULT_MARKER_COLOR;
      const importantHighlightMarkerColor = String(window.__mdexplorePersistentHighlightImportantMarkerColor || '').trim() || DEFAULT_IMPORTANT_MARKER_COLOR;
      highlightOverlay.replaceChildren();
      const marks = Array.from(document.querySelectorAll('span[data-mdexplore-persistent-highlight="1"]'));
      const scrollHeight = Math.max(
        1,
        document.documentElement ? document.documentElement.scrollHeight : 0,
        document.body ? document.body.scrollHeight : 0,
      );
      if (!marks.length || scrollHeight <= window.innerHeight) {
        highlightOverlay.classList.remove('mdexplore-visible');
        syncOverlayHorizontalPosition();
        return;
      }

      const trackHeight = window.innerHeight;
      const groups = new Map();
      for (const mark of marks) {
        const id = String(mark.getAttribute('data-mdexplore-persistent-highlight-id') || '').trim();
        const kind = String(mark.getAttribute('data-mdexplore-persistent-highlight-kind') || PREVIEW_HIGHLIGHT_KIND_NORMAL).trim().toLowerCase() === PREVIEW_HIGHLIGHT_KIND_IMPORTANT
          ? PREVIEW_HIGHLIGHT_KIND_IMPORTANT
          : PREVIEW_HIGHLIGHT_KIND_NORMAL;
        if (!id) {
          continue;
        }
        const rects = Array.from(mark.getClientRects ? mark.getClientRects() : []);
        const usableRects = rects.length ? rects.filter((rect) => rect && rect.height > 0 && rect.width > 0) : [];
        const sourceRects = usableRects.length
          ? usableRects
          : [mark.getBoundingClientRect()].filter((rect) => rect && rect.height > 0 && rect.width > 0);
        if (!sourceRects.length) {
          continue;
        }
        let group = groups.get(id);
        if (!group) {
          group = { kind, minTop: null, maxBottom: null, targets: [] };
          groups.set(id, group);
        }
        for (const rect of sourceRects) {
          const absoluteTop = window.scrollY + rect.top;
          const absoluteBottom = absoluteTop + rect.height;
          group.minTop = group.minTop === null ? absoluteTop : Math.min(group.minTop, absoluteTop);
          group.maxBottom = group.maxBottom === null ? absoluteBottom : Math.max(group.maxBottom, absoluteBottom);
          group.targets.push({
            element: mark,
            center: ((absoluteTop + absoluteBottom) * 0.5 / scrollHeight) * trackHeight,
          });
        }
      }

      const markerPositions = [];
      for (const group of groups.values()) {
        if (!Number.isFinite(group.minTop) || !Number.isFinite(group.maxBottom)) {
          continue;
        }
        const topPx = Math.max(0, Math.min(trackHeight - 4, (group.minTop / scrollHeight) * trackHeight));
        const bottomPx = Math.max(topPx + 3, Math.min(trackHeight, (group.maxBottom / scrollHeight) * trackHeight));
        markerPositions.push({
          top: Math.round(topPx),
          bottom: Math.round(bottomPx),
          kind: group.kind,
          color: group.kind === PREVIEW_HIGHLIGHT_KIND_IMPORTANT ? importantHighlightMarkerColor : highlightMarkerColor,
          targets: Array.isArray(group.targets) ? group.targets : [],
        });
      }

      if (!markerPositions.length) {
        highlightOverlay.classList.remove('mdexplore-visible');
        syncOverlayHorizontalPosition();
        return;
      }

      markerPositions.sort((left, right) => left.top - right.top);
      const merged = [];
      for (const item of markerPositions) {
        const previous = merged.length ? merged[merged.length - 1] : null;
        if (previous && previous.kind === item.kind && item.top <= previous.bottom + 2) {
          previous.bottom = Math.max(previous.bottom, item.bottom);
          previous.targets.push(...item.targets);
          continue;
        }
        merged.push({
          top: item.top,
          bottom: item.bottom,
          kind: item.kind,
          color: item.color,
          targets: Array.isArray(item.targets) ? [...item.targets] : [],
        });
      }

      for (const item of merged) {
        const marker = document.createElement('div');
        marker.className = 'mdexplore-scroll-highlight-marker';
        marker.style.top = `${item.top}px`;
        marker.style.height = `${Math.max(5, item.bottom - item.top)}px`;
        marker.style.background = item.color;
        const activateMarker = (event) => {
          if (typeof event.button === 'number' && event.button !== 0) {
            return;
          }
          event.preventDefault();
          event.stopPropagation();
          if (typeof event.stopImmediatePropagation === 'function') {
            event.stopImmediatePropagation();
          }
          const clickY = Number.isFinite(event.clientY) ? event.clientY : (item.top + item.bottom) * 0.5;
          const targetInfo = Array.isArray(item.targets) && item.targets.length
            ? item.targets.reduce((best, candidate) => {
                if (!best) return candidate;
                return Math.abs(candidate.center - clickY) < Math.abs(best.center - clickY) ? candidate : best;
              }, null)
            : null;
          const target = targetInfo && targetInfo.element ? targetInfo.element : null;
          if (target) {
            jumpToTarget(target);
          }
        };
        marker.addEventListener('mousedown', activateMarker);
        marker.addEventListener('pointerdown', activateMarker);
        highlightOverlay.appendChild(marker);
      }
      syncOverlayHorizontalPosition();
      highlightOverlay.classList.add('mdexplore-visible');
    };

    const scheduleSearchHitRefresh = (delayMs = 40) => {
      if (state.searchRefreshTimer) {
        window.clearTimeout(state.searchRefreshTimer);
      }
      state.searchRefreshTimer = window.setTimeout(() => {
        state.searchRefreshTimer = null;
        refreshSearchHitMarkers();
      }, delayMs);
    };

    const schedulePersistentHighlightRefresh = (delayMs = 40) => {
      if (!hasPersistentHighlights() && !highlightOverlay.classList.contains('mdexplore-visible')) {
        return;
      }
      if (state.highlightRefreshTimer) {
        window.clearTimeout(state.highlightRefreshTimer);
      }
      state.highlightRefreshTimer = window.setTimeout(() => {
        state.highlightRefreshTimer = null;
        refreshPersistentHighlightMarkers();
      }, delayMs);
    };

    window.__mdexploreRefreshScrollHitMarkers = () => scheduleSearchHitRefresh(0);
    window.__mdexploreRefreshPersistentHighlightMarkers = () => schedulePersistentHighlightRefresh(0);

    window.addEventListener('resize', () => {
      scheduleSearchHitRefresh(0);
      if (hasPersistentHighlights() || highlightOverlay.classList.contains('mdexplore-visible')) {
        schedulePersistentHighlightRefresh(0);
      }
    }, { passive: true });

    const observer = new MutationObserver((mutationList) => {
      for (const mutation of mutationList) {
        const targetNode = mutation && mutation.target ? mutation.target : null;
        if (
          targetNode === hitOverlay
          || targetNode === highlightOverlay
          || (targetNode instanceof Node && (hitOverlay.contains(targetNode) || highlightOverlay.contains(targetNode)))
        ) {
          continue;
        }
        if (mutation.type === 'childList' || mutation.type === 'attributes') {
          scheduleSearchHitRefresh();
          if (hasPersistentHighlights() || highlightOverlay.classList.contains('mdexplore-visible')) {
            schedulePersistentHighlightRefresh();
          }
          return;
        }
      }
    });
    observer.observe(document.body, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: [
        'data-mdexplore-search-mark',
        'data-mdexplore-persistent-highlight',
        'data-mdexplore-persistent-highlight-id',
      ],
    });

    scheduleSearchHitRefresh(0);
    schedulePersistentHighlightRefresh(0);
  }

  function applyPersistentHighlights(options = {}) {
    const incoming = Array.isArray(options.entries) ? options.entries : [];
    const highlightColor = options.color || DEFAULT_COLOR;
    const importantHighlightColor = options.importantColor || DEFAULT_IMPORTANT_COLOR;
    const importantHighlightTextColor = options.importantTextColor || DEFAULT_IMPORTANT_TEXT_COLOR;
    const highlightMarkerColor = options.markerColor || DEFAULT_MARKER_COLOR;
    const importantHighlightMarkerColor = options.importantMarkerColor || DEFAULT_IMPORTANT_MARKER_COLOR;
    const previewOffsetSpace = PREVIEW_HIGHLIGHT_OFFSET_SPACE_PREVIEW;
    const sourceOffsetSpace = PREVIEW_HIGHLIGHT_OFFSET_SPACE_SOURCE;

    window.__mdexplorePersistentHighlightMarkerColor = highlightMarkerColor;
    window.__mdexplorePersistentHighlightImportantMarkerColor = importantHighlightMarkerColor;

    const root = previewRoot();
    if (!root) {
      window.__mdexplorePersistentHighlights = [];
      return { applied: 0, entries: 0, resolvedEntries: [] };
    }

    function normalizeEntries(raw) {
      if (!Array.isArray(raw)) {
        return [];
      }
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
        if (!item || typeof item !== 'object') {
          continue;
        }
        const id = typeof item.id === 'string' ? item.id.trim() : '';
        const start = Number(item.start);
        const end = Number(item.end);
        const kind = String(item.kind || PREVIEW_HIGHLIGHT_KIND_NORMAL).trim().toLowerCase() === PREVIEW_HIGHLIGHT_KIND_IMPORTANT
          ? PREVIEW_HIGHLIGHT_KIND_IMPORTANT
          : PREVIEW_HIGHLIGHT_KIND_NORMAL;
        const offsetSpace = String(item.offset_space || item.offsetSpace || '').trim().toLowerCase();
        if (!id || !Number.isFinite(start) || !Number.isFinite(end) || end <= start || start < 0) {
          continue;
        }
        prepared.push({
          id,
          start: Math.floor(start),
          end: Math.floor(end),
          kind,
          offsetSpace: offsetSpace === previewOffsetSpace || offsetSpace === sourceOffsetSpace ? offsetSpace : '',
          previewAnchorText: normalizeSearchText(item.anchor_text || item.anchorText || item.previewAnchorText || ''),
          legacyAnchorText: normalizeSearchText(item.legacy_anchor_text || item.legacyAnchorText || ''),
          legacyDirectCandidates: normalizeCandidateArray(item.legacy_direct_candidates || item.legacyDirectCandidates),
          legacyContextCandidates: normalizeCandidateArray(item.legacy_context_candidates || item.legacyContextCandidates),
        });
      }

      if (!prepared.length) {
        return [];
      }
      prepared.sort((left, right) => (left.start - right.start) || (left.end - right.end));
      const merged = [];
      for (const item of prepared) {
        const previous = merged[merged.length - 1];
        if (previous && item.kind === previous.kind && item.start <= previous.end) {
          previous.end = Math.max(previous.end, item.end);
          continue;
        }
        merged.push(item);
      }
      return merged;
    }

    function isPersistentHighlightSpan(node) {
      return node instanceof Element
        && node.tagName === 'SPAN'
        && node.getAttribute('data-mdexplore-persistent-highlight') === '1';
    }

    function canMergePersistentHighlightRuns(leftNode, rightNode) {
      if (!isPersistentHighlightSpan(leftNode) || !isPersistentHighlightSpan(rightNode)) {
        return false;
      }
      const leftId = String(leftNode.getAttribute('data-mdexplore-persistent-highlight-id') || '');
      const rightId = String(rightNode.getAttribute('data-mdexplore-persistent-highlight-id') || '');
      if (leftId !== rightId) {
        return false;
      }
      const leftKind = String(leftNode.getAttribute('data-mdexplore-persistent-highlight-kind') || '');
      const rightKind = String(rightNode.getAttribute('data-mdexplore-persistent-highlight-kind') || '');
      return leftKind === rightKind;
    }

    function mergeAdjacentPersistentHighlightRuns(container) {
      if (!(container instanceof Node)) {
        return 0;
      }
      let merged = 0;
      let node = container.firstChild;
      while (node) {
        if (!isPersistentHighlightSpan(node)) {
          node = node.nextSibling;
          continue;
        }
        let next = node.nextSibling;
        while (canMergePersistentHighlightRuns(node, next)) {
          node.textContent = `${node.textContent || ''}${next.textContent || ''}`;
          const removable = next;
          next = next.nextSibling;
          if (removable && removable.parentNode === container) {
            container.removeChild(removable);
            merged += 1;
          }
        }
        node = next;
      }
      return merged;
    }

    for (const mark of Array.from(root.querySelectorAll('span[data-mdexplore-persistent-highlight="1"]'))) {
      const parent = mark.parentNode;
      if (!parent) {
        continue;
      }
      parent.replaceChild(document.createTextNode(mark.textContent || ''), mark);
      parent.normalize();
    }

    const skipTags = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEXTAREA']);
    const walker = document.createTreeWalker(
      root,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          if (shouldSkipTextNode(node)) {
            return NodeFilter.FILTER_REJECT;
          }
          const parent = node.parentElement;
          if (!parent || skipTags.has(parent.tagName)) {
            return NodeFilter.FILTER_REJECT;
          }
          return NodeFilter.FILTER_ACCEPT;
        },
      },
    );

    const nodeRecords = [];
    let logicalText = '';
    let totalLength = 0;
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const text = node.nodeValue || '';
      const pieces = splitTextPieces(text).map((piece) => {
        const record = { ...piece, start: totalLength, end: totalLength };
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
      if (!entry || typeof entry !== 'object') {
        return entry;
      }
      const candidateGroups = [];
      if (entry.offsetSpace === previewOffsetSpace) {
        const previewCandidates = buildPreviewAnchorCandidates(entry.previewAnchorText);
        if (!previewCandidates.length) {
          return entry;
        }
        candidateGroups.push(previewCandidates);
      } else {
        candidateGroups.push(Array.isArray(entry.legacyDirectCandidates) ? entry.legacyDirectCandidates : []);
        candidateGroups.push(
          Array.isArray(entry.legacyContextCandidates) && entry.legacyContextCandidates.length
            ? entry.legacyContextCandidates
            : buildAnchorCandidates(entry.legacyAnchorText),
        );
      }

      const compactCurrentText = String(logicalText.slice(entry.start, entry.end) || '').replace(/\s+/g, '');
      const currentOffsetsAlreadyMatch = entry.offsetSpace !== sourceOffsetSpace
        && !!compactCurrentText
        && Array.isArray(candidateGroups[0])
        && candidateGroups[0].some((candidate) => {
          const compactCandidate = candidate.replace(/\s+/g, '');
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
          const compactCandidate = candidate.replace(/\s+/g, '');
          if (compactCandidate.length < (groupIndex === 0 ? 3 : 8)) {
            continue;
          }
          let matchIndex = compactIndex.text.indexOf(compactCandidate);
          while (matchIndex >= 0) {
            const rawStart = compactIndex.map[matchIndex];
            const rawEnd = compactIndex.map[matchIndex + compactCandidate.length - 1] + 1;
            const score = [
              Math.abs(candidate.length - entryLength),
              -candidate.length,
              Math.abs(rawStart - entry.start),
              rawStart,
            ];
            if (
              !groupBest
              || score[0] < groupBest.score[0]
              || (score[0] === groupBest.score[0] && score[1] < groupBest.score[1])
              || (score[0] === groupBest.score[0] && score[1] === groupBest.score[1] && score[2] < groupBest.score[2])
              || (score[0] === groupBest.score[0] && score[1] === groupBest.score[1] && score[2] === groupBest.score[2] && score[3] < groupBest.score[3])
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
      if (!best) {
        return entry;
      }
      return {
        ...entry,
        start: Math.floor(best.rawStart),
        end: Math.floor(best.rawEnd),
        offsetSpace: previewOffsetSpace,
      };
    }

    const compactIndex = buildCompactSearchIndex(logicalText);
    const resolvedEntries = normalizeEntries(incoming.map((entry) => resolveLegacyEntry(entry, compactIndex)));
    window.__mdexplorePersistentHighlights = resolvedEntries;

    if (!resolvedEntries.length) {
      if (typeof window.__mdexploreRefreshPersistentHighlightMarkers === 'function') {
        window.__mdexploreRefreshPersistentHighlightMarkers();
      }
      return { applied: 0, entries: 0, resolvedEntries: [] };
    }
    if (!nodeRecords.length) {
      return { applied: 0, entries: resolvedEntries.length, resolvedEntries: [] };
    }

    let applied = 0;
    let mergedSpans = 0;
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
        const localRanges = [];
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
        localRanges.sort((left, right) => left.start - right.start);
        let cursor = 0;
        for (const range of localRanges) {
          if (range.start > cursor) {
            fragment.appendChild(document.createTextNode(piece.text.slice(cursor, range.start)));
          }
          const mark = document.createElement('span');
          mark.setAttribute('data-mdexplore-persistent-highlight', '1');
          mark.setAttribute('data-mdexplore-persistent-highlight-id', range.id);
          mark.setAttribute('data-mdexplore-persistent-highlight-kind', range.kind === PREVIEW_HIGHLIGHT_KIND_IMPORTANT ? PREVIEW_HIGHLIGHT_KIND_IMPORTANT : PREVIEW_HIGHLIGHT_KIND_NORMAL);
          const isImportant = range.kind === PREVIEW_HIGHLIGHT_KIND_IMPORTANT;
          mark.style.backgroundColor = isImportant ? importantHighlightColor : highlightColor;
          mark.style.color = isImportant ? importantHighlightTextColor : '';
          mark.style.borderRadius = '2px';
          mark.style.padding = '0 1px';
          mark.style.boxDecorationBreak = 'clone';
          mark.style.webkitBoxDecorationBreak = 'clone';
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
        mergedSpans += mergeAdjacentPersistentHighlightRuns(parent);
      }
    }

    if (typeof window.__mdexploreRefreshPersistentHighlightMarkers === 'function') {
      window.__mdexploreRefreshPersistentHighlightMarkers();
    }

    return {
      applied,
      mergedSpans,
      entries: resolvedEntries.length,
      segments: countableSegmentCount,
      totalLength,
      resolvedEntries: resolvedEntries.map((entry) => ({
        id: entry.id,
        start: entry.start,
        end: entry.end,
        kind: entry.kind,
        anchor_text: entry.previewAnchorText || '',
        offset_space: entry.offsetSpace === sourceOffsetSpace ? sourceOffsetSpace : previewOffsetSpace,
      })),
    };
  }

  window.__mdextHighlightCompat = {
    applyPersistentHighlights,
    currentSelectionText,
    getLiveSelectionOffsets,
    installMarkerOverlays,
  };
})();