'use strict';

(() => {
  function previewRoot() {
    return document.getElementById('preview-content') || document.querySelector('main') || document.body;
  }

  function clearSearchHighlights() {
    const root = previewRoot();
    if (!root) {
      return 0;
    }
    const marks = root.querySelectorAll('span[data-mdexplore-search-mark="1"]');
    for (const mark of marks) {
      const parent = mark.parentNode;
      if (!parent) {
        continue;
      }
      parent.replaceChild(document.createTextNode(mark.textContent || ''), mark);
      parent.normalize();
    }
    if (typeof window.__mdexploreRefreshScrollHitMarkers === 'function') {
      window.__mdexploreRefreshScrollHitMarkers();
    }
    return marks.length;
  }

  function highlightSearchTerms(options = {}) {
    const terms = Array.isArray(options.terms) ? options.terms : [];
    const shouldScroll = !!options.scrollToFirst;
    const nearWordGap = Math.max(1, Number(options.nearWordGap) || 50);
    const nearTermGroups = Array.isArray(options.nearTermGroups) ? options.nearTermGroups : [];
    const root = previewRoot();
    if (!root) {
      return { matches: 0 };
    }

    clearSearchHighlights();

    const markSelector = 'span[data-mdexplore-search-mark="1"]';
    const skipTags = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEXTAREA']);

    function escapeRegExp(input) {
      return String(input || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    function escapeTermText(input) {
      return Array.from(String(input || ''), (character) => {
        if (/['\u2018\u2019\u02bc]/u.test(character)) {
          return "['\\u2018\\u2019\\u02bc]";
        }
        return escapeRegExp(character);
      }).join('');
    }

    function shouldUseCloseWordBoundaries(termText) {
      return typeof termText === 'string' && !!termText && !/\s/.test(termText) && /^\w+$/u.test(termText);
    }

    function buildTermPattern(termText, caseSensitive, enforceWordBoundaries = false) {
      const raw = String(termText || '');
      const leadingSpaceMatch = raw.match(/^ +/);
      const trailingSpaceMatch = raw.match(/ +$/);
      const leadingSpaceCount = leadingSpaceMatch ? leadingSpaceMatch[0].length : 0;
      const trailingSpaceCount = trailingSpaceMatch ? trailingSpaceMatch[0].length : 0;
      const useLeadingBoundarySpace = leadingSpaceCount === 1;
      const useTrailingBoundarySpace = trailingSpaceCount === 1;
      const leftTrim = useLeadingBoundarySpace ? 1 : 0;
      const rightTrim = useTrailingBoundarySpace ? 1 : 0;
      const core =
        useLeadingBoundarySpace || useTrailingBoundarySpace
          ? raw.slice(leftTrim, rightTrim ? raw.length - rightTrim : raw.length)
          : raw;

      let source = escapeTermText(core || raw);
      const canUseBoundarySpaceMode = !!core;
      if (canUseBoundarySpaceMode && useLeadingBoundarySpace) {
        source = `(?:^|(?<=[^\\w]))${source}`;
      }
      if (canUseBoundarySpaceMode && useTrailingBoundarySpace) {
        source = `${source}(?=$|(?=[^\\w]))`;
      }
      if (
        enforceWordBoundaries
        && !useLeadingBoundarySpace
        && !useTrailingBoundarySpace
        && shouldUseCloseWordBoundaries(core)
      ) {
        source = `(?<!\\w)${source}(?!\\w)`;
      }
      return new RegExp(source, caseSensitive ? 'g' : 'gi');
    }

    function upperBound(values, target) {
      let low = 0;
      let high = values.length;
      while (low < high) {
        const middle = (low + high) >> 1;
        if (values[middle] <= target) {
          low = middle + 1;
        } else {
          high = middle;
        }
      }
      return low;
    }

    function normalizeTerms(items) {
      const normalized = [];
      for (const item of items) {
        if (!item || typeof item.text !== 'string') {
          continue;
        }
        if (!item.text.trim()) {
          continue;
        }
        normalized.push({
          text: item.text,
          caseSensitive: !!item.caseSensitive,
        });
      }
      return normalized;
    }

    function normalizeNearGroups(groups) {
      const normalized = [];
      for (const group of groups) {
        if (!Array.isArray(group)) {
          continue;
        }
        const nextGroup = [];
        for (const item of group) {
          if (!item || typeof item.text !== 'string' || !item.text.trim()) {
            continue;
          }
          nextGroup.push({
            text: item.text,
            caseSensitive: !!item.caseSensitive,
          });
        }
        if (nextGroup.length >= 2) {
          normalized.push(nextGroup);
        }
      }
      return normalized;
    }

    const walker = document.createTreeWalker(
      root,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          if (!node || !node.nodeValue || !node.nodeValue.trim()) {
            return NodeFilter.FILTER_REJECT;
          }
          const parent = node.parentElement;
          if (!parent || skipTags.has(parent.tagName) || parent.closest(markSelector)) {
            return NodeFilter.FILTER_REJECT;
          }
          return NodeFilter.FILTER_ACCEPT;
        },
      },
    );

    const segments = [];
    let fullText = '';
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const value = node.nodeValue || '';
      if (!value) {
        continue;
      }
      const start = fullText.length;
      fullText += value;
      const end = fullText.length;
      segments.push({ node, text: value, start, end });
      fullText += '\n';
    }
    if (!segments.length) {
      if (typeof window.__mdexploreRefreshScrollHitMarkers === 'function') {
        window.__mdexploreRefreshScrollHitMarkers();
      }
      return { matches: 0 };
    }

    const normalizedTerms = normalizeTerms(terms);
    const normalizedNearGroups = normalizeNearGroups(nearTermGroups);
    const nearFocusWindows = [];

    if (normalizedNearGroups.length) {
      const wordMatches = [];
      const wordRegex = /\S+/g;
      let wordMatch = null;
      while ((wordMatch = wordRegex.exec(fullText)) !== null) {
        wordMatches.push({ start: wordMatch.index, end: wordMatch.index + wordMatch[0].length });
        if (wordRegex.lastIndex <= wordMatch.index) {
          wordRegex.lastIndex = wordMatch.index + 1;
        }
      }

      if (wordMatches.length) {
        const wordStarts = wordMatches.map((item) => item.start);

        function earliestWindowForGroup(group, minStartChar = 0) {
          const occurrencesByTerm = Array.from({ length: group.length }, () => []);
          for (let termIndex = 0; termIndex < group.length; termIndex += 1) {
            const termInfo = group[termIndex];
            const pattern = buildTermPattern(termInfo.text, termInfo.caseSensitive, true);
            let match = null;
            while ((match = pattern.exec(fullText)) !== null) {
              const startChar = match.index;
              const endChar = startChar + match[0].length;
              if (startChar < minStartChar) {
                if (pattern.lastIndex <= startChar) {
                  pattern.lastIndex = startChar + 1;
                }
                continue;
              }
              const startWord = upperBound(wordStarts, startChar) - 1;
              if (startWord >= 0) {
                const endProbe = endChar > startChar ? endChar - 1 : startChar;
                let endWord = upperBound(wordStarts, endProbe) - 1;
                if (endWord < startWord) {
                  endWord = startWord;
                }
                occurrencesByTerm[termIndex].push({
                  startWord,
                  endWord,
                  start: startChar,
                  end: endChar,
                });
              }
              if (pattern.lastIndex <= startChar) {
                pattern.lastIndex = startChar + 1;
              }
            }
          }

          if (occurrencesByTerm.some((occurrences) => !occurrences.length)) {
            return null;
          }

          for (const occurrences of occurrencesByTerm) {
            occurrences.sort((left, right) => {
              if (left.start !== right.start) return left.start - right.start;
              if (left.startWord !== right.startWord) return left.startWord - right.startWord;
              if (left.endWord !== right.endWord) return left.endWord - right.endWord;
              return left.end - right.end;
            });
          }

          const orderedTerms = occurrencesByTerm
            .map((occurrences, termIndex) => ({ termIndex, count: occurrences.length }))
            .sort((left, right) => left.count - right.count)
            .map((entry) => entry.termIndex);

          let best = null;
          const search = (orderIndex, usedStarts, minStartWord, maxEndWord, minStartCharValue, maxEndChar) => {
            if (orderIndex >= orderedTerms.length) {
              if (
                minStartWord === null
                || maxEndWord === null
                || minStartCharValue === null
                || maxEndChar === null
              ) {
                return;
              }
              const span = maxEndWord - minStartWord;
              if (
                !best
                || minStartCharValue < best.startChar
                || (
                  minStartCharValue === best.startChar
                  && (span < best.span || (span === best.span && maxEndChar < best.endChar))
                )
              ) {
                best = {
                  span,
                  startChar: minStartCharValue,
                  endChar: maxEndChar,
                };
              }
              return;
            }

            const termIndex = orderedTerms[orderIndex];
            for (const occurrence of occurrencesByTerm[termIndex]) {
              if (usedStarts.has(occurrence.start)) {
                continue;
              }
              const nextMinStartWord = minStartWord === null
                ? occurrence.startWord
                : Math.min(minStartWord, occurrence.startWord);
              const nextMaxEndWord = maxEndWord === null
                ? occurrence.endWord
                : Math.max(maxEndWord, occurrence.endWord);
              const span = nextMaxEndWord - nextMinStartWord;
              if (span > nearWordGap) {
                continue;
              }
              const nextMinStartChar = minStartCharValue === null
                ? occurrence.start
                : Math.min(minStartCharValue, occurrence.start);
              const nextMaxEndChar = maxEndChar === null
                ? occurrence.end
                : Math.max(maxEndChar, occurrence.end);
              if (best) {
                if (nextMinStartChar > best.startChar) {
                  continue;
                }
                if (nextMinStartChar === best.startChar && span > best.span) {
                  continue;
                }
              }
              usedStarts.add(occurrence.start);
              search(
                orderIndex + 1,
                usedStarts,
                nextMinStartWord,
                nextMaxEndWord,
                nextMinStartChar,
                nextMaxEndChar,
              );
              usedStarts.delete(occurrence.start);
            }
          };

          search(0, new Set(), null, null, null, null);
          return best ? { ...best, terms: group } : null;
        }

        for (const group of normalizedNearGroups) {
          let nextMinStartChar = 0;
          while (true) {
            const candidate = earliestWindowForGroup(group, nextMinStartChar);
            if (!candidate) {
              break;
            }
            nearFocusWindows.push(candidate);
            nextMinStartChar = Math.max(nextMinStartChar + 1, candidate.endChar);
          }
        }

        nearFocusWindows.sort((left, right) => {
          if (left.startChar !== right.startChar) return left.startChar - right.startChar;
          if (left.span !== right.span) return left.span - right.span;
          return left.endChar - right.endChar;
        });
      }
    }

    if (normalizedNearGroups.length && !nearFocusWindows.length) {
      if (typeof window.__mdexploreRefreshScrollHitMarkers === 'function') {
        window.__mdexploreRefreshScrollHitMarkers();
      }
      return { matches: 0 };
    }
    if (!normalizedNearGroups.length && !normalizedTerms.length) {
      if (typeof window.__mdexploreRefreshScrollHitMarkers === 'function') {
        window.__mdexploreRefreshScrollHitMarkers();
      }
      return { matches: 0 };
    }

    function addTermRanges(segment, ranges, termInfo, focusWindow = null) {
      const pattern = buildTermPattern(termInfo.text, termInfo.caseSensitive, !!focusWindow);
      let match = null;
      while ((match = pattern.exec(segment.text)) !== null) {
        const localStart = match.index;
        const localEnd = localStart + match[0].length;
        const absoluteStart = segment.start + localStart;
        const absoluteEnd = segment.start + localEnd;
        if (focusWindow && (absoluteStart < focusWindow.startChar || absoluteEnd > focusWindow.endChar)) {
          if (pattern.lastIndex <= localStart) {
            pattern.lastIndex = localStart + 1;
          }
          continue;
        }
        ranges.push({ start: localStart, end: localEnd });
        if (pattern.lastIndex <= localStart) {
          pattern.lastIndex = localStart + 1;
        }
      }
    }

    function collectRanges(segment) {
      const ranges = [];
      if (nearFocusWindows.length) {
        for (const focusWindow of nearFocusWindows) {
          if (segment.end <= focusWindow.startChar || segment.start >= focusWindow.endChar) {
            continue;
          }
          for (const termInfo of focusWindow.terms) {
            addTermRanges(segment, ranges, termInfo, focusWindow);
          }
        }
      } else {
        for (const termInfo of normalizedTerms) {
          addTermRanges(segment, ranges, termInfo);
        }
      }
      if (!ranges.length) {
        return [];
      }
      ranges.sort((left, right) => {
        if (left.start !== right.start) return left.start - right.start;
        return (right.end - right.start) - (left.end - left.start);
      });
      const deduped = [];
      let lastEnd = -1;
      for (const item of ranges) {
        if (item.start < lastEnd) {
          continue;
        }
        deduped.push(item);
        lastEnd = item.end;
      }
      return deduped;
    }

    let firstMark = null;
    let matchCount = 0;
    for (const segment of segments) {
      const ranges = collectRanges(segment);
      if (!ranges.length) {
        continue;
      }
      const text = segment.text;
      let cursor = 0;
      const fragment = document.createDocumentFragment();
      for (const range of ranges) {
        if (range.start > cursor) {
          fragment.appendChild(document.createTextNode(text.slice(cursor, range.start)));
        }
        const mark = document.createElement('span');
        mark.setAttribute('data-mdexplore-search-mark', '1');
        mark.style.backgroundColor = '#f5d34f';
        mark.style.color = '#111827';
        mark.style.padding = '0 1px';
        mark.style.borderRadius = '2px';
        mark.textContent = text.slice(range.start, range.end);
        fragment.appendChild(mark);
        if (!firstMark) {
          firstMark = mark;
        }
        matchCount += 1;
        cursor = range.end;
      }
      if (cursor < text.length) {
        fragment.appendChild(document.createTextNode(text.slice(cursor)));
      }
      const parent = segment.node.parentNode;
      if (parent) {
        parent.replaceChild(fragment, segment.node);
      }
    }

    if (typeof window.__mdexploreRefreshScrollHitMarkers === 'function') {
      window.__mdexploreRefreshScrollHitMarkers();
    }
    if (firstMark && shouldScroll) {
      firstMark.scrollIntoView({ behavior: 'auto', block: 'center', inline: 'nearest' });
    }
    return { matches: matchCount };
  }

  window.__mdextSearchCompat = {
    clearSearchHighlights,
    highlightSearchTerms,
  };
})();