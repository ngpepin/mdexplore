(() => {
  // Rebuild highlight spans from plain text each pass so updates are idempotent.
  const terms = __TERMS_JSON__;
  const shouldScroll = __SCROLL_BOOL__;
  const nearWordGap = __NEAR_WORD_GAP__;
  const nearTermGroups = __NEAR_GROUPS_JSON__;
  const root = document.querySelector("main") || document.body;
  if (!root || !terms.length) return 0;

  const markSelector = 'span[data-mdexplore-search-mark="1"]';
  for (const oldMark of root.querySelectorAll(markSelector)) {
    const parent = oldMark.parentNode;
    if (!parent) continue;
    parent.replaceChild(document.createTextNode(oldMark.textContent || ""), oldMark);
    parent.normalize();
  }

  const skipTags = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEXTAREA"]);
  const walker = document.createTreeWalker(
    root,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        if (!node || !node.nodeValue || !node.nodeValue.trim()) {
          return NodeFilter.FILTER_REJECT;
        }
        const parent = node.parentElement;
        if (!parent) {
          return NodeFilter.FILTER_REJECT;
        }
        if (skipTags.has(parent.tagName)) {
          return NodeFilter.FILTER_REJECT;
        }
        if (parent.closest(markSelector)) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    },
  );

  const segments = [];
  let fullText = "";
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const value = node.nodeValue || "";
    if (!value) {
      continue;
    }
    const start = fullText.length;
    fullText += value;
    const end = fullText.length;
    segments.push({ node, text: value, start, end });
    // Separate nodes to avoid accidental cross-node token merging.
    fullText += "\n";
  }
  if (!segments.length) return 0;

  function escapeRegExp(input) {
    return String(input || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function upperBound(values, target) {
    let lo = 0;
    let hi = values.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (values[mid] <= target) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  }

  function normalizeTerms(items) {
    if (!Array.isArray(items)) return [];
    const normalized = [];
    for (const item of items) {
      if (!item || typeof item.text !== "string") continue;
      const text = item.text;
      if (!text.trim()) continue;
      normalized.push({
        text,
        caseSensitive: !!item.caseSensitive,
      });
    }
    return normalized;
  }

  function normalizeNearGroups(groups) {
    if (!Array.isArray(groups)) return [];
    const normalized = [];
    for (const group of groups) {
      if (!Array.isArray(group)) continue;
      const next = [];
      for (const item of group) {
        if (!item || typeof item.text !== "string") continue;
        const text = item.text;
        if (!text.trim()) continue;
        next.push({
          text,
          caseSensitive: !!item.caseSensitive,
        });
      }
      if (next.length >= 2) normalized.push(next);
    }
    return normalized;
  }

  function hasDistinctStartAssignment(windowOccurrences, termCount) {
    const startsByTerm = Array.from({ length: termCount }, () => []);
    for (const occurrence of windowOccurrences) {
      if (!occurrence) continue;
      const termIndex = Number(occurrence.term);
      const startChar = Number(occurrence.start);
      if (!Number.isInteger(termIndex) || termIndex < 0 || termIndex >= termCount) {
        continue;
      }
      if (!Number.isFinite(startChar)) {
        continue;
      }
      const starts = startsByTerm[termIndex];
      if (!starts.includes(startChar)) {
        starts.push(startChar);
      }
    }

    const orderedTerms = startsByTerm
      .map((starts, term) => ({ term, starts }))
      .sort((left, right) => left.starts.length - right.starts.length);
    if (orderedTerms.some((item) => !item.starts.length)) {
      return false;
    }

    const usedStarts = new Set();
    const assign = (orderIndex) => {
      if (orderIndex >= orderedTerms.length) {
        return true;
      }
      const entry = orderedTerms[orderIndex];
      for (const startChar of entry.starts) {
        if (usedStarts.has(startChar)) {
          continue;
        }
        usedStarts.add(startChar);
        if (assign(orderIndex + 1)) {
          return true;
        }
        usedStarts.delete(startChar);
      }
      return false;
    };

    return assign(0);
  }

  function shouldUseCloseWordBoundaries(termText) {
    return typeof termText === "string" && !!termText && !/\s/.test(termText) && /^\w+$/u.test(termText);
  }

  function isWordCharAt(text, index) {
    if (index < 0 || index >= text.length) {
      return false;
    }
    return /\w/u.test(text.charAt(index));
  }

  const normalizedTerms = normalizeTerms(terms);
  const normalizedCloseGroups = normalizeNearGroups(nearTermGroups);
  const closeFocusWindows = [];

  if (normalizedCloseGroups.length) {
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
          const enforceWordBoundaries = shouldUseCloseWordBoundaries(termInfo.text);
          const pattern = new RegExp(
            escapeRegExp(termInfo.text),
            termInfo.caseSensitive ? "g" : "gi"
          );
          let m = null;
          while ((m = pattern.exec(fullText)) !== null) {
            const startChar = m.index;
            const endChar = startChar + m[0].length;
            if (startChar < minStartChar) {
              if (pattern.lastIndex <= startChar) {
                pattern.lastIndex = startChar + 1;
              }
              continue;
            }
            if (
              enforceWordBoundaries &&
              (isWordCharAt(fullText, startChar - 1) || isWordCharAt(fullText, endChar))
            ) {
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

        const search = (
          orderIndex,
          usedStarts,
          minStartWord,
          maxEndWord,
          minStartCharValue,
          maxEndChar
        ) => {
          if (orderIndex >= orderedTerms.length) {
            if (
              minStartWord === null ||
              maxEndWord === null ||
              minStartCharValue === null ||
              maxEndChar === null
            ) {
              return;
            }
            const span = maxEndWord - minStartWord;
            if (
              !best ||
              minStartCharValue < best.startChar ||
              (
                minStartCharValue === best.startChar &&
                (
                  span < best.span ||
                  (span === best.span && maxEndChar < best.endChar)
                )
              )
            ) {
              best = {
                span,
                startWord: minStartWord,
                endWord: maxEndWord,
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
            const nextMinStartWord =
              minStartWord === null
                ? occurrence.startWord
                : Math.min(minStartWord, occurrence.startWord);
            const nextMaxEndWord =
              maxEndWord === null
                ? occurrence.endWord
                : Math.max(maxEndWord, occurrence.endWord);
            const span = nextMaxEndWord - nextMinStartWord;
            if (span > nearWordGap) {
              continue;
            }
            const nextMinStartChar =
              minStartCharValue === null
                ? occurrence.start
                : Math.min(minStartCharValue, occurrence.start);
            const nextMaxEndChar =
              maxEndChar === null
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
              nextMaxEndChar
            );
            usedStarts.delete(occurrence.start);
          }
        };

        search(0, new Set(), null, null, null, null);

        if (!best) {
          return null;
        }
        return {
          span: best.span,
          startChar: best.startChar,
          endChar: best.endChar,
          terms: group,
        };
      }

      for (const group of normalizedCloseGroups) {
        let nextMinStartChar = 0;
        while (true) {
          const candidate = earliestWindowForGroup(group, nextMinStartChar);
          if (!candidate) {
            break;
          }
          closeFocusWindows.push(candidate);
          nextMinStartChar = Math.max(nextMinStartChar + 1, candidate.endChar);
        }
      }

      closeFocusWindows.sort((left, right) => {
        if (left.startChar !== right.startChar) return left.startChar - right.startChar;
        if (left.span !== right.span) return left.span - right.span;
        return left.endChar - right.endChar;
      });
    }
  }

  // If NEAR(...) is present, highlight every qualifying NEAR window.
  if (normalizedCloseGroups.length && !closeFocusWindows.length) {
    return 0;
  }

  if (!normalizedCloseGroups.length && !normalizedTerms.length) return 0;

  function addTermRanges(segment, ranges, termInfo, focusWindow = null) {
    const rawText = termInfo && typeof termInfo.text === "string" ? termInfo.text : "";
    const termText = rawText;
    if (!termText.trim()) return;
    const enforceWordBoundaries = !!focusWindow && shouldUseCloseWordBoundaries(termText);
    const pattern = new RegExp(
      escapeRegExp(termText),
      termInfo.caseSensitive ? "g" : "gi"
    );
    let m = null;
    while ((m = pattern.exec(segment.text)) !== null) {
      const localStart = m.index;
      const localEnd = localStart + m[0].length;
      if (
        enforceWordBoundaries &&
        ((localStart > 0 && /\w/u.test(segment.text.charAt(localStart - 1))) ||
          (localEnd < segment.text.length && /\w/u.test(segment.text.charAt(localEnd))))
      ) {
        if (pattern.lastIndex <= localStart) {
          pattern.lastIndex = localStart + 1;
        }
        continue;
      }
      const absoluteStart = segment.start + localStart;
      const absoluteEnd = segment.start + localEnd;
      if (focusWindow) {
        if (absoluteStart < focusWindow.startChar || absoluteEnd > focusWindow.endChar) {
          if (pattern.lastIndex <= localStart) {
            pattern.lastIndex = localStart + 1;
          }
          continue;
        }
      }
      ranges.push({ start: localStart, end: localEnd });
      if (pattern.lastIndex <= localStart) {
        pattern.lastIndex = localStart + 1;
      }
    }
  }

  function collectRanges(segment) {
    const ranges = [];
    if (closeFocusWindows.length) {
      for (const focusWindow of closeFocusWindows) {
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

    ranges.sort((a, b) => {
      if (a.start !== b.start) return a.start - b.start;
      return (b.end - b.start) - (a.end - a.start);
    });

    const deduped = [];
    let lastEnd = -1;
    for (const item of ranges) {
      if (item.start < lastEnd) continue;
      deduped.push(item);
      lastEnd = item.end;
    }
    return deduped;
  }

  let firstMark = null;
  let matchCount = 0;
  for (const segment of segments) {
    const ranges = collectRanges(segment);
    if (!ranges.length) continue;

    const text = segment.text;
    let cursor = 0;
    const fragment = document.createDocumentFragment();
    for (const range of ranges) {
      if (range.start > cursor) {
        fragment.appendChild(document.createTextNode(text.slice(cursor, range.start)));
      }
      const mark = document.createElement("span");
      mark.setAttribute("data-mdexplore-search-mark", "1");
      mark.style.backgroundColor = "#f5d34f";
      mark.style.color = "#111827";
      mark.style.padding = "0 1px";
      mark.style.borderRadius = "2px";
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

  if (firstMark && shouldScroll) {
    firstMark.scrollIntoView({ behavior: "auto", block: "center", inline: "nearest" });
  }
  return matchCount;
})();
