(() => {
  const topBandY = 12;
  const viewportHeight = Math.max(1, Number(window.innerHeight) || 0);
  const taggedNodes = Array.from(document.querySelectorAll('[data-md-line-start]'));
  let crossingLine = null;
  let crossingTop = -Infinity;
  let aboveLine = null;
  let aboveBottom = -Infinity;
  let belowLine = null;
  let belowTop = Infinity;
  for (const node of taggedNodes) {
    const rawValue = parseInt(node.getAttribute('data-md-line-start') || "", 10);
    if (Number.isNaN(rawValue)) continue;
    const lineValue = rawValue + 1;
    const rect = node.getBoundingClientRect();
    if (!rect) continue;
    if (!Number.isFinite(rect.top) || !Number.isFinite(rect.bottom)) continue;
    if (rect.height <= 0) continue;
    if (rect.bottom <= 0 || rect.top >= viewportHeight) continue;
    if (rect.top <= topBandY && rect.bottom > topBandY) {
      if (
        rect.top > crossingTop
        || (rect.top === crossingTop && (crossingLine === null || lineValue < crossingLine))
      ) {
        crossingTop = rect.top;
        crossingLine = lineValue;
      }
      continue;
    }
    if (rect.bottom <= topBandY) {
      if (
        rect.bottom > aboveBottom
        || (rect.bottom === aboveBottom && (aboveLine === null || lineValue > aboveLine))
      ) {
        aboveBottom = rect.bottom;
        aboveLine = lineValue;
      }
      continue;
    }
    if (
      rect.top < belowTop
      || (rect.top === belowTop && (belowLine === null || lineValue < belowLine))
    ) {
      belowTop = rect.top;
      belowLine = lineValue;
    }
  }
  if (crossingLine !== null) return crossingLine;
  if (aboveLine !== null) return aboveLine;
  if (belowLine !== null) return belowLine;
  return 1;
})();
