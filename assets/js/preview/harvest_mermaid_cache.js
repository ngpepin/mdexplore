(() => {
  const cache = window.__mdexploreMermaidSvgCacheByMode;
  if (!cache || typeof cache !== "object") {
    return {};
  }
  return cache;
})();
