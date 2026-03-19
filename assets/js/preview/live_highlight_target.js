(() => {
  const rawId =
    typeof window.__mdexploreLastPersistentHighlightId === "string"
      ? window.__mdexploreLastPersistentHighlightId.trim()
      : "";
  const rawOffset =
    typeof window.__mdexploreLastPersistentHighlightOffset === "number" &&
    Number.isFinite(window.__mdexploreLastPersistentHighlightOffset)
      ? Math.max(0, Math.floor(window.__mdexploreLastPersistentHighlightOffset))
      : null;
  return {
    clickedHighlightId: rawId,
    clickedOffset: rawOffset,
  };
})()
