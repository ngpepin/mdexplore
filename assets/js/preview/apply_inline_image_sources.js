(() => {
  const urlsByDigest = __INLINE_IMAGE_URLS_JSON__;
  if (!urlsByDigest || typeof urlsByDigest !== "object") {
    return 0;
  }
  let updated = 0;
  for (const [digest, nextUrl] of Object.entries(urlsByDigest)) {
    if (typeof digest !== "string" || typeof nextUrl !== "string" || !digest || !nextUrl) {
      continue;
    }
    const selector = `img[data-mdexplore-inline-image-digest="${digest}"]`;
    for (const node of Array.from(document.querySelectorAll(selector))) {
      try {
        node.setAttribute("src", nextUrl);
        node.removeAttribute("data-mdexplore-inline-image-pending");
        updated += 1;
      } catch (_err) {
        // Ignore per-node update failures so remaining images still hydrate.
      }
    }
  }
  return updated;
})();
