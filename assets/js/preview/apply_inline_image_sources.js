(() => {
  const urlsByDigest = __INLINE_IMAGE_URLS_JSON__;
  if (!urlsByDigest || typeof urlsByDigest !== "object") {
    return { updated: 0, appliedDigests: [] };
  }
  let updated = 0;
  const appliedDigests = [];
  for (const [digest, nextUrl] of Object.entries(urlsByDigest)) {
    if (typeof digest !== "string" || typeof nextUrl !== "string" || !digest || !nextUrl) {
      continue;
    }
    const selector = `img[data-mdexplore-inline-image-digest="${digest}"]`;
    let digestApplied = false;
    for (const node of Array.from(document.querySelectorAll(selector))) {
      try {
        node.setAttribute("src", nextUrl);
        node.removeAttribute("data-mdexplore-inline-image-pending");
        updated += 1;
        digestApplied = true;
      } catch (_err) {
        // Ignore per-node update failures so remaining images still hydrate.
      }
    }
    if (digestApplied) {
      appliedDigests.push(digest);
    }
  }
  return { updated, appliedDigests };
})();
