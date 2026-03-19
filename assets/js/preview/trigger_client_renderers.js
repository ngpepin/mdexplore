(() => {
  if (window.__mdexploreRunClientRenderers) {
    window.__mdexploreRunClientRenderers();
  } else if (window.__mdexploreTryTypesetMath) {
    window.__mdexploreTryTypesetMath();
  }
})();
