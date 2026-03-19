(() => {
  const mermaidBlocks = Array.from(document.querySelectorAll(".mermaid"));
  const hasMermaidNodes = mermaidBlocks.length > 0;
  const hasPendingMermaid = mermaidBlocks.some((block) =>
    block instanceof HTMLElement && block.classList.contains("mermaid-pending")
  );
  const mermaidReady =
    !hasMermaidNodes || (!!window.__mdexploreMermaidReady && !hasPendingMermaid);

  const hasPendingPlantUml = !!document.querySelector(".plantuml-pending");
  const plantumlImages = Array.from(document.querySelectorAll("img.plantuml"));
  const plantumlImagesReady = plantumlImages.every((img) =>
    !(img instanceof HTMLImageElement) || (img.complete && img.naturalWidth > 0)
  );
  const plantumlDomReady = !hasPendingPlantUml && plantumlImagesReady;

  return { mermaidReady, plantumlDomReady };
})();
