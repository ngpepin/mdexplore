(() => {
  const hasMathNodes = !!document.querySelector(".mdexplore-math-block, mjx-container, .MathJax");
  const hasMermaidNodes = !!document.querySelector(".mermaid");
  return {
    hasMathNodes,
    hasMermaidNodes,
    mathReady: !hasMathNodes || !!window.__mdexploreMathReady,
    mermaidReady: !hasMermaidNodes || !!window.__mdexploreMermaidReady
  };
})();
