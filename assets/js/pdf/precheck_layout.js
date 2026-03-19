(() => {
  // Print-only math tuning to avoid cramped/squished glyph appearance in PDF.
  if (!document.getElementById("__mdexplore_pdf_math_style")) {
    const style = document.createElement("style");
    style.id = "__mdexplore_pdf_math_style";
    style.textContent = `
@media print {
  @page {
    size: Letter portrait;
  }
  @page mdexploreLandscape {
    size: Letter landscape;
  }
  main {
    max-width: none !important;
    width: auto !important;
    margin: 0 !important;
    padding: 1.1rem 0.9rem 4rem 0.9rem !important;
  }
  .mdexplore-print-block {
    display: block;
    margin-left: auto !important;
    margin-right: auto !important;
  }
  .mdexplore-print-block.mdexplore-print-keep {
    break-inside: avoid-page !important;
    page-break-inside: avoid !important;
  }
  .mdexplore-print-block.mdexplore-print-heading-keep {
    break-inside: avoid-page !important;
    page-break-inside: avoid !important;
  }
  .mdexplore-print-block.mdexplore-print-break-before {
    break-before: page !important;
    page-break-before: always !important;
  }
  .mdexplore-fence.mdexplore-print-break-before {
    display: block !important;
    break-before: page !important;
    page-break-before: always !important;
  }
  .mdexplore-print-sequence-page-break {
    display: block !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    break-before: page !important;
    page-break-before: always !important;
  }
  .mdexplore-print-landscape-start {
    display: block !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    break-before: page !important;
    page-break-before: always !important;
    page: mdexploreLandscape !important;
  }
  hr.mdexplore-print-skip {
    display: none !important;
  }
  .mdexplore-print-block.mdexplore-print-landscape-page {
    break-after: page;
    page-break-after: always;
  }
  .mdexplore-print-block.mdexplore-print-landscape-page:last-child {
    break-after: auto;
    page-break-after: auto;
  }
  .mdexplore-print-heading-anchor {
    break-after: avoid-page;
    page-break-after: avoid;
  }
  .mdexplore-print-heading-break-before {
    break-before: page !important;
    page-break-before: always !important;
  }
  .mdexplore-print-heading-landscape {
    page: mdexploreLandscape;
  }
  .mdexplore-print-heading-landscape + .mdexplore-fence.mdexplore-print-landscape-page,
  .mdexplore-print-heading-landscape + .mdexplore-print-block.mdexplore-print-landscape-page {
    break-before: avoid-page !important;
    page-break-before: avoid !important;
  }
  .mdexplore-fence.mdexplore-print-with-heading {
    break-before: avoid-page;
    page-break-before: avoid;
  }
  .mdexplore-fence {
    margin-left: auto !important;
    margin-right: auto !important;
  }
  .mdexplore-fence.mdexplore-print-keep {
    display: inline-block !important;
    break-inside: avoid-page !important;
    page-break-inside: avoid !important;
  }
  .mdexplore-fence.mdexplore-print-landscape-page {
    break-after: page;
    page-break-after: always;
  }
  .mdexplore-fence.mdexplore-print-landscape-page:last-child {
    break-after: auto;
    page-break-after: auto;
  }
  .mdexplore-fence.mdexplore-print-keep .mermaid,
  .mdexplore-fence.mdexplore-print-keep img.plantuml,
  .mdexplore-fence.mdexplore-print-keep svg.mdexplore-plantuml-inline,
  .mdexplore-fence.mdexplore-print-keep .mermaid svg {
    break-inside: avoid-page !important;
    page-break-inside: avoid !important;
  }
  .mdexplore-fence.mdexplore-print-allow-break {
    break-inside: auto !important;
    page-break-inside: auto !important;
  }
  .mdexplore-fence .mermaid {
    display: block;
    width: 100% !important;
    max-width: 100% !important;
    min-height: var(--mdexplore-print-diagram-reserved-height, auto);
  }
  .mdexplore-fence img.plantuml,
  .mdexplore-fence svg.mdexplore-plantuml-inline {
    display: block;
    width: var(--mdexplore-print-diagram-width, auto) !important;
    max-width: var(--mdexplore-print-diagram-max-width, 100%) !important;
    height: var(--mdexplore-print-diagram-height, auto) !important;
    margin: 0 auto !important;
  }
  .mdexplore-fence {
    width: var(--mdexplore-print-section-width, auto) !important;
    max-width: 100% !important;
    min-height: var(--mdexplore-print-diagram-reserved-height, auto);
  }
  .mdexplore-fence .mermaid svg {
    display: block;
    width: var(--mdexplore-print-diagram-width, auto) !important;
    max-width: var(--mdexplore-print-diagram-max-width, 100%) !important;
    height: var(--mdexplore-print-diagram-height, auto) !important;
    margin: 0 auto !important;
  }
  .mdexplore-fence.mdexplore-print-keep img.plantuml,
  .mdexplore-fence.mdexplore-print-keep svg.mdexplore-plantuml-inline,
  .mdexplore-fence.mdexplore-print-keep .mermaid svg {
    max-height: var(--mdexplore-print-diagram-max-height, 86vh) !important;
    object-fit: contain;
  }
  .mdexplore-fence.mdexplore-print-allow-break img.plantuml,
  .mdexplore-fence.mdexplore-print-allow-break svg.mdexplore-plantuml-inline,
  .mdexplore-fence.mdexplore-print-allow-break .mermaid svg {
    max-height: none !important;
  }
  main img:not(.plantuml) {
    display: block !important;
    width: auto !important;
    max-width: 100% !important;
    height: auto !important;
  }
  pre {
    overflow: visible !important;
    max-width: 100% !important;
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
  }
  pre.mdexplore-print-pre-keep {
    break-inside: avoid-page !important;
    page-break-inside: avoid !important;
  }
  pre > code,
  pre code,
  code[class*="language-"],
  pre code[class*="language-"] {
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
  }
  pre::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
  }
  mjx-container[jax="SVG"] {
    font-size: 1.08em !important;
    text-rendering: geometricPrecision;
    page-break-inside: avoid;
    break-inside: avoid;
  }
  mjx-container[jax="SVG"] > svg {
    overflow: visible;
    shape-rendering: geometricPrecision;
    text-rendering: geometricPrecision;
  }
  mjx-container[jax="SVG"][display="true"] {
    margin: 0.9em 0 1.05em 0 !important;
  }
  mjx-container[jax="CHTML"] {
    font-family: "STIX Two Math", "STIXGeneral", "Cambria Math", "Noto Sans Math", "Latin Modern Math", serif !important;
    font-kerning: normal !important;
    text-rendering: geometricPrecision;
  }
  mjx-container[jax="CHTML"] mjx-mi,
  mjx-container[jax="CHTML"] mjx-mo,
  mjx-container[jax="CHTML"] mjx-mn,
  mjx-container[jax="CHTML"] mjx-mtext {
    letter-spacing: 0.01em !important;
  }
}
`;
    document.head.appendChild(style);
  }

  if (document.documentElement) {
    document.documentElement.classList.add("mdexplore-pdf-export-mode");
  }
  document.body.classList.add("mdexplore-pdf-export-mode");
  if (!document.getElementById("__mdexplore_pdf_mermaid_light_override")) {
    const style = document.createElement("style");
    style.id = "__mdexplore_pdf_mermaid_light_override";
    style.textContent = `
body.mdexplore-pdf-export-mode .mdexplore-mermaid-toolbar {
  display: none !important;
}
body.mdexplore-pdf-export-mode .mdexplore-mermaid-viewport {
  overflow: hidden !important;
  scrollbar-width: none !important;
  -ms-overflow-style: none !important;
}
html.mdexplore-pdf-export-mode,
body.mdexplore-pdf-export-mode {
  --fg: #1a1a1a !important;
  --bg: #ffffff !important;
  --code-bg: #efefef !important;
  --border: #7a7a7a !important;
  --link: #2d2d2d !important;
  --callout-note-border: #666666 !important;
  --callout-note-bg: #f2f2f2 !important;
  --callout-tip-border: #666666 !important;
  --callout-tip-bg: #f2f2f2 !important;
  --callout-important-border: #666666 !important;
  --callout-important-bg: #f2f2f2 !important;
  --callout-warning-border: #666666 !important;
  --callout-warning-bg: #f2f2f2 !important;
  --callout-caution-border: #666666 !important;
  --callout-caution-bg: #f2f2f2 !important;
  color: #1a1a1a !important;
  background: #ffffff !important;
}
html.mdexplore-pdf-export-mode,
body.mdexplore-pdf-export-mode main {
  color: #1a1a1a !important;
  background: #ffffff !important;
}
body.mdexplore-pdf-export-mode a {
  color: #2d2d2d !important;
}
body.mdexplore-pdf-export-mode code,
body.mdexplore-pdf-export-mode pre {
  color: #1a1a1a !important;
  background: #efefef !important;
  border-color: #7a7a7a !important;
}
body.mdexplore-pdf-export-mode pre {
  overflow: visible !important;
  max-width: 100% !important;
  white-space: pre-wrap !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
}
body.mdexplore-pdf-export-mode pre > code,
body.mdexplore-pdf-export-mode pre code,
body.mdexplore-pdf-export-mode code[class*="language-"],
body.mdexplore-pdf-export-mode pre code[class*="language-"] {
  white-space: pre-wrap !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
}
body.mdexplore-pdf-export-mode pre::-webkit-scrollbar {
  width: 0 !important;
  height: 0 !important;
  display: none !important;
}
body.mdexplore-pdf-export-mode main {
  max-width: none !important;
  width: auto !important;
  margin: 0 !important;
  padding: 1.1rem 0.9rem 4rem 0.9rem !important;
}
body.mdexplore-pdf-export-mode main img:not(.plantuml) {
  display: block !important;
  width: auto !important;
  max-width: 100% !important;
  height: auto !important;
}
body.mdexplore-pdf-export-mode table,
body.mdexplore-pdf-export-mode th,
body.mdexplore-pdf-export-mode td,
body.mdexplore-pdf-export-mode blockquote,
body.mdexplore-pdf-export-mode .mdexplore-callout,
body.mdexplore-pdf-export-mode .mdexplore-fence {
  border-color: #7a7a7a !important;
}
`;
    document.head.appendChild(style);
  }

  const decodeSvgDataUriForPdf = (src) => {
    const raw = String(src || "");
    if (!raw.startsWith("data:image/svg+xml")) {
      return "";
    }
    const commaIndex = raw.indexOf(",");
    if (commaIndex < 0) {
      return "";
    }
    const meta = raw.slice(0, commaIndex).toLowerCase();
    const payload = raw.slice(commaIndex + 1);
    try {
      if (meta.includes(";base64")) {
        return atob(payload);
      }
      return decodeURIComponent(payload);
    } catch (_error) {
      return "";
    }
  };

  const intrinsicSvgSizeFromNodeForPdf = (svg) => {
    if (!(svg instanceof SVGElement)) {
      return { width: 0, height: 0 };
    }
    const viewBox = String(svg.getAttribute("viewBox") || "").trim();
    if (viewBox) {
      const parts = viewBox.split(/[\\s,]+/).map((part) => Number.parseFloat(part));
      if (parts.length === 4 && parts.every((num) => Number.isFinite(num))) {
        const width = Math.abs(parts[2]);
        const height = Math.abs(parts[3]);
        if (width > 0 && height > 0) {
          return { width, height };
        }
      }
    }
    const width = Number.parseFloat(String(svg.getAttribute("width") || "").replace(/px$/i, ""));
    const height = Number.parseFloat(String(svg.getAttribute("height") || "").replace(/px$/i, ""));
    if (Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0) {
      return { width, height };
    }
    return { width: 0, height: 0 };
  };

  const normalizePlantUmlInlineSvgForPdf = (svg) => {
    if (!(svg instanceof SVGElement)) {
      return;
    }
    svg.classList.add("plantuml", "mdexplore-plantuml-inline");
    svg.style.removeProperty("transform");
    svg.style.setProperty("display", "block", "important");
    svg.style.setProperty("width", "var(--mdexplore-print-diagram-width, auto)", "important");
    svg.style.setProperty("max-width", "var(--mdexplore-print-diagram-max-width, 100%)", "important");
    svg.style.setProperty("height", "var(--mdexplore-print-diagram-height, auto)", "important");
    svg.style.setProperty("max-height", "var(--mdexplore-print-diagram-max-height, none)", "important");
    svg.style.setProperty("margin", "0 auto", "important");
  };

  const replacePlantUmlImageWithInlineSvgForPdf = (img) => {
    if (!(img instanceof HTMLImageElement)) {
      return null;
    }
    if (img.dataset.mdexplorePlantumlInline === "1") {
      return null;
    }
    const markup = decodeSvgDataUriForPdf(img.currentSrc || img.src || "");
    if (!markup) {
      return null;
    }
    try {
      const template = document.createElement("template");
      template.innerHTML = markup.trim();
      const svg = template.content.firstElementChild;
      if (!(svg instanceof SVGElement) || String(svg.tagName || "").toLowerCase() !== "svg") {
        return null;
      }
      const replacement = svg;
      replacement.setAttribute("data-mdexplore-original-img", img.outerHTML);
      replacement.setAttribute("data-mdexplorePlantumlInline", "1");
      replacement.dataset.mdexplorePlantumlInline = "1";
      normalizePlantUmlInlineSvgForPdf(replacement);
      img.replaceWith(replacement);
      return replacement;
    } catch (_error) {
      return null;
    }
  };

  const normalizeDiagramStateForPdf = () => {
    // Flatten interactive wrappers so current scroll/pan/zoom cannot leak into PDF.
    for (const shell of Array.from(document.querySelectorAll(".mdexplore-mermaid-shell"))) {
      if (!(shell instanceof HTMLElement)) {
        continue;
      }
      const host = shell.parentElement;
      if (!(host instanceof HTMLElement)) {
        continue;
      }
      host.style.setProperty("display", "block", "important");
      host.style.setProperty("width", "100%", "important");
      host.style.setProperty("max-width", "100%", "important");
      const viewport = shell.querySelector(".mdexplore-mermaid-viewport");
      const svg = viewport instanceof HTMLElement ? viewport.querySelector("svg") : shell.querySelector("svg");
      const plantImg =
        viewport instanceof HTMLElement ? viewport.querySelector("img.plantuml") : shell.querySelector("img.plantuml");
      const plantSvg =
        viewport instanceof HTMLElement
          ? viewport.querySelector("svg.mdexplore-plantuml-inline")
          : shell.querySelector("svg.mdexplore-plantuml-inline");
      if (svg instanceof SVGElement) {
        svg.style.removeProperty("transform");
        svg.removeAttribute("width");
        svg.removeAttribute("height");
        svg.style.removeProperty("width");
        svg.style.setProperty("width", "var(--mdexplore-print-diagram-width, auto)", "important");
        svg.style.setProperty("max-width", "var(--mdexplore-print-diagram-max-width, 100%)", "important");
        svg.style.setProperty("height", "var(--mdexplore-print-diagram-height, auto)", "important");
        svg.style.setProperty("max-height", "var(--mdexplore-print-diagram-max-height, none)", "important");
        host.innerHTML = "";
        host.appendChild(svg);
        continue;
      }
      const normalizedPlant =
        plantSvg instanceof SVGElement ? plantSvg : replacePlantUmlImageWithInlineSvgForPdf(plantImg);
      if (normalizedPlant instanceof SVGElement) {
        normalizePlantUmlInlineSvgForPdf(normalizedPlant);
        host.innerHTML = "";
        host.appendChild(normalizedPlant);
      }
    }

    for (const viewport of Array.from(document.querySelectorAll(".mdexplore-mermaid-viewport"))) {
      if (!(viewport instanceof HTMLElement)) {
        continue;
      }
      viewport.scrollLeft = 0;
      viewport.scrollTop = 0;
      viewport.style.setProperty("overflow", "hidden", "important");
      viewport.style.setProperty("scrollbar-width", "none", "important");
      viewport.style.setProperty("-ms-overflow-style", "none", "important");
    }

    for (const img of Array.from(document.querySelectorAll("img.plantuml"))) {
      if (!(img instanceof HTMLImageElement)) {
        continue;
      }
      replacePlantUmlImageWithInlineSvgForPdf(img);
    }

    for (const svg of Array.from(document.querySelectorAll("svg.mdexplore-plantuml-inline"))) {
      normalizePlantUmlInlineSvgForPdf(svg);
    }

    for (const svg of Array.from(document.querySelectorAll(".mermaid svg"))) {
      if (!(svg instanceof SVGElement)) {
        continue;
      }
      svg.style.removeProperty("transform");
      svg.removeAttribute("width");
      svg.removeAttribute("height");
      svg.style.removeProperty("width");
      svg.style.removeProperty("max-width");
      svg.style.removeProperty("height");
      svg.style.setProperty("display", "block", "important");
      svg.style.setProperty("width", "var(--mdexplore-print-diagram-width, auto)", "important");
      svg.style.setProperty("max-width", "var(--mdexplore-print-diagram-max-width, 100%)", "important");
      svg.style.setProperty("height", "var(--mdexplore-print-diagram-height, auto)", "important");
      svg.style.setProperty("max-height", "var(--mdexplore-print-diagram-max-height, none)", "important");
    }

    // Expand regular markdown raster images to full printable content width.
    for (const img of Array.from(document.querySelectorAll("main img"))) {
      if (!(img instanceof HTMLImageElement)) {
        continue;
      }
      if (img.classList.contains("plantuml")) {
        continue;
      }
      if (img.closest(".mermaid")) {
        continue;
      }
      img.style.removeProperty("transform");
      img.style.removeProperty("width");
      img.style.setProperty("display", "block", "important");
      img.style.setProperty("width", "auto", "important");
      img.style.setProperty("max-width", "100%", "important");
      img.style.setProperty("height", "auto", "important");
      img.style.setProperty("object-fit", "contain", "important");
    }

    // Prevent horizontal-scroll snapshots in PDF for long command/code lines.
    for (const preNode of Array.from(document.querySelectorAll("pre"))) {
      if (!(preNode instanceof HTMLElement)) {
        continue;
      }
      preNode.style.setProperty("overflow", "visible", "important");
      preNode.style.setProperty("max-width", "100%", "important");
      preNode.style.setProperty("white-space", "pre-wrap", "important");
      preNode.style.setProperty("overflow-wrap", "anywhere", "important");
      preNode.style.setProperty("word-break", "break-word", "important");
    }
    for (const codeNode of Array.from(document.querySelectorAll("pre code, code[class*='language-']"))) {
      if (!(codeNode instanceof HTMLElement)) {
        continue;
      }
      codeNode.style.setProperty("white-space", "pre-wrap", "important");
      codeNode.style.setProperty("overflow-wrap", "anywhere", "important");
      codeNode.style.setProperty("word-break", "break-word", "important");
    }
  };

    const forceMermaidSvgMonochromeForPdf = (svgNode, options = null) => {
      if (!(svgNode instanceof SVGElement)) {
        return;
      }
      const TEXT_DARK = "#1a1a1a";
    const isSequenceDiagram = !!(options && typeof options === "object" && options.isSequence === true);
    const TRANSPARENT_VALUES = new Set(["none", "transparent", "rgba(0, 0, 0, 0)", "rgba(0,0,0,0)"]);
    const textTags = new Set(["text", "tspan"]);
    const paintableSelector = "path, line, polyline, polygon, rect, circle, ellipse, text, tspan, g, stop, marker";
    const clampByte = (value) => Math.max(0, Math.min(255, Math.round(value)));
    const parseRgbaText = (raw) => {
      const text = String(raw || "").trim().toLowerCase();
      const rgbMatch = text.match(/^rgba?\\(([^)]+)\\)$/);
      if (!(rgbMatch && rgbMatch[1])) {
        return null;
      }
      const parts = rgbMatch[1]
        .split(",")
        .map((part) => Number.parseFloat(String(part).trim()))
        .filter((part) => Number.isFinite(part));
      if (parts.length < 3) {
        return null;
      }
      return {
        r: clampByte(parts[0]),
        g: clampByte(parts[1]),
        b: clampByte(parts[2]),
        a: parts.length >= 4 ? Math.max(0, Math.min(1, parts[3])) : 1,
      };
    };
    const parseColorToRgba = (value) => {
      const raw = String(value || "").trim().toLowerCase();
      if (!raw || TRANSPARENT_VALUES.has(raw) || raw.startsWith("url(")) {
        return null;
      }
      const rgbaDirect = parseRgbaText(raw);
      if (rgbaDirect) {
        return rgbaDirect;
      }
      const hex = raw.startsWith("#") ? raw.slice(1) : "";
      if (hex.length === 3 || hex.length === 4) {
        const r = parseInt(hex[0] + hex[0], 16);
        const g = parseInt(hex[1] + hex[1], 16);
        const b = parseInt(hex[2] + hex[2], 16);
        const a = hex.length === 4 ? parseInt(hex[3] + hex[3], 16) / 255 : 1;
        if ([r, g, b, a].every((v) => Number.isFinite(v))) {
          return { r, g, b, a };
        }
      }
      if (hex.length === 6 || hex.length === 8) {
        const r = parseInt(hex.slice(0, 2), 16);
        const g = parseInt(hex.slice(2, 4), 16);
        const b = parseInt(hex.slice(4, 6), 16);
        const a = hex.length === 8 ? parseInt(hex.slice(6, 8), 16) / 255 : 1;
        if ([r, g, b, a].every((v) => Number.isFinite(v))) {
          return { r, g, b, a };
        }
      }
      // Fallback to browser color parser for color functions we don't parse.
      try {
        if (!window.__mdexploreColorProbeEl || !(window.__mdexploreColorProbeEl instanceof HTMLElement)) {
          const probe = document.createElement("span");
          probe.style.position = "absolute";
          probe.style.left = "-10000px";
          probe.style.top = "-10000px";
          probe.style.visibility = "hidden";
          probe.style.pointerEvents = "none";
          probe.textContent = ".";
          document.body.appendChild(probe);
          window.__mdexploreColorProbeEl = probe;
        }
        const probe = window.__mdexploreColorProbeEl;
        probe.style.color = raw;
        const resolved = window.getComputedStyle(probe).color;
        const rgbaResolved = parseRgbaText(resolved);
        if (rgbaResolved) {
          return rgbaResolved;
        }
      } catch (_error) {
        // Ignore parser fallback failures and use static fallback gray.
      }
      return null;
    };
    const rgbToLuma = (r, g, b) => (0.2126 * r) + (0.7152 * g) + (0.0722 * b);
    const rgbSaturation = (r, g, b) => {
      const maxV = Math.max(r, g, b);
      const minV = Math.min(r, g, b);
      if (maxV <= 0.0001) {
        return 0;
      }
      return (maxV - minV) / maxV;
    };
    const grayRgbFromSource = (sourceColor, grayMin, grayMax, fallbackGray) => {
      const parsed = parseColorToRgba(sourceColor);
      if (!parsed || parsed.a <= 0.001) {
        const f = clampByte(fallbackGray);
        return `rgb(${f}, ${f}, ${f})`;
      }
      const luma = rgbToLuma(parsed.r, parsed.g, parsed.b);
      // Slightly lower high-saturation colors so colored fills with similar
      // luminance don't collapse into the same gray band.
      const sat = rgbSaturation(parsed.r, parsed.g, parsed.b);
      const adjustedLuma = Math.max(0, Math.min(255, luma - (sat * 26)));
      const mapped = clampByte(grayMin + ((grayMax - grayMin) * (adjustedLuma / 255)));
      return `rgb(${mapped}, ${mapped}, ${mapped})`;
    };
    const colorIsTransparent = (value) => {
      const normalized = String(value || "").trim().toLowerCase();
      return !normalized || TRANSPARENT_VALUES.has(normalized);
    };
    const lightenTowardWhite = (colorText, ratio = 0.75, fallback = 240) => {
      const parsed = parseColorToRgba(colorText);
      if (!parsed || parsed.a <= 0.001) {
        const f = clampByte(fallback);
        return `rgb(${f}, ${f}, ${f})`;
      }
      const t = Math.max(0, Math.min(1, Number(ratio)));
      const r = clampByte(parsed.r + ((255 - parsed.r) * t));
      const g = clampByte(parsed.g + ((255 - parsed.g) * t));
      const b = clampByte(parsed.b + ((255 - parsed.b) * t));
      return `rgb(${r}, ${g}, ${b})`;
    };
    const resolveSvgBounds = () => {
      const vb = String(svgNode.getAttribute("viewBox") || "").trim();
      if (vb) {
        const parts = vb.split(/[\\s,]+/).map((part) => Number.parseFloat(part));
        if (parts.length === 4 && parts.every((part) => Number.isFinite(part))) {
          return { x: parts[0], y: parts[1], width: Math.abs(parts[2]), height: Math.abs(parts[3]) };
        }
      }
      try {
        const bbox = svgNode.getBBox();
        if (bbox && Number.isFinite(bbox.width) && Number.isFinite(bbox.height) && bbox.width > 0 && bbox.height > 0) {
          return { x: bbox.x, y: bbox.y, width: bbox.width, height: bbox.height };
        }
      } catch (_error) {
        // Ignore and use fallback below.
      }
      return null;
    };
    const isFullDiagramBackground = (node, svgBounds) => {
      if (!(node instanceof SVGElement) || !svgBounds) {
        return false;
      }
      const computed = window.getComputedStyle(node);
      if (colorIsTransparent(computed.fill || "")) {
        return false;
      }
      let bbox = null;
      try {
        bbox = node.getBBox();
      } catch (_error) {
        bbox = null;
      }
      if (!bbox || bbox.width <= 0 || bbox.height <= 0) {
        return false;
      }

      const margin = Math.max(3, Math.min(svgBounds.width, svgBounds.height) * 0.012);
      const touchesLeft = bbox.x <= (svgBounds.x + margin);
      const touchesTop = bbox.y <= (svgBounds.y + margin);
      const touchesRight = (bbox.x + bbox.width) >= (svgBounds.x + svgBounds.width - margin);
      const touchesBottom = (bbox.y + bbox.height) >= (svgBounds.y + svgBounds.height - margin);
      const coversWidth = bbox.width >= (svgBounds.width * 0.92);
      const coversHeight = bbox.height >= (svgBounds.height * 0.92);
      return touchesLeft && touchesTop && touchesRight && touchesBottom && coversWidth && coversHeight;
    };

    const tightenSvgCanvasToContent = () => {
      let currentViewBox = null;
      const vbText = String(svgNode.getAttribute("viewBox") || "").trim();
      if (vbText) {
        const parts = vbText
          .split(/[\\s,]+/)
          .map((part) => Number.parseFloat(part))
          .filter((part) => Number.isFinite(part));
        if (parts.length === 4 && parts[2] > 1 && parts[3] > 1) {
          currentViewBox = { x: parts[0], y: parts[1], width: parts[2], height: parts[3] };
        }
      }
      let bbox = null;
      try {
        bbox = svgNode.getBBox();
      } catch (_error) {
        bbox = null;
      }
      if (!bbox || bbox.width <= 1 || bbox.height <= 1) {
        return;
      }
      const widthGain = currentViewBox ? (currentViewBox.width / Math.max(1, bbox.width)) : 1;
      const heightGain = currentViewBox ? (currentViewBox.height / Math.max(1, bbox.height)) : 1;
      if (widthGain < 1.02 && heightGain < 1.02) {
        return;
      }
      // Generic crop for all Mermaid diagrams in PDF mode with a small cushion.
      // Sequence diagrams keep a bit more room for arrow heads/labels.
      const padXRatio = isSequenceDiagram ? 0.05 : 0.028;
      const padYRatio = isSequenceDiagram ? 0.055 : 0.034;
      const padX = Math.max(7, bbox.width * padXRatio);
      const padY = Math.max(7, bbox.height * padYRatio);
      svgNode.setAttribute(
        "viewBox",
        `${bbox.x - padX} ${bbox.y - padY} ${bbox.width + (2 * padX)} ${bbox.height + (2 * padY)}`
      );
      svgNode.removeAttribute("width");
      svgNode.removeAttribute("height");
      svgNode.style.removeProperty("width");
      svgNode.style.removeProperty("max-width");
      svgNode.style.removeProperty("height");
    };

    // PDF must not inherit GUI viewport assumptions; re-fit canvas to content.
    tightenSvgCanvasToContent();

    svgNode.style.setProperty("background", "#ffffff", "important");
    svgNode.style.setProperty("color", TEXT_DARK, "important");
    svgNode.style.removeProperty("filter");
    svgNode.style.removeProperty("-webkit-filter");

    for (const node of Array.from(svgNode.querySelectorAll(paintableSelector))) {
      if (!(node instanceof SVGElement)) {
        continue;
      }
      const tag = String(node.tagName || "").toLowerCase();
      const computed = window.getComputedStyle(node);
      const computedFill = String(computed.fill || "").trim();
      const computedStroke = String(computed.stroke || "").trim();

      if (tag === "stop") {
        const stopGray = grayRgbFromSource(computed.stopColor || computedFill, 125, 246, 212);
        node.style.setProperty("stop-color", stopGray, "important");
        node.style.setProperty("stop-opacity", "1", "important");
        continue;
      }

      if (textTags.has(tag)) {
        // PDF monochrome mode keeps Mermaid text consistently dark.
        node.style.setProperty("fill", TEXT_DARK, "important");
        node.style.setProperty("stroke", "none", "important");
        node.style.setProperty("color", TEXT_DARK, "important");
        node.style.setProperty("opacity", "1", "important");
        continue;
      }

      if (!colorIsTransparent(computedFill)) {
        const inLabel = !!node.closest(".edgeLabel, .labelBkg, .messageText");
        const fillGray = inLabel
          ? grayRgbFromSource(computedFill, 204, 250, 234)
          : grayRgbFromSource(computedFill, 88, 242, 206);
        node.style.setProperty("fill", fillGray, "important");
        node.style.setProperty("fill-opacity", "1", "important");
      } else if (node.hasAttribute("fill")) {
        node.style.setProperty("fill", "none", "important");
      }

      if (!colorIsTransparent(computedStroke)) {
        const strokeGray = grayRgbFromSource(computedStroke, 28, 168, 96);
        node.style.setProperty("stroke", strokeGray, "important");
        node.style.setProperty("stroke-opacity", "1", "important");
      } else if (node.hasAttribute("stroke")) {
        node.style.setProperty("stroke", "none", "important");
      }
      node.style.setProperty("opacity", "1", "important");
    }

    // Some Mermaid renderers emit labels via foreignObject HTML instead of
    // pure SVG text nodes. Force readable print colors there as well.
    for (const foreign of Array.from(svgNode.querySelectorAll("foreignObject"))) {
      if (!(foreign instanceof SVGElement)) {
        continue;
      }
      foreign.style.setProperty("color", TEXT_DARK, "important");
      foreign.style.setProperty("opacity", "1", "important");
      const htmlLabels = Array.from(foreign.querySelectorAll("*"));
      for (const element of htmlLabels) {
        if (!(element instanceof HTMLElement)) {
          continue;
        }
        element.style.setProperty("color", TEXT_DARK, "important");
        element.style.setProperty("fill", TEXT_DARK, "important");
        // Keep fills transparent so grayscale shape treatment remains visible.
        element.style.setProperty("background", "transparent", "important");
        element.style.setProperty("border-color", "rgb(96, 96, 96)", "important");
      }
    }

    // If a Mermaid diagram has an edge-to-edge shaded background panel,
    // lighten that panel strongly (~75% toward white) to reduce page tint.
    const svgBounds = resolveSvgBounds();
    const backgroundCandidates = Array.from(
      svgNode.querySelectorAll("rect, path, polygon, circle, ellipse")
    );
    for (const node of backgroundCandidates) {
      if (!(node instanceof SVGElement)) {
        continue;
      }
      if (!isFullDiagramBackground(node, svgBounds)) {
        continue;
      }
      const computed = window.getComputedStyle(node);
      const lighterFill = lightenTowardWhite(computed.fill || node.getAttribute("fill") || "", 0.75, 244);
      node.style.setProperty("fill", lighterFill, "important");
      node.style.setProperty("fill-opacity", "1", "important");
    }
  };

  const forceAllMermaidMonochromeForPdf = () => {
    for (const block of Array.from(document.querySelectorAll(".mermaid"))) {
      if (!(block instanceof HTMLElement)) {
        continue;
      }
      const sourceText = String(
        (block.dataset && block.dataset.mdexploreMermaidSource) ||
          block.getAttribute("data-mdexplore-mermaid-source") ||
          "",
      );
      const explicitKind = String((block.dataset && block.dataset.mdexploreMermaidKind) || "")
        .trim()
        .toLowerCase();
      const detectedKind =
        explicitKind ||
        (typeof window.__mdexploreDetectMermaidKind === "function"
          ? String(window.__mdexploreDetectMermaidKind(sourceText) || "").toLowerCase()
          : "");
      const isSequenceDiagram =
        detectedKind === "sequence" || /^\\s*sequenceDiagram\\b/im.test(sourceText);
      const svg = block.querySelector("svg");
      if (svg instanceof SVGElement) {
        forceMermaidSvgMonochromeForPdf(svg, { isSequence: isSequenceDiagram });
      }
    }
  };

  const startPdfMermaidCleanRender = (forceRender = false) => {
    const mermaidBlocks = Array.from(document.querySelectorAll(".mermaid")).filter(
      (block) => block instanceof HTMLElement
    );
    const backend = String(window.__mdexploreMermaidBackend || "js").toLowerCase();
    if (backend === "rust") {
      if (window.__mdexplorePdfMermaidInFlight) {
        return;
      }
      window.__mdexplorePdfMermaidInFlight = true;
      window.__mdexplorePdfMermaidReady = false;
      window.__mdexplorePdfMermaidError = "";
      try {
        // Rust PDF mode uses the dedicated PDF SVG cache produced by Python
        // with default mmdr theming (no GUI post-processing).
        const cacheByMode = window.__mdexploreMermaidSvgCacheByMode;
        if (!cacheByMode || typeof cacheByMode !== "object") {
          window.__mdexploreMermaidSvgCacheByMode = {};
        }
        if (
          !window.__mdexploreMermaidSvgCacheByMode.auto ||
          typeof window.__mdexploreMermaidSvgCacheByMode.auto !== "object"
        ) {
          window.__mdexploreMermaidSvgCacheByMode.auto = {};
        }
        if (
          !window.__mdexploreMermaidSvgCacheByMode.pdf ||
          typeof window.__mdexploreMermaidSvgCacheByMode.pdf !== "object"
        ) {
          window.__mdexploreMermaidSvgCacheByMode.pdf = {};
        }
        const autoCache = window.__mdexploreMermaidSvgCacheByMode.auto;
        const pdfCache =
          window.__mdexploreMermaidSvgCacheByMode.pdf &&
          typeof window.__mdexploreMermaidSvgCacheByMode.pdf === "object"
            ? window.__mdexploreMermaidSvgCacheByMode.pdf
            : null;
        let missingCount = 0;
        for (const block of mermaidBlocks) {
          if (!(block instanceof HTMLElement)) {
            continue;
          }
          const hashKey = String(block.getAttribute("data-mdexplore-mermaid-hash") || "").trim().toLowerCase();
          if (hashKey) {
            const existingSvg = block.querySelector("svg");
            if (
              existingSvg instanceof SVGElement &&
              typeof existingSvg.outerHTML === "string" &&
              existingSvg.outerHTML.indexOf("<svg") >= 0 &&
              typeof autoCache[hashKey] !== "string"
            ) {
              // Snapshot preview SVG before replacing with PDF variant so
              // post-export restore can reliably return to GUI styling.
              autoCache[hashKey] = existingSvg.outerHTML;
            }
          }
          const cachedSvg = hashKey && pdfCache && typeof pdfCache[hashKey] === "string" ? pdfCache[hashKey] : "";
          if (cachedSvg && cachedSvg.indexOf("<svg") >= 0) {
            block.removeAttribute("data-mdexplore-mermaid-render");
            block.classList.remove("mermaid-pending", "mermaid-error", "mermaid-rust-fallback");
            block.classList.add("mermaid-ready");
            block.innerHTML = cachedSvg;
            continue;
          }
          const existingSvg = block.querySelector("svg");
          if (existingSvg instanceof SVGElement) {
            block.removeAttribute("data-mdexplore-mermaid-render");
            block.classList.remove("mermaid-pending", "mermaid-error", "mermaid-rust-fallback");
            block.classList.add("mermaid-ready");
            continue;
          }
          missingCount += 1;
          const rustError = (block.getAttribute("data-mdexplore-rust-error") || "").trim();
          block.removeAttribute("data-mdexplore-mermaid-render");
          block.classList.remove("mermaid-pending", "mermaid-ready", "mermaid-rust-fallback");
          block.classList.add("mermaid-error");
          block.textContent = rustError
            ? `Mermaid render failed: Rust renderer: ${rustError}`
            : "Mermaid render failed: Rust PDF SVG unavailable";
        }
        if (missingCount > 0) {
          window.__mdexplorePdfMermaidError = `${missingCount} Rust Mermaid block(s) missing cached PDF SVG`;
        }
        window.__mdexploreMermaidReady = true;
        window.__mdexploreMermaidPaletteMode = "pdf";
      } catch (error) {
        window.__mdexplorePdfMermaidError =
          error && error.message ? error.message : String(error || "Rust PDF Mermaid render failed");
        window.__mdexploreMermaidReady = false;
      } finally {
        window.__mdexplorePdfMermaidInFlight = false;
        window.__mdexplorePdfMermaidReady = true;
      }
      return;
    }
    if (mermaidBlocks.length === 0) {
      window.__mdexplorePdfMermaidReady = true;
      window.__mdexploreMermaidReady = true;
      window.__mdexploreMermaidPaletteMode = "pdf";
      return;
    }
    if (!forceRender && window.__mdexplorePdfMermaidReady && !window.__mdexplorePdfMermaidInFlight) {
      return;
    }
    if (window.__mdexplorePdfMermaidInFlight) {
      return;
    }
    window.__mdexplorePdfMermaidInFlight = true;
    window.__mdexplorePdfMermaidReady = false;
    window.__mdexplorePdfMermaidError = "";

    const normalizeMermaidSource = (value) => String(value || "").replace(/\\r\\n/g, "\\n").trim();

    (async () => {
      try {
        if (!window.__mdexploreLoadMermaidScript) {
          throw new Error("Mermaid loader unavailable in preview page");
        }
        const loaded = await window.__mdexploreLoadMermaidScript();
        if (!loaded || !window.mermaid) {
          throw new Error("Mermaid script failed to load for PDF render");
        }
        const config =
          (window.__mdexploreMermaidInitConfig && window.__mdexploreMermaidInitConfig("pdf")) || {
            startOnLoad: false,
            securityLevel: "loose",
            theme: "default",
            darkMode: false,
          };
        mermaid.initialize(config);

        let renderFailures = 0;
        for (let index = 0; index < mermaidBlocks.length; index += 1) {
          const block = mermaidBlocks[index];
          if (!(block instanceof HTMLElement)) {
            continue;
          }
          let sourceText = normalizeMermaidSource(block.dataset && block.dataset.mdexploreMermaidSource);
          if (!sourceText) {
            const hasRenderedDiagram = !!block.querySelector("svg");
            if (!hasRenderedDiagram) {
              sourceText = normalizeMermaidSource(block.textContent || "");
            }
            if (sourceText) {
              block.dataset.mdexploreMermaidSource = sourceText;
            }
          }
          if (!sourceText) {
            renderFailures += 1;
            block.classList.remove("mermaid-pending", "mermaid-ready");
            block.classList.add("mermaid-error");
            block.textContent = "Mermaid source unavailable for PDF render";
            continue;
          }
          block.classList.remove("mermaid-ready", "mermaid-error");
          block.classList.add("mermaid-pending");
          block.textContent = "Mermaid rendering...";
          try {
            const renderId = `mdexplore_pdf_mermaid_${Date.now()}_${index}`;
            const renderResult = await mermaid.render(renderId, sourceText);
            const svgMarkup =
              renderResult && typeof renderResult === "object" && typeof renderResult.svg === "string"
                ? renderResult.svg
                : String(renderResult || "");
            if (!svgMarkup || svgMarkup.indexOf("<svg") < 0) {
              throw new Error("Mermaid returned empty SVG for PDF render");
            }
            block.innerHTML = svgMarkup;
            const renderedSvg = block.querySelector("svg");
            forceMermaidSvgMonochromeForPdf(renderedSvg);
            block.classList.remove("mermaid-pending", "mermaid-error");
            block.classList.add("mermaid-ready");
          } catch (renderError) {
            renderFailures += 1;
            block.classList.remove("mermaid-pending", "mermaid-ready");
            block.classList.add("mermaid-error");
            const message =
              renderError && renderError.message ? renderError.message : String(renderError || "Unknown Mermaid error");
            block.textContent = `Mermaid render failed: ${message}`;
          }
        }

        window.__mdexploreMermaidReady = true;
        window.__mdexploreMermaidPaletteMode = "pdf";
        if (renderFailures > 0) {
          window.__mdexplorePdfMermaidError = `${renderFailures} Mermaid block(s) failed during PDF clean render`;
        }
      } catch (error) {
        window.__mdexplorePdfMermaidError = error && error.message ? error.message : String(error);
        window.__mdexploreMermaidReady = false;
      } finally {
        window.__mdexplorePdfMermaidReady = true;
        window.__mdexplorePdfMermaidInFlight = false;
      }
    })();
  };

  if (__MDEXPLORE_RESET_MERMAID__) {
    window.__mdexploreMermaidReady = false;
    window.__mdexploreMermaidPaletteMode = "";
    window.__mdexplorePdfMermaidReady = false;
    window.__mdexplorePdfMermaidInFlight = false;
    window.__mdexplorePdfMermaidError = "";
  }
  startPdfMermaidCleanRender(__MDEXPLORE_FORCE_MERMAID__);
  if (window.__mdexploreTryTypesetMath) {
    window.__mdexploreTryTypesetMath();
  }
  if (window.__mdexploreApplyPlantUmlZoomControls) {
    window.__mdexploreApplyPlantUmlZoomControls("pdf");
  }
  normalizeDiagramStateForPdf();
  // Apply print-safe grayscale for both Mermaid backends. Rust still uses the
  // dedicated PDF SVG source; this pass only normalizes print contrast.
  forceAllMermaidMonochromeForPdf();
  // Ensure interactive zoom/pan toolbars never appear in PDF snapshots.
  for (const toolbar of Array.from(document.querySelectorAll(".mdexplore-mermaid-toolbar"))) {
    if (!(toolbar instanceof HTMLElement)) {
      continue;
    }
    toolbar.dataset.mdexplorePdfHidden = "1";
    toolbar.style.setProperty("display", "none", "important");
  }
  // Hide diagram viewport scrollbars for PDF output.
  for (const viewport of Array.from(document.querySelectorAll(".mdexplore-mermaid-viewport"))) {
    if (!(viewport instanceof HTMLElement)) {
      continue;
    }
    viewport.dataset.mdexplorePdfViewportHidden = "1";
    viewport.style.setProperty("overflow", "hidden", "important");
    viewport.style.setProperty("scrollbar-width", "none", "important");
    viewport.style.setProperty("-ms-overflow-style", "none", "important");
    viewport.scrollLeft = 0;
    viewport.scrollTop = 0;
  }

  // Decide how each diagram section should paginate for PDF output. This pass
  // is the bridge between DOM content and the later Qt print snapshot: it
  // classifies sections as portrait/landscape and keep/spill, moves heading
  // clusters into diagram fences so they paginate as one unit, and records the
  // resulting layout hints for footer stamping.
  const markDiagramPrintLayout = () => {
    const PRINT_LAYOUT_KNOBS = __MDEXPLORE_PRINT_LAYOUT_KNOBS__;
    const HEADING_TO_DIAGRAM_GAP_PX = PRINT_LAYOUT_KNOBS.headingToDiagramGapPx;
    const PRINT_LAYOUT_SAFETY_PX = PRINT_LAYOUT_KNOBS.layoutSafetyPx;
    // PDF Mermaid shrink floor for one-page fit decisions. This is
    // intentionally user-tweakable and controls when tall flow/activity
    // diagrams are allowed to stay on one page instead of spilling.
    const MIN_PRINT_DIAGRAM_FONT_PT = PRINT_LAYOUT_KNOBS.minPrintDiagramFontPt;
    const minPrintDiagramFontPx = MIN_PRINT_DIAGRAM_FONT_PT * (4 / 3);
    const maxPrintDiagramFontPx = PRINT_LAYOUT_KNOBS.maxPrintDiagramFontPt * (4 / 3);
    // PDF page selection must be based on print-page geometry, not the live
    // GUI viewport. The export target is US Letter, so use Letter CSS-pixel
    // dimensions here; otherwise the keep/landscape classifier makes the wrong
    // tradeoffs for wide diagrams and heading orphan control.
    const letterPortraitWidthPx = PRINT_LAYOUT_KNOBS.portraitLetterWidthPx;
    const letterPortraitHeightPx = PRINT_LAYOUT_KNOBS.portraitLetterHeightPx;
    const letterLandscapeWidthPx = PRINT_LAYOUT_KNOBS.landscapeLetterWidthPx;
    const letterLandscapeHeightPx = PRINT_LAYOUT_KNOBS.landscapeLetterHeightPx;
    const printableWidthPortrait = Math.max(
      PRINT_LAYOUT_KNOBS.portraitMinWidthPx,
      letterPortraitWidthPx - PRINT_LAYOUT_KNOBS.horizontalMarginPx,
    );
    const printableHeightPortrait = Math.max(
      PRINT_LAYOUT_KNOBS.portraitMinHeightPx,
      letterPortraitHeightPx - PRINT_LAYOUT_KNOBS.verticalMarginPx,
    );
    const printableWidthLandscape = Math.max(
      PRINT_LAYOUT_KNOBS.landscapeMinWidthPx,
      letterLandscapeWidthPx - PRINT_LAYOUT_KNOBS.horizontalMarginPx,
    );
    const printableHeightLandscape = Math.max(
      PRINT_LAYOUT_KNOBS.landscapeMinHeightPx,
      letterLandscapeHeightPx - PRINT_LAYOUT_KNOBS.verticalMarginPx,
    );
    let diagramCount = 0;
    let keepCount = 0;
    let allowBreakCount = 0;
    let landscapeCount = 0;
    const landscapeHeadings = [];
    const diagramHeadings = [];

    // Reset any previous preflight wrappers before recomputing layout. PDF
    // export retries can run this block more than once.
    const unwrapPrintBlocks = () => {
      for (const block of Array.from(document.querySelectorAll(".mdexplore-print-block[data-mdexplore-print-block='1']"))) {
        if (!(block instanceof HTMLElement)) {
          continue;
        }
        const parent = block.parentNode;
        if (!parent) {
          continue;
        }
        while (block.firstChild) {
          parent.insertBefore(block.firstChild, block);
        }
        parent.removeChild(block);
      }
    };

    const parseLength = (value) => {
      const num = Number.parseFloat(String(value || "").replace(/px$/i, "").trim());
      return Number.isFinite(num) ? num : 0;
    };

    const headingLevel = (node) => {
      if (!(node instanceof HTMLElement)) {
        return 0;
      }
      const match = String(node.tagName || "").match(/^H([1-6])$/i);
      return match ? Number.parseInt(match[1], 10) : 0;
    };

    const previousMeaningfulElement = (node) => {
      let current = node ? node.previousSibling : null;
      while (current) {
        if (current instanceof HTMLElement) {
          return current;
        }
        if (current instanceof Text && String(current.textContent || "").trim()) {
          return null;
        }
        current = current.previousSibling;
      }
      return null;
    };

    const nextMeaningfulElement = (node) => {
      let current = node ? node.nextSibling : null;
      while (current) {
        if (current instanceof HTMLElement) {
          return current;
        }
        if (current instanceof Text && String(current.textContent || "").trim()) {
          return null;
        }
        current = current.nextSibling;
      }
      return null;
    };

    // Heading height is measured against an explicit printable width so orphan
    // control uses stable numbers instead of live viewport geometry.
    const stableHeadingHeight = (heading, widthPx) => {
      if (!(heading instanceof HTMLElement)) {
        return 0;
      }
      const restore = {
        display: heading.style.getPropertyValue("display"),
        width: heading.style.getPropertyValue("width"),
        maxWidth: heading.style.getPropertyValue("max-width"),
        boxSizing: heading.style.getPropertyValue("box-sizing"),
        whiteSpace: heading.style.getPropertyValue("white-space"),
      };
      heading.style.setProperty("display", "block", "important");
      if (widthPx > 0) {
        const widthText = `${Math.round(widthPx)}px`;
        heading.style.setProperty("width", widthText, "important");
        heading.style.setProperty("max-width", widthText, "important");
      }
      heading.style.setProperty("box-sizing", "border-box", "important");
      heading.style.setProperty("white-space", "normal", "important");
      const rect = heading.getBoundingClientRect();
      const computed = window.getComputedStyle(heading);
      const marginTop = parseLength(computed.marginTop);
      const marginBottom = parseLength(computed.marginBottom);
      const height = Math.max(0, rect.height + marginTop + marginBottom);
      if (restore.display) {
        heading.style.setProperty("display", restore.display);
      } else {
        heading.style.removeProperty("display");
      }
      if (restore.width) {
        heading.style.setProperty("width", restore.width);
      } else {
        heading.style.removeProperty("width");
      }
      if (restore.maxWidth) {
        heading.style.setProperty("max-width", restore.maxWidth);
      } else {
        heading.style.removeProperty("max-width");
      }
      if (restore.boxSizing) {
        heading.style.setProperty("box-sizing", restore.boxSizing);
      } else {
        heading.style.removeProperty("box-sizing");
      }
      if (restore.whiteSpace) {
        heading.style.setProperty("white-space", restore.whiteSpace);
      } else {
        heading.style.removeProperty("white-space");
      }
      return height;
    };

    const detectMermaidKindForFence = (fence) => {
      if (!(fence instanceof HTMLElement)) {
        return "";
      }
      const mermaid = fence.querySelector(".mermaid");
      if (!(mermaid instanceof HTMLElement)) {
        return "";
      }
      const explicitKind = String(mermaid.dataset.mdexploreMermaidKind || "")
        .trim()
        .toLowerCase();
      if (explicitKind) {
        return explicitKind;
      }
      const sourceText = String(mermaid.dataset.mdexploreMermaidSource || mermaid.textContent || "");
      if (/^\s*sequenceDiagram\b/im.test(sourceText)) {
        return "sequence";
      }
      if (/^\s*classDiagram\b/im.test(sourceText)) {
        return "class";
      }
      if (/^\s*erDiagram\b/im.test(sourceText)) {
        return "er";
      }
      if (/^\s*stateDiagram(?:-v2)?\b/im.test(sourceText)) {
        return "state";
      }
      if (/^\s*(?:graph|flowchart)\b/im.test(sourceText)) {
        return "flowchart";
      }
      return "";
    };

    // Diagram fit decisions operate on intrinsic SVG/image dimensions rather
    // than already-scaled CSS boxes so one-page and spill decisions are
    // reproducible.
    const intrinsicDiagramSize = (fence) => {
      if (!(fence instanceof HTMLElement)) {
        return { width: 0, height: 0 };
      }
      const svg = fence.querySelector("svg");
      if (svg instanceof SVGElement) {
        const viewBox = String(svg.getAttribute("viewBox") || "").trim();
        if (viewBox) {
          const parts = viewBox.split(/[\s,]+/).map((part) => Number.parseFloat(part));
          if (parts.length === 4 && parts.every((num) => Number.isFinite(num))) {
            const width = Math.abs(parts[2]);
            const height = Math.abs(parts[3]);
            if (width > 0 && height > 0) {
              return { width, height };
            }
          }
        }
        const width = parseLength(svg.getAttribute("width"));
        const height = parseLength(svg.getAttribute("height"));
        if (width > 0 && height > 0) {
          return { width, height };
        }
        try {
          const box = svg.getBBox();
          if (box && box.width > 0 && box.height > 0) {
            return { width: box.width, height: box.height };
          }
        } catch (_error) {
          // Ignore SVG bbox failures.
        }
      }
      const img = fence.querySelector("img.plantuml");
      if (img instanceof HTMLImageElement) {
        const width = Number(img.naturalWidth || 0);
        const height = Number(img.naturalHeight || 0);
        if (width > 0 && height > 0) {
          return { width, height };
        }
      }
      const rect = fence.getBoundingClientRect();
      return { width: Math.max(0, rect.width), height: Math.max(0, rect.height) };
    };

    // Font-size bounds drive the "shrink vs spill" decision. We cap at 12pt
    // on enlargement and compare against the configurable floor when deciding
    // whether a one-page shrink remains legible.
    const maxSvgFontPxForFence = (fence) => {
      if (!(fence instanceof HTMLElement)) {
        return 12;
      }
      let maxFontPx = 0;
      for (const node of Array.from(fence.querySelectorAll("svg text, svg tspan, svg foreignObject, svg foreignObject *"))) {
        if (!(node instanceof Element)) {
          continue;
        }
        const rawFont =
          node.getAttribute("font-size") ||
          (node instanceof HTMLElement || node instanceof SVGElement
            ? window.getComputedStyle(node).fontSize
            : "");
        const fontPx = parseLength(rawFont);
        if (fontPx > maxFontPx) {
          maxFontPx = fontPx;
        }
      }
      return Math.max(12, maxFontPx);
    };

    const collectHeadingClusterBeforeFence = (fence) => {
      const headings = [];
      let cursor = previousMeaningfulElement(fence);
      const isShortLeadParagraph = (node) => {
        if (!(node instanceof HTMLElement)) {
          return false;
        }
        if (String(node.tagName || "").toLowerCase() !== "p") {
          return false;
        }
        const leadText = String(node.textContent || "").trim();
        const leadWords = leadText ? leadText.split(/\s+/).filter(Boolean).length : 0;
        return leadWords > 0 && leadWords <= 48;
      };
      if (
        cursor instanceof HTMLElement &&
        cursor.classList.contains("mdexplore-print-block")
      ) {
        const children = Array.from(cursor.children).filter(
          (child) => child instanceof HTMLElement
        );
        let headingCount = 0;
        let paragraphCount = 0;
        let valid = children.length > 0;
        for (const child of children) {
          if (headingLevel(child) > 0) {
            headingCount += 1;
            continue;
          }
          if (isShortLeadParagraph(child)) {
            paragraphCount += 1;
            continue;
          }
          valid = false;
          break;
        }
        if (valid && headingCount > 0 && paragraphCount <= 1) {
          return children;
        }
      }
      let leadParagraph = null;
      if (isShortLeadParagraph(cursor)) {
        leadParagraph = cursor;
        cursor = previousMeaningfulElement(cursor);
      }
      while (cursor instanceof HTMLElement) {
        const level = headingLevel(cursor);
        if (level <= 0) {
          break;
        }
        headings.unshift(cursor);
        cursor = previousMeaningfulElement(cursor);
      }
      if (headings.length <= 0) {
        return [];
      }
      if (leadParagraph instanceof HTMLElement) {
        headings.push(leadParagraph);
      }
      return headings;
    };

    const clearFenceLayout = (fence) => {
      if (!(fence instanceof HTMLElement)) {
        return;
      }
      fence.classList.remove(
        "mdexplore-print-keep",
        "mdexplore-print-allow-break",
        "mdexplore-print-landscape-page",
        "mdexplore-print-with-heading",
        "mdexplore-print-break-before",
      );
      fence.style.removeProperty("--mdexplore-print-section-width");
      fence.style.removeProperty("--mdexplore-print-diagram-width");
      fence.style.removeProperty("--mdexplore-print-diagram-max-width");
      fence.style.removeProperty("--mdexplore-print-diagram-height");
      fence.style.removeProperty("--mdexplore-print-diagram-max-height");
      fence.style.removeProperty("--mdexplore-print-diagram-reserved-height");
      fence.style.removeProperty("width");
      fence.style.removeProperty("max-width");
      fence.style.removeProperty("min-height");
      for (const child of Array.from(fence.children)) {
        if (!(child instanceof HTMLElement)) {
          continue;
        }
        child.classList.remove(
          "mdexplore-print-heading-anchor",
          "mdexplore-print-heading-break-before",
          "mdexplore-print-heading-landscape",
        );
      }
    };

    const addSequenceBreakMarker = (fence) => {
      if (!(fence instanceof HTMLElement) || !fence.parentNode) {
        return;
      }
      const previousSibling = fence.previousSibling;
      const alreadyMarked =
        previousSibling instanceof HTMLElement &&
        previousSibling.classList.contains("mdexplore-print-sequence-page-break");
      if (alreadyMarked) {
        return;
      }
      const marker = document.createElement("div");
      marker.className = "mdexplore-print-sequence-page-break";
      fence.parentNode.insertBefore(marker, fence);
    };

    // Landscape page name assignment is separated into an explicit start marker
    // so Chromium does not name the preceding prose page as landscape.
    const ensureLandscapeStartMarker = (targetBlock) => {
      if (!(targetBlock instanceof HTMLElement) || !targetBlock.parentNode) {
        return null;
      }
      const previousSibling = targetBlock.previousSibling;
      const alreadyMarked =
        previousSibling instanceof HTMLElement &&
        previousSibling.classList.contains("mdexplore-print-landscape-start");
      if (alreadyMarked) {
        return previousSibling;
      }
      const marker = document.createElement("div");
      marker.className = "mdexplore-print-landscape-start";
      marker.dataset.mdexplorePrintLandscapeStart = "1";
      targetBlock.parentNode.insertBefore(marker, targetBlock);
      return marker;
    };

    // Landscape sections need an explicit block wrapper so Chromium applies
    // page breaks to the section boundary instead of letting adjacent prose
    // remain on the later rotated page.
    const ensurePrintBlockWrapper = (node, classNames = []) => {
      if (!(node instanceof HTMLElement) || !node.parentNode) {
        return null;
      }
      const parent = node.parentElement;
      if (parent instanceof HTMLElement && parent.classList.contains("mdexplore-print-block")) {
        parent.dataset.mdexplorePrintBlock = "1";
        for (const className of classNames) {
          if (className) {
            parent.classList.add(className);
          }
        }
        return parent;
      }
      const wrapper = document.createElement("div");
      wrapper.className = "mdexplore-print-block";
      wrapper.dataset.mdexplorePrintBlock = "1";
      for (const className of classNames) {
        if (className) {
          wrapper.classList.add(className);
        }
      }
      node.parentNode.insertBefore(wrapper, node);
      wrapper.appendChild(node);
      return wrapper;
    };

    // Short lead-in lists read poorly when Chromium leaves only the tail of
    // the list on a fresh page. Keep a small intro paragraph and the
    // immediately-following short list together when practical.
    const maybeWrapLeadInList = (listNode) => {
      if (!(listNode instanceof HTMLElement) || !listNode.parentNode) {
        return null;
      }
      if (listNode.closest(".mdexplore-fence")) {
        return null;
      }
      const parent = listNode.parentElement;
      if (parent instanceof HTMLElement && parent.classList.contains("mdexplore-print-block")) {
        return null;
      }
      const items = Array.from(listNode.children).filter((child) => child instanceof HTMLElement);
      if (items.length <= 0 || items.length > 3) {
        return null;
      }
      const previous = previousMeaningfulElement(listNode);
      if (!(previous instanceof HTMLElement) || previous.parentNode !== listNode.parentNode) {
        return null;
      }
      if (previous.closest(".mdexplore-fence")) {
        return null;
      }
      const previousParent = previous.parentElement;
      if (previousParent instanceof HTMLElement && previousParent.classList.contains("mdexplore-print-block")) {
        return null;
      }
      const previousTag = String(previous.tagName || "").toLowerCase();
      const previousText = String(previous.textContent || "").trim();
      if (previousTag !== "p" || !previousText || !/[.:]$/.test(previousText)) {
        return null;
      }
      const wrapper = document.createElement("div");
      wrapper.className = "mdexplore-print-block mdexplore-print-keep";
      wrapper.dataset.mdexplorePrintBlock = "1";
      listNode.parentNode.insertBefore(wrapper, previous);
      wrapper.appendChild(previous);
      wrapper.appendChild(listNode);
      const next = nextMeaningfulElement(wrapper);
      if (next instanceof HTMLElement && next.parentNode === wrapper.parentNode) {
        const nextLevel = headingLevel(next);
        if (nextLevel > 0 && nextLevel <= 6) {
          const nextFence = nextMeaningfulElement(next);
          if (
            nextFence instanceof HTMLElement &&
            nextFence.classList.contains("mdexplore-fence") &&
            nextFence.parentNode === wrapper.parentNode &&
            items.length <= 2
          ) {
            wrapper.appendChild(next);
            wrapper.appendChild(nextFence);
          }
        }
      }
      return wrapper;
    };

    const markShortPreKeep = (preNode) => {
      if (!(preNode instanceof HTMLElement) || preNode.closest(".mdexplore-fence")) {
        return;
      }
      const text = String(preNode.textContent || "");
      const lines = text
        .split(/\\r?\\n/)
        .map((line) => line.trim())
        .filter(Boolean);
      if (lines.length > 0 && lines.length <= 12) {
        preNode.classList.add("mdexplore-print-pre-keep");
      }
    };

    // Keep-together sections receive explicit width/height CSS variables so
    // the print snapshot path honors the same geometry this solver chose.
    const applyKeepSizing = (fence, sectionWidth, diagramWidth, diagramHeight, headingHeight) => {
      if (!(fence instanceof HTMLElement)) {
        return;
      }
      const widthText = `${Math.max(1, Math.round(diagramWidth))}px`;
      const heightText = `${Math.max(1, Math.round(diagramHeight))}px`;
      const sectionWidthText = `${Math.max(1, Math.round(sectionWidth))}px`;
      const reservedHeight = Math.max(1, Math.round(headingHeight + HEADING_TO_DIAGRAM_GAP_PX + diagramHeight));
      fence.style.setProperty("--mdexplore-print-section-width", sectionWidthText);
      fence.style.setProperty("--mdexplore-print-diagram-width", widthText);
      fence.style.setProperty("--mdexplore-print-diagram-max-width", widthText);
      fence.style.setProperty("--mdexplore-print-diagram-height", heightText);
      fence.style.setProperty("--mdexplore-print-diagram-max-height", heightText);
      fence.style.setProperty("--mdexplore-print-diagram-reserved-height", `${reservedHeight}px`);
    };

    // Spillable sections still get an explicit width budget, but height is left
    // unconstrained so Chromium can paginate through the diagram vertically.
    const applyBreakSizing = (fence, sectionWidth, diagramWidth) => {
      if (!(fence instanceof HTMLElement)) {
        return;
      }
      const widthText = `${Math.max(1, Math.round(diagramWidth))}px`;
      const sectionWidthText = `${Math.max(1, Math.round(sectionWidth))}px`;
      fence.style.setProperty("--mdexplore-print-section-width", sectionWidthText);
      fence.style.setProperty("--mdexplore-print-diagram-width", widthText);
      fence.style.setProperty("--mdexplore-print-diagram-max-width", widthText);
      fence.style.removeProperty("--mdexplore-print-diagram-height");
      fence.style.removeProperty("--mdexplore-print-diagram-max-height");
      fence.style.removeProperty("--mdexplore-print-diagram-reserved-height");
    };

    const remainingOnCurrentPage = (targetBlock, printableHeight) => {
      if (!(targetBlock instanceof HTMLElement)) {
        return printableHeight;
      }
      const contentRoot = document.querySelector("main") || document.body || document.documentElement;
      const contentTop =
        contentRoot instanceof HTMLElement
          ? contentRoot.getBoundingClientRect().top + window.scrollY
          : 0;
      const targetTop = targetBlock.getBoundingClientRect().top + window.scrollY;
      const relativeTop = Math.max(0, targetTop - contentTop);
      const pageOffset = relativeTop % Math.max(180, printableHeight);
      return {
        pageOffset,
        remaining: Math.max(0, printableHeight - pageOffset),
      };
    };

    // Preflight can rerun while waiting for async readiness; clear any prior
    // landscape tokens first so wrapper unwrapping cannot strand stale tokens
    // as standalone nodes that later paginate into blank pages.
    for (const token of Array.from(document.querySelectorAll(".mdexplore-pdf-landscape-token"))) {
      if (token instanceof HTMLElement) {
        token.remove();
      }
    }
    unwrapPrintBlocks();
    for (const marker of Array.from(document.querySelectorAll(".mdexplore-print-sequence-page-break"))) {
      if (marker instanceof HTMLElement) {
        marker.remove();
      }
    }
    for (const marker of Array.from(document.querySelectorAll(".mdexplore-print-landscape-start[data-mdexplore-print-landscape-start='1']"))) {
      if (marker instanceof HTMLElement) {
        marker.remove();
      }
    }
    for (const preNode of Array.from(document.querySelectorAll("pre.mdexplore-print-pre-keep"))) {
      if (preNode instanceof HTMLElement) {
        preNode.classList.remove("mdexplore-print-pre-keep");
      }
    }
    for (const fence of Array.from(document.querySelectorAll(".mdexplore-fence"))) {
      clearFenceLayout(fence);
    }

    for (const heading of Array.from(document.querySelectorAll("h1, h2, h3, h4, h5, h6"))) {
      if (!(heading instanceof HTMLElement)) {
        continue;
      }
      const parent = heading.parentElement;
      if (parent instanceof HTMLElement && parent.classList.contains("mdexplore-print-block")) {
        continue;
      }
      if (!heading.parentNode) {
        continue;
      }

      const level = headingLevel(heading);
      if (level <= 0 || level > 6) {
        continue;
      }
      if (heading.closest(".mdexplore-fence")) {
        continue;
      }

      const cluster = [heading];
      if (level <= 3) {
        let cursor = heading;
        while (true) {
          const next = nextMeaningfulElement(cursor);
          const nextLevel = headingLevel(next);
          if (
            !(next instanceof HTMLElement) ||
            next.parentNode !== heading.parentNode ||
            (next.parentElement instanceof HTMLElement &&
              next.parentElement.classList.contains("mdexplore-print-block")) ||
            nextLevel <= 0 ||
            nextLevel > 3
          ) {
            break;
          }
          cluster.push(next);
          cursor = next;
        }
      }

      // Prevent orphaned section headings: when a major heading is followed by
      // a subordinate heading (for example H3 then H4), keep them together so
      // the parent heading is not left alone at a page boundary.
      if (level <= 3) {
        const clusterTail = cluster[cluster.length - 1];
        const nextHeading = nextMeaningfulElement(clusterTail);
        const nextHeadingLevel = headingLevel(nextHeading);
        if (
          nextHeading instanceof HTMLElement &&
          nextHeading.parentNode === heading.parentNode &&
          !(nextHeading.parentElement instanceof HTMLElement &&
            nextHeading.parentElement.classList.contains("mdexplore-print-block")) &&
          nextHeadingLevel > level &&
          nextHeadingLevel <= 6 &&
          !nextHeading.classList.contains("mdexplore-fence")
        ) {
          cluster.push(nextHeading);
        }
      }

      const lastClusterItem = cluster[cluster.length - 1];
      const nextBlock = nextMeaningfulElement(lastClusterItem);
      if (
        nextBlock instanceof HTMLElement &&
        nextBlock.parentNode === heading.parentNode &&
        !(nextBlock.parentElement instanceof HTMLElement &&
          nextBlock.parentElement.classList.contains("mdexplore-print-block")) &&
        !nextBlock.classList.contains("mdexplore-fence") &&
        !/^H[1-6]$/.test(nextBlock.tagName)
      ) {
        cluster.push(nextBlock);
      }

      if (cluster.length <= 1) {
        continue;
      }

      const wrapper = document.createElement("div");
      wrapper.className = "mdexplore-print-block mdexplore-print-keep mdexplore-print-heading-keep";
      wrapper.dataset.mdexplorePrintBlock = "1";
      heading.parentNode.insertBefore(wrapper, heading);
      for (const element of cluster) {
        wrapper.appendChild(element);
      }
      for (const element of cluster) {
        if (/^H[1-6]$/.test(element.tagName)) {
          element.classList.add("mdexplore-print-heading-anchor");
        }
      }
    }

    for (const rule of Array.from(document.querySelectorAll("hr"))) {
      if (!(rule instanceof HTMLElement)) {
        continue;
      }
      const parent = rule.parentElement;
      if (parent instanceof HTMLElement && parent.classList.contains("mdexplore-print-block")) {
        continue;
      }
      if (!rule.parentNode) {
        continue;
      }
      const nextBlock = nextMeaningfulElement(rule);
      if (!(nextBlock instanceof HTMLElement)) {
        rule.classList.add("mdexplore-print-skip");
        continue;
      }
      const nextParent = nextBlock.parentElement;
      const nextWrapped =
        nextBlock.classList.contains("mdexplore-print-block") ||
        (nextParent instanceof HTMLElement && nextParent.classList.contains("mdexplore-print-block"));
      const nextIsBreakable =
        nextBlock.classList.contains("mdexplore-print-allow-break") ||
        nextBlock.classList.contains("mdexplore-print-landscape-page") ||
        !!nextBlock.querySelector(".mdexplore-print-allow-break, .mdexplore-print-landscape-page");
      if (nextIsBreakable) {
        rule.classList.add("mdexplore-print-skip");
        continue;
      }
      if (nextBlock.parentNode !== rule.parentNode && !nextWrapped) {
        rule.classList.add("mdexplore-print-skip");
        continue;
      }
      if (nextBlock.classList.contains("mdexplore-print-block")) {
        nextBlock.insertBefore(rule, nextBlock.firstChild);
        nextBlock.classList.add("mdexplore-print-keep", "mdexplore-print-heading-keep");
        continue;
      }
      const wrapper = document.createElement("div");
      wrapper.className = "mdexplore-print-block mdexplore-print-keep mdexplore-print-heading-keep";
      wrapper.dataset.mdexplorePrintBlock = "1";
      rule.parentNode.insertBefore(wrapper, rule);
      wrapper.appendChild(rule);
      wrapper.appendChild(nextBlock);
    }

    for (const listNode of Array.from(document.querySelectorAll("ol, ul"))) {
      maybeWrapLeadInList(listNode);
    }
    for (const preNode of Array.from(document.querySelectorAll("pre"))) {
      markShortPreKeep(preNode);
    }

    for (const fence of Array.from(document.querySelectorAll(".mdexplore-fence"))) {
      if (!(fence instanceof HTMLElement)) {
        continue;
      }
      const mermaid = fence.querySelector(".mermaid");
      const plantuml = fence.querySelector("img.plantuml, svg.mdexplore-plantuml-inline");
      const hasMermaid = mermaid instanceof HTMLElement;
      const hasPlantUml = plantuml instanceof Element;
      if (!hasMermaid && !hasPlantUml) {
        continue;
      }
      // Move any immediately-preceding heading cluster into the fence so the
      // pagination solver can treat "heading + diagram" as one print section.
      const headingSourceBlock = previousMeaningfulElement(fence);
      const headingCluster = collectHeadingClusterBeforeFence(fence);
      if (headingCluster.length > 0) {
        for (const node of Array.from(headingCluster).reverse()) {
          fence.insertBefore(node, fence.firstChild);
          if (headingLevel(node) > 0) {
            node.classList.add("mdexplore-print-heading-anchor");
          }
        }
        if (
          headingSourceBlock instanceof HTMLElement &&
          headingSourceBlock.classList.contains("mdexplore-print-block") &&
          headingSourceBlock.childElementCount <= 0 &&
          !String(headingSourceBlock.textContent || "").trim()
        ) {
          headingSourceBlock.remove();
        }
        fence.classList.add("mdexplore-print-with-heading");
      }

      const headingNodes = Array.from(fence.children).filter(
        (child) => child instanceof HTMLElement && headingLevel(child) > 0
      );
      const headingText = headingNodes.map((node) => String(node.textContent || "").trim()).filter(Boolean).join(" / ");
      if (headingText) {
        diagramHeadings.push(headingText);
      }

      const size = intrinsicDiagramSize(fence);
      if (size.width <= 0 || size.height <= 0) {
        continue;
      }

      const mermaidKind = hasMermaid ? detectMermaidKindForFence(fence) : "";
      const isSequenceMermaid = mermaidKind === "sequence";
      const isFlowchartMermaid = mermaidKind === "flowchart";
      const aspectRatio = size.width / Math.max(1, size.height);
      const maxFontPx = maxSvgFontPxForFence(fence);
      const fontCapScale = maxFontPx > 0 ? (maxPrintDiagramFontPx / maxFontPx) : 1;

      const headingHeightPortrait = headingNodes.reduce(
        (sum, node) => sum + stableHeadingHeight(node, printableWidthPortrait),
        0,
      );
      const headingHeightLandscape = headingNodes.reduce(
        (sum, node) => sum + stableHeadingHeight(node, printableWidthLandscape),
        0,
      );

      const availableHeightPortrait = Math.max(
        PRINT_LAYOUT_KNOBS.keepMinHeightPx,
        printableHeightPortrait - headingHeightPortrait - HEADING_TO_DIAGRAM_GAP_PX - PRINT_LAYOUT_SAFETY_PX,
      );
      const availableHeightLandscape = Math.max(
        PRINT_LAYOUT_KNOBS.keepMinHeightPx,
        printableHeightLandscape - headingHeightLandscape - HEADING_TO_DIAGRAM_GAP_PX - PRINT_LAYOUT_SAFETY_PX,
      );

      const portraitScale = Math.min(
        printableWidthPortrait / size.width,
        availableHeightPortrait / size.height,
        fontCapScale,
      );
      const landscapeScale = Math.min(
        printableWidthLandscape / size.width,
        availableHeightLandscape / size.height,
        fontCapScale,
      );
      const portraitWidthScale = printableWidthPortrait / Math.max(1, size.width);
      const portraitHeightScale = availableHeightPortrait / Math.max(1, size.height);

      const portraitFontPx = portraitScale * maxFontPx;
      const landscapeFontPx = landscapeScale * maxFontPx;
      const canKeepPortrait = portraitScale > 0 && (maxFontPx <= 0 || portraitFontPx >= minPrintDiagramFontPx);
      const canKeepLandscape = landscapeScale > 0 && (maxFontPx <= 0 || landscapeFontPx >= minPrintDiagramFontPx);
      const wideLandscapeCandidate =
        aspectRatio >= PRINT_LAYOUT_KNOBS.wideDiagramAspectRatio &&
        landscapeScale > (portraitScale * PRINT_LAYOUT_KNOBS.wideDiagramLandscapeGain);
      const shortWideFlowchartCandidate =
        isFlowchartMermaid &&
        landscapeScale >= (portraitScale * 0.99) &&
        size.width > (printableWidthPortrait * 0.62) &&
        aspectRatio >= 1.45;
      const sequenceLandscapeCandidate =
        isSequenceMermaid &&
        aspectRatio >= 1.35 &&
        portraitScale <= 0.45 &&
        landscapeScale >= (portraitScale * 0.93) &&
        size.width > (printableWidthPortrait * 0.9);
      const plantumlWideLandscapeCandidate =
        hasPlantUml &&
        aspectRatio >= 1.08 &&
        portraitWidthScale <= (portraitHeightScale + 0.01) &&
        landscapeScale >= (portraitScale * 1.03) &&
        size.width > (printableWidthPortrait * 0.84);

      // Landscape is only selected when it provides a meaningful improvement;
      // portrait should remain the default so later pages resume normal flow.
      let useLandscape = false;
      if (
        canKeepLandscape &&
        (
          !canKeepPortrait ||
          wideLandscapeCandidate ||
          shortWideFlowchartCandidate ||
          sequenceLandscapeCandidate ||
          plantumlWideLandscapeCandidate
        )
      ) {
        useLandscape = true;
      } else if (
        !canKeepPortrait &&
        hasPlantUml &&
        aspectRatio >= PRINT_LAYOUT_KNOBS.plantumlLandscapeAspectRatio &&
        landscapeScale > portraitScale
      ) {
        useLandscape = true;
      }

      const keepOnOnePage = useLandscape ? canKeepLandscape : (canKeepPortrait || (!useLandscape && canKeepLandscape));
      const chosenScale =
        keepOnOnePage
          ? (useLandscape ? landscapeScale : (canKeepPortrait ? portraitScale : landscapeScale))
          : Math.min(
              (useLandscape ? printableWidthLandscape : printableWidthPortrait) / size.width,
              fontCapScale,
            );
      const sectionWidth = useLandscape ? printableWidthLandscape : printableWidthPortrait;
      const chosenHeadingHeight = useLandscape ? headingHeightLandscape : headingHeightPortrait;
      const diagramWidth = Math.max(1, Math.round(size.width * chosenScale));
      const diagramHeight = Math.max(1, Math.round(size.height * chosenScale));

      if (useLandscape) {
        const landscapeBlock = ensurePrintBlockWrapper(fence, ["mdexplore-print-landscape-page"]);
        ensureLandscapeStartMarker(landscapeBlock || fence);
        if (headingText) {
          landscapeHeadings.push(headingText);
        }
      }

      if (keepOnOnePage) {
        fence.classList.add("mdexplore-print-keep");
        applyKeepSizing(fence, sectionWidth, diagramWidth, diagramHeight, chosenHeadingHeight);
        keepCount += 1;
      } else {
        // Once the font floor is reached, preserve width and let Chromium spill
        // vertically rather than shrinking the diagram into illegibility.
        fence.classList.add("mdexplore-print-allow-break");
        applyBreakSizing(fence, sectionWidth, diagramWidth);
        allowBreakCount += 1;
      }

      const pageBudget = useLandscape ? printableHeightLandscape : printableHeightPortrait;
      const pageMetrics = remainingOnCurrentPage(fence, pageBudget);
      const projectedSectionHeight = chosenHeadingHeight + HEADING_TO_DIAGRAM_GAP_PX + diagramHeight;
      // Landscape sections already force a dedicated page via the named-page
      // CSS rule. Adding the generic pre-break marker on top of that can
      // trigger an extra intermediate page before the actual landscape page,
      // which shows up as a heading-only orphan for very wide UML sections.
      const shouldBreakBefore =
        !useLandscape &&
        pageMetrics.pageOffset > 1 &&
        (keepOnOnePage
          ? projectedSectionHeight > pageMetrics.remaining + 1
          : headingNodes.length > 0);
      if (shouldBreakBefore) {
        fence.classList.add("mdexplore-print-break-before");
        if (isSequenceMermaid) {
          addSequenceBreakMarker(fence);
        }
      }

      diagramCount += 1;
      if (useLandscape) {
        landscapeCount += 1;
      }
    }

    return {
      diagramCount,
      keepCount,
      allowBreakCount,
      landscapeCount,
      landscapeHeadings: Array.from(new Set(landscapeHeadings)),
      diagramHeadings: Array.from(new Set(diagramHeadings)),
    };
  };

  const diagramLayout = markDiagramPrintLayout();

  const landscapeTokenText = __MDEXPLORE_LANDSCAPE_PAGE_TOKEN_JSON__;
  // Landscape decisions are passed back to Python via tiny hidden tokens so
  // the footer-stamping pass can rotate only those pages after printToPdf().
  for (const block of Array.from(document.querySelectorAll(".mdexplore-print-landscape-page"))) {
    if (!(block instanceof HTMLElement)) {
      continue;
    }
    let token = block.querySelector(":scope > .mdexplore-pdf-landscape-token");
    if (!(token instanceof HTMLElement)) {
      token = document.createElement("div");
      token.className = "mdexplore-pdf-landscape-token";
      token.textContent = landscapeTokenText;
      token.style.setProperty("display", "block", "important");
      token.style.setProperty("font-size", "2px", "important");
      token.style.setProperty("line-height", "2px", "important");
      token.style.setProperty("margin", "0", "important");
      token.style.setProperty("padding", "0", "important");
      token.style.setProperty("color", "#ffffff", "important");
      token.style.setProperty("pointer-events", "none", "important");
      block.insertBefore(token, block.firstChild);
    }
  }

  const hasMath = !!document.querySelector("mjx-container, .MathJax");
  const hasMermaid = !!document.querySelector(".mermaid");
  const plantumlImages = Array.from(document.querySelectorAll("img.plantuml"));
  const inlinePlantumlSvgs = Array.from(document.querySelectorAll("svg.mdexplore-plantuml-inline"));
  const hasPlantUml =
    plantumlImages.length > 0 || inlinePlantumlSvgs.length > 0 || !!document.querySelector(".plantuml-pending");
  const plantumlReady =
    !hasPlantUml ||
    (!!window.__mdexplorePdfPlantUmlReady) ||
    (
      !document.querySelector(".plantuml-pending") &&
      plantumlImages.every((img) => {
        if (!(img instanceof HTMLImageElement)) {
          return true;
        }
        return !!img.complete && Number(img.naturalWidth || 0) > 1 && Number(img.naturalHeight || 0) > 1;
      }) &&
      inlinePlantumlSvgs.every((svg) => svg instanceof SVGElement)
    );
  const mathReady = !hasMath || !!window.__mdexploreMathReady;
  const mermaidReady = !hasMermaid || !!window.__mdexplorePdfMermaidReady;
  const fontsReady = !document.fonts || document.fonts.status === "loaded";

  return JSON.stringify({
    mathReady,
    mermaidReady,
    plantumlReady,
    fontsReady,
    hasMath,
    hasMermaid,
    hasPlantUml,
    diagramLayout,
  });
})();
