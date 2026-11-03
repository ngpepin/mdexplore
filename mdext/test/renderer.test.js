'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const path = require('node:path');
const fs = require('node:fs');
const { renderMarkdown, transformCallouts, diagramShell } = require('../src/renderer');
const { DiagramRenderers } = require('../src/diagramRenderers');
const { resolveRelativeFile, isMarkdownPath } = require('../src/utils');

const fakeRenderers = {
  async renderMermaid(source) {
    return {
      renderer: 'rust',
      svg: `<svg viewBox="0 0 20 10"><text>${source.length}</text></svg>`,
      error: '',
    };
  },
  async renderPlantUml() {
    return {
      svg: '<svg viewBox="0 0 20 10"><text>PlantUML</text></svg>',
      error: '',
    };
  },
  renderEmbeddedSvg(source) {
    return {
      dataUri: `data:image/svg+xml;base64,${Buffer.from(source, 'utf8').toString('base64')}`,
      error: '',
    };
  },
};

test('renders Markdown, callouts, source metadata, and diagram controls', async () => {
  const markdown = [
    '# Title',
    '',
    '> [!NOTE] Read this',
    '> Body text.',
    '',
    '```mermaid',
    'flowchart LR',
    '  A --> B',
    '```',
    '',
    '```plantuml',
    'Alice -> Bob: hello',
    '```',
  ].join('\n');
  const result = await renderMarkdown({
    markdown,
    documentPath: '/tmp/docs/sample.md',
    resolveAsset: (_documentPath, value) => value,
    diagramRenderers: fakeRenderers,
  });

  assert.match(result.body, /<h1[^>]*data-md-line-start="0"/);
  assert.match(result.body, /mdext-callout-note/);
  assert.match(result.body, /renderer-toggle[^>]*>R<\/button>/);
  assert.match(result.body, />Fit<\/button>/);
  assert.match(result.body, /plantuml-shell/);
  assert.equal(result.diagramCount, 2);
});

test('renders fenced and raw embedded SVG as images while leaving SVG in code fences literal', async () => {
  const markdown = [
    '```svg',
    '<svg xmlns="http://www.w3.org/2000/svg"><text>Fenced</text></svg>',
    '```',
    '',
    '<svg xmlns="http://www.w3.org/2000/svg">',
    '<text>Raw</text>',
    '</svg>',
    '',
    '```text',
    '<svg xmlns="http://www.w3.org/2000/svg"><text>Literal</text></svg>',
    '```',
  ].join('\n');
  const result = await renderMarkdown({
    markdown,
    documentPath: '/tmp/docs/sample.md',
    resolveAsset: (_documentPath, value) => value,
    diagramRenderers: fakeRenderers,
  });

  assert.equal(result.embeddedSvgCount, 2);
  assert.equal((result.body.match(/class="mdext-embedded-svg"/g) || []).length, 2);
  assert.match(result.body, /src="data:image\/svg\+xml;base64,/);
  assert.match(result.body, /language-text/);
  assert.doesNotMatch(result.body, /mdext-svg-placeholder/);
});

test('embedded SVG sanitizer and cache reuse unchanged sources and invalidate changed sources', () => {
  const context = {
    asAbsolutePath(relativePath) {
      return path.join('/tmp/mdext-svg-test', relativePath);
    },
  };
  const configuration = { get(_key, fallback) { return fallback; } };
  const renderers = new DiagramRenderers(context, () => configuration);
  const original = '<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"><script>alert(2)</script><a href="javascript:alert(3)"><text>Safe</text></a></svg>';
  const changed = original.replace('Safe', 'Changed');

  const first = renderers.renderEmbeddedSvg(original);
  const second = renderers.renderEmbeddedSvg(original);
  const third = renderers.renderEmbeddedSvg(changed);
  const decoded = Buffer.from(first.dataUri.split(',', 2)[1], 'base64').toString('utf8').toLowerCase();

  assert.equal(first.error, '');
  assert.equal(first.dataUri, second.dataUri);
  assert.notEqual(first.dataUri, third.dataUri);
  assert.equal(renderers.embeddedSvgCache.size, 2);
  assert.doesNotMatch(decoded, /<script/i);
  assert.doesNotMatch(decoded, /onload=/i);
  assert.doesNotMatch(decoded, /javascript:/i);
  assert.match(decoded, />safe</);
});

test('renders both legacy SVG diagrams from the reported Markdown issue', async () => {
  const source = fs.readFileSync(path.join(__dirname, '../../issues/SVG-rendering/2026-11-02 Create AIgile Summary.md'), 'utf8');
  const context = { asAbsolutePath(relativePath) { return path.join(__dirname, relativePath); } };
  const renderers = new DiagramRenderers(context, () => ({ get(_key, fallback) { return fallback; } }));
  const result = await renderMarkdown({
    markdown: source,
    documentPath: 'reported-issue.md',
    resolveAsset: (_documentPath, value) => value,
    diagramRenderers: renderers,
  });
  const encoded = [...result.body.matchAll(/<img class="mdext-embedded-svg" src="data:image\/svg\+xml;base64,([^"\s]+)"/g)];
  assert.equal(result.embeddedSvgCount, 2);
  assert.equal(encoded.length, 2);
  assert.doesNotMatch(result.body, /mdext-svg-error/);
  const images = encoded.map((match) => Buffer.from(match[1], 'base64').toString('utf8'));
  for (const svg of images) {
    assert.match(svg, /xmlns="http:\/\/www\.w3\.org\/2000\/svg"/);
    assert.equal((svg.match(/<rect\b/g) || []).length, 4);
    assert.doesNotMatch(svg, /\{#(?:each|if)|\$\{/);
  }
  // Embedded image documents cannot inherit currentColor from the webview.
  // Verify their own light/dark foreground rules survive SVG serialization.
  for (const svg of images) {
    assert.match(svg, /<style>:root\{color:#111827\}/);
    assert.match(svg, /@media \(prefers-color-scheme:dark\)\{:root\{color:#f8fafc\}\}/);
    assert.match(svg, /fill="currentColor"/);
  }
  assert.match(images[0], /Requirements/);
  assert.match(images[0], /Code/);
  assert.match(images[1], /Intent/);
  assert.match(images[1], /Feedback revises earlier assumptions/);
});

test('legacy ChatGPT layout in reported document renders as prose and links, not code', async () => {
  const source = fs.readFileSync(path.join(__dirname, '../../issues/SVG-rendering/2026-11-02 Create AIgile Summary.md'), 'utf8');
  const context = { asAbsolutePath(relativePath) { return path.join(__dirname, relativePath); } };
  const renderers = new DiagramRenderers(context, () => ({ get(_key, fallback) { return fallback; } }));
  const result = await renderMarkdown({
    markdown: source,
    documentPath: 'reported-issue.md',
    resolveAsset: (_documentPath, value) => value,
    diagramRenderers: renderers,
  });
  const tree = result.body.split('10.1 Recommended repository organization')[1].split('10.2 Comparing the three implementation paths')[0];
  const section = result.body.split('10.2 Comparing the three implementation paths')[1].split('10.3 Representative Claude Code workflow')[0];
  // The exported CodeBlock must retain a preformatted repository tree, not
  // flatten its indentation or its nested branches into ordinary prose.
  assert.match(tree, /<pre\b/);
  assert.match(tree, /\.aigile\/\n├── constitution\.md\n├── standards\/\n│   ├── api\.md/);
  assert.doesNotMatch(tree, /<CodeBlock/);
  assert.doesNotMatch(section, /\*\*\*\*/);
  assert.match(section, /<p[^>]*>The adapter installs an extension/);
  assert.match(section, /<p[^>]*>The main advantage is process enforcement/);
  assert.match(section, /<p[^>]*>Best suited to teams wanting/);
  assert.match(section, /data-mdext-href="https:\/\/www\.goaigile\.net\/docs\/tutorials\/speckit\/"/);
  // An inline <code> is legitimate for `aigile-loop`; only code BLOCKS
  // indicate the exported DIL layout was mistakenly parsed as a fence.
  assert.doesNotMatch(section, /<pre\b|&lt;text|<Link\b|```/);
  assert.equal(result.embeddedSvgCount, 2);
  assert.doesNotMatch(result.body, /mdext-svg-error/);
});

test('legacy DIL layout cleanup does not consume actual fenced code', async () => {
  const result = await renderMarkdown({
    markdown: '<box gap={2}>\n\n```javascript\nconst answer = 42;\n```\n\n  Normal prose.\n</box>\n',
    documentPath: '/tmp/docs/sample.md',
    resolveAsset: (_documentPath, value) => value,
    diagramRenderers: fakeRenderers,
  });
  assert.match(result.body, /language-javascript/);
  assert.match(result.body, /hljs-keyword[^>]*>const<\/span> answer/);
  assert.match(result.body, /<p[^>]*>Normal prose\.<\/p>/);
});

test('unsupported DIL expressions fail closed instead of executing code', () => {
  const context = { asAbsolutePath(relativePath) { return path.join(__dirname, relativePath); } };
  const renderers = new DiagramRenderers(context, () => ({ get(_key, fallback) { return fallback; } }));
  const source = [
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 55">',
    '  {#each ["Plan", "Build"] as label,i}',
    '    <text x={eval("not safe")}>{label}</text>',
    '  {/each}',
    '</svg>',
  ].join('\n');
  const result = renderers.renderEmbeddedSvg(source);
  assert.equal(result.dataUri, '');
  assert.match(result.error, /Unsupported ChatGPT DIL/);
});

test('currency dollar signs are not interpreted as inline math', async () => {
  const result = await renderMarkdown({
    markdown: '$57,217 annual base; $2,193.23 biweekly base',
    documentPath: '/tmp/docs/sample.md',
    resolveAsset: (_documentPath, value) => value,
    diagramRenderers: fakeRenderers,
  });

  assert.match(result.body, /\$57,217 annual base; \$2,193\.23 biweekly base/);
  assert.doesNotMatch(result.body, /mdext-math-inline/);
});

test('legitimate single-dollar inline math is emitted with explicit MathJax delimiters', async () => {
  const result = await renderMarkdown({
    markdown: 'The relation $x^2 + y^2$ is useful.',
    documentPath: '/tmp/docs/sample.md',
    resolveAsset: (_documentPath, value) => value,
    diagramRenderers: fakeRenderers,
  });

  assert.match(result.body, /class="mdext-math-inline"/);
  assert.match(result.body, /\\\(x\^2 \+ y\^2\\\)/);
});

test('sanitizes executable source HTML while retaining supported structure', async () => {
  const result = await renderMarkdown({
    markdown: '<script>globalThis.pwned = true</script>\n\n<img src="javascript:alert(1)" onerror="alert(1)">',
    documentPath: '/tmp/docs/sample.md',
    resolveAsset: (_documentPath, value) => value,
    diagramRenderers: fakeRenderers,
  });
  assert.doesNotMatch(result.body, /<script/i);
  assert.doesNotMatch(result.body, /onerror/i);
  assert.doesNotMatch(result.body, /javascript:/i);
});

test('transforms GitHub and Obsidian style callouts', () => {
  const input = '<blockquote><p>[!WARNING] Careful</p><p>Details</p></blockquote>';
  const output = transformCallouts(input);
  assert.match(output, /mdext-callout-warning/);
  assert.match(output, /Careful/);
  assert.match(output, /Details/);
});

test('diagram shell puts the renderer toggle first and carries cached Rust SVG', () => {
  const output = diagramShell({
    id: 'mermaid-1',
    kind: 'mermaid',
    source: 'flowchart LR\nA-->B',
    renderedSvg: '<svg></svg>',
    renderer: 'rust',
    error: '',
  });
  const toggle = output.indexOf('renderer-toggle');
  const fit = output.indexOf('fit-button');
  assert.ok(toggle >= 0 && fit > toggle);
  assert.match(output, /data-renderer="rust"/);
  assert.match(output, />R<\/button>/);
  assert.match(output, /data-rust-svg-b64="PHN2Zz48L3N2Zz4="/);
});

test('resolves relative file paths and recognizes Markdown extensions', () => {
  assert.equal(resolveRelativeFile('/tmp/docs/a.md', '../img/p.png'), '/tmp/img/p.png');
  assert.equal(isMarkdownPath('README.md'), true);
  assert.equal(isMarkdownPath('notes.markdown#section'), true);
  assert.equal(isMarkdownPath('image.png'), false);
});
