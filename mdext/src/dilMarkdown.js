'use strict';

// A non-executable Markdown fallback for legacy ChatGPT DIL layout markup.
// It preserves source line counts and leaves real fenced code untouched.
const { escapeHtml } = require('./utils');

function normalizeDilMarkdown(markdown) {
  const source = String(markdown || '');
  if (!/<(?:box|row|grid|grid-item|CodeBlock|AsyncImage)\b/.test(source)) return source;

  const converted = source.replace(/<CodeBlock\b([^>]*)>([\s\S]*?)<\/CodeBlock>/g, (_match, attrs, content) => {
    const language = /\blanguage="([A-Za-z0-9_-]{1,32})"/.exec(attrs);
    const lang = language ? language[1] : 'text';
    return `<pre><code class="language-${lang}">${escapeHtml(content)}</code></pre>`;
  });
  const lines = converted.match(/[^\n]*\n|[^\n]+$/g) || [];
  const result = [];
  let depth = 0;
  let fenceChar = '';
  let fenceLength = 0;
  let inCodeBlock = false;

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    const content = raw.replace(/[\r\n]+$/, '');
    const ending = raw.slice(content.length);
    if (content.includes('<pre><code class="language-')) inCodeBlock = true;
    if (inCodeBlock) {
      result.push(raw);
      if (content.includes('</code></pre>')) inCodeBlock = false;
      continue;
    }

    const fence = /^[ \t]{0,3}(`{3,}|~{3,})(.*)$/.exec(content);
    if (fenceChar) {
      result.push(raw);
      if (fence && fence[1][0] === fenceChar && fence[1].length >= fenceLength && !fence[2].trim()) {
        fenceChar = '';
        fenceLength = 0;
      }
      continue;
    }
    if (fence) {
      if (depth && fence[1] === '```' && !fence[2].trim()) {
        result.push(ending);
      } else {
        fenceChar = fence[1][0];
        fenceLength = fence[1].length;
        result.push(raw);
      }
      continue;
    }

    const loop = /^\s*\{#each\s+(\[[^\n]*\])\s+as\s+(\w+)\s*,\s*i\s*\}\s*$/.exec(content);
    if (depth && loop) {
      let end = i + 1;
      while (end < lines.length && lines[end].trim() !== '{/each}') end++;
      if (end < lines.length && end - i < 200) {
        let labels;
        try { labels = JSON.parse(loop[1]); } catch { labels = null; }
        if (Array.isArray(labels) && labels.length <= 32 && labels.every(v => typeof v === 'string' && v.length < 256)) {
          result.push(labels.map((label, n) => `${n + 1}. ${escapeHtml(label)}`).join(' · ') + ending);
          for (let j = i + 1; j <= end; j++) result.push(lines[j].replace(/[^\r\n]/g, ''));
          i = end;
          continue;
        }
      }
    }

    let normalized = content.replace(/<Link\b([^>]*?)\/>/g, (_match, attrs) => {
      const title = /\btitle="([^"]{0,500})"/.exec(attrs)?.[1] || '';
      const rawUrl = /\burl="([^"]{0,2048})"/.exec(attrs)?.[1] || '';
      // Legacy exports can contain a Markdown link inside the url attribute.
      const urls = rawUrl.match(/https?:\/\/[^\s<>"()\]]+/g) || [];
      const url = urls[urls.length - 1] || '';
      return url ? `[${title.replace(/[\[\]]/g, '')}](${url})` : title;
    });
    normalized = normalized.replace(/<(?:AsyncImage|icon)\b[^>]*?\/>/g, '');
    normalized = normalized.replace(/<divider\b[^>]*?\/>/g, '---');
    normalized = normalized.replace(/\\*<Cite\b[^>]*?\/>/g, '');
    normalized = normalized.replace(/<\/?(?:box|row|grid|grid-item|text|caption)\b[^>]*>/gi, '');
    const insideLayout = depth > 0 || /<(?:box|row|grid|grid-item)\b/.test(content);
    depth = Math.max(0, depth + (content.match(/<(?:box|row|grid|grid-item)\b/g) || []).length - (content.match(/<\/(?:box|row|grid|grid-item)\s*>/g) || []).length);
    // DIL layout indentation must not become an indented Markdown code block.
    if (insideLayout) normalized = normalized.trimStart();
    result.push(normalized + ending);
  }
  return result.join('');
}

module.exports = { normalizeDilMarkdown };
