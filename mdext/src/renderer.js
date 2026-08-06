'use strict';

const MarkdownIt = require('markdown-it');
const taskLists = require('markdown-it-task-lists');
const hljs = require('highlight.js');
const sanitizeHtml = require('sanitize-html');
const { escapeHtml, sha1, textToBase64 } = require('./utils');

const CALLOUTS = {
  NOTE: ['note', 'ℹ', 'Note'],
  TIP: ['tip', '✓', 'Tip'],
  IMPORTANT: ['important', '!', 'Important'],
  WARNING: ['warning', '⚠', 'Warning'],
  CAUTION: ['caution', '⛔', 'Caution'],
};

function installDollarMath(md) {
  md.inline.ruler.before('escape', 'mdext_dollar_math', (state, silent) => {
    const start = state.pos;
    if (state.src[start] !== '$' || state.src[start + 1] === '$' || state.src[start - 1] === '\\') {
      return false;
    }

    const first = state.src[start + 1];
    if (!first || /\s|\d/.test(first)) {
      return false;
    }

    let end = start + 1;
    while ((end = state.src.indexOf('$', end)) >= 0) {
      if (state.src[end - 1] === '\\') {
        end += 1;
        continue;
      }
      if (state.src[end + 1] === '$' || /\s/.test(state.src[end - 1] || '')) {
        end += 1;
        continue;
      }
      break;
    }
    if (end < 0) {
      return false;
    }

    if (!silent) {
      const token = state.push('mdext_math_inline', 'math', 0);
      token.content = state.src.slice(start + 1, end);
    }
    state.pos = end + 1;
    return true;
  });

  md.renderer.rules.mdext_math_inline = (tokens, index) => {
    return `<span class="mdext-math-inline">\\(${escapeHtml(tokens[index].content)}\\)</span>`;
  };
}

function createMarkdownRenderer() {
  const md = new MarkdownIt({
    html: true,
    linkify: true,
    typographer: true,
    highlight(code, language) {
      const requested = String(language || '').trim();
      try {
        if (requested && hljs.getLanguage(requested)) {
          return hljs.highlight(code, { language: requested, ignoreIllegals: true }).value;
        }
        return hljs.highlightAuto(code).value;
      } catch {
        return escapeHtml(code);
      }
    },
  }).enable(['table', 'strikethrough']);
  md.use(taskLists, { enabled: true, label: true, labelAfter: true });
  installDollarMath(md);
  return md;
}

function addSourceLineMetadata(md) {
  const original = md.renderer.renderToken.bind(md.renderer);
  md.renderer.renderToken = (tokens, index, options) => {
    const token = tokens[index];
    if (token.nesting === 1 && Array.isArray(token.map) && token.map.length === 2) {
      token.attrSet('data-md-line-start', String(token.map[0]));
      token.attrSet('data-md-line-end', String(token.map[1]));
    }
    return original(tokens, index, options);
  };
}

function sanitizeRenderedMarkdown(body) {
  return sanitizeHtml(body, {
    allowedTags: sanitizeHtml.defaults.allowedTags.concat([
      'img', 'details', 'summary', 'mark', 'kbd', 's', 'del', 'ins', 'sub', 'sup',
      'aside', 'section', 'input',
    ]),
    allowedAttributes: {
      '*': ['class', 'id', 'title', 'aria-label', 'role', 'data-*'],
      a: ['href', 'title', 'data-mdext-href', 'data-mdext-external'],
      img: ['src', 'alt', 'title', 'width', 'height', 'loading'],
      code: ['class'],
      ol: ['start'],
      input: ['type', 'checked', 'disabled'],
    },
    allowedSchemes: ['http', 'https', 'mailto', 'tel', 'data'],
    allowedSchemesByTag: {
      img: ['http', 'https', 'data'],
    },
    allowProtocolRelative: false,
    disallowedTagsMode: 'discard',
  });
}

function transformCallouts(body) {
  return body.replace(
    /<blockquote([^>]*)>\s*<p([^>]*)>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*([^<]*)<\/p>([\s\S]*?)<\/blockquote>/gi,
    (_match, blockAttrs, _pAttrs, type, customTitle, rest) => {
      const [cssKind, icon, defaultTitle] = CALLOUTS[String(type).toUpperCase()] || CALLOUTS.NOTE;
      const title = String(customTitle || '').trim() || defaultTitle;
      return `<aside class="mdext-callout mdext-callout-${cssKind}"${blockAttrs}>` +
        `<div class="mdext-callout-title"><span class="mdext-callout-icon">${icon}</span>${escapeHtml(title)}</div>` +
        `<div class="mdext-callout-content">${rest}</div></aside>`;
    },
  );
}

function diagramShell({ id, kind, source, renderedSvg, renderer, error }) {
  const rustSvgB64 = renderedSvg ? textToBase64(renderedSvg) : '';
  const sourceB64 = textToBase64(source);
  const rendererLabel = kind === 'mermaid' ? (renderer === 'rust' ? 'R' : 'J') : '';
  const toggle = kind === 'mermaid'
    ? `<button class="diagram-button renderer-toggle" title="Toggle Rust/JavaScript renderer" ${rustSvgB64 ? '' : 'data-rust-unavailable="1"'}>${rendererLabel}</button>`
    : '';
  const content = renderedSvg
    ? renderedSvg
    : kind === 'mermaid'
      ? '<div class="diagram-loading">Rendering Mermaid…</div>'
      : `<div class="diagram-error">${escapeHtml(error || 'Diagram rendering failed.')}</div>`;
  const errorNote = error
    ? `<div class="diagram-notice" title="${escapeHtml(error)}">${escapeHtml(error)}</div>`
    : '';

  return `<section class="diagram-shell ${kind}-shell" data-diagram-id="${id}" data-kind="${kind}" ` +
    `data-renderer="${renderer}" data-source-b64="${sourceB64}" data-rust-svg-b64="${rustSvgB64}">` +
    `<div class="diagram-toolbar">${toggle}` +
    '<button class="diagram-button fit-button">Fit</button>' +
    '<button class="diagram-button zoom-out-button" aria-label="Zoom out">−</button>' +
    '<button class="diagram-button zoom-in-button" aria-label="Zoom in">+</button>' +
    '<button class="diagram-button pan-left-button" aria-label="Pan left">←</button>' +
    '<button class="diagram-button pan-right-button" aria-label="Pan right">→</button>' +
    '<button class="diagram-button pan-up-button" aria-label="Pan up">↑</button>' +
    '<button class="diagram-button pan-down-button" aria-label="Pan down">↓</button>' +
    '<span class="diagram-zoom-label">100%</span></div>' +
    `${errorNote}<div class="diagram-viewport" tabindex="0"><div class="diagram-canvas">${content}</div></div></section>`;
}

async function renderMarkdown({ markdown, documentPath, resolveAsset, diagramRenderers }) {
  const md = createMarkdownRenderer();
  addSourceLineMetadata(md);
  const diagrams = [];

  const defaultFence = md.renderer.rules.fence.bind(md.renderer.rules);
  md.renderer.rules.fence = (tokens, index, options, env, self) => {
    const token = tokens[index];
    const language = String(token.info || '').trim().split(/\s+/, 1)[0].toLowerCase();
    if (language === 'mermaid' || ['plantuml', 'puml', 'uml'].includes(language)) {
      const kind = language === 'mermaid' ? 'mermaid' : 'plantuml';
      const source = String(token.content || '');
      const id = `${kind}-${diagrams.length}-${sha1(source).slice(0, 12)}`;
      diagrams.push({ id, kind, source });
      return `<div class="mdext-diagram-placeholder" data-diagram-placeholder="${id}"></div>`;
    }
    return defaultFence(tokens, index, options, env, self);
  };

  const defaultImage = md.renderer.rules.image;
  md.renderer.rules.image = (tokens, index, options, env, self) => {
    const token = tokens[index];
    const srcIndex = token.attrIndex('src');
    if (srcIndex >= 0) {
      const original = token.attrs[srcIndex][1];
      const resolved = resolveAsset(documentPath, original);
      if (resolved) {
        token.attrs[srcIndex][1] = resolved;
      }
      token.attrSet('loading', 'lazy');
    }
    return defaultImage(tokens, index, options, env, self);
  };

  const defaultLinkOpen = md.renderer.rules.link_open || ((tokens, index, options, _env, self) => self.renderToken(tokens, index, options));
  md.renderer.rules.link_open = (tokens, index, options, env, self) => {
    const token = tokens[index];
    const hrefIndex = token.attrIndex('href');
    if (hrefIndex >= 0) {
      const href = token.attrs[hrefIndex][1];
      if (!String(href).startsWith('#')) {
        token.attrSet('data-mdext-href', href);
        token.attrSet('href', '#');
      }
    }
    return defaultLinkOpen(tokens, index, options, env, self);
  };

  let body = md.render(String(markdown || ''));
  body = transformCallouts(body);
  body = sanitizeRenderedMarkdown(body);

  const rendered = await Promise.all(diagrams.map(async (diagram) => {
    if (diagram.kind === 'mermaid') {
      const result = await diagramRenderers.renderMermaid(diagram.source);
      return {
        ...diagram,
        renderedSvg: result.svg,
        renderer: result.renderer,
        error: result.error,
      };
    }
    const result = await diagramRenderers.renderPlantUml(diagram.source);
    return {
      ...diagram,
      renderedSvg: result.svg,
      renderer: 'plantuml',
      error: result.error,
    };
  }));

  for (const diagram of rendered) {
    const placeholder = `<div class="mdext-diagram-placeholder" data-diagram-placeholder="${diagram.id}"></div>`;
    body = body.replace(placeholder, diagramShell(diagram));
  }

  return { body, diagramCount: diagrams.length };
}

module.exports = {
  renderMarkdown,
  transformCallouts,
  diagramShell,
};
