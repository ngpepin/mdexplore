'use strict';

// A deliberately limited, non-executable compiler for SVG snippets from
// older ChatGPT DIL exports. No eval, Function, DOM injection or JS execution.
function compileDilSvg(source) {
  if (!source.includes('{#each') && !source.includes('{#if')) return source;
  if (source.length > 100000) throw new Error('DIL SVG template is too large.');
  const lines = source.match(/[^\n]*\n|[^\n]+$/g) || [];
  const escape = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  function number(expr, i) {
    const match = /^i(?:\s*\*\s*(\d+))?(?:\s*([+-])\s*(\d+))?$/.exec(expr.trim());
    if (!match) throw new Error('Unsupported DIL SVG arithmetic expression.');
    return i * Number(match[1] || 1) + (match[2] === '-' ? -1 : 1) * Number(match[3] || 0);
  }
  function predicate(expr, i) {
    const clauses = expr.trim().split('||');
    if (clauses.length > 4) throw new Error('Unsupported DIL SVG condition.');
    return clauses.some(clause => {
      const match = /^\s*i\s*(===|==|<|<=|>|>=)\s*(\d+)\s*$/.exec(clause);
      if (!match) throw new Error('Unsupported DIL SVG condition.');
      const n = Number(match[2]);
      switch (match[1]) {
        case '===': case '==': return i === n;
        case '<': return i < n;
        case '<=': return i <= n;
        case '>': return i > n;
        case '>=': return i >= n;
        default: return false;
      }
    });
  }
  function expression(expr, i, label) {
    expr = expr.trim();
    if (expr === 'label') return escape(label);
    if (expr.startsWith('`') && expr.endsWith('`')) {
      const inner = expr.slice(1, -1);
      if (inner.includes('`')) throw new Error('Unsupported SVG template literal.');
      return inner.replace(/\$\{([^{}]+)\}/g, (_m, value) => String(number(value, i)));
    }
    const ternary = /^(.+?)\s*\?\s*("[^"\n]*"|'[^'\n]*')\s*:\s*("[^"\n]*"|'[^'\n]*')$/.exec(expr);
    if (ternary) {
      const selected = predicate(ternary[1], i) ? ternary[2] : ternary[3];
      return escape(selected.slice(1, -1));
    }
    return String(number(expr, i));
  }
  function interpolate(line, i, label) {
    // JSX unquoted expressions must become quoted XML attributes; path data
    // contains spaces and otherwise fails to parse even after interpolation.
    return line.replace(/(\s+[A-Za-z][\w:.-]*)=\{(`[^`\n]*`|[^{}\n]*)\}/g,
      (_m, name, value) => `${name}="${expression(value, i, label)}"`)
      .replace(/\{([^{}\n]+)\}/g, (_m, value) => expression(value, i, label));
  }
  function block(start, stop, index, label) {
    const output = [];
    let size = 0;
    for (let pos = start; pos < stop;) {
      const marker = lines[pos].trim();
      const each = /^\{#each\s+(\[[^\n]*\])\s+as\s+label\s*,\s*i\s*\}$/.exec(marker);
      const condition = /^\{#if\s+([^{}\n]+)\}$/.exec(marker);
      if (each || condition) {
        const closing = each ? '{/each}' : '{/if}';
        const opening = each ? '{#each' : '{#if';
        let end = pos + 1;
        let depth = 1;
        while (end < stop && depth) {
          const current = lines[end].trim();
          if (current.startsWith(opening)) depth++;
          else if (current === closing) depth--;
          end++;
        }
        if (depth) throw new Error('Unclosed DIL SVG directive.');
        if (each) {
          let labels;
          try { labels = JSON.parse(each[1]); } catch { throw new Error('Invalid literal SVG labels.'); }
          if (!Array.isArray(labels) || labels.length > 32 || labels.some(x => typeof x !== 'string' || x.length >= 256)) {
            throw new Error('Unsupported SVG label list.');
          }
          labels.forEach((newLabel, newIndex) => output.push(...block(pos + 1, end - 1, newIndex, newLabel)));
        } else if (predicate(condition[1], index)) {
          output.push(...block(pos + 1, end - 1, index, label));
        }
        pos = end;
      } else {
        if (/^\{[#/:]/.test(marker)) throw new Error('Unsupported DIL SVG directive.');
        const line = interpolate(lines[pos++], index, label);
        size += line.length;
        if (size > 200000) throw new Error('Expanded DIL SVG is too large.');
        output.push(line);
      }
    }
    return output;
  }
  let rendered = block(0, lines.length, 0, '').join('');
  const namespace = ['http:', '', 'www.w3.org', '2000', 'svg'].join('/');
  rendered = rendered.replace(/\bxmlns\s*=\s*"[^"\n]*"/g,
    match => match.includes('w3.org/2000/svg') ? `xmlns="${namespace}"` : match);
  const names = { fillOpacity: 'fill-opacity', strokeOpacity: 'stroke-opacity', strokeWidth: 'stroke-width', strokeDasharray: 'stroke-dasharray', textAnchor: 'text-anchor', fontSize: 'font-size' };
  rendered = rendered.replace(/\b(fillOpacity|strokeOpacity|strokeWidth|strokeDasharray|textAnchor|fontSize)(?=\s*=)/g, name => names[name]);
  if (/\{#(?:each|if)|\{\/(?:each|if)\}|\$\{/.test(rendered)) throw new Error('Unexpanded SVG template expression.');
  return rendered;
}

module.exports = { compileDilSvg };
