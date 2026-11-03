'use strict';

// Conservative support for the two exported AIgile widgets. This is an
// explicit data/template fallback, NOT a JavaScript/DIL evaluator.
const { escapeHtml } = require('./utils');

function replaceSameLines(source, start, end, html) {
  return source.slice(0, start) + html + '\n'.repeat((source.slice(start, end).match(/\n/g) || []).length) + source.slice(end);
}

function roadmap(source) {
  const marker = '<box border radius="lg" padding={3} gap={3}>';
  let start = source.indexOf(marker);
  while (start >= 0 && !source.slice(start, start + 190).includes('<title size="lg">90-day roadmap</title>')) {
    start = source.indexOf(marker, start + marker.length);
  }
  if (start < 0) return source;
  const close = source.indexOf('\n</box>', start);
  if (close < 0 || close - start > 8500) return source;
  const end = close + '\n</box>'.length;
  const part = source.slice(start, end);
  const dayLabels = [...part.matchAll(/<text weight="medium" size="xs">(Days \d+-\d+)<\/text>/g)].map(m => m[1]);
  const dayDetails = [...part.matchAll(/<text color="secondary" size="3xs">([^<>]+)<\/text>/g)].map(m => m[1]);
  const titles = [...part.matchAll(/<text weight="medium">(Phase [123] - [^<>]+)<\/text>/g)];
  if (dayLabels.length !== 3 || dayDetails.length !== 3 || titles.length !== 3) return source;
  const descriptions = titles.map((title, i) => {
    const from = title.index + title[0].length;
    const to = i + 1 < titles.length ? titles[i + 1].index : part.length;
    return [...part.slice(from, to).matchAll(/<text size="sm">([^<>]+)<\/text>/g)].map(m => m[1]);
  });
  if (descriptions.some(group => group.length !== 2)) return source;
  const cards = dayLabels.map((day, i) => `<div class="aigile-roadmap-day"><strong>${escapeHtml(day)}</strong><span>${escapeHtml(dayDetails[i])}</span></div>`).join('');
  const phases = titles.map((title, i) => `<div class="aigile-roadmap-phase"><h4>${escapeHtml(title[1])}</h4>${descriptions[i].map(v => `<p>${escapeHtml(v)}</p>`).join('')}</div>`).join('');
  return replaceSameLines(source, start, end, `<section class="aigile-roadmap" data-aigile-roadmap="1"><header><h3>90-day roadmap</h3><span class="aigile-proposed">Proposed</span></header><div class="aigile-roadmap-days">${cards}</div>${phases}</section>`);
}

function readiness(source) {
  const start = source.indexOf('{@body const dimensions = ');
  if (start < 0) return source;
  const end = source.indexOf('\n---', start);
  if (end < 0 || end - start > 10000) return source;
  const part = source.slice(start, end);
  const required = ['Readiness self-assessment', 'DIL.useState(Array(7).fill(0))', '{#each dimensions as d,i}', '{#if answered===7}', 'Lowest-scoring priorities', 'Analyze my readiness results'];
  if (!required.every(v => part.includes(v))) return source;
  const dimensions = [...part.matchAll(/\{name:"([^"\\\n]{1,160})",detail:"([^"\\\n]{1,700})"\}/g)].map(m => [m[1], m[2]]);
  if (dimensions.length !== 7 || new Set(dimensions.map(d => d[0])).size !== 7) return source;
  const areas = dimensions.map(([name, detail], i) => {
    const buttons = [1, 2, 3, 4, 5].map(n => `<button type="button" class="aigile-rating" data-aigile-rate="${n}" data-aigile-index="${i}" aria-pressed="false">${n}</button>`).join('');
    return `<div class="aigile-readiness-area" data-aigile-area="${i}"><div class="aigile-readiness-label"><strong>${escapeHtml(name)}</strong><span data-aigile-area-score>Not rated</span></div><p>${escapeHtml(detail)}</p><div class="aigile-rating-row" role="group" aria-label="${escapeHtml(name + ' readiness rating')}">${buttons}</div></div>`;
  }).join('');
  const html = '<section class="aigile-readiness" data-aigile-readiness="1" data-aigile-complete="false">' +
    '<header><div><h3>Readiness self-assessment</h3><p>Rate each area from 1 (weak) to 5 (strong).</p></div>' +
    '<div class="aigile-readiness-total"><strong data-aigile-total>0/7</strong><span data-aigile-total-label>Areas rated</span></div></header>' + areas +
    '<p class="aigile-readiness-hint">The results will identify the three lowest-scoring areas and suggest how to prioritize the implementation effort. These are indicative management scores, not empirically validated maturity thresholds.</p>' +
    '<div class="aigile-readiness-results"><h4>Interpretation</h4><p data-aigile-interpretation></p><h4>Lowest-scoring priorities</h4><div data-aigile-priorities></div>' +
    '<button type="button" class="aigile-prepare" data-aigile-prepare>Prepare analysis request</button><div class="aigile-request-output" data-aigile-request role="textbox" aria-label="Readiness analysis request" tabindex="0"></div>' +
    '<p class="aigile-readiness-help">Copy this request into your AI assistant for a tailored 90-day implementation plan.</p></div></section>';
  return replaceSameLines(source, start, end, html);
}

function renderKnownDilWidgets(markdown) {
  return readiness(roadmap(String(markdown || '')));
}

module.exports = { renderKnownDilWidgets };
