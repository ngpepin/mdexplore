"""Non-executing fallback for two known exported AIgile DIL widgets.

The source's JavaScript is *never* evaluated. Only an exact, bounded data
shape is admitted and HTML is escaped. Preserve newlines for source mapping.
"""
from __future__ import annotations

import html
import re

_DIMENSION = re.compile(r'\{name:"([^"\\\n]{1,160})",detail:"([^"\\\n]{1,700})"\}')


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _replace_with_same_lines(source: str, start: int, end: int, replacement: str) -> str:
    # Newline count, not just visible layout, is needed for source scroll sync.
    return source[:start] + replacement + ('\n' * source[start:end].count('\n')) + source[end:]


def _roadmap(source: str) -> str:
    marker = '<box border radius="lg" padding={3} gap={3}>'
    start = source.find(marker)
    while start >= 0 and '<title size="lg">90-day roadmap</title>' not in source[start:start + 190]:
        start = source.find(marker, start + len(marker))
    if start < 0:
        return source
    close = source.find('\n</box>', start)
    if close < 0 or close - start > 8500:
        return source
    end = close + len('\n</box>')
    part = source[start:end]
    day_labels = re.findall(r'<text weight="medium" size="xs">(Days \d+-\d+)</text>', part)
    day_details = re.findall(r'<text color="secondary" size="3xs">([^<>]+)</text>', part)
    titles = list(re.finditer(r'<text weight="medium">(Phase [123] - [^<>]+)</text>', part))
    if len(day_labels) != 3 or len(day_details) != 3 or len(titles) != 3:
        return source
    descriptions = []
    for i, title in enumerate(titles):
        portion = part[title.end():titles[i + 1].start() if i + 1 < 3 else len(part)]
        descriptions.append(re.findall(r'<text size="sm">([^<>]+)</text>', portion))
    if any(len(group) != 2 for group in descriptions):
        return source
    cards = ''.join(
        '<div class="aigile-roadmap-day"><strong>' + _escape(day) + '</strong><span>' + _escape(detail) + '</span></div>'
        for day, detail in zip(day_labels, day_details)
    )
    phases = ''.join(
        '<div class="aigile-roadmap-phase"><h4>' + _escape(match.group(1)) + '</h4>'
        + ''.join('<p>' + _escape(text) + '</p>' for text in descriptions[index]) + '</div>'
        for index, match in enumerate(titles)
    )
    markup = '<section class="aigile-roadmap" data-aigile-roadmap="1"><header><h3>90-day roadmap</h3><span class="aigile-proposed">Proposed</span></header><div class="aigile-roadmap-days">' + cards + '</div>' + phases + '</section>'
    return _replace_with_same_lines(source, start, end, markup)


def _readiness(source: str) -> str:
    start = source.find('{@body const dimensions = ')
    if start < 0:
        return source
    end = source.find('\n---', start)
    if end < 0 or end - start > 10000:
        return source
    part = source[start:end]
    required = ('Readiness self-assessment', 'DIL.useState(Array(7).fill(0))',
                '{#each dimensions as d,i}', '{#if answered===7}',
                'Lowest-scoring priorities', 'Analyze my readiness results')
    if not all(marker in part for marker in required):
        return source
    dimensions = _DIMENSION.findall(part)
    if len(dimensions) != 7 or len({name for name, _ in dimensions}) != 7:
        return source
    areas = []
    for index, (name, detail) in enumerate(dimensions):
        buttons = ''.join(
            '<button type="button" class="aigile-rating" data-aigile-rate="' + str(score) + '" data-aigile-index="' + str(index) + '" aria-pressed="false">' + str(score) + '</button>'
            for score in range(1, 6)
        )
        areas.append('<div class="aigile-readiness-area" data-aigile-area="' + str(index) + '"><div class="aigile-readiness-label"><strong>' + _escape(name) + '</strong><span data-aigile-area-score>Not rated</span></div><p>' + _escape(detail) + '</p><div class="aigile-rating-row" role="group" aria-label="' + _escape(name + ' readiness rating') + '">' + buttons + '</div></div>')
    markup = (
        '<section class="aigile-readiness" data-aigile-readiness="1" data-aigile-complete="false">'
        '<header><div><h3>Readiness self-assessment</h3><p>Rate each area from 1 (weak) to 5 (strong).</p></div>'
        '<div class="aigile-readiness-total"><strong data-aigile-total>0/7</strong><span data-aigile-total-label>Areas rated</span></div></header>'
        + ''.join(areas)
        + '<p class="aigile-readiness-hint">The results will identify the three lowest-scoring areas and suggest how to prioritize the implementation effort. These are indicative management scores, not empirically validated maturity thresholds.</p>'
        '<div class="aigile-readiness-results"><h4>Interpretation</h4><p data-aigile-interpretation></p><h4>Lowest-scoring priorities</h4><div data-aigile-priorities></div>'
        '<button type="button" class="aigile-prepare" data-aigile-prepare>Prepare analysis request</button><div class="aigile-request-output" data-aigile-request role="textbox" aria-label="Readiness analysis request" tabindex="0"></div>'
        '<p class="aigile-readiness-help">Copy this request into your AI assistant for a tailored 90-day implementation plan.</p></div></section>'
    )
    return _replace_with_same_lines(source, start, end, markup)


def render_known_dil_widgets(source: str) -> str:
    """Render supported exported widgets without executing arbitrary DIL."""
    return _readiness(_roadmap(source))
