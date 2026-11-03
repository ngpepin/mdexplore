"""Bounded, non-executable expansion of legacy ChatGPT DIL SVG templates.

This is deliberately not a JavaScript/JSX interpreter. Only literal string lists,
integer index arithmetic, simple index predicates and literal color choices are
accepted; other expressions fail closed.
"""

from __future__ import annotations

import html
import json
import re


class DilSvgError(ValueError):
    pass


def _index_number(expression: str, index: int) -> int:
    value = re.fullmatch(r"i(?:\s*\*\s*(\d+))?(?:\s*([+-])\s*(\d+))?", expression.strip())
    if not value:
        raise DilSvgError("Unsupported DIL SVG arithmetic expression")
    number = index * int(value.group(1) or '1')
    if value.group(2):
        number += (1 if value.group(2) == '+' else -1) * int(value.group(3))
    return number


def _predicate(expression: str, index: int) -> bool:
    clauses = expression.strip().split('||')
    if not 1 <= len(clauses) <= 4:
        raise DilSvgError("Unsupported DIL SVG condition")
    for clause in clauses:
        match = re.fullmatch(r"\s*i\s*(===|==|<|<=|>|>=)\s*(\d+)\s*", clause)
        if not match:
            raise DilSvgError("Unsupported DIL SVG condition")
        op, raw = match.groups()
        number = int(raw)
        if {'===': index == number, '==': index == number, '<': index < number,
            '<=': index <= number, '>': index > number, '>=': index >= number}[op]:
            return True
    return False


def _expression(expression: str, index: int, label: str) -> str:
    expression = expression.strip()
    if expression == 'label':
        return html.escape(label, quote=True)
    if expression.startswith('`') and expression.endswith('`'):
        inner = expression[1:-1]
        if '`' in inner:
            raise DilSvgError("Unsupported SVG template literal")
        return re.sub(r"\$\{([^{}]+)\}", lambda m: str(_index_number(m.group(1), index)), inner)
    choice = re.fullmatch(r'(.+?)\s*\?\s*("[^"\n]*"|\'[^\'\n]*\')\s*:\s*("[^"\n]*"|\'[^\'\n]*\')', expression)
    if choice:
        selected = choice.group(2) if _predicate(choice.group(1), index) else choice.group(3)
        return html.escape(selected[1:-1], quote=True)
    return str(_index_number(expression, index))


def _interpolate(line: str, index: int, label: str) -> str:
    # SVG attributes must be quoted XML strings after JSX interpolation,
    # especially path data containing whitespace. Never evaluate arbitrary JS.
    line = re.sub(
        r'(\s+[A-Za-z][\w:.-]*)=\{(`[^`\n]*`|[^{}\n]*)\}',
        lambda m: m.group(1) + '="' + _expression(m.group(2), index, label) + '"',
        line,
    )
    return re.sub(r'\{([^{}\n]+)\}',
                  lambda m: _expression(m.group(1), index, label), line)


def expand_dil_svg(source: str) -> str:
    """Return static SVG markup, or raise DilSvgError for unsupported DIL.

    No execution, filesystem access, or Python eval is involved. Both loops
    and emitted markup are bounded to avoid untrusted input amplification.
    """
    if '{#each' not in source and '{#if' not in source:
        return source
    if len(source) > 100_000:
        raise DilSvgError("DIL SVG template is too large")
    lines = source.splitlines(keepends=True)

    def block(start: int, stop: int, index: int, label: str) -> list[str]:
        result: list[str] = []
        pos = start
        while pos < stop:
            stripped = lines[pos].strip()
            each = re.fullmatch(r'\{#each\s+(\[[^\n]*\])\s+as\s+label\s*,\s*i\s*\}', stripped)
            cond = re.fullmatch(r'\{#if\s+([^{}\n]+)\}', stripped)
            if each or cond:
                closing = '{/each}' if each else '{/if}'
                opening = '{#each' if each else '{#if'
                depth = 1
                end = pos + 1
                while end < stop and depth:
                    marker = lines[end].strip()
                    if marker.startswith(opening):
                        depth += 1
                    elif marker == closing:
                        depth -= 1
                    end += 1
                if depth:
                    raise DilSvgError("Unclosed DIL SVG directive")
                if each:
                    try:
                        labels = json.loads(each.group(1))
                    except (ValueError, TypeError) as exc:
                        raise DilSvgError("Invalid literal SVG labels") from exc
                    if not isinstance(labels, list) or len(labels) > 32 or not all(isinstance(x, str) and len(x) < 256 for x in labels):
                        raise DilSvgError("Unsupported SVG label list")
                    for new_index, new_label in enumerate(labels):
                        result.extend(block(pos + 1, end - 1, new_index, new_label))
                elif _predicate(cond.group(1), index):
                    result.extend(block(pos + 1, end - 1, index, label))
                pos = end
                continue
            if stripped.startswith(('{#', '{/', '{:')):
                raise DilSvgError("Unsupported DIL SVG directive")
            result.append(_interpolate(lines[pos], index, label))
            if sum(map(len, result)) > 200_000:
                raise DilSvgError("Expanded DIL SVG is too large")
            pos += 1
        return result

    rendered = ''.join(block(0, len(lines), 0, ''))
    # Legacy markdown exports sometimes autolinked the XML namespace value.
    namespace = 'http:' + '/' * 2 + 'www.w3.org/2000/svg'
    rendered = re.sub(
        r'\bxmlns\s*=\s*"[^"\n]*"',
        lambda m: 'xmlns="' + namespace + '"' if 'w3.org/2000/svg' in m.group(0) else m.group(0),
        rendered,
    )
    names = {'fillOpacity': 'fill-opacity', 'strokeOpacity': 'stroke-opacity',
             'strokeWidth': 'stroke-width', 'strokeDasharray': 'stroke-dasharray',
             'textAnchor': 'text-anchor', 'fontSize': 'font-size'}
    rendered = re.sub(r'\b(' + '|'.join(names) + r')(?=\s*=)',
                      lambda m: names[m.group(1)], rendered)
    if any(marker in rendered for marker in ('{#each', '{#if', '{/each}', '{/if}', '${')):
        raise DilSvgError("Unexpanded SVG template expression")
    return rendered
