"""Conservative Markdown fallback for legacy ChatGPT DIL presentation components.

This is a text/markup normalizer, not a DIL interpreter: it never evaluates
expressions or fetches external images. The original Markdown is unchanged.
"""

from __future__ import annotations

import html
import json
import re

_STRUCTURE = re.compile(r"<(?:box|row|grid|grid-item|CodeBlock|AsyncImage)\b")
_CODE_BLOCK = re.compile(r'<CodeBlock\b([^>]*)>(.*?)</CodeBlock>', re.DOTALL)
_TAGS = re.compile(r'</?(?:box|row|grid|grid-item|text|caption)\b[^>]*>', re.IGNORECASE)
_COMPONENT_OPEN = re.compile(r'<(?:box|row|grid|grid-item)\b')
_COMPONENT_CLOSE = re.compile(r'</(?:box|row|grid|grid-item)\s*>')
_FENCE = re.compile(r'^[ \t]{0,3}(`{3,}|~{3,})(.*)$')
_LOOP = re.compile(r'^\s*\{#each\s+(\[[^\n]*\])\s+as\s+(\w+)\s*,\s*i\s*\}\s*$')


def normalize_dil_markdown(markdown: str) -> str:
    """Make known exported DIL layout readable in ordinary Markdown preview.

    Keep line counts stable for source navigation. Only bare triple backticks
    within DIL layouts are treated as export artifacts; normal fenced code
    blocks, including SVG, remain verbatim.
    """
    if not _STRUCTURE.search(markdown):
        return markdown

    def code_block(match: re.Match[str]) -> str:
        language = re.search(r'\blanguage="([A-Za-z0-9_-]{1,32})"', match.group(1))
        lang = language.group(1) if language else 'text'
        return f'<pre><code class="language-{lang}">{html.escape(match.group(2))}</code></pre>'

    source = _CODE_BLOCK.sub(code_block, markdown)
    lines = source.splitlines(keepends=True)
    result: list[str] = []
    depth = 0
    fence_char = ''
    fence_length = 0
    in_code_block = False
    i = 0
    while i < len(lines):
        raw = lines[i]
        content = raw.rstrip('\r\n')
        ending = raw[len(content):]
        if '<pre><code class="language-' in content:
            in_code_block = True
        if in_code_block:
            result.append(raw)
            if '</code></pre>' in content:
                in_code_block = False
            i += 1
            continue

        fence = _FENCE.match(content)
        if fence_char:
            result.append(raw)
            if fence and fence.group(1)[0] == fence_char and len(fence.group(1)) >= fence_length and not fence.group(2).strip():
                fence_char = ''
                fence_length = 0
            i += 1
            continue
        if fence:
            if depth and fence.group(1) == '```' and not fence.group(2).strip():
                result.append(ending)
            else:
                fence_char, fence_length = fence.group(1)[0], len(fence.group(1))
                result.append(raw)
            i += 1
            continue

        loop = _LOOP.match(content)
        if depth and loop:
            end = i + 1
            while end < len(lines) and lines[end].strip() != '{/each}':
                end += 1
            if end < len(lines) and end - i < 200:
                try:
                    labels = json.loads(loop.group(1))
                except (ValueError, TypeError):
                    labels = None
                if isinstance(labels, list) and len(labels) <= 32 and all(isinstance(v, str) and len(v) < 256 for v in labels):
                    result.append(' · '.join(f'{n}. {html.escape(label)}' for n, label in enumerate(labels, 1)) + ending)
                    result.extend(x[len(x.rstrip('\r\n')):] for x in lines[i + 1:end + 1])
                    i = end + 1
                    continue

        def link(match: re.Match[str]) -> str:
            attrs = match.group(1)
            title_match = re.search(r'\btitle="([^"]{0,500})"', attrs)
            url_match = re.search(r'\burl="([^"]{0,2048})"', attrs)
            title = title_match.group(1) if title_match else ''
            value = url_match.group(1) if url_match else ''
            # Exports may wrap a URL in Markdown link syntax inside the
            # component attribute. Use the final URL without re-parsing its
            # surrounding brackets as part of the destination.
            urls = re.findall(r'https?://[^\s<>"()\]]+', value)
            url = urls[-1] if urls else ''
            if not url:
                return title
            return f'[{title.replace("[", "").replace("]", "")}]({url})'

        normalized = re.sub(r'<Link\b([^>]*?)/>', link, content)
        normalized = re.sub(r'<(?:AsyncImage|icon)\b[^>]*?/>', '', normalized)
        normalized = re.sub(r'<divider\b[^>]*?/>', '---', normalized)
        normalized = re.sub(r'\\*<Cite\b[^>]*?/>', '', normalized)
        normalized = _TAGS.sub('', normalized)
        inside_layout = depth > 0 or bool(_COMPONENT_OPEN.search(content))
        depth = max(0, depth + len(_COMPONENT_OPEN.findall(content)) - len(_COMPONENT_CLOSE.findall(content)))
        # Source indentation positions DIL components, not Markdown code.
        # Leaving four spaces turns ordinary prose and links into code blocks.
        if inside_layout:
            normalized = normalized.lstrip(' \t')
        result.append(normalized + ending)
        i += 1
    return ''.join(result)
