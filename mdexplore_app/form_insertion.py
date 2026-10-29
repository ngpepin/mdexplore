"""Conservative insertion at a preview selection; never replace selected text."""
import html
import re
import secrets
from .form_fields import revision, parse_fields, code_ranges, PUNCTUATION


def cells(line):
    """Source spans of GFM cells; escaped pipes are content, not separators."""
    cuts = [m.start() for m in re.finditer(r'\|', line)
            if (len(line[:m.start()]) - len(line[:m.start()].rstrip('\\'))) % 2 == 0]
    if not cuts:
        return []
    bounds = [-1] + cuts + [len(line)]
    result = [(a + 1, b) for a, b in zip(bounds, bounds[1:])]
    if not line[:cuts[0]].lstrip('\ufeff').strip(): result.pop(0)
    if not line[cuts[-1] + 1:].strip(): result.pop()
    return result


def apply_insert(source, request):
    if request.get('revision') != revision(source):
        raise ValueError('The file changed. Refresh before inserting a field.')
    kind = request.get('kind')
    label = request.get('label', 'Done' if kind == 'check' else 'Click to fill in')
    if kind not in ('field', 'check') or not isinstance(label, str) or len(label) > 200 or re.search(r'[\r\n\x00]|<!--|-->|mdform:', label, re.I):
        raise ValueError('Use a single-line label of up to 200 characters without form markers.')
    label = label.strip() or ('Done' if kind == 'check' else 'Click to fill in')
    selection = request.get('selection') or {}
    text = selection.get('text')
    if not isinstance(text, str) or not text.strip() or len(text) > 20000:
        raise ValueError('Select text in one paragraph or table cell first.')
    lines = source.splitlines(keepends=True)
    start, end = selection.get('startLine'), selection.get('endLine')
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(lines):
        raise ValueError('Cannot locate this selection in Markdown. Select a shorter text span.')
    offset = sum(map(len, lines[:start])); segment = ''.join(lines[start:end])
    table = selection.get('table')
    table_rows = None
    if table is not None:
        top, row, column = table.get('startLine'), table.get('row'), table.get('column')
        if any(type(n) is not int or n < 0 for n in (top, row, column)) or top + 1 >= len(lines):
            raise ValueError('Cannot locate the selected Markdown table.')
        header = cells(lines[top].rstrip('\r\n')); sep = lines[top+1].rstrip('\r\n'); separators = cells(sep)
        if not header or len(header) != len(separators) or column >= len(header) or any(not re.fullmatch(r'\s*:?-+:?\s*', sep[a:b]) for a,b in separators):
            raise ValueError('Only ordinary pipe tables support column insertion.')
        table_rows = [top, top+1]
        i = top+2
        while i < len(lines) and lines[i].strip() and cells(lines[i].rstrip('\r\n')):
            table_rows.append(i); i += 1
        selected_row = top if row == 0 else top + row + 1
        if selected_row not in table_rows or selected_row == top+1:
            raise ValueError('The selected table row is not present in the source.')
        bounds = cells(lines[selected_row].rstrip('\r\n'))
        if len(bounds) != len(header):
            raise ValueError('Normalize ragged table rows in the source before inserting.')
        a,b = bounds[column]
        offset = sum(map(len, lines[:selected_row])) + a
        segment = lines[selected_row][a:b]
    # Contiguous literal text, including backslash-escaped punctuation. More
    # complex selections fail closed rather than guessing a destructive offset.
    pattern = ''.join((r'\\?' if re.fullmatch(PUNCTUATION, c) else '') + re.escape(c) for c in text)
    matches = list(re.finditer(pattern, segment))
    occurrence, total = selection.get('occurrence', 0), selection.get('occurrences', 1)
    if type(occurrence) is not int or len(matches) != total or not 0 <= occurrence < len(matches):
        raise ValueError('Selection cannot be mapped unambiguously. Select a shorter plain-text span within one cell or paragraph.')
    match = matches[occurrence]; begin, finish = offset + match.start(), offset + match.end()
    fields = parse_fields(source)
    if any(a < finish and b > begin for a,b in code_ranges(source)) or any(f['start'] < finish and f['end'] > begin for f in fields):
        raise ValueError('Choose ordinary text outside code and existing form controls.')
    identity = 'field-' + secrets.token_hex(8)
    encoded = re.sub(PUNCTUATION, lambda m: '\\' + m[0], label)
    placeholder = html.escape(label, quote=True).replace("|", "&#124;").replace("`", "&#96;")
    element = (f'<!-- mdform:field id="{identity}" placeholder="{placeholder}" -->____<!-- /mdform -->'
               if kind == 'field' else f'<!-- mdform:check id="{identity}" -->[ ] {encoded}<!-- /mdform -->')
    if not request.get('newColumn'):
        updated = source[:finish] + ' ' + element + source[finish:]
    else:
        if table_rows is None: raise ValueError('Select text within one table cell to add a column.')
        width = len(header)
        for index in reversed(table_rows):
            raw = lines[index].rstrip('\r\n'); bounds = cells(raw)
            if len(bounds) != width: raise ValueError('Normalize ragged table rows in the source before inserting.')
            pos = bounds[column][1]
            value = '---' if index == top+1 else element if index == selected_row else ''
            lines[index] = lines[index][:pos] + ' | ' + value + ' ' + lines[index][pos:]
        updated = ''.join(lines)
    parse_fields(updated)
    return updated
