"""Explicitly tagged Markdown form regions; source-bounded, backed-up local saves."""
from __future__ import annotations
import hashlib
import html
import os
from pathlib import Path
import re
import secrets

MARKER = re.compile(r'<!--\s*(?:mdform:(field|text|check)\s+id="([A-Za-z][A-Za-z0-9_-]{0,63})"(?:\s+placeholder="([^"<>]*)")?\s*|/mdform\s*)-->')
PUNCTUATION = r'[\\`*_{}\[\]<>()#+.!|~$>\-]'

def revision(source: str) -> str:
    return hashlib.sha256(source.encode('utf-8')).hexdigest()

def code_ranges(source):
    ranges, fence, offset = [], None, 0
    for line in source.splitlines(keepends=True):
        m = re.match(r'^( {0,3})(`{3,}|~{3,})(.*)', line)
        if fence:
            if m and m[2][0] == fence[1] and len(m[2]) >= fence[2] and not m[3].strip():
                ranges.append((fence[0], offset + len(line))); fence = None
        elif m:
            fence = (offset, m[2][0], len(m[2]))
        elif re.match(r'^(?: {4}|\t)', line):
            ranges.append((offset, offset + len(line)))
        offset += len(line)
    if fence:
        ranges.append((fence[0], len(source)))
    ranges.extend((m.start(), m.end()) for m in re.finditer(r'(?<!\\)(`+)[\s\S]*?(?<!\\)\1', source))
    return ranges

def parse_fields(source: str) -> list[dict]:
    if "mdform" not in source:
        return []
    ranges, fields, ids, active = code_ranges(source), [], set(), None
    for m in MARKER.finditer(source):
        if any(a <= m.start() < b for a, b in ranges):
            continue
        if m[1]:
            if active or m[2] in ids:
                raise ValueError('Nested or duplicate mdform markers. Fix the source before editing.')
            active = dict(id=m[2], kind=m[1], placeholder=html.unescape(m[3] or 'Click to fill in'), start=m.start(), valueStart=m.end()); ids.add(m[2])
        else:
            if active is None:
                raise ValueError('Unmatched mdform closing marker.')
            raw = source[active['valueStart']:m.start()]
            if active['kind'] == 'check':
                check = re.fullmatch(r'(\s*(?:- +)?\[)([ xX])(\][^\r\n]*\s*)', raw)
                if not check:
                    raise ValueError('A check region must contain one [ ] or [x] checkbox, optionally a task-list item.')
                value = check[2].lower() == 'x'
                active['label'] = re.sub(r'\\(' + PUNCTUATION + ')', r'\1', check[3][1:].strip()) or active['id']
                active['checkOffset'] = active['valueStart'] + len(check[1])
            else:
                if active['kind'] == 'field' and re.search(r'[\r\n]', raw):
                    raise ValueError('Use mdform:text for multiline fields.')
                value = re.sub(r'\\(' + PUNCTUATION + ')', r'\1', raw.strip())
                if value == '____': value = ''
            fields.append(dict(active, valueEnd=m.start(), end=m.end(), value=value)); active = None
    if active:
        raise ValueError('Unclosed mdform marker.')
    return fields

def apply_field(source: str, request: dict) -> str:
    if request.get('revision') != revision(source):
        raise ValueError('The file changed. Refresh and review your draft before saving again.')
    field = next((f for f in parse_fields(source) if f['id'] == request.get('id')), None)
    if not field: raise ValueError('This is not a designated editable field.')
    value = request.get('value')
    if field['kind'] == 'check':
        if not isinstance(value, bool): raise ValueError('Checkbox value must be true or false.')
        pos = field['checkOffset']; return source[:pos] + ('x' if value else ' ') + source[pos+1:]
    if not isinstance(value, str) or len(value) > 20000 or re.search(r'\x00|<!--|-->|mdform:', value, re.I):
        raise ValueError('Invalid field text (limit 20,000 characters; form markers are not allowed).')
    value = value.replace('\r\n', '\n').replace('\r', '\n')
    if field['kind'] == 'field' and '\n' in value: raise ValueError('This field accepts a single line.')
    value = re.sub(PUNCTUATION, lambda m: '\\' + m[0], value) or '____'
    if field['kind'] == 'text':
        eol = '\r\n' if '\r\n' in source else '\n'; value = eol + value.replace('\n', eol) + eol
    return source[:field['valueStart']] + value + source[field['valueEnd']:]

def prepare_forms(source):
    try: fields = parse_fields(source)
    except ValueError as error: return source, [], str(error)
    token = 'MDFORM' + secrets.token_hex(12)
    prepared = source
    for i in range(len(fields)-1, -1, -1):
        f = fields[i]; f['token'] = token + str(i) + 'END'
        prepared = prepared[:f['start']] + f['token'] + '\n' * source[f['start']:f['end']].count('\n') + prepared[f['end']:]
    return prepared, fields, ''

def finish_forms(body, fields):
    for f in fields:
        value = str(f['value']).lower() if isinstance(f['value'], bool) else f['value']
        data = f'data-mdform-id="{f["id"]}" data-mdform-kind="{f["kind"]}" data-mdform-value="{html.escape(value, quote=True)}" data-mdform-placeholder="{html.escape(f["placeholder"], quote=True)}"'
        if f['kind'] == 'check':
            control = f'<label class="mdform-region"><input type="checkbox" {data} {"checked" if f["value"] else ""}> {html.escape(f["label"])}</label>'
        else:
            empty_class = ' mdform-empty' if not value else ''
            control = f'<button type="button" class="mdform-region mdform-{f["kind"]}{empty_class}" {data} aria-label="Edit {f["id"]}">{html.escape(value or f["placeholder"])}</button>'
        body = body.replace(f['token'], control, 1)
    return body

def save_field(file: Path, request: dict) -> dict:
    file = file.resolve(strict=True)
    if file.suffix.lower() not in ('.md', '.markdown') or file.name.lower().endswith('.preedit.md'):
        raise ValueError('Only original local Markdown documents can be edited.')
    lock = Path(str(file) + '.mdform.lock'); fd = None; temporary = None
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        data = file.read_bytes(); source = data.decode('utf-8')
        if request.get('operation') == 'insert':
            from .form_insertion import apply_insert
            updated = apply_insert(source, request)
        else:
            updated = apply_field(source, request)
        if updated == source: return dict(text=source, revision=revision(source), unchanged=True)
        backup = file.with_suffix('.preedit.md')
        try:
            out = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, file.stat().st_mode & 0o777)
        except FileExistsError:
            if backup.is_symlink() or not backup.is_file(): raise ValueError('Backup path is not a regular file.')
        else:
            try:
                with os.fdopen(out, 'wb') as handle:
                    os.fchmod(handle.fileno(), file.stat().st_mode & 0o777)
                    handle.write(data); handle.flush(); os.fsync(handle.fileno())
            except BaseException:
                backup.unlink(missing_ok=True)
                raise
        if file.read_bytes() != data: raise ValueError('The file changed during save. Nothing was overwritten.')
        temporary = file.with_name('.' + file.name + '.mdform-' + secrets.token_hex(8))
        out = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, file.stat().st_mode & 0o777)
        with os.fdopen(out, 'wb') as handle:
            os.fchmod(handle.fileno(), file.stat().st_mode & 0o777)
            handle.write(updated.encode('utf-8')); handle.flush(); os.fsync(handle.fileno())
        if file.read_bytes() != data: raise ValueError('The file changed during save. Nothing was overwritten.')
        os.replace(temporary, file); temporary = None
        return dict(text=updated, revision=revision(updated), backup=str(backup), lineDelta=updated.count("\n")-source.count("\n"))
    except FileExistsError as error:
        raise ValueError('Another form save is in progress (.mdform.lock). Retry shortly.') from error
    finally:
        if temporary: temporary.unlink(missing_ok=True)
        if fd is not None: os.close(fd); lock.unlink(missing_ok=True)
