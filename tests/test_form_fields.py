from pathlib import Path
import tempfile
import unittest
from mdexplore_app.form_fields import parse_fields, apply_field, save_field, revision, prepare_forms, finish_forms

SOURCE = '# Fixed\n\n| Date | <!-- mdform:field id="date" -->____<!-- /mdform --> |\n\n<!-- mdform:check id="done" -->\n- [ ] Called CRA\n<!-- /mdform -->\n\n<!-- mdform:text id="notes" -->\n____\n<!-- /mdform -->\nEND\n'

def request(s, field, value): return dict(revision=revision(s), id=field, value=value)

class FormFieldsTests(unittest.TestCase):
    def test_boundaries_and_round_trip(self):
        value = 'Sept 24 | café <script> & `note`'
        updated = apply_field(SOURCE, request(SOURCE, 'date', value))
        before, after = parse_fields(SOURCE)[0], parse_fields(updated)[0]
        self.assertEqual(after['value'], value)
        self.assertEqual(SOURCE[:before['valueStart']], updated[:after['valueStart']])
        self.assertEqual(SOURCE[before['valueEnd']:], updated[after['valueEnd']:])
        checked = apply_field(updated, request(updated, 'done', True))
        self.assertIn('- [x] Called CRA', checked)
        self.assertEqual(apply_field(checked, request(checked, 'done', False)), updated)
        notes = apply_field(updated, request(updated, 'notes', 'First\n\n- Second | item'))
        self.assertEqual(parse_fields(notes)[2]['value'], 'First\n\n- Second | item')

    def test_rejects_stale_unmarked_and_invalid(self):
        for req in [dict(id='date',value='x',revision='old'), request(SOURCE,'missing','x'), request(SOURCE,'date','a\nb'), request(SOURCE,'notes','<!-- /mdform -->'), request(SOURCE,'done','yes')]:
            with self.assertRaises(ValueError): apply_field(SOURCE, req)
        for text in [SOURCE+SOURCE, '<!-- mdform:field id="x" -->oops', '<!-- mdform:text id="x" --><!-- mdform:field id="y" -->x<!-- /mdform --><!-- /mdform -->']:
            with self.assertRaises(ValueError): parse_fields(text)

    def test_code_examples_inert(self):
        field='<!-- mdform:field id="example" -->____<!-- /mdform -->'
        self.assertEqual(parse_fields('```md\n'+field+'\n```\n`'+field+'`\n    '+field+'\n'), [])

    def test_backups_crlf_bom_and_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'form.md'; original='\ufeff'+SOURCE.replace('\n','\r\n'); file.write_bytes(original.encode())
            saved=save_field(file,request(original,'date','Réponse'))
            self.assertEqual(file.with_suffix('.preedit.md').read_bytes(),original.encode())
            self.assertIn('\r\n',saved['text'])
            save_field(file,request(saved['text'],'done',True))
            self.assertEqual(file.with_suffix('.preedit.md').read_bytes(),original.encode())
            with self.assertRaises(ValueError): save_field(file,request(original,'date','stale'))
            lock=Path(str(file)+'.mdform.lock');lock.write_text('other')
            with self.assertRaises(ValueError):save_field(file,request(original,'date','busy'))
            self.assertEqual(lock.read_text(),'other')

    def test_backup_failure_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'form.md';file.write_text(SOURCE);file.with_suffix('.preedit.md').mkdir()
            with self.assertRaises(ValueError):save_field(file,request(SOURCE,'date','bad'))
            self.assertEqual(file.read_text(),SOURCE)
            self.assertFalse(Path(str(file)+'.mdform.lock').exists())

    def test_controls_escape_source_text(self):
        source=apply_field(SOURCE,request(SOURCE,'date','<img onerror=evil>'))
        prepared,fields,error=prepare_forms(source)
        rendered=finish_forms(prepared,fields)
        self.assertFalse(error);self.assertIn('&lt;img',rendered);self.assertNotIn('<img',rendered)
        self.assertEqual(rendered.count('data-mdform-id='),3)
        self.assertEqual(rendered.count('mdform-empty'), 1)
        self.assertNotIn('mdform-field mdform-empty', rendered)

    def test_empty_text_controls_receive_placeholder_class(self):
        prepared, fields, error = prepare_forms(SOURCE)
        rendered = finish_forms(prepared, fields)
        self.assertFalse(error)
        self.assertEqual(rendered.count('mdform-empty'), 2)
        self.assertIn('mdform-field mdform-empty', rendered)
        self.assertIn('mdform-text mdform-empty', rendered)
