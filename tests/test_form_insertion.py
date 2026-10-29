import tempfile
from pathlib import Path
import unittest
from mdexplore_app.form_fields import revision, parse_fields, save_field
from mdexplore_app.form_insertion import apply_insert, cells

class FormInsertionTests(unittest.TestCase):
    def request(self, source, text='Hello', **extra):
        return dict(operation='insert', revision=revision(source), kind='field', selection=dict(text=text,startLine=0,endLine=1,occurrence=0,occurrences=1), **extra)

    def test_preserves_text_custom_placeholder_and_backup(self):
        source='Hello world!\n'
        request=self.request(source,label='Name | "quoted" & café')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'form.md';p.write_text(source)
            result=save_field(p,request)
            field=parse_fields(result['text'])[0]
            self.assertEqual(field['placeholder'],'Name | "quoted" & café')
            self.assertEqual(field['value'],'')
            self.assertEqual(result['text'][:field['start']], 'Hello ')
            self.assertEqual(result['text'][field['end']:], ' world!\n')
            self.assertEqual(p.with_suffix('.preedit.md').read_text(),source)

    def test_check_default_and_custom_label(self):
        source='Hello\n';r=self.request(source);r['kind']='check'
        self.assertEqual(parse_fields(apply_insert(source,r))[0]['label'],'Done')
        r['label']='Received | confirmed'
        self.assertEqual(parse_fields(apply_insert(source,r))[0]['label'],r['label'])

    def test_insert_middle_and_append_table_column(self):
        source='| Left | Right |\r\n| :--- | ---: |\r\n| A \\| B | Target |\r\n| C | D |\r\n'
        for col,text in [(0,'A | B'),(1,'Target')]:
            r=self.request(source,text,newColumn=True)
            r['selection']['endLine']=4
            r['selection']['table']=dict(startLine=0,row=1,column=col)
            result=apply_insert(source,r); lines=result.splitlines()
            self.assertTrue(all(len(cells(line))==3 for line in lines))
            self.assertIn('A \\| B',result);self.assertIn('Target',result)
            self.assertIn('---:',result);self.assertEqual(result.count('\r\n'),4)
            spans=cells(lines[2]); self.assertIn('mdform:field',lines[2][slice(*spans[col+1])])
            self.assertEqual(len(parse_fields(result)),1)

    def test_repeated_selection_and_fail_closed(self):
        source='Hello Hello\n';r=self.request(source)
        with self.assertRaises(ValueError):apply_insert(source,r)
        r['selection'].update(occurrence=1,occurrences=2)
        self.assertTrue(apply_insert(source,r).startswith('Hello Hello <!--'))
        r['revision']='stale'
        with self.assertRaises(ValueError):apply_insert(source,r)
        r=self.request('`Hello`\n')
        with self.assertRaises(ValueError):apply_insert('`Hello`\n',r)
        r=self.request('Hello\n',newColumn=True)
        with self.assertRaises(ValueError):apply_insert('Hello\n',r)
