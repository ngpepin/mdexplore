"""Real Chromium UI + Qt host save contract (temporary documents only)."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
import mdexplore
from mdexplore_app.form_fields import parse_fields
from test_form_fields import SOURCE

class FormPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='mdform-ui-')
        self.root = Path(self.temp.name)
        self.file = self.root / 'form.md'
        self.file.write_text(SOURCE + '\n- [ ] Ordinary read-only task\n')
        self.original = self.file.read_bytes()
        self.window = mdexplore.MdExploreWindow(root=self.root, app_icon=QIcon(), config_path=self.root / '.config', gpu_context_available=False)
        self.window.show()
        self.window._load_preview(self.file)
        self.wait(lambda: self.js("document.querySelectorAll('[data-mdform-id]').length") == 3)

    def tearDown(self):
        self.window._form_edit_active = False
        self.window._form_refresh_pending = False
        QTest.qWait(1600)  # Drain delayed scroll restoration before deleting Qt objects.
        self.window.close()
        # Single-shot scroll callbacks outlive a closed test window.
        self.window._restore_current_preview_scroll = lambda *args: None
        self.window._enable_preview_scroll_capture_for = lambda *args: None
        self.window.preview.page().deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.temp.cleanup()

    def wait(self, predicate, timeout=15):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.app.processEvents()
            if predicate(): return
            QTest.qWait(30)
        self.fail('Timed out waiting for preview state')

    def js(self, code):
        result = []
        self.window.preview.page().runJavaScript(code, lambda value: result.append(value))
        end = time.monotonic() + 5
        while not result and time.monotonic() < end:
            self.app.processEvents(); QTest.qWait(5)
        return result[0] if result else None

    def edit(self, field, value):
        self.wait(lambda: self.js("!!document.querySelector('[data-mdform-id=\""+field+"\"]')"))
        self.js("document.querySelector('[data-mdform-id=\""+field+"\"]').click()")
        self.wait(lambda: self.js("!!document.querySelector('#mdform-editor')"))
        self.js('document.querySelector("#mdform-editor").value='+json.dumps(value))
        self.js("document.querySelector('.mdform-actions button:last-child').click()")

    def test_save_checkbox_notes_cancel_and_first_backup(self):
        self.assertTrue(self.js("document.querySelector('input:not([data-mdform-id])').disabled"))
        value = 'Sept 24 | café 100%20 %25 #1 <note>'
        self.edit('date',value)
        self.wait(lambda: parse_fields(self.file.read_text())[0]['value'] == value)
        self.wait(lambda: not self.window._form_edit_active)
        self.assertEqual(self.file.with_suffix('.preedit.md').read_bytes(),self.original)
        self.wait(lambda: self.js("document.querySelector('[data-mdform-id=date]')?.dataset.mdformValue") == value)
        self.js("document.querySelector('[data-mdform-id=done]').click()")
        self.wait(lambda: parse_fields(self.file.read_text())[1]['value'] is True)
        self.wait(lambda: self.js("document.querySelector('[data-mdform-id=done]')?.hasAttribute('checked')") is True)
        self.wait(lambda: not self.window._form_edit_active)
        self.edit('notes','Line one\n\nSecond | line with `literal` text')
        self.wait(lambda: parse_fields(self.file.read_text())[2]['value'].startswith('Line one'))
        self.wait(lambda: not self.window._form_edit_active)
        self.assertEqual(self.file.with_suffix('.preedit.md').read_bytes(),self.original)
        self.wait(lambda: self.js("document.querySelector('[data-mdform-id=notes]')?.dataset.mdformValue") == 'Line one\n\nSecond | line with `literal` text')
        self.wait(lambda: not self.window._preview_load_in_progress and not self.window._active_render_workers)
        QTest.qWait(100)
        before = self.file.read_bytes()
        self.js("document.querySelector('[data-mdform-id=date]').click()")
        self.wait(lambda: self.window._form_edit_active)
        self.js("document.querySelector('#mdform-editor').value='Cancelled draft'")
        self.window._refresh_current_preview(reason='test external refresh')
        self.assertEqual(self.js("document.querySelector('#mdform-editor').value"),'Cancelled draft')
        self.js("document.querySelector('.mdform-actions button').click()")
        self.wait(lambda: not self.window._form_edit_active)
        self.assertEqual(self.file.read_bytes(),before)

    def test_stale_save_keeps_draft_and_external_change(self):
        self.js("document.querySelector('[data-mdform-id=date]').click()")
        self.wait(lambda: self.window._form_edit_active)
        self.js("document.querySelector('#mdform-editor').value='Keep my draft'")
        external = self.file.read_text() + '\nExternal change\n'
        self.file.write_text(external)
        self.js("document.querySelector('.mdform-actions button:last-child').click()")
        self.wait(lambda: 'changed' in (self.js("document.querySelector('.mdform-message')?.textContent") or ''))
        self.assertEqual(self.js("document.querySelector('#mdform-editor').value"),'Keep my draft')
        self.assertEqual(self.file.read_text(),external)
        self.assertFalse(self.file.with_suffix('.preedit.md').exists())
        self.js("document.querySelector('.mdform-actions button').click()")
        self.wait(lambda: not self.window._form_edit_active)

    def test_placeholder_font_and_form_saves_preserve_viewport_anchor(self):
        self.file.write_text(
            "\n\n".join(f"## Before {index}\nParagraph {index}." for index in range(45))
            + "\n\n" + SOURCE
            + "\n\n".join(f"## After {index}\nParagraph {index}." for index in range(45))
        )
        stat = self.file.stat()
        self.window._set_current_preview_signature(
            str(self.file.resolve()), int(stat.st_mtime_ns), int(stat.st_size)
        )
        self.window._refresh_current_preview(reason='test long form document')
        self.wait(lambda: self.js("document.body.textContent.includes('Before 44')"))
        self.wait(lambda: self.js("!!document.querySelector('[data-mdform-id=date]')"))
        placeholder_style = json.loads(self.js("""
(() => {
  const field = document.querySelector('[data-mdform-id=date]');
  return JSON.stringify({
    empty: field.classList.contains('mdform-empty'),
    field: parseFloat(getComputedStyle(field).fontSize),
    surrounding: parseFloat(getComputedStyle(field.parentElement).fontSize),
  });
})()
"""))
        QTest.qWait(1600)  # Let initial saved-scroll restoration settle before user interaction.
        self.assertTrue(placeholder_style['empty'])
        self.assertAlmostEqual(placeholder_style['surrounding'] - placeholder_style['field'], 2 * 96 / 72, delta=0.1)

        self.js("document.querySelector('[data-mdform-id=date]').scrollIntoView({block:'center'})")
        before = self.js("document.querySelector('[data-mdform-id=date]').getBoundingClientRect().top")
        render_request = self.window._render_request_id
        self.edit('date', 'A filled value')
        self.wait(lambda: parse_fields(self.file.read_text())[0]['value'] == 'A filled value')
        self.wait(lambda: not self.window._form_edit_active)
        after = self.js("document.querySelector('[data-mdform-id=date]').getBoundingClientRect().top")
        self.assertAlmostEqual(after, before, delta=2.0)
        self.assertEqual(self.window._render_request_id, render_request)
        self.assertFalse(self.js("document.querySelector('[data-mdform-id=date]').classList.contains('mdform-empty')"))

        self.js("document.querySelector('[data-mdform-id=done]').scrollIntoView({block:'center'})")
        before = self.js("document.querySelector('[data-mdform-id=done]').getBoundingClientRect().top")
        self.js("document.querySelector('[data-mdform-id=done]').click()")
        self.wait(lambda: parse_fields(self.file.read_text())[1]['value'] is True)
        self.wait(lambda: not self.window._form_edit_active)
        after = self.js("document.querySelector('[data-mdform-id=done]').getBoundingClientRect().top")
        self.assertAlmostEqual(after, before, delta=2.0)
        self.assertEqual(self.window._render_request_id, render_request)

    def test_insert_from_context_menu_into_unmarked_table(self):
        from unittest.mock import patch
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QMenu
        source = '# Test\n\nHello world\n\n| Left | Right |\n| --- | --- |\n| Alpha | Beta |\n| Gamma | Delta |\n'
        self.file.write_text(source)
        self.window._load_preview(self.file)
        self.wait(lambda: self.js("document.querySelector('td')?.textContent.trim()") == 'Alpha')
        self.wait(lambda: self.js("!!window.MdForm"))
        self.js("var r=document.createRange();r.selectNodeContents(document.querySelector('td'));getSelection().removeAllRanges();getSelection().addRange(r)")
        context = json.loads(self.js('JSON.stringify(window.MdForm.selectionContext())') or 'null')
        # QVariant serializes objects for the desktop menu probe.
        self.assertIsInstance(context,dict)
        self.assertEqual(context['table']['column'],0)
        self.assertEqual(context['table']['row'],1)
        seen=[]
        def choose(menu, *args):
            seen.extend(a.text() for a in menu.actions())
            return next(a for a in menu.actions() if a.text()=='Add text field in new column')
        class TestMenu(QMenu):
            def exec(self, *args):
                return choose(self, *args)
        with patch.object(mdexplore,'QMenu',TestMenu):
            self.window._show_preview_context_menu_with_cached_selection(QPoint(10,10),{'formSelection':context,'hasSelection':True},'Alpha',QMenu(self.window.preview))
        self.assertIn('Add checkbox',seen)
        self.assertIn('Add checkbox in new column',seen)
        self.wait(lambda:self.js("document.querySelector('#mdform-editor')?.value")=='Click to fill in')
        self.js("document.querySelector('#mdform-editor').value='Enter | name `here`'")
        self.js("document.querySelector('.mdform-actions button:last-child').click()")
        self.wait(lambda:len(parse_fields(self.file.read_text()))==1)
        field=parse_fields(self.file.read_text())[0]
        self.assertEqual(field['placeholder'],'Enter | name `here`')
        self.wait(lambda:self.js("document.querySelectorAll('tbody tr:first-child td').length")==3)
        self.assertEqual(self.file.with_suffix('.preedit.md').read_text(),source)
        self.assertEqual(self.js("document.querySelectorAll('tbody tr:first-child td')[1].textContent.trim()"),'Enter | name `here`')
        # Append a checkbox after text, preserving the source's existing space.
        self.wait(lambda:not self.window._form_edit_active)
        self.js("var p=Array.from(document.querySelectorAll('p')).find(p=>p.textContent==='Hello world');var r=document.createRange();r.setStart(p.firstChild,0);r.setEnd(p.firstChild,5);getSelection().removeAllRanges();getSelection().addRange(r)")
        self.js("window.MdForm.openInsert('check',false,window.MdForm.selectionContext())")
        self.wait(lambda:self.js("document.querySelector('#mdform-editor')?.value")=='Done')
        self.js("document.querySelector('#mdform-editor').value='Approved'")
        self.js("document.querySelector('.mdform-actions button:last-child').click()")
        self.wait(lambda:len(parse_fields(self.file.read_text()))==2)
        self.assertIn('Hello <!-- mdform:check',self.file.read_text())
        self.assertIn('<!-- /mdform --> world',self.file.read_text())
        self.assertEqual(parse_fields(self.file.read_text())[0]['label'],'Approved')
        self.assertEqual(self.file.with_suffix('.preedit.md').read_text(),source)
