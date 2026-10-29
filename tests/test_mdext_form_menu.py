"""Exercise the shipped mdext HTML/JS menu in real Chromium, with local files."""
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from PySide6.QtCore import QUrl, QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView
import test_form_preview as preview_helpers

class MdextFormMenuTests(unittest.TestCase):
    js = preview_helpers.FormPreviewTests.js
    wait = preview_helpers.FormPreviewTests.wait

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_actual_webview_menu_default_custom_cancel_and_write(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='mdext-menu-') as tmp:
            file=Path(tmp)/'source.md';page=Path(tmp)/'preview.html'
            original='Hello world\n\n| Left | Right |\n| --- | --- |\n| Alpha | Beta |\n'
            file.write_text(original)
            subprocess.run(['node',str(root/'mdext/test/integration/insertionWebview.js'),str(page),str(file)],check=True)
            self.view=QWebEngineView();self.window=SimpleNamespace(preview=self.view)
            self.view.resize(1000,700);self.view.show();self.view.load(QUrl.fromLocalFile(str(page)))
            try:
                self.wait(lambda:self.js("typeof applyRender === 'function'"))
                message=Path(str(page)+'.json').read_text()
                self.js('window.dispatchEvent(new MessageEvent("message",{data:'+message+'}))')
                self.wait(lambda:self.js("document.querySelector('td')?.textContent")=='Alpha')
                def menu(selector):
                    self.js("var el=document.querySelector("+json.dumps(selector)+");var r=document.createRange();r.selectNodeContents(el);getSelection().removeAllRanges();getSelection().addRange(r);el.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true,clientX:30,clientY:30}))")
                menu('p')
                self.assertEqual(self.js("getComputedStyle(document.getElementById('context-add-field-column')).display"),'none')
                self.js("document.getElementById('context-add-field').click()")
                self.assertEqual(self.js("document.getElementById('mdform-editor').value"),'Click to fill in')
                self.js("document.querySelector('.mdform-actions button').click()")
                self.assertEqual(self.js("outMessages.filter(m=>m.type==='saveFormField').length"),0)
                menu('td')
                self.assertNotEqual(self.js("getComputedStyle(document.getElementById('context-add-check-column')).display"),'none')
                self.js("document.getElementById('context-add-check-column').click()")
                self.assertEqual(self.js("document.getElementById('mdform-editor').value"),'Done')
                self.js("document.getElementById('mdform-editor').value='Received | checked'")
                self.js("document.querySelector('.mdform-actions button:last-child').click()")
                request=json.loads(self.js("JSON.stringify(outMessages.findLast(m=>m.type==='saveFormField'))"))
                self.assertEqual(request['kind'],'check');self.assertTrue(request['newColumn'])
                self.assertEqual(request['label'],'Received | checked')
                script='const {saveField}=require(process.argv[1]);const r=saveField(process.argv[2],JSON.parse(process.argv[3]));process.stdout.write(JSON.stringify(r));'
                result=json.loads(subprocess.check_output(['node','-e',script,str(root/'mdext/src/formFields.js'),str(file),json.dumps(request)],text=True))
                self.assertIn('[ ] Received \\| checked',file.read_text())
                self.assertEqual(file.with_suffix('.preedit.md').read_text(),original)
                self.js('window.dispatchEvent(new MessageEvent("message",{data:'+json.dumps(dict(type='formSaved',ok=True,revision=result['revision']))+'}))')
                self.assertFalse(self.js("!!document.querySelector('dialog')"))
            finally:
                self.view.close();self.view.page().deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
                self.view.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
