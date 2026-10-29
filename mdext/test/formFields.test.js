"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { hash, parseFields, applyField, saveField } = require("../src/formFields");
const { renderMarkdown } = require("../src/renderer");
const source = '# Fixed\n\n| Call | <!-- mdform:field id="call" -->____<!-- /mdform --> |\n\n<!-- mdform:check id="done" -->\n- [ ] Called CRA\n<!-- /mdform -->\n\n<!-- mdform:text id="notes" -->\n____\n<!-- /mdform -->\nEND\n';
function request(s, id, value) {
  return { revision: hash(s), id, value };
}
test("bounded edits preserve everything outside the requested region", () => {
  const next = applyField(source, request(source, "call", "Sept 24 | caf\xE9 <script> & `note`"));
  assert.equal(parseFields(next)[0].value, "Sept 24 | caf\xE9 <script> & `note`");
  const a = parseFields(source)[0], b = parseFields(next)[0];
  assert.equal(source.slice(0, a.valueStart), next.slice(0, b.valueStart));
  assert.equal(source.slice(a.valueEnd), next.slice(b.valueEnd));
  const checked = applyField(next, request(next, "done", true));
  assert.ok(checked.includes("- [x] Called CRA"));
  assert.equal(applyField(checked, request(checked, "done", false)), next);
  const notes = applyField(next, request(next, "notes", "Two lines\n\n- Second | item"));
  assert.equal(parseFields(notes)[2].value, "Two lines\n\n- Second | item");
});
test("stale, unmarked, duplicate, nested and injected edits are rejected", () => {
  assert.throws(() => applyField(source, { id: "call", value: "bad", revision: "old" }), /changed/);
  assert.throws(() => applyField(source, request(source, "missing", "bad")), /designated/);
  assert.throws(() => applyField(source, request(source, "call", "a\nb")), /single line/);
  assert.throws(() => applyField(source, request(source, "notes", "<!-- /mdform -->")), /Invalid/);
  assert.throws(() => parseFields(source + source), /duplicate/);
  assert.throws(() => parseFields('<!-- mdform:text id="a" --><!-- mdform:field id="b" -->x<!-- /mdform --><!-- /mdform -->'), /Nested/);
  assert.throws(() => parseFields('<!-- mdform:field id="a" -->oops'), /Unclosed/);
});
test("examples inside fenced, inline and indented code remain inert", () => {
  const field = '<!-- mdform:field id="example" -->____<!-- /mdform -->';
  assert.equal(parseFields("```md\n" + field + "\n```\n`" + field + "`\n    " + field + "\n").length, 0);
});
test("saves preserve first preedit backup, CRLF, BOM, Unicode and lock ownership", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "mdform-"));
  const file = path.join(dir, "form.md");
  try {
    const original = "\uFEFF" + source.replace(/\n/g, "\r\n");
    fs.writeFileSync(file, original);
    const saved = saveField(file, request(original, "call", "R\xE9ponse"));
    assert.equal(fs.readFileSync(path.join(dir, "form.preedit.md"), "utf8"), original);
    assert.ok(saved.text.includes("\r\n"));
    const second = saveField(file, request(saved.text, "done", true));
    assert.equal(fs.readFileSync(path.join(dir, "form.preedit.md"), "utf8"), original);
    assert.throws(() => saveField(file, request(original, "call", "stale")), /changed/);
    fs.writeFileSync(file + ".mdform.lock", "owned by other process");
    assert.throws(() => saveField(file, request(second.text, "call", "busy")), /progress/);
    assert.equal(fs.readFileSync(file + ".mdform.lock", "utf8"), "owned by other process");
    fs.unlinkSync(file + ".mdform.lock");
    assert.throws(() => saveField(path.join(dir, "form.preedit.md"), request(original, "call", "no")), /original/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
test("failed backup leaves source untouched and missing permissions propagate", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "mdform-"));
  const file = path.join(dir, "form.md");
  try {
    fs.writeFileSync(file, source);
    fs.mkdirSync(path.join(dir, "form.preedit.md"));
    assert.throws(() => saveField(file, request(source, "call", "bad")));
    assert.equal(fs.readFileSync(file, "utf8"), source);
    assert.equal(fs.existsSync(file + ".mdform.lock"), false);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
test("renderer makes only valid marked fields interactive with escaped values", async () => {
  const result = await renderMarkdown({ markdown: source, documentPath: "/tmp/form.md", resolveAsset: (_p, v) => v, diagramRenderers: {} });
  assert.equal((result.body.match(/data-mdform-id=/g) || []).length, 3);
  assert.match(result.body, /class="mdform-region/);
  assert.equal((result.body.match(/mdform-empty/g) || []).length, 2);
  assert.equal(result.formRevision, hash(source));
  assert.doesNotMatch(result.body, /MDFORM[0-9a-f]/);
  const invalid = await renderMarkdown({ markdown: source + source, documentPath: "/tmp/form.md", resolveAsset: (_p, v) => v, diagramRenderers: {} });
  assert.match(invalid.formError, /duplicate/);
  assert.doesNotMatch(invalid.body, /data-mdform-id=/);
});

test("shared form UI and preview retain a field viewport anchor across rerender", () => {
  const shared = fs.readFileSync(path.join(__dirname, "../../assets/js/preview/form_fields.js"), "utf8");
  const preview = fs.readFileSync(path.join(__dirname, "../media/preview.js"), "utf8");
  assert.match(shared, /captureScrollAnchor\(element\)/);
  assert.match(shared, /restoreScrollAnchor/);
  assert.match(shared, /mdform-region\.mdform-empty\{font-size:calc\(1em - 2pt\)\}/);
  assert.match(preview, /MdForm\?\.restoreScrollAnchor\(content\)/);
});

test('host rejects untrusted, dirty and wrong-document requests before writing', async () => {
  const Module = require('node:module');
  const originalLoad = Module._load;
  const workspace = {isTrusted: true, textDocuments: []};
  let PreviewCoordinator;
  try {
    Module._load = function(id, ...args) {
      return id === 'vscode' ? {workspace} : originalLoad.call(this, id, ...args);
    };
    ({PreviewCoordinator} = require('../src/previewCoordinator'));
  } finally { Module._load = originalLoad; }
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mdform-host-'));
  const file = path.join(dir, 'form.md');
  fs.writeFileSync(file, source);
  const uri = {scheme:'file', fsPath:file, toString:()=>file};
  const messages = [];
  const surface = {uri, formRevision:hash(source), webview:{postMessage:m=>messages.push(m)}};
  const save = message => PreviewCoordinator.prototype.saveFormField.call({}, surface, message);
  const message = {...request(source,'call','Host saved'), documentKey:file};
  try {
    workspace.isTrusted = false;
    await save(message);
    assert.match(messages.pop().error,/trusted/);
    workspace.isTrusted = true;
    workspace.textDocuments = [{uri, isDirty:true}];
    await save(message);
    assert.match(messages.pop().error,/unsaved/);
    workspace.textDocuments = [];
    await save({...message, documentKey:'/different.md'});
    assert.match(messages.pop().error,/preview changed/);
    assert.equal(fs.readFileSync(file,'utf8'),source);
    assert.equal(fs.existsSync(path.join(dir,'form.preedit.md')),false);
    await save(message);
    assert.equal(messages.pop().ok,true);
    assert.equal(parseFields(fs.readFileSync(file,'utf8'))[0].value,'Host saved');
  } finally {fs.rmSync(dir,{recursive:true,force:true});}
});


test('an unchanged save creates neither a backup nor a lingering lock', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mdform-noop-'));
  const file = path.join(dir, 'form.md');
  try {
    fs.writeFileSync(file, source);
    assert.equal(saveField(file, request(source, 'call', '')).unchanged, true);
    assert.deepEqual(fs.readdirSync(dir), ['form.md']);
    assert.equal(fs.readFileSync(file, 'utf8'), source);
  } finally {fs.rmSync(dir, {recursive:true, force:true});}
});
