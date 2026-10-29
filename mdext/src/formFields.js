"use strict";
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { TextDecoder } = require("node:util");
const hash = (text) => crypto.createHash("sha256").update(text, "utf8").digest("hex");
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const punctuation = /[\\`*_{}\[\]<>()#+.!|~$>-]/g;
const decode = (s) => s.replace(/\\([\\`*_{}\[\]<>()#+.!|~$>-])/g, "$1");
function decodeEntities(s) { return s.replace(/&(?:amp|lt|gt|quot|apos|#39|#x27|#124|#96);/g, c => ({"&amp;":"&","&lt;":"<","&gt;":">","&quot;":'"',"&apos;":"'","&#39;":"'","&#x27;":"'","&#124;":"|","&#96;":"`"}[c])); }
function codeRanges(source) {
  const ranges = [];
  let fence = null;
  let offset = 0;
  for (const line of source.match(/[^\n]*\n|[^\n]+$/g) || []) {
    const m2 = /^( {0,3})(`{3,}|~{3,})(.*)/.exec(line);
    if (fence) {
      if (m2 && m2[2][0] === fence.char && m2[2].length >= fence.length && !m2[3].trim()) {
        ranges.push([fence.start, offset + line.length]);
        fence = null;
      }
    } else if (m2) fence = { start: offset, char: m2[2][0], length: m2[2].length };
    else if (/^(?: {4}|\t)/.test(line)) ranges.push([offset, offset + line.length]);
    offset += line.length;
  }
  if (fence) ranges.push([fence.start, source.length]);
  const ticks = /(?<!\\)(`+)[\s\S]*?(?<!\\)\1/g;
  let m;
  while (m = ticks.exec(source)) ranges.push([m.index, ticks.lastIndex]);
  return ranges;
}
function parseFields(source) {
  if (!source.includes("mdform")) return [];
  const ranges = codeRanges(source);
  const fields = [];
  const ids = /* @__PURE__ */ new Set();
  let active = null;
  const marker = /<!--\s*(?:mdform:(field|text|check)\s+id="([A-Za-z][A-Za-z0-9_-]{0,63})"(?:\s+placeholder="([^"<>]*)")?\s*|\/mdform\s*)-->/g;
  let m;
  while (m = marker.exec(source)) {
    if (ranges.some(([a, b]) => m.index >= a && m.index < b)) continue;
    if (m[1]) {
      if (active || ids.has(m[2])) throw new Error("Nested or duplicate mdform markers. Fix the source before editing.");
      active = { id: m[2], kind: m[1], start: m.index, placeholder: decodeEntities(m[3] || "Click to fill in"), valueStart: marker.lastIndex };
      ids.add(m[2]);
    } else {
      if (!active) throw new Error("Unmatched mdform closing marker.");
      const raw = source.slice(active.valueStart, m.index);
      let value;
      if (active.kind === "check") {
        const check = /^(\s*(?:- +)?\[)([ xX])(\][^\r\n]*\s*)$/.exec(raw);
        if (!check) throw new Error("A check region must contain one [ ] or [x] checkbox, optionally a task-list item.");
        value = check[2].toLowerCase() === "x";
        active.label = decode(check[3].slice(1).trim()) || active.id;
        active.checkOffset = active.valueStart + check[1].length;
      } else {
        if (active.kind === "field" && /[\r\n]/.test(raw)) throw new Error("Use mdform:text for multiline fields.");
        value = decode(raw.trim());
        if (value === "____") value = "";
      }
      fields.push({ ...active, valueEnd: m.index, end: marker.lastIndex, value });
      active = null;
    }
  }
  if (active) throw new Error("Unclosed mdform marker.");
  return fields;
}
function applyField(source, request) {
  if (!request || request.revision !== hash(source)) throw new Error("The file changed. Refresh and review your draft before saving again.");
  const f = parseFields(source).find((f2) => f2.id === request.id);
  if (!f) throw new Error("This is not a designated editable field.");
  if (f.kind === "check") {
    if (typeof request.value !== "boolean") throw new Error("Checkbox value must be true or false.");
    return source.slice(0, f.checkOffset) + (request.value ? "x" : " ") + source.slice(f.checkOffset + 1);
  }
  if (typeof request.value !== "string" || request.value.length > 2e4 || /\x00|<!--|-->|mdform:/i.test(request.value)) throw new Error("Invalid field text (limit 20,000 characters; form markers are not allowed).");
  let value = request.value.replace(/\r\n?/g, "\n");
  if (f.kind === "field" && value.includes("\n")) throw new Error("This field accepts a single line.");
  value = value.replace(punctuation, "\\$&") || "____";
  if (f.kind === "text") {
    const eol = source.includes("\r\n") ? "\r\n" : "\n";
    value = eol + value.replace(/\n/g, eol) + eol;
  }
  return source.slice(0, f.valueStart) + value + source.slice(f.valueEnd);
}
function prepareForms(source) {
  let fields;
  try {
    fields = parseFields(source);
  } catch (error) {
    return { source, fields: [], error: error.message };
  }
  const token = "MDFORM" + crypto.randomBytes(12).toString("hex");
  let prepared = source;
  for (let i = fields.length - 1; i >= 0; i--) {
    const f = fields[i];
    f.token = token + i + "END";
    prepared = prepared.slice(0, f.start) + f.token + "\n".repeat((source.slice(f.start, f.end).match(/\n/g) || []).length) + prepared.slice(f.end);
  }
  return { source: prepared, fields, revision: hash(source) };
}
function finishForms(body, forms) {
  for (const f of forms.fields) {
    const data = `data-mdform-id="${f.id}" data-mdform-kind="${f.kind}" data-mdform-value="${escapeHtml(String(f.value))}" data-mdform-placeholder="${escapeHtml(f.placeholder)}"`;
    const emptyClass = f.value ? "" : " mdform-empty";
    const control = f.kind === "check" ? `<label class="mdform-region"><input type="checkbox" ${data} ${f.value ? "checked" : ""}> ${escapeHtml(f.label)}</label>` : `<button type="button" class="mdform-region mdform-${f.kind}${emptyClass}" ${data} aria-label="Edit ${f.id}">${escapeHtml(f.value || f.placeholder)}</button>`;
    body = body.replace(f.token, control);
  }
  return body;
}
// VS Code omits a UTF-8 BOM from TextDocument.getText(). Bind saves to the
// exact disk bytes only when the rendered text still matches that source.
async function documentRevision(file, renderedSource) {
  const bytes = await fs.promises.readFile(file);
  const raw = new TextDecoder("utf-8", {fatal:true, ignoreBOM:true}).decode(bytes);
  return hash(raw.replace(/^\uFEFF/, "") === renderedSource.replace(/^\uFEFF/, "") ? raw : renderedSource);
}
function saveField(file, request, guard = () => {
}) {
  const resolved = fs.realpathSync(file);
  if (!/\.(md|markdown)$/i.test(resolved) || /\.preedit\.md$/i.test(resolved)) throw new Error("Only original local Markdown documents can be edited.");
  const lock = resolved + ".mdform.lock";
  let fd;
  let temp;
  try {
    fd = fs.openSync(lock, "wx", 384);
    guard();
    const bytes = fs.readFileSync(resolved);
    const source = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes);
    const next = request.operation === "insert" ? require("./formInsertion").applyInsert(source, request) : applyField(source, request);
    if (next === source) return { text: source, revision: hash(source), unchanged: true };
    const backup = resolved.replace(/\.(md|markdown)$/i, ".preedit.md");
    try {
      fs.copyFileSync(resolved, backup, fs.constants.COPYFILE_EXCL);
    } catch (e) {
      if (e.code !== "EEXIST") throw e;
      const st = fs.lstatSync(backup);
      if (!st.isFile() || st.isSymbolicLink()) throw new Error("Backup path is not a regular file.");
    }
    guard();
    if (!fs.readFileSync(resolved).equals(bytes)) throw new Error("The file changed during save. Nothing was overwritten.");
    temp = path.join(path.dirname(resolved), "." + path.basename(resolved) + ".mdform-" + crypto.randomBytes(8).toString("hex"));
    const out = fs.openSync(temp, "wx", fs.statSync(resolved).mode & 511);
    try {
      fs.fchmodSync(out, fs.statSync(resolved).mode & 511);
      fs.writeFileSync(out, next, "utf8");
      fs.fsyncSync(out);
    } finally {
      fs.closeSync(out);
    }
    if (!fs.readFileSync(resolved).equals(bytes)) throw new Error("The file changed during save. Nothing was overwritten.");
    fs.renameSync(temp, resolved);
    temp = null;
    return { text: next, revision: hash(next), backup };
  } catch (e) {
    if (e.code === "EEXIST") throw new Error("Another form save is in progress (.mdform.lock). Retry shortly.");
    throw e;
  } finally {
    if (temp) try {
      fs.unlinkSync(temp);
    } catch {
    }
    if (fd !== void 0) {
      fs.closeSync(fd);
      fs.unlinkSync(lock);
    }
  }
}
module.exports = { codeRanges, hash, parseFields, applyField, prepareForms, finishForms, saveField, documentRevision };
