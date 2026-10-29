"use strict";
const crypto = require("node:crypto");
const { hash, parseFields, codeRanges } = require("./formFields");
const punctuation = /[\\`*_{}\[\]<>()#+.!|~$>\-]/;
const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
function cells(line) {
  const cuts = [];
  for (let i = 0; i < line.length; i++) if (line[i] === "|") {
    let slashes = 0;
    for (let j = i - 1; j >= 0 && line[j] === "\\"; j--) slashes++;
    if (slashes % 2 === 0) cuts.push(i);
  }
  if (!cuts.length) return [];
  const bounds = [-1, ...cuts, line.length], result = bounds.slice(1).map((b, i) => [bounds[i] + 1, b]);
  if (!line.slice(0, cuts[0]).trim()) result.shift();
  if (!line.slice(cuts.at(-1) + 1).trim()) result.pop();
  return result;
}
function applyInsert(source, request) {
  if (request.revision !== hash(source)) throw new Error("The file changed. Refresh before inserting a field.");
  const kind = request.kind;
  let label = request.label ?? (kind === "check" ? "Done" : "Click to fill in");
  if (!["field", "check"].includes(kind) || typeof label !== "string" || label.length > 200 || /[\r\n\x00]|<!--|-->|mdform:/i.test(label)) throw new Error("Use a single-line label of up to 200 characters without form markers.");
  label = label.trim() || (kind === "check" ? "Done" : "Click to fill in");
  const selection = request.selection || {}, text = selection.text;
  if (typeof text !== "string" || !text.trim() || text.length > 2e4) throw new Error("Select text in one paragraph or table cell first.");
  const lines = source.match(/[^\n]*\n|[^\n]+$/g) || [];
  const { startLine: start, endLine: end } = selection;
  if (!Number.isInteger(start) || !Number.isInteger(end) || !(0 <= start && start < end && end <= lines.length)) throw new Error("Cannot locate this selection in Markdown. Select a shorter text span.");
  const sumTo = (n) => lines.slice(0, n).join("").length;
  let offset = sumTo(start), segment = lines.slice(start, end).join(""), tableRows = null, header, top, row, column, selectedRow;
  if (selection.table != null) {
    ({ startLine: top, row, column } = selection.table);
    if ([top, row, column].some((n) => !Number.isInteger(n) || n < 0) || top + 1 >= lines.length) throw new Error("Cannot locate the selected Markdown table.");
    header = cells(lines[top].replace(/[\r\n]+$/, ""));
    const sep = lines[top + 1].replace(/[\r\n]+$/, ""), separators = cells(sep);
    if (!header.length || header.length !== separators.length || column >= header.length || separators.some(([a2, b2]) => !/^\s*:?-+:?\s*$/.test(sep.slice(a2, b2)))) throw new Error("Only ordinary pipe tables support column insertion.");
    tableRows = [top, top + 1];
    let i = top + 2;
    while (i < lines.length && lines[i].trim() && cells(lines[i].replace(/[\r\n]+$/, "")).length) tableRows.push(i++);
    selectedRow = row === 0 ? top : top + row + 1;
    if (!tableRows.includes(selectedRow) || selectedRow === top + 1) throw new Error("The selected table row is not present in the source.");
    const bounds = cells(lines[selectedRow].replace(/[\r\n]+$/, ""));
    if (bounds.length !== header.length) throw new Error("Normalize ragged table rows in the source before inserting.");
    const [a, b] = bounds[column];
    offset = sumTo(selectedRow) + a;
    segment = lines[selectedRow].slice(a, b);
  }
  const pattern = Array.from(text, (c) => (punctuation.test(c) ? "\\\\?" : "") + escapeRegex(c)).join("");
  const matches = [...segment.matchAll(new RegExp(pattern, "g"))];
  const occurrence = selection.occurrence ?? 0, total = selection.occurrences ?? 1;
  if (!Number.isInteger(occurrence) || matches.length !== total || occurrence < 0 || occurrence >= matches.length) throw new Error("Selection cannot be mapped unambiguously. Select a shorter plain-text span within one cell or paragraph.");
  const match = matches[occurrence], begin = offset + match.index, finish = begin + match[0].length;
  const fields = parseFields(source);
  if (codeRanges(source).some(([a, b]) => a < finish && b > begin) || fields.some((f) => f.start < finish && f.end > begin)) throw new Error("Choose ordinary text outside code and existing form controls.");
  const id = "field-" + crypto.randomBytes(8).toString("hex");
  const encoded = Array.from(label, (c) => punctuation.test(c) ? "\\" + c : c).join("");
  const element = kind === "field" ? `<!-- mdform:field id="${id}" placeholder="${escapeHtml(label).replace(/\|/g, "&#124;").replace(/`/g, "&#96;")}" -->____<!-- /mdform -->` : `<!-- mdform:check id="${id}" -->[ ] ${encoded}<!-- /mdform -->`;
  let updated;
  if (!request.newColumn) updated = source.slice(0, finish) + " " + element + source.slice(finish);
  else {
    if (!tableRows) throw new Error("Select text within one table cell to add a column.");
    for (const index of [...tableRows].reverse()) {
      const bounds = cells(lines[index].replace(/[\r\n]+$/, ""));
      if (bounds.length !== header.length) throw new Error("Normalize ragged table rows in the source before inserting.");
      const pos = bounds[column][1], value = index === top + 1 ? "---" : index === selectedRow ? element : "";
      lines[index] = lines[index].slice(0, pos) + " | " + value + " " + lines[index].slice(pos);
    }
    updated = lines.join("");
  }
  parseFields(updated);
  return updated;
}
module.exports = { applyInsert, cells };
