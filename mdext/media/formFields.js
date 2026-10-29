// Shared form UI; authoritative write validation remains in each host.
(() => {
  "use strict";
  if (window.MdForm) return;
  let config = {}, dialog = null, editor = null, active = null, pending = false, deferred = null, scrollAnchor = null, scrollAnchorExpiry = null;
  const style = document.createElement("style");
  style.textContent = `.mdform-region{background:rgba(128,128,128,.13)!important;color:inherit!important;border:1px dashed #888!important;border-radius:4px;padding:3px 7px;cursor:pointer;font:inherit;text-align:left;white-space:pre-wrap;overflow-wrap:anywhere;max-width:100%}.mdform-region.mdform-empty{font-size:calc(1em - 2pt)}.mdform-text{display:block;width:100%;min-height:3em}.mdform-region:hover,.mdform-region:focus-visible{outline:2px solid #888;outline-offset:2px;background:rgba(128,128,128,.23)!important}.mdform-region input{accent-color:#777}.mdform-dialog{color:CanvasText;background:Canvas;border:1px solid #888;border-radius:8px;padding:20px;width:min(650px,85vw);box-shadow:0 8px 40px #0006}.mdform-dialog::backdrop{background:#0005}.mdform-dialog label{display:block;margin-bottom:10px;font-weight:bold}.mdform-dialog textarea,.mdform-dialog input{box-sizing:border-box;width:100%;padding:10px;color:CanvasText;background:Canvas;border:1px solid #888;font:inherit}.mdform-dialog textarea{min-height:180px;resize:vertical}.mdform-actions{display:flex;gap:10px;justify-content:flex-end;margin-top:12px}.mdform-actions button{padding:6px 16px;font:inherit;cursor:pointer}.mdform-message{white-space:pre-wrap;margin:10px 0;font-size:.9em}.mdform-toast{position:fixed;bottom:16px;right:16px;z-index:2147483647;background:Canvas;color:CanvasText;border:1px solid #888;border-radius:4px;padding:10px;max-width:70vw}@media print{.mdform-dialog,.mdform-toast{display:none!important}.mdform-region{border:0!important;background:transparent!important;padding:0;cursor:default}}`;
  document.head.appendChild(style);
  function notify(text) {
    let n = document.querySelector(".mdform-toast");
    if (!n) {
      n = document.createElement("div");
      n.className = "mdform-toast";
      n.setAttribute("role", "status");
      document.body.appendChild(n);
    }
    n.textContent = text;
  }
  function editing(value) {
    config.editing?.(value);
  }
  function captureScrollAnchor(element) {
    const rect = element?.getBoundingClientRect?.();
    if (!rect || !Number.isFinite(rect.top)) return;
    scrollAnchor = {
      id: element.dataset.mdformId,
      top: rect.top,
      documentKey: config.documentKey,
    };
    if (scrollAnchorExpiry) clearTimeout(scrollAnchorExpiry);
    scrollAnchorExpiry = setTimeout(() => { scrollAnchor = null; scrollAnchorExpiry = null; }, 2500);
  }
  function restoreScrollAnchor(root = document) {
    if (!scrollAnchor) return false;
    if (scrollAnchor.documentKey !== config.documentKey) {
      scrollAnchor = null;
      return false;
    }
    const element = Array.from(root.querySelectorAll?.("[data-mdform-id]") || [])
      .find((candidate) => candidate.dataset.mdformId === scrollAnchor.id);
    const rect = element?.getBoundingClientRect?.();
    if (!rect || !Number.isFinite(rect.top)) return false;
    const delta = rect.top - scrollAnchor.top;
    if (Math.abs(delta) > .5) window.scrollBy({ top: delta, left: 0, behavior: "auto" });
    return true;
  }
  function close() {
    if (pending) return;
    dialog?.remove();
    dialog = null;
    editor = null;
    active = null;
    editing(false);
    if (deferred) {
      const cb = deferred;
      deferred = null;
      cb();
    }
  }
  function submit(element, value) {
    if (pending) return;
    captureScrollAnchor(element);
    pending = true;
    active = element;
    editing(true);
    element.disabled = true;
    if (dialog) {
      dialog.querySelectorAll("button").forEach((b) => b.disabled = true);
      dialog.querySelector(".mdform-message").textContent = "Saving...";
    }
    try {
      config.send(element.mdformInsertion
        ? {...element.mdformInsertion, label:value, revision:config.revision, documentKey:config.documentKey}
        : { id: element.dataset.mdformId, value, revision: config.revision, documentKey: config.documentKey });
    } catch (e) {
      result({ ok: false, error: e.message });
    }
  }
  function result(message) {
    if (!pending) return;
    pending = false;
    if (active) active.disabled = false;
    if (!message.ok) {
      if (active?.type === "checkbox") active.checked = active.dataset.mdformValue === "true";
      const error = message.error || "Save failed.";
      if (dialog) {
        dialog.querySelectorAll("button").forEach((b) => b.disabled = false);
        dialog.querySelector(".mdform-message").textContent = error + " Your draft is still here; copy it before refreshing.";
      } else {
        close();
        notify(error);
      }
      return;
    }
    config.revision = message.revision;
    config.reloading = Boolean(message.reload);
    if (config.reloading) document.querySelectorAll("[data-mdform-id]").forEach(el => el.disabled = true);
    if (active) {
      const value = active.type === "checkbox" ? active.checked : editor.value;
      active.dataset.mdformValue = String(value);
      if (active.type === "checkbox") active.toggleAttribute("checked", value);
      else {
        active.textContent = value || active.dataset.mdformPlaceholder || "Click to fill in";
        active.classList.toggle("mdform-empty", !value);
      }
      restoreScrollAnchor();
    }
    close();
    notify(message.unchanged ? "No change; nothing written." : "Saved. Original snapshot retained in .preedit.md.");
  }
  function open(element) {
    if (dialog || pending || config.reloading || !config.send) return;
    active = element;
    editing(true);
    dialog = document.createElement("dialog");
    dialog.className = "mdform-dialog";
    const label = document.createElement("label");
    label.textContent = element.closest("tr")?.cells[0]?.textContent || "Edit " + element.dataset.mdformId.replaceAll("-", " ");
    label.htmlFor = "mdform-editor";
    editor = document.createElement(element.dataset.mdformKind === "text" ? "textarea" : "input");
    editor.id = "mdform-editor";
    editor.maxLength = 2e4;
    editor.value = element.dataset.mdformValue || "";
    const msg = document.createElement("div");
    msg.className = "mdform-message";
    msg.setAttribute("role", "status");
    msg.textContent = "Only this designated area will change. Save updates Markdown; Cancel leaves it unchanged.";
    const actions = document.createElement("div");
    actions.className = "mdform-actions";
    const cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    cancel.type = "button";
    cancel.addEventListener("click", close);
    const save = document.createElement("button");
    save.textContent = "Save";
    save.type = "button";
    save.addEventListener("click", () => submit(element, editor.value));
    actions.append(cancel, save);
    dialog.append(label, editor, msg, actions);
    document.body.appendChild(dialog);
    dialog.addEventListener("cancel", (e) => {
      e.preventDefault();
      close();
    });
    dialog.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey || editor.tagName === "INPUT")) {
        e.preventDefault();
        submit(element, editor.value);
      }
    });
    dialog.showModal();
    editor.focus();
  }
  document.addEventListener("click", (e) => {
    const el = e.target.closest?.("button[data-mdform-id]");
    if (el) {
      e.preventDefault();
      e.stopPropagation();
      open(el);
    }
  }, true);
  document.addEventListener("change", (e) => {
    const el = e.target;
    if (el.matches?.('input[type="checkbox"][data-mdform-id]')) {
      e.stopPropagation();
      submit(el, el.checked);
    }
  }, true);
  function selectionContext() {
    if (config.reloading) return null;
    const selected = window.getSelection();
    if (!selected || selected.isCollapsed || !selected.rangeCount) return null;
    const range = selected.getRangeAt(0), text = range.toString();
    const element = n => n.nodeType === Node.ELEMENT_NODE ? n : n.parentElement;
    const first = element(range.startContainer), last = element(range.endContainer);
    if (!text.trim() || first?.closest('.mdform-region,pre,code,dialog') || last?.closest('.mdform-region,pre,code,dialog')) return null;
    const cell = first?.closest('td,th');
    const block = cell || first?.closest('p,h1,h2,h3,h4,h5,h6,li');
    if (!block || !block.contains(range.endContainer) || (cell && cell !== last?.closest('td,th'))) return null;
    const tagged = block.closest('[data-md-line-start]');
    if (!tagged) return null;
    const before = range.cloneRange(); before.selectNodeContents(block); before.setEnd(range.startContainer, range.startOffset);
    const result = {text, occurrence:before.toString().split(text).length-1, occurrences:block.textContent.split(text).length-1,
      startLine:Number(tagged.dataset.mdLineStart), endLine:Number(tagged.dataset.mdLineEnd)};
    if (cell) {
      const table = cell.closest('table');
      if (!table?.hasAttribute('data-md-line-start')) return null;
      result.table = {startLine:Number(table.dataset.mdLineStart), row:cell.parentElement.rowIndex, column:cell.cellIndex};
    }
    return result;
  }
  function openInsert(kind, newColumn, selection) {
    if (dialog || pending || !selection) return;
    const placeholder = kind === 'check' ? 'Done' : 'Click to fill in';
    const element = document.createElement('button');
    element.dataset.mdformKind = 'field';
    element.dataset.mdformId = 'new-field';
    element.dataset.mdformValue = placeholder;
    element.mdformInsertion = {operation:'insert', kind, newColumn:Boolean(newColumn), selection};
    open(element);
    if (!dialog) return;
    dialog.querySelector('label').textContent = kind === 'check' ? 'Checkbox label' : 'Text shown when the field is empty';
    editor.maxLength = 200;
    editor.select();
    dialog.querySelector('.mdform-actions button:last-child').textContent = 'Add';
    dialog.querySelector('.mdform-message').textContent = newColumn
      ? 'Insert a column to the right, with this control in the selected row. Other cells in the new column stay blank.'
      : 'Keep the selected text and add a space followed by this control.';
  }

  window.MdForm = { selectionContext, openInsert, configure(options) {
    config = { ...config, ...options };
  }, result, restoreScrollAnchor, busy() {
    return Boolean(dialog || pending);
  }, deferRender(cb) {
    if (dialog || pending) {
      deferred = cb;
      return true;
    }
    return false;
  }, disableOrdinaryCheckboxes(root = document) {
    root.querySelectorAll('input[type="checkbox"]:not([data-mdform-id])').forEach((el) => el.disabled = true);
  } };
})();
