# Fillable Markdown in mdexplore and mdext

Markdown previews remain read-only except for explicitly marked form regions and user-invoked insertion of new controls. Gray backgrounds with dashed borders identify editable values. Empty fields show a smaller **Click to fill in** placeholder; saved values use the normal document type size. Click a gray value, type, then **Save** or **Cancel**; checkboxes save as soon as they are toggled. Tab and Enter work with the controls. Ctrl/Cmd+Enter saves multiline notes. Escape cancels an open editor.

Save writes back to the original local `.md` or `.markdown` file. On the **first changed save**, `name.preedit.md` is created beside it with the original bytes. Later saves retain that first snapshot; it is not a rolling undo file. Cancel and unchanged saves create no backup. To begin a new backup cycle, deliberately move the existing backup into your history folder first. Backup previews cannot be saved through form controls.

## Add controls from the preview (0.1.18)

Select text, then right-click **Add text field** or **Add checkbox**. The selected text stays intact; a space and the new control are inserted immediately after it. A popup lets you accept or change the visible text. Text fields default to **Click to fill in**; checkboxes default to **Done**. The text-field choice is an empty-field placeholder, not an answer. Empty-field text is two points smaller than surrounding text; entered answers retain the normal size.

Within one Markdown table cell, the menu also offers **Add text field in new column** and **Add checkbox in new column**. A column is inserted immediately to the right of the selected cell's column, or appended if it is already the final column. The chosen control goes in the selected row; the other new cells are blank. The header and delimiter row are updated, and existing column contents/alignment are preserved. Selecting a header puts the control in the new header cell.

The popup's **Add** action saves immediately using the same first `.preedit.md` backup and stale-file checks as other form saves. **Cancel** writes nothing. Labels may contain up to 200 characters on one line. A blank label uses the default. Generated IDs are unique. Column actions appear only for a selection inside one table cell.

Selection mapping is conservative: select contiguous text within one paragraph, heading, list item or table cell. Code and existing form controls are excluded. Selections spanning formatting boundaries, ambiguous source matches, HTML tables and ragged pipe tables can be rejected with an explanatory error; select a shorter plain-text span or normalize the source table. A failed insertion never replaces the selected text or saves a partial table.

For a custom empty-field placeholder, the generated marker includes an optional attribute:

```markdown
Name <!-- mdform:field id="example-name" placeholder="Enter name" -->____<!-- /mdform -->
```

Quotes and special characters in that attribute are HTML-escaped; pipes and backticks are stored as numeric entities to preserve table parsing. Checkbox labels are stored beside `[ ]` as escaped plain text.

## Authoring regions

Tags are HTML comments: ordinary Markdown viewers hide the markers and display the saved content. No plugin is required to read the document. IDs must be unique within a document, start with a letter and contain only letters, digits, underscores or hyphens (maximum 64 characters). Do not nest regions. Examples inside code blocks stay inert.

Single-line fields work within table cells:

```markdown
| Item | Answer |
|---|---|
| Call date | <!-- mdform:field id="call-date" -->____<!-- /mdform --> |
```

Free-form notes use a standalone multiline region:

```markdown
<!-- mdform:text id="call-notes" -->
____
<!-- /mdform -->
```

Checklists designate each checkbox separately:

```markdown
<!-- mdform:check id="follow-up-complete" -->
- [ ] Follow-up completed
<!-- /mdform -->
```

Use `____` for an empty answer. Fields store **plain text**: Markdown punctuation is escaped so pipes do not split table cells and typed text cannot introduce executable HTML or additional form regions. The text limit is 20,000 characters per region. Use standalone multiline notes for longer answers, not multiline text inside tables. Keep region markers outside blockquotes, nested lists and indented code.

## Save behavior and limits

- Each value edit changes only the requested region; a checkbox toggle changes only its `[ ]`/`[x]` character. Explicit insertion adds a region and, when requested, a full table column. Surrounding content, line endings and UTF-8 BOM are preserved.
- Saving a field or toggling a checkbox preserves that control's position in the preview viewport, including when mdext rerenders after its file watcher observes the save.
- Both hosts check the original source hash, unique field ID and field type. Duplicate, nested or unmatched markers disable form editing for the document. Never rename IDs while a form is open.
- A changed file is rejected rather than overwritten. A failed text save keeps the draft in its editor: copy it, cancel, refresh, review the current file and try again. Preview refresh is deferred while an editor is open. Unsaved drafts are not persisted across app shutdown or webview disposal.
- mdext requires a trusted VS Code workspace, a local file and no unsaved changes in that file's source editor. Save or discard source edits before using the form. Source editor undo is not a form-save history; use the backup to recover the original.
- The hosts share an exclusive `.mdform.lock`, make the backup before changing the source and replace the source through a temporary file in the same directory. Failed saves leave the original intact. External editors do not honor this lock; avoid editing the same document simultaneously in other applications.
- A leftover lock after a crash blocks saves. Confirm neither host is saving, then remove that document's `.mdform.lock`. Never remove another active writer's lock.
- Read-only folders or backup failures produce an error. Existing backups are never overwritten. Symlink/invalid backup targets are rejected.
- Exported PDFs show saved field values. Editor dialogs and gray affordances are hidden in print. PDFs are snapshots, not new fillable PDFs.

## User acceptance test

Use a **copy** of `CRA-Call-Script-Kashana-Indromohan.md` from the insolvency repository for practice so fictional results never enter the actual call record.

1. In VS Code, reload after installing mdext 0.1.18. Open the copy with **mdExt Markdown Preview**. In desktop mdexplore, restart and open the same copy.
2. Scroll to **Call results**. Confirm response cells, checkboxes and the notes area are gray; the read-aloud script and financial figures are ordinary text.
3. Enter a test call date and Save. Confirm the source shows that answer, the adjacent `.preedit.md` exactly preserves the original, and the preview remains at the same reading position.
4. Check and uncheck a task. Enter two paragraphs in the notes area, including `|`, `%20`, accents and backticks. Save, close/reopen the preview and confirm values persist.
5. Type another answer and Cancel. Confirm the file is unchanged. Confirm later saves did not replace the original backup.
6. Open a field, change the file externally, then try Save. Confirm an error, preserved draft and preserved external change. Cancel and refresh before continuing.
7. In mdext, make an unsaved source edit and try saving a field. Confirm the save is refused. Save/discard the source edit before retrying.
8. View the copy in a conventional Markdown preview and export a PDF. Confirm answers remain readable and tags are hidden. Discard the practice files when done.

The automated tests use temporary documents only. They do not record a real call, complete a follow-up or establish a CRA hold. User acceptance remains for the user to confirm.

## Maintenance

`mdexplore_app/form_fields.py` and `mdext/src/formFields.js` implement the same grammar and save contract; their companion `form_insertion.py` / `formInsertion.js` modules implement selection and table insertion. Keep both in step. `assets/js/preview/form_fields.js` is the shared UI source; the mdext build copies it to `mdext/media/formFields.js`. Do not edit the generated copy independently. Qt uses a narrow QWebChannel bridge; mdext uses webview messages with host-side validation.

Tests: Python `test_form_fields.py` and real Chromium `test_form_preview.py`; Node `mdext/test/formFields.test.js`. Run the full preview regression suite after renderer/UI changes. Preserve pagination, source-line metadata, search and highlighting behavior.


## Validation record — September 24, 2026

- Node suite: 50 tests passed, including bounded edits, backup preservation, stale requests, lock ownership, rendering, viewport-anchor wiring and host trust/dirty-buffer/document checks.
- Python form tests: seven writer/parser tests and three real Chromium/Qt preview tests passed, including placeholder sizing and viewport retention after text and checkbox saves. A temporary copy of the actual CRA call script exercised all 41 controls at render time and successfully saved a response, checkbox and multiline notes while retaining the first backup. The actual case record remained blank and unchecked. Gray response cells were visually inspected.
- A real VS Code extension-host test passed for UTF-8 BOM source hashing, disk save, exact first backup, automatic clean-buffer reload and rejection of unsaved source changes. The integration runner is `mdext/test/integration/forms.js`; run with an isolated workspace/user-data directory, not the case repository.
- Shell syntax, Python compilation, Node checks and extension build passed. Existing user changes in `assets/templates/preview/document.html` and `tests/test_template_assets.py` were preserved.
- The full Python discovery run was attempted twice and encountered native Qt crashes. Running the 34 test modules separately yielded 27 passing modules. Outstanding modules: hfind settings/progress assertions; PDF search-scope crash; PDF marker-identity assertion; PDF window geometry assertion; preview regressions missing `test/testdoc.md`; recent-root count assertion; main-window Qt crash. The recent-root failure and main-window crash reproduced with the pre-change `mdexplore.py`; other listed failures involve unchanged modules or absent fixtures. These prevent a clean full-suite claim. They are not concealed as passing checks.
- User acceptance is still pending. Use the practice-copy checklist above after reloading VS Code or restarting mdexplore. Changes remain uncommitted.


### Insertion acceptance checks

On a practice copy, select a word in ordinary text. Confirm only the two regular insertion actions appear, accept the default text-field placeholder, and verify the word plus its surrounding text remain unchanged. Repeat for a checkbox, confirming **Done** is prefilled and can be replaced. Cancel once and verify no write occurs. In a two-column table, select a body cell in the first column and add a new column; verify the previous right-hand column moves right intact. Repeat from the final column to test appending. Reopen the file and confirm labels persist and the first backup has not changed. Never put practice answers into the actual CRA record.


## Version 0.1.18 validation and handoff

The September 24 follow-up adds selection-based insertion in both context menus, table-column insertion, configurable placeholder/checkbox labels and an exact two-point reduction for empty-field text. Checkbox labels default to **Done**, as explicitly requested. All write operations reuse the original backup, locking, revision and dirty-buffer checks. Qt refreshes after structural insertion or line-count changes to keep source positions accurate; ordinary single-line value edits retain their viewport anchor.

Validation: 54 Node tests passed; 16 targeted Python tests passed, including the desktop menu/popup, the actual mdext webview HTML/JS menu in Chromium, escaped-pipe columns, middle/last-column insertion, custom labels, cancelling, backups, stale source and viewport retention. A real VS Code extension-host test passed field insertion and clean-buffer reload, with the original backup retained and dirty-buffer writes rejected. The full Python discovery command was attempted again and crashed in the unchanged PDF search-scope tests; the broader baseline limitations above remain. No case record was filled with test data. Changes remain uncommitted, and user acceptance is pending.
