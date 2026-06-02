# pdfexplore TO-DO

## Target

`pdfexplore` should feel like the PDF analog of `mdexplore`: same root-navigation
shell, same scope rules where PDFs make sense, same sidecar philosophy, and the
same habit of preserving work state without modifying source documents.

## Audit Summary

### Already working

- PDF-only tree with directory navigation.
- Embedded local `pdf.js` preview.
- File highlight colors with sidecar persistence.
- Preview text highlights with two-tier sidecar persistence using
  `.pdfexplore-highlighting.json`.
- Multi-view tabs with per-document session restore.
- Single-view return-to-document scroll/session restore.
- Preview zoom in/out/reset with `mdexplore`-matching shortcuts and overlay.
- Copy actions with directory metadata merge.
- Auto-refresh when the currently previewed PDF changes on disk.

### Main UX gaps vs `mdexplore`

1. Top-left shell is incomplete.
   - Missing `Recent`.
   - Missing `PDF`.
   - Missing `Edit`.
   - Current order does not match the canonical `mdexplore` shell.

2. Root persistence is behind `mdexplore`.
   - `~/.pdfexplore.cfg` still behaves like a legacy single-path file.
   - No JSON payload with `default_root` + `recent_roots`.
   - No lock file / multi-instance-safe reload behavior.
   - No 30-second dwell rule before recording departed roots.

3. Search scope does not match `mdexplore`.
   - Current search scans only direct-child PDFs of one selected directory.
   - Target behavior is visible-tree scope: current root plus expanded branches.
   - Search should rerun automatically when tree visibility changes.

4. Search/tree styling parity is incomplete.
   - Effective-root row is not wired like `mdexplore`’s bold aqua/yellow state.
   - Search-hit summary for the active scope is not surfaced the same way.
   - Filename-term match styling parity needs confirmation after search rework.

5. Highlight/preview parity needs manual UX confirmation.
   - Two-tier preview highlight colors now use the shared `mdexplore` palette.
   - Scroll/session return is wired, but needs manual feel-check in real PDFs.
   - Zoom behavior is implemented, but needs manual parity review against
     `mdexplore`.

6. External-file actions are not settled.
   - `Edit` analog needs a defined PDF-safe behavior.
   - `PDF` button needs a PDF-specific analog or explicit no-op/disabled behavior
     that still preserves the shell layout contract.

7. Tests do not yet cover the missing shell/config/search contracts.

## Work Plan

### Phase 1

- [x] Add `Recent` button and dropdown before `^`.
- [x] Move `pdfexplore` config to JSON payload with legacy plain-text read support.
- [x] Add recent-root storage, menu presentation, reload-on-open, dwell timing,
      cap of 35, and lock-file writes.
- [x] Add tests for recent-root behavior and config persistence.

### Phase 2

- [x] Rework search to use visible-tree PDF scope instead of one-directory
      direct children.
- [x] Rerun active search automatically on expand/collapse/root/scope changes.
- [ ] Manual UX verification for effective-root and search-hit pill styling.
      Current wiring now uses the shared effective-scope tree model path.

### Phase 3

- [x] Persist preview text highlights in `.pdfexplore-highlighting.json`.
- [x] Use the shared two-tier preview highlight colors from `mdexplore`.
- [x] Preserve scroll/session position when returning to a PDF.
- [x] Implement preview zoom in/out/reset with the `mdexplore` shortcut set.
- [ ] Manual UX verification of highlight visuals, scroll return, and zoom feel.

### Phase 4

- [ ] Define and implement `PDF` button behavior for a PDF-native workflow.
- [ ] Define and implement `Edit` behavior for PDFs.
- [ ] Add tests for those controls once behavior is finalized.

### Phase 5

- [ ] Review whether more `mdexplore_app` helpers should be shared to reduce
      divergence between explorers.
