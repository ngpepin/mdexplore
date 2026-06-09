# pdfexplore TO-DO

## Target

`pdfexplore` should feel like the PDF analog of `mdexplore`: same root-navigation
shell, same scope rules where PDFs make sense, same sidecar philosophy, and the
same habit of preserving work state without modifying source documents.

## Current Snapshot

- Search and prefetch now run on worker pools with cancellation support.
- Text extraction uses layered cache (memory + disk) and badge visibility.
- Tree marker state merges sidecar scan + live state to reduce stale badge drops.
- Interaction-first throttling is in place (scroll and tree mutation pauses).
- Viewer right-rail search markers now build progressively with document-scoped page-text caching.
- Marker builds now include interaction-priority behavior so click-to-jump remains responsive during long scans.

## Roadmap (2026)

This roadmap is organized by execution horizon so contributors can prioritize
work without reinterpreting old phase notes.

### Now (next 1-3 weeks)

- [ ] Finalize `PDF` top-button behavior and confirm UX contract against
   `mdexplore` parity expectations.
- [ ] Run manual UX pass for search-hit/effective-root pill styling and capture
   any visual regressions.
- [ ] Run manual UX pass for highlight visibility, marker badge continuity,
   scroll return, and zoom-feel parity.
- [ ] Add focused regression tests for marker badge continuity under file switch
   and search rerun scenarios.

### Next (next 1-2 months)

- [ ] Add performance regression coverage for root changes in medium/large trees.
- [ ] Add deterministic tests for interaction-heavy scenarios where prefetch,
   marker scan, and search overlap.
- [ ] Introduce optional timing instrumentation for hot paths:
   marker sync, visible-tree listing, and search aggregation.
- [ ] Refresh top-row control tests after `PDF` behavior is finalized.

### Later (quarterly hardening)

- [ ] Evaluate more helper extraction into `mdexplore_app` to reduce divergence
   and maintenance overhead.
- [ ] Continue multi-instance soak testing for config/recent-roots lock behavior.
- [ ] Reassess cache/prefetch heuristics for very large directory trees and
   document practical tuning defaults.

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

1. Shell parity still needs final polish.
   - `Recent` and `Edit` are now present.
   - `PDF` behavior/design parity is still open.

2. Root persistence parity needs continued soak testing.
   - JSON payload and lock strategy are implemented.
   - Recent-root behavior should continue to be validated under multi-instance use.

3. Search/marker performance tuning remains ongoing.
   - Search scope now follows visible-tree behavior.
   - Marker generation throughput and click responsiveness were both improved.
   - Remaining work is mostly calibration/timing for very large PDFs and edge hardware.

4. Visual consistency needs additional regression checks.
   - Badge coexistence (search pill + marker + cache badge) should be regression-tested more thoroughly.

5. Highlight/preview parity still needs manual UX confirmation.
   - Two-tier preview highlight colors use shared palette.
   - Scroll/session return and zoom are implemented; parity feel-check still open.

6. External-file actions are partially settled.
   - `Edit` opens current document externally.
   - `PDF` button behavior remains open for long-term parity/design.

7. Test coverage gaps remain for complex UI interactions.
   - Need stronger regressions for marker badge continuity and heavy-interaction throttling.

## Detailed Phase Plan

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
- [ ] Manual UX verification of highlight visuals, marker badge continuity,
      scroll return, and zoom feel.

### Phase 4

- [ ] Define and implement final `PDF` button behavior for a PDF-native workflow.
- [x] Implement `Edit` behavior for PDFs.
- [ ] Add/refresh tests for top-row controls once `PDF` behavior is finalized.

### Phase 5

- [ ] Review whether more `mdexplore_app` helpers should be shared to reduce
      divergence between explorers.

### Phase 6 (Stability / Responsiveness)

- [ ] Add focused performance regression tests around root changes with medium-size directories.
- [ ] Add deterministic marker-badge tests for transitions: add highlight -> switch file -> switch back.
- [ ] Add viewer-bridge regression tests for marker click responsiveness during in-progress marker builds.
- [ ] Add lightweight optional timing instrumentation hooks for hot paths
   (tree marker sync, visible-tree listing, search aggregation).
