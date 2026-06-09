# pdfexplore Notes

## Mission

- Keep `pdfexplore` read-only for source PDFs; persist only sidecar metadata.
- Maintain mdexplore-style interaction contracts where behavior maps cleanly to PDF workflows.
- Keep GUI responsive under medium/large trees by prioritizing user interaction over background cache work.

## Product Narrative

`pdfexplore` exists to make high-volume PDF triage feel as fluid as markdown exploration while preserving strict document safety. The core idea is simple:

- source PDFs are immutable,
- user intent is captured as sidecar metadata,
- expensive work is pushed off the UI thread,
- the interface always favors interaction smoothness over background throughput.

The system should feel predictable under stress: opening a new root, expanding folders, scrolling dense PDFs, and running searches should not cause behavioral surprises (lost markers, stale badges, blocked navigation) even when background workers are active.

## Non-Negotiables

- Never mutate source PDF content.
- Keep bundled `pdf.js` assets isolated under `pdfexplore/vendor/pdfjs/`.
- Sidecar failures must degrade gracefully; malformed JSON must never block browsing.
- Preserve backward compatibility for legacy sidecar payloads where currently supported.

## System Invariants

- UI responsiveness invariant:
    - User interaction has higher priority than prefetch and marker scans.
    - Background jobs may be delayed/canceled but UI actions should remain immediate.
- Metadata durability invariant:
    - Highlight and view metadata must survive file switches, refreshes, and restarts.
    - Missing/invalid sidecars must never crash or block browsing.
- Badge continuity invariant:
    - Cache, view, and highlight badges must remain coherent across search updates,
        tree changes, and document switches.
- Scope correctness invariant:
    - Operations that are scope-dependent (search/copy/highlight actions) must use
        the same effective-scope semantics as the tree model.

## Key Sidecars

- `.pdfexplore-colors.json`: tree color assignments.
- `.pdfexplore-views.json`: per-document multi-view tab sessions.
- `.pdfexplore-highlighting.json`: persistent in-document text highlight ranges.

### Sidecar Philosophy

- Sidecars store intent/state, not source transformations.
- Writes should avoid noisy churn (do not persist trivial default state unless required).
- Reads should be permissive:
    - tolerate partial payloads,
    - accept legacy shapes when feasible,
    - sanitize malformed entries instead of failing the pipeline.

## C4 Context

```mermaid
C4Context
    title pdfexplore System Context

    Person(user, "User", "Browses and annotates PDFs")
    System(pdfexplore, "pdfexplore", "Qt desktop PDF explorer")
    System_Ext(filesystem, "Filesystem", "PDF files + sidecar metadata")
    System_Ext(pdfjs, "Bundled pdf.js Viewer", "Local web viewer for rendering/search overlays")

    Rel(user, pdfexplore, "Browse, search, highlight, copy metadata")
    Rel(pdfexplore, filesystem, "Read PDFs, read/write sidecars")
    Rel(pdfexplore, pdfjs, "Embed viewer and bridge commands")
```

## C4 Container

```mermaid
C4Container
    title pdfexplore Container View

    Person(user, "User")
    System_Boundary(pdfexploreBoundary, "pdfexplore") {
        Container(gui, "Qt GUI App", "PySide6", "Main window, tree, controls, orchestration")
        Container(workerPool, "Worker Pools", "QThreadPool", "Search scan, prefetch, sidecar marker scan")
        Container(modelLayer, "Tree Model/Delegate", "Shared mdexplore_app", "Icons, badges, row styling")
        Container(viewerBridge, "Viewer Bridge", "JS + QWebEngine", "State sync, highlights, search term bridge")
        Container(cacheLayer, "Text Cache", "Memory + Disk", "Extracted PDF text for search/prefetch")
    }
    System_Ext(filesystem, "Filesystem", "PDF files + sidecars")

    Rel(user, gui, "Interacts")
    Rel(gui, workerPool, "Dispatches async jobs")
    Rel(gui, modelLayer, "Updates row state + badges")
    Rel(gui, viewerBridge, "Sends viewer commands")
    Rel(gui, cacheLayer, "Reads/writes cache")
    Rel(workerPool, cacheLayer, "Warms text cache")
    Rel(cacheLayer, filesystem, "Stores/loads compressed text cache")
    Rel(gui, filesystem, "Reads/writes sidecars")
```

## C4 Component (GUI App)

```mermaid
C4Component
    title pdfexplore GUI Components

    Container_Boundary(guiBoundary, "Qt GUI App") {
        Component(window, "PdfExploreWindow", "pdfexplore/app.py", "Main orchestration and UI wiring")
        Component(treeModel, "ColorizedPdfModel + Delegate", "pdfexplore/tree.py + mdexplore_app/file_tree.py", "Tree rows, icon composition, badge visuals")
        Component(searchCtrl, "Search Controller", "pdfexplore/app.py", "Query parsing, worker fan-out, result aggregation")
        Component(prefetchCtrl, "Prefetch Controller", "pdfexplore/app.py", "Idle-aware cache warming and throttling")
        Component(markerCtrl, "Marker Sync", "pdfexplore/app.py", "Merges sidecar/live marker sets into model")
        Component(workers, "Worker Jobs", "pdfexplore/workers.py", "Search, prefetch, marker scan jobs")
        Component(viewer, "Viewer Bridge", "pdfexplore/assets/viewer_bridge.js", "Highlight/search/view state bridge")
    }

    Rel(window, searchCtrl, "Delegates")
    Rel(window, prefetchCtrl, "Delegates")
    Rel(window, markerCtrl, "Delegates")
    Rel(searchCtrl, workers, "Starts PdfSearchWorker")
    Rel(prefetchCtrl, workers, "Starts PdfTextPrefetchWorker")
    Rel(markerCtrl, workers, "Starts PdfTreeMarkerScanWorker")
    Rel(markerCtrl, treeModel, "Sets marker path keys")
    Rel(window, viewer, "Sends/receives state + highlight commands")
```

## Runtime Lifecycle (Narrative)

1. Boot

- App resolves startup root.
- Tree model/delegate are attached.
- Viewer bridge source is loaded.
- Timers and worker pools are initialized.

2. Root Activation

- Root path is set in the filesystem model.
- Tree state and marker caches are rebuilt.
- Marker sidecar scan is launched in background.

3. Interactive Work

- User navigates tree, opens PDFs, adds highlights, runs searches.
- Search workers fan out by visible-scope candidate chunks.
- Prefetch worker warms text cache opportunistically when idle.

4. Synchronization

- Marker sync merges scan output with live/in-memory state.
- Tree model receives merged badge sets.
- Viewer bridge applies highlight/search overlays.

5. Shutdown

- Active view state is captured.
- Sidecars/config are persisted.
- No source PDF modifications occur.

## Worker Coordination Rules

- Search workers:
    - should be cancellation-aware by request id,
    - must never manipulate widgets directly.
- Prefetch workers:
    - run low-priority,
    - should pause/cancel under interaction pressure,
    - should prefer current-document warmup before broader scope.
- Marker scan workers:
    - produce sidecar-derived marker sets,
    - must be merged with live state to avoid transient badge regressions.

## UI Performance Guardrails

- Avoid introducing filesystem `resolve()` loops inside frequently called marker-sync paths.
- Keep repaint pressure low during heavy tree mutation.
- Coalesce badge updates where practical.
- Prefer root-scoped filtering before expensive merging.

## Contributor Playbook

When changing any of the following, update both behavior and docs in the same change:

- Search orchestration or query semantics.
- Prefetch scheduling/throttling behavior.
- Sidecar format interpretation.
- Tree badge rendering/priority rules.
- Viewer bridge highlight/search interactions.

Recommended validation sequence:

1. `python -m py_compile` for touched modules.
2. Ensure diagnostics are clean.
3. Manual smoke checks:
    - open root with medium file count,
    - expand/collapse multiple folders,
    - run search and clear search,
    - add/remove highlight, switch files, return,
    - verify cache and marker badges remain coherent.

## Failure Modes to Watch

- Marker disappearance after search or tab switch.
- Prefetch starving indefinitely while UI is idle.
- Excessive UI stalls on root change due to marker merge overhead.
- Crash paths around viewer creation or event/filter hooks.
- Sidecar parse errors cascading into badge/model state resets.

## Change Boundaries

- Keep generic tree rendering logic in shared `mdexplore_app` when possible.
- Keep PDF-specific orchestration in `pdfexplore/app.py` and `pdfexplore/workers.py`.
- Keep viewer bridge-specific behavior in JS bridge surface, not scattered widget callbacks.

## Maintenance Guidance

- Prefer sharing generic behavior through `mdexplore_app` for long-term parity.
- Keep prefetch throttling interaction-first; regressions should bias toward smooth UI.
- Treat marker badge continuity as correctness-critical: highlight and cache badges must survive tab switches, searches, and root navigation.
- Preserve current UX contracts:
  - scope-aware operations align with visible tree behavior,
  - copy metadata merge semantics match mdexplore expectations,
  - view-tab model remains compatible with existing sidecars.

## Documentation Contract

Any change to runtime behavior in these areas must be reflected in `pdfexplore/README.md`:

- Sidecar behavior or compatibility.
- Search/prefetch throttling semantics.
- Marker badge derivation/merge behavior.
- User-visible workflow changes in top controls or preview context menu.
