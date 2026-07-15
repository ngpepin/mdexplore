# pdfexplore

`pdfexplore` is the read-only PDF companion to `mdexplore`, with the same UI shell and operational model where it is applicable to PDFs.

It is designed for people who review many PDFs and need stable annotation/session context without touching source documents. `pdfexplore` treats the PDF file as immutable input and stores user state in sidecars so workflows remain reproducible and reversible.

## Settings Cheat Sheet

| What you want to change | File | Jump to details |
| --- | --- | --- |
| Performance and responsiveness (cache, prefetch, timer cadence, thread pools) | [pdfexplore.settings.json](../pdfexplore.settings.json) | [app section key reference](#app-section-key-reference) |
| Viewer search-marker behavior and bridge tuning | [pdfexplore.settings.json](../pdfexplore.settings.json) | [viewer_bridge section key reference](#viewer_bridge-section-key-reference) |
| Sidecar/session behavior and file naming | [pdfexplore.settings.json](../pdfexplore.settings.json) | [app section key reference](#app-section-key-reference) |
| Tree extension/icon/color sidecar constants | [pdfexplore.settings.json](../pdfexplore.settings.json) | [tree section key reference](#tree-section-key-reference) |
| Disk-cache trim cadence and retention policy details | [pdfexplore.settings.json](../pdfexplore.settings.json) | [pdf_text_disk_cache_trim_interval deep dive](#pdf_text_disk_cache_trim_interval-deep-dive) |
| Cache-size number derivations (MiB to bytes) | [pdfexplore.settings.json](../pdfexplore.settings.json) | [Runtime value derivation notes](#runtime-value-derivation-notes) |
| Full pdfexplore runtime defaults (all sections) | [pdfexplore.settings.json](../pdfexplore.settings.json) | [Runtime Settings (JSON)](#runtime-settings-json) |
| Shared cross-app constants imported from mdexplore | [mdexplore.settings.json](../mdexplore.settings.json) | [shared global settings file note](#runtime-settings-json) |

## What This Tool Prioritizes

- Read-only handling of source PDFs.
- Fast directory navigation with interaction-first scheduling.
- Sidecar-based persistence for colors, view sessions, and text highlights.
- UI parity with mdexplore interaction patterns where they fit PDF workflows.

## Experience Goals

- Predictable state continuity:
  - highlights, view tabs, and marker badges should persist through navigation,
    refreshes, and restarts.
- Interaction-first responsiveness:
  - scrolling, expanding folders, and selecting files should remain smooth even
    while background cache/scan work is active.
- Transparent metadata model:
  - all durable user state lives in sidecars that are easy to inspect and back up.

## Features

- Left-pane tree rooted at the chosen directory, showing folders plus `*.pdf` files.
- Right-pane embedded PDF preview via bundled local `pdf.js`.
- Top actions:
  - `^` to move root up one directory.
  - `Refresh` to rescan the current tree root.
  - `Dark` to render PDF pages dark gray with light text; while active, the
    button reads `Light` and restores normal PDF colors when clicked.
  - `Add View` to create another tabbed view of the same PDF at the current page/scroll location.
- `F5` refresh shortcut (same behavior as `Refresh`).
- Search box and color-apply controls matching `mdexplore` search syntax:
  - supports tokenized/quoted boolean queries via shared parser helpers,
  - searches PDFs currently visible in the tree (root files plus expanded branches),
  - highlights in-tree match counts and in-view search hits for matched files,
  - opening another matched PDF while search is active jumps to its first hit.
- Right-rail search indicators in preview:
  - markers are generated progressively from top pages to bottom pages,
  - markers become clickable as soon as they appear,
  - dense nearby markers are clustered into larger pills for easier click targets,
  - clicking a marker temporarily prioritizes navigation/rendering over marker-build throughput,
  - marker generation resumes automatically after the interaction settles.
- Left-rail persistent-highlight indicators in preview:
  - normal and important highlights use the same distinct marker colors as `mdexplore`,
  - marker length reflects the highlighted span when rendered text geometry is available,
  - markers are available for unrendered pages from persisted page/range metadata,
  - clicking a marker opens its page and centers the corresponding highlight.
- In-view `NEAR(...)` highlighting mirrors `mdexplore` by highlighting and marking
  only qualifying proximity windows, including variadic groups of two or more terms.
- Tree highlight colors and clear actions:
  - per-file color assignment via tree context menu,
  - `Clear in Directory` and `Clear All` with confirmation prompts,
  - persisted per-directory in sidecars.
- Copy controls aligned with `mdexplore`:
  - destination mode `Clipboard` or `Directory`,
  - pin action copies current PDF,
  - color actions copy all files with a selected highlight color in active scope,
  - directory copy merges color/view/highlight metadata into destination sidecars.
- Preview context menu:
  - `Copy Selected Text`,
  - `Highlight` / `Highlight Important`,
  - `Remove Highlight`.
- Persistent in-view text highlights with per-file sidecar storage.
- View tabs aligned with `mdexplore` behavior:
  - per-document multi-view tabs (max 8),
  - tab position progress bar driven by page progress,
  - custom tab labels via tab context menu,
  - labeled-tab `Return to beginning` support,
  - home/reset icon actions from shared `ViewTabBar`,
  - hide tab strip when only one unlabeled default view remains,
  - per-document tab sessions persisted/restored from `.pdfexplore-views.json`.
- PDF preview widget caching:
  - live `QWebEngineView`/pdf.js pages are retained in a bounded least-recently-used
    cache (two documents by default),
  - switching away and back avoids viewer/widget re-instantiation while the PDF
    remains in that cache,
  - an eviction discards the least-recently-used WebEngine page so returning to
    that PDF reloads its viewer instead of retaining unbounded browser memory.
- Extracted-text cache + optional idle prefetch:
  - automatic visible-scope prefetch is disabled by default to avoid competing
    with navigation and scrolling, and can be enabled with `prefetch_enabled`,
  - when enabled, likely-next PDFs are warmed in small, interaction-aware batches,
  - cached-text badge reflects both memory and disk cache hits,
  - idle garbage collection removes memory/disk text entries whose source PDF
    has been deleted and clears the corresponding cache badge.
- Tree marker badges:
  - view badge indicates multi-view/open-session state,
  - marker badge indicates persistent text highlights,
  - marker sync merges scan results with live/in-memory metadata to reduce badge flicker.
- If the currently previewed PDF changes on disk, preview auto-refreshes and reports that in the status bar.
- `Refresh` preserves expanded folders/selection when possible and clears the preview if the active PDF was deleted.
- Effective root persisted to `~/.pdfexplore.cfg` on close (interactive terminal launch without path defaults to current working directory).

## User Workflow (Narrative)

1. Choose a root directory (or launch on a single PDF).
2. Navigate tree folders and open PDFs in the preview panel.
3. Use search to identify matching PDFs in visible scope.
4. Apply file highlight colors or preview text highlights as needed.
5. Add extra views (tabs) for parallel reading positions in one document.
6. Copy files and merge metadata to another directory when curating output sets.
7. Close and reopen later with session and highlight context restored from sidecars.

## Architecture Overview

- Main GUI orchestration: `pdfexplore/app.py` (`PdfExploreWindow`).
- Tree model/delegate: `pdfexplore/tree.py` + shared `mdexplore_app/file_tree.py`.
- Background jobs: `pdfexplore/workers.py`.
- Embedded viewer bridge: `pdfexplore/assets/viewer_bridge.js` + local `pdf.js` bundle.
- Text caching:
  - memory cache: bounded `OrderedDict`,
  - disk cache: compressed entries under `${XDG_CACHE_HOME:-~/.cache}/mdexplore/pdfexplore-text-cache`,
  - atomic `.txt.gz.meta.json` companions map compressed entries back to source
    PDFs for idle garbage collection; legacy entries gain metadata when loaded.

### Data/Control Flow Summary

- Tree selection/open action:
  - GUI updates active file/tab state,
  - preview widget is reused or created and the bounded viewer LRU evicts old
    WebEngine pages as needed,
  - viewer bridge restores view state and highlights.
- Search action:
  - visible candidate set is built,
  - the extraction pool evaluates candidates in chunks (one thread and eight
    PDFs per worker by default),
  - partial worker results are coalesced before the model receives hit counts
    and filename match styling.
- Search-indicator action (viewer bridge):
  - term signature is computed from normalized query + current document key,
  - per-page text extraction reuses document-scoped in-memory cache entries,
  - pages are scanned in bounded concurrent batches,
  - marker entries are published progressively with DOM updates rate-limited,
  - periodic event-loop yields keep input/paint responsive,
  - click interactions can interrupt and resume builds to keep navigation immediate,
  - identical search/highlight payloads are ignored rather than rebuilding overlays.
- Prefetch action:
  - when explicitly enabled, an idle timer evaluates interaction/search pressure
    before scheduling optional cache warming,
  - the independent GC timer continues to offer bounded missing-source cleanup
    even when prefetch is disabled.
- Marker update action:
  - sidecar scan worker emits marker sets,
  - app merges scan state with live state,
  - model receives final marker path keys for rendering.

## Run

From repo root:

```bash
./pdfexplore.sh [PATH]
```

- `PATH` may be a directory or a single PDF file.
- If omitted:
  - interactive terminal launch uses current working directory,
  - otherwise falls back to `~/.pdfexplore.cfg`, then `$HOME`.

## Configuration

- Primary config file: `~/.pdfexplore.cfg`.
- Supports modern JSON payload with default root/recent roots.
- Includes compatibility handling for legacy single-path config payloads.
- Recent roots are maintained using short lock-based writes for safer multi-instance behavior.

### Runtime Settings (JSON)

- Shared global settings file: `mdexplore.settings.json`
  - contains shared defaults used by both apps (for example zoom and highlight-related constants imported from `mdexplore_app.constants`).
- pdfexplore-specific settings file: `pdfexplore.settings.json`
  - contains pdfexplore-only settings in three sections:
    - `app`: timers, worker limits, file/session/cache behavior.
    - `viewer_bridge`: in-view marker/search tuning and progressive marker controls.
    - `tree`: PDF tree model constants (sidecar file name, extension target, icon name/color).

This split keeps shared/global behavior in one place while preserving a
separate, app-local tuning surface for pdfexplore.

#### Loading and fallback behavior

- `pdfexplore/settings.py` reads `pdfexplore.settings.json` from repo root.
- If the file is missing or invalid, each section (`app`, `viewer_bridge`, `tree`) starts empty.
- `pdfexplore/app.py` applies per-key defaults with `_app_setting(...)` for missing `app` values.
- `pdfexplore/assets/viewer_bridge.js` applies numeric defaults and clamp ranges for missing or out-of-range `viewer_bridge` values.
- `pdfexplore/tree.py` reads `tree` values with per-key fallback defaults.

#### Units and naming conventions

- `_ms`: milliseconds.
- `_seconds`: seconds.
- `_bytes`: bytes.
- `_max_threads`: thread count.
- Ratio/divisor keys are unitless numbers.

#### `app` section key reference

##### Files, sidecars, and root/session behavior

| Key | Default | Purpose |
| --- | --- | --- |
| `config_file_name` | `.pdfexplore.cfg` | User config filename in home directory. |
| `views_file_name` | `.pdfexplore-views.json` | Sidecar filename for per-document view sessions. |
| `highlighting_file_name` | `.pdfexplore-highlighting.json` | Sidecar filename for persistent text highlights. |
| `viewer_html_relative` | `vendor/pdfjs/web/viewer.html` | Relative path to embedded pdf.js viewer HTML. |
| `viewer_bridge_js_relative` | `assets/viewer_bridge.js` | Relative path to bridge script injected into viewer. |
| `okular_edit_launcher` | `/home/npepin/.local/share/applications-scripts/run-okular.sh` | External launcher script path for edit/open integration. |
| `max_document_views` | `8` | Maximum number of tabs/views per document. |
| `max_recent_root_directories` | `35` | Recent-roots persistence cap. |
| `recent_root_menu_mru_count` | `10` | Number of MRU roots shown first in menu. |
| `min_recent_root_dwell_seconds` | `30.0` | Minimum active-root dwell before history retention. |
| `config_default_root_key` | `default_root` | Config JSON key for default root. |
| `config_recent_roots_key` | `recent_roots` | Config JSON key for recent roots. |
| `config_lock_stale_seconds` | `120.0` | Lock-file stale threshold for cleanup. |

##### Text cache, prefetch, and search cadence

| Key | Default | Purpose |
| --- | --- | --- |
| `pdf_text_cache_max_entries` | `768` | Max in-memory text cache entries. |
| `pdf_text_cache_max_chars` | `201326592` | Max total characters in memory text cache. |
| `pdf_text_disk_cache_max_files` | `8000` | Max disk cache files before trim pressure. |
| `pdf_text_disk_cache_max_bytes` | `1610612736` | Max disk cache byte budget before trim pressure. |
| `pdf_text_disk_cache_trim_interval` | `240` | Disk-cache trim trigger interval measured in successful store operations (every Nth store runs a trim pass). |
| `pdf_text_cache_gc_interval_seconds` | `30.0` | Cadence of the dedicated missing-source garbage-collection timer. |
| `pdf_text_cache_gc_idle_seconds` | `10.0` | Sustained no-input period required before a GC batch may start or continue. |
| `pdf_text_cache_gc_batch_size` | `128` | Maximum memory entries and disk metadata companions checked per idle GC batch. |
| `prefetch_enabled` | `false` | Enables optional automatic visible-scope extracted-text warming; disabled by default for responsiveness. |
| `prefetch_batch_size` | `1` | Number of PDFs prefetched per idle cycle. |
| `prefetch_idle_seconds` | `0.8` | Idle-delay threshold before prefetch can run. |
| `prefetch_heavy_use_window_seconds` | `1.2` | Sliding window for interaction-pressure checks. |
| `prefetch_heavy_use_event_threshold` | `14` | Interaction count threshold that triggers temporary prefetch pause. |
| `prefetch_heavy_use_pause_seconds` | `1.2` | Pause duration after heavy interaction threshold is exceeded. |
| `prefetch_viewer_activity_pause_seconds` | `1.6` | Pause duration after active viewer interaction. |
| `prefetch_tree_mutation_pause_seconds` | `0.85` | Pause duration after tree mutation/expand/collapse activity. |
| `tree_search_refresh_debounce_ms` | `900` | Debounce delay for tree-driven search refreshes. |
| `tree_interaction_visual_relax_ms` | `320` | UI relax window after interaction bursts. |
| `cached_badge_sync_interval_ms` | `200` | Cadence for cache badge synchronization. |
| `match_timer_interval_ms` | `320` | Match worker dispatch cadence. |
| `search_worker_chunk_size` | `8` | Number of candidate PDFs evaluated by each search worker job. |
| `search_progress_publish_interval_ms` | `100` | Minimum interval for coalescing partial search results into model/UI updates. |
| `scope_prefetch_timer_interval_ms` | `550` | Prefetch timer cadence for current scope. |
| `viewer_ready_timer_interval_ms` | `160` | Viewer-ready polling cadence. |
| `view_state_poll_timer_interval_ms` | `900` | View-state synchronization polling cadence. |
| `file_change_watch_interval_ms` | `1200` | Poll interval for source PDF signature changes. |

##### Thread pools and palettes

| Key | Default | Purpose |
| --- | --- | --- |
| `search_thread_pool_max_threads` | `1` | Search/extraction worker pool max thread count; the single-thread default reduces extraction contention with the GUI. |
| `prefetch_thread_pool_max_threads` | `1` | Prefetch worker pool max thread count. |
| `preview_widget_cache_max_entries` | `2` | Maximum live per-document `QWebEngineView` pages retained by the preview LRU. |
| `tree_marker_scan_thread_pool_max_threads` | `1` | Tree marker scan worker pool max thread count. |
| `highlight_colors` | `Yellow, Green, Blue, Orange, Purple, Light Gray, Medium Gray, Red` | File-highlight palette entries (`name`, `value`). |

##### pdf_text_disk_cache_trim_interval deep dive

`pdf_text_disk_cache_trim_interval` controls how often the on-disk text cache is
trimmed after writes.

- A counter increments after each successful compressed cache write.
- When `store_count % pdf_text_disk_cache_trim_interval == 0`, a trim pass runs.
- Default `240` means trim runs on the 240th, 480th, 720th, ... successful store.
- Trim pass ordering is recency-first (newest `mtime` kept first).
- A file is kept only if both limits remain satisfied:
  - `pdf_text_disk_cache_max_files`
  - `pdf_text_disk_cache_max_bytes`
- Any remaining older entries are deleted.
- Cache reads update file `mtime`, so recently used entries are less likely to
  be removed during future trims.

Tuning guidance:

- Lower interval (for example `80` to `160`):
  - more frequent trim work,
  - tighter disk-footprint control,
  - slightly higher background IO overhead.
- Higher interval (for example `480` to `960`):
  - less frequent trim work,
  - larger transient disk growth between trims,
  - occasional larger cleanup bursts.
- Keep this value as an integer >= `1`.
  - `0` or negative values are invalid for modulo-trigger logic.

#### `viewer_bridge` section key reference

These values are consumed in browser context by `viewer_bridge.js`.
Each numeric value is validated with clamp bounds before use.

| Key | Default | Clamp | Purpose |
| --- | --- | --- | --- |
| `three_up_divisor` | `3` | `1..12` | Divisor used to scale three-up layout width. |
| `min_zoom_scale` | `0.1` | `0.01..100` | Minimum allowed zoom scale in bridge operations. |
| `max_zoom_scale` | `10.0` | `min_zoom_scale..100` | Maximum allowed zoom scale in bridge operations. |
| `restore_stabilize_ms` | `2800` | `50..30000` | Stabilization window after restore actions. |
| `search_indicator_resume_delay_ms` | `180` | `1..5000` | Delay before paused marker generation resumes. |
| `search_indicator_click_retry_delay_ms` | `80` | `1..5000` | First retry delay for click-driven marker navigation. |
| `search_indicator_click_final_retry_delay_ms` | `180` | `1..5000` | Final retry delay for click-driven marker navigation. |
| `search_indicator_max_entries` | `2400` | `50..50000` | Upper cap of marker entries rendered in right rail. |
| `search_indicator_concurrency_min` | `2` | `1..64` | Minimum page-scan concurrency for marker build. |
| `search_indicator_concurrency_max` | `4` | `search_indicator_concurrency_min..64` | Maximum page-scan concurrency for marker build. |
| `search_indicator_markers_per_page_max` | `48` | `1..500` | Per-page marker cap before clustering/pruning pressure. |
| `search_indicator_yield_every_batches` | `3` | `1..50` | Event-loop yield cadence for responsive progressive builds. |
| `search_indicator_publish_interval_ms` | `90` | `16..2000` | Minimum interval between progressive right-rail DOM publications. |

#### `tree` section key reference

| Key | Default | Purpose |
| --- | --- | --- |
| `color_file_name` | `.pdfexplore-colors.json` | Sidecar filename for per-directory tree color assignments. |
| `target_extension` | `.pdf` | Extension filter used by PDF tree model. |
| `primary_icon_name` | `pdf.svg` | Primary icon asset used for target rows. |
| `primary_icon_color` | `#e86060` | Tint color applied to primary icon. |

#### Runtime value derivation notes

Some cache limits were previously expressed in code as MiB formulas and are
now explicit integers in JSON.

- `app.pdf_text_cache_max_chars = 201326592`
  - derived from `192 * 1024 * 1024` (192 MiB).
- `app.pdf_text_disk_cache_max_bytes = 1610612736`
  - derived from `1536 * 1024 * 1024` (1536 MiB).

General conversion rule: `MiB * 1024 * 1024`.

#### Practical tuning notes

- Change one setting group at a time (cache, prefetch, bridge markers) and verify interaction responsiveness after each change.
- Keep `viewer_bridge` concurrency and entry limits balanced: high values increase throughput but can delay interaction-priority jumps.
- For low-memory systems, reduce `pdf_text_cache_max_chars` first, then `pdf_text_disk_cache_max_bytes`.
- Leave `prefetch_enabled` off unless automatic warming is worth its extraction
  cost; if enabled, lower `prefetch_batch_size` and/or increase pause-related
  seconds values when heavy-document navigation slows.
- Lower `preview_widget_cache_max_entries` when browser-process memory is more
  important than instant return to recently opened PDFs.

## Sidecars

`pdfexplore` writes sidecars beside PDFs/directories:

- `.pdfexplore-colors.json`
- `.pdfexplore-views.json`
- `.pdfexplore-highlighting.json`

`views` sidecars store multi-view sessions (tabs, active tab, per-tab state, custom labels when present). As with `mdexplore`, default untouched single-view sessions are not kept as persistent entries.

### Sidecar Compatibility and Safety

- Sidecar parsing is defensive: malformed entries are ignored where possible.
- Legacy and partial shapes are tolerated in key paths (especially views).
- Sidecars are the only durable writes; source PDFs are never edited.

### Sidecar Quick Reference

- `.pdfexplore-colors.json`
  - file highlight colors for tree rows.
- `.pdfexplore-views.json`
  - multi-view tab sessions and active view placement.
- `.pdfexplore-highlighting.json`
  - persistent in-document text highlight ranges (normal/important).

## Performance Notes

- Search is worker-based and cancellation-aware. The one-thread default avoids
  Python PDF extraction jobs contending with each other, workers receive eight
  candidates at a time, and intermediate partial results use a 100 ms debounced
  UI publication by default (final completion publishes immediately).
- Automatic extracted-text prefetch is off by default. When enabled, it remains
  low-priority and is throttled by interaction detection.
- Extracted-text garbage collection uses an independent 30-second cadence,
  requires 10 seconds of sustained input idle, scans bounded batches on the
  low-priority pool, and yields when interaction or search pressure returns.
  It stays available when automatic prefetch is disabled and never wakes or
  rescans prefetch scope itself.
- A source is treated as missing only on a definite `FileNotFoundError`; permission
  or transient filesystem errors preserve the cached text.
- Tree marker scans run in background workers and are merged with live state.
- Viewer search-marker generation uses document-scoped page-text cache plus bounded concurrency.
- Marker builds publish partial rails progressively, with DOM publications
  coalesced on a 90 ms interval by default (with a final immediate flush), so
  long documents become navigable early without repeatedly rebuilding the rail
  for every page batch.
- Marker clicks are interaction-prioritized: active builds can pause briefly so page jumps feel immediate.
- Persistent-highlight gutter markers are derived immediately from sidecar page/range
  metadata and refine their placement from live text geometry as pages render.
- Bridge-owned overlay DOM mutations are ignored by the viewer mutation observer,
  preventing highlight painting from scheduling itself indefinitely.
- Ordinary scrolling updates navigation state without invalidating every page-text
  index, repainting all overlays, or scanning every page rectangle. Whole-document
  geometry scans remain limited to modes such as three-up that require them.
- The pdf.js `#viewerContainer` remains an absolute, viewport-bounded scroll host;
  letting it expand to document height defeats pdf.js navigation and virtualization.
- Repeated identical persistent-highlight or search payloads are idempotent and do
  not recreate overlay DOM.
- View-state restore retries are generation-guarded so a newer zoom/layout action
  cancels stale callbacks instead of fighting the current viewer state.
- The preview LRU retains at most two live WebEngine pages by default; evicted
  viewers are discarded to cap cumulative browser memory.
- Temporary reduced-paint mode may be used during tree mutation bursts, but marker sync forces full marker visibility when marker state is updated.

`pdfexplore` inherits `SEARCH_CLOSE_WORD_GAP`,
`PREVIEW_PERSISTENT_HIGHLIGHT_MARKER_COLOR`, and
`PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_MARKER_COLOR` from the shared
`mdexplore.settings.json` values so proximity and marker visuals stay aligned
between the two apps.

### Performance Tradeoffs

- Throughput is intentionally sacrificed during active interaction periods to keep UI feel stable.
- Optional prefetch uses small batches to avoid large contiguous extraction spikes.
- Marker sync work is root-scoped to reduce unnecessary model churn after root changes.

### What Optional “Idle Prefetch” Means Here

- Set `prefetch_enabled` to `true` to opt into automatic extracted-text warming.
- The app is considered idle enough when no recent interaction pressure or active
  search scan is blocking that optional prefetch.
- Enabled prefetch warms text cache for likely-next PDFs and updates cache badges progressively.
- Independently, every 30 seconds the GC timer may offer one bounded
  missing-source cleanup pass after at least 10 seconds without input; it does
  not enable, wake, or rescan scope prefetch.

## Troubleshooting

- If browsing feels stalled after a root change, verify that no long-running search query is active.
- If marker/cache badges look stale, trigger `Refresh`; badge state is rebuilt from sidecars plus in-memory live state.
- If a sidecar gets malformed, browsing still works; affected metadata (colors/views/highlights) for that directory may be ignored until fixed.

### Badge Troubleshooting Checklist

If highlight or cache badges seem wrong:

1. Confirm sidecar files exist beside the relevant PDFs/directories.
2. Trigger `Refresh` and re-open the affected file.
3. Verify active root/scope matches where metadata was created.
4. Re-run search and clear search to force model icon state refresh.

### Crash/Freeze Triage Hints

- Repro patterns involving tree expansion and preview switching often indicate scheduling contention between marker sync and prefetch/search workers.
- Repro patterns involving white preview flashes often indicate viewer-bridge or preview-widget lifecycle timing.

## Notes

- PDF content remains read-only; sidecars are the only persisted metadata outputs.
- The embedded viewer bundle is local under `pdfexplore/vendor/pdfjs/` (no remote dependency for core PDF viewing).

## Contributor Notes

- Keep behavioral parity with mdexplore where users expect consistency.
- Prefer shared helper reuse from `mdexplore_app` for generic tree/model concerns.
- Update `pdfexplore/AGENTS.md` when architecture contracts or major workflow guarantees change.
