# mdexplore — Code Audit, Issues, and Resolutions

**Audit date:** 2026-10-29
**Scope:** `/workspace/Projects/mdexplore` — mdexplore desktop app, `mdexplore_app` support package, `pdfexplore`, `mdext` VS Code extension, launchers, tests, docs, and vendored assets.
**Mode:** Read-only audit. No source code was modified; this document is the only file added.

---

## 1. Executive summary

The project is functionally rich and well covered for a desktop tool of this shape: 454
Python tests collect successfully, the selected non-GUI subset (64 tests) passes, the 54
Node tests for the `mdext` extension pass, all Python files compile, all audited
JavaScript passes `node --check`, and all shell launchers pass `bash -n` plus
`shellcheck -S warning` with no reported warnings.

The main risks are not "the app is broken" but **sustainability, release hygiene, and
crash recovery**:

1. The working tree is mid-feature. The new Markdown forms feature is partly untracked
   (`mdexplore_app/form_fields.py`, `mdexplore_app/form_insertion.py`,
   `mdext/src/formFields.js`, four new Python test modules, and several Node test files),
   while 19 tracked files are modified. Nothing enforces that this state is coherent or
   committable.
2. There is **no CI configuration and no repository-level test configuration**
   (`pyproject.toml`, `pytest.ini`, `setup.cfg`, and `conftest.py` are all absent). The
   454-test suite and the 103 current `ruff` findings are therefore advisory, not gates.
3. Static analysis reports **103 `ruff` errors**, concentrated in the new form modules
   (`E701`/`E702` one-line statements), plus unused imports and unused test variables.
   The backlog is mechanical, but it masks real signal.
4. Architecture is concentrated in two very large modules: `mdexplore.py` (13,788 lines,
   488 functions, 5 classes) and `pdfexplore/app.py` (7,506 lines, 295 functions). Both
   re-implement overlapping window, search, and persistence behavior, and they already
   **diverge on WebEngine security settings**.
5. Runtime code writes into the source checkout (`fastbase64-adaptive.json`,
   `fastbase64-benchmark.log`) and can compile native code with `make` from inside the
   application process. That is fragile for packaged or read-only installs and is a
   supply-chain concern.
6. The new form-save lock (`.mdform.lock`) has **no stale-lock recovery**. A crash or
   `SIGKILL` between lock creation and the `finally` block leaves a permanent lock that
   blocks all future edits of that document until a human deletes the file.

None of these require a rewrite. Section 10 proposes a phased plan: roughly one day of
hygiene, then automated gates, then targeted decomposition and hardening.

### Baseline measurements (commands actually run)

| Check | Command | Result |
|---|---|---|
| Python syntax | `python3 -m py_compile mdexplore.py mdexplore_app/*.py pdfexplore/*.py hfind.py` | clean |
| Python test discovery | `.venv/bin/python -m pytest --collect-only -q` | **454 tests collected** |
| Non-GUI Python tests | `pytest -q` over the base64, file-coordination, form, search-query, highlight-label, and PDF-layout modules | **64 passed, 16 subtests passed** |
| Python lint | `ruff check mdexplore.py mdexplore_app pdfexplore hfind.py tests` | **103 errors** (25 auto-fixable) |
| Node tests | `npm --prefix mdext test` | **54 passed, 0 failed** |
| JS syntax | `node --check` over `mdext/src/*.js` and `assets/js/preview/*.js` | clean |
| Shell syntax | `bash -n` on four launcher scripts | clean |
| Shell lint | `shellcheck -S warning` on the four launcher scripts | no warnings reported |
| Module imports | `.venv/bin/python -c "import ..."` for 10 core modules | all import |
| Version control | `git status --short --branch` | 19 modified, 12 untracked entries |

### Severity legend

| Level | Meaning |
|---|---|
| **P0 — Critical** | Data loss, security exposure, or blocks all work; fix before next release |
| **P1 — High** | Release/CI correctness, crash recovery, or a defect users will hit |
| **P2 — Medium** | Maintainability, performance, or hygiene that compounds over time |
| **P3 — Low** | Polish, documentation, and consistency |

---

## 2. Findings at a glance

| ID | Severity | Area | Issue |
|---|---|---|---|
| REPO-01 | P1 | Repo hygiene | `npm test` runs test files that are not under version control |
| REPO-02 | P1 | Repo hygiene | Runtime writes `fastbase64-adaptive.json` and a benchmark log into the source tree |
| REPO-03 | P2 | Repo hygiene | Large binaries and build output are tracked (~44 MB `.git`) |
| REPO-04 | P2 | Repo hygiene | Generated duplicates are committed next to their sources |
| REPO-05 | P2 | Repo hygiene | Bare `test` ignore pattern is over-broad |
| REPO-06 | P1 | Release process | Large unreviewed mid-feature working tree with no change gate |
| LINT-01 | P2 | Static analysis | 103 `ruff` violations; the new form modules are the bulk |
| LINT-02 | P3 | Static analysis | Unused imports and unused locals in app and tests |
| LINT-03 | P3 | Static analysis | No committed lint, format, or type-check configuration |
| ARCH-01 | P1 | Architecture | `mdexplore.py` is a 13.8k-line God module |
| ARCH-02 | P1 | Architecture | `pdfexplore/app.py` duplicates window logic and has already drifted |
| ARCH-03 | P2 | Architecture | Two tree modules; `tree.py` is a thin subclass, which is acceptable |
| ARCH-04 | P2 | Architecture | Required Python/JS form-parser parity has no cross-language test |
| CONC-01 | P0 | Persistence | `.mdform.lock` has no stale-lock recovery; a crash blocks edits permanently |
| CONC-02 | P2 | Persistence | Sidecar and form writes never `fsync` the parent directory |
| CONC-03 | P2 | Reliability | Broad `except Exception` (246 in `mdexplore.py`, 132 in `pdfexplore/app.py`) |
| SEC-01 | P1 | Supply chain | `make` is invoked from the application runtime to build a native library |
| SEC-02 | P1 | Supply chain | Setup downloads executable assets without checksums, including `latest` PlantUML |
| SEC-03 | P1 | Security | WebEngine local/remote URL access settings diverge between the two apps |
| SEC-04 | P3 | Security | Hardcoded `/usr/bin/code` and unconditional clipboard helper subprocesses |
| TEST-01 | P1 | Testing | No CI and no automated lint/test gate |
| TEST-02 | P1 | Testing | No `conftest.py`; 16 GUI test modules have no headless skip policy |
| TEST-03 | P2 | Testing | Dev dependencies are undeclared; `pytest-timeout` is absent |
| CFG-01 | P3 | Configuration | A malformed `mdexplore.settings.json` is silently replaced by defaults |
| CFG-02 | P2 | Dependencies | Wide version ranges, no lock file, no hashes |
| PERF-01 | P3 | Performance | Production base64 calls carry benchmarking and routing overhead |
| DOC-01 | P3 | Documentation | Documentation drift (backend default, duplicated forms doc, generated copies) |
| DOC-02 | P3 | Documentation | No application version string or changelog surface |

---

## 3. Repository hygiene and version control

### REPO-01 (P1) — `npm test` executes tests that are not tracked

`.gitignore` contains `mdext/test/*` plus negations that un-ignore `formFields.test.js`,
`manifest.test.js`, `formInsertion.test.js`, and `integration/forms.js` /
`integration/insertionWebview.js`. The remaining files — `highlightStore.test.js`,
`previewCoordinator.test.js`, `renderer.test.js`, `searchQuery.test.js` — stay ignored,
yet `mdext/package.json` declares `"test": "node --test test/*.test.js"`, which runs all
eight files locally.

Verified with `git check-ignore -v`:

```text
.gitignore:91:!mdext/test/formFields.test.js      mdext/test/formFields.test.js
.gitignore:90:mdext/test/*                        mdext/test/highlightStore.test.js
```

Consequences: a fresh clone runs a strictly smaller suite than the author does; the 54
passing Node tests are not reproducible from the repository; and `git status` reports the
whole `mdext/test/` directory as untracked, so new tests are easy to lose.

**Resolution.** Pick one policy and encode it in `.gitignore`:

1. Preferred: track `mdext/test/**` wholesale (`!mdext/test/`, then re-ignore only
   genuinely local scratch files) and commit all eight test files.
2. If some tests must stay local, split them into `test/unit/` (tracked) and
   `test/local/` (ignored), then change the npm script to
   `node --test test/unit/*.test.js` so the command matches what CI can run.

Add a release check that compares `git ls-files mdext/test | wc -l` with the number of
files matched by the test glob.

### REPO-02 (P1) — Runtime state is written into the source checkout

`mdexplore_app/fast_base64.py` defines both the adaptive-routing state path and the
benchmark log path relative to the repository root:

```python
_VENDOR_FASTBASE64_BENCHMARK_LOG_PATH = (
    Path(__file__).resolve().parents[1] / "fastbase64-benchmark.log"
)
_VENDOR_FASTBASE64_ADAPTIVE_STATE_PATH = (
    Path(__file__).resolve().parents[1] / "fastbase64-adaptive.json"
)
```

`fastbase64-adaptive.json` is **tracked**, so merely running the application mutates a
tracked file (`git status` showed `M fastbase64-adaptive.json` after the test run in this
audit). That yields spurious diffs, merge conflicts between machines with different CPUs,
and a guaranteed write failure when the install directory is read-only (system-wide
install, container image, or Flatpak bundle).

**Resolution.**

- Move both paths to `~/.cache/mdexplore/` (honoring `XDG_CACHE_HOME`), matching the
  existing icon-cache convention documented in `AGENTS.md`.
- Keep a read-only fallback that swallows write errors when the cache is unavailable.
- `git rm --cached fastbase64-adaptive.json`, add it to `.gitignore`, and delete the
  root-level `fastbase64-benchmark.log` from the working tree.
- Make adaptive routing opt-in (`MDEXPLORE_BASE64_ADAPTIVE=1`) so the default path never
  touches disk.

### REPO-03 (P2) — Large tracked binaries

`git ls-files | du -k` reported the largest tracked objects:

| Size | Path |
|---|---|
| 17 MB | `vendor/plantuml/plantuml.jar` |
| 4.3 MB | `DEVELOPERS-AGENTS.pdf` |
| 4.0 MB | `COURSE.pdf` |
| 3.1 MB | `README.pdf` |
| 2.8 MB | `vendor/mermaid/dist/mermaid.min.js` |
| 2.8 MB | `mdext/media/vendor/mermaid.min.js` (duplicate) |
| 2.4 MB | `mdext/dist/extension.js` (build output) |
| 2.1 MB | `vendor/mathjax/es5/tex-svg.js` and its `mdext/media` duplicate |

`.git` is 44 MB. That is not fatal yet, but every edit to a vendored bundle adds a full
copy to history, and the duplicate pairs double the cost.

**Resolution.** Download at setup time (already supported by `setup-mdexplore.sh`) and
git-ignore the vendored runtime directory; or, if offline reproducibility matters more,
keep one copy and copy/symlink it at build time. Never commit `dist/extension.js`; build
it in CI and attach it to a release.

### REPO-04 (P2) — Generated files edited as sources

`mdext/package.json` `build` copies `../assets/js/preview/form_fields.js` to
`media/formFields.js` and `../MARKDOWN-FORMS.md` to `MARKDOWN-FORMS.md`. Both copies are
tracked and both differ from their parents in the current working tree. Additionally,
`mdext/media/vendor/mermaid.min.js` and `mdext/media/vendor/tex-svg.js` duplicate
`vendor/`. `AGENTS.md` documents the regeneration step, which is the signature of a
manual process that will eventually be skipped.

**Resolution.** Either (a) ignore the generated copies and produce them in
`vscode:prepublish` and CI, or (b) keep them tracked but add a check that regenerates into
a temp directory and fails when the committed copy differs (a "generated files are
current" test). Option (b) is smaller and fits the existing source-structure test style
in `tests/test_js_assets.py`.

### REPO-05 (P2) — Over-broad ignore pattern

`.gitignore` contains a bare `test` line. In gitignore syntax a pattern without a slash
matches at any depth, so any file or directory named `test` anywhere in the tree is
ignored, including future `src/test/` or `docs/test/`. The same file lists
`mermaid-rs-renderer`, `LOC.sh`, and `bugs` without a leading slash.

**Resolution.** Anchor them: `/test`, `/mermaid-rs-renderer`, `/LOC.sh`, `/bugs`.

### REPO-06 (P1) — Unreviewed mid-feature working tree

At audit time `git status` reported 19 modified tracked files (including `mdexplore.py`,
`mdext/src/renderer.js`, `mdext/src/previewCoordinator.js`,
`assets/templates/preview/document.html`, and `tests/test_template_assets.py`) and 12
untracked entries. The forms feature straddles that boundary: its Python modules and four
test modules are untracked while the call sites in `mdexplore.py` are modified. There is
no branch, no PR, and no CI to prove the combination works.

**Resolution.** Create a feature branch (`feature/markdown-forms`), commit the forms work
as one coherent unit (implementation + tests + docs + generated copies), and run the full
suite before merging. See section 10, step 0.

---

## 4. Static analysis and lint

### LINT-01 (P2) — 103 `ruff` violations

`ruff 0.12.1` reports 103 errors across `mdexplore.py`, `mdexplore_app`, `pdfexplore`,
`hfind.py`, and `tests`. Breakdown:

- **~85 `E701`/`E702`** — multiple statements on one line, almost entirely in the new
  `mdexplore_app/form_fields.py` (24), `mdexplore_app/form_insertion.py` (13), and the new
  test modules (`tests/test_form_fields.py`, `test_form_insertion.py`,
  `test_form_preview.py`, `test_mdext_form_menu.py`). Also one in `mdexplore.py:9623`
  (`standard_menu.deleteLater(); menu.deleteLater(); return`).
- **4 `F401`** — unused imports: `pathlib.Path` in `mdexplore_app/js.py`,
  `QTreeView` in `pdfexplore/app.py`, `QPoint` in `tests/test_window_layout.py`.
- **3 `F841`** — unused locals in `tests/test_pdfexplore_highlighting.py`,
  `test_pdfexplore_multi_instance.py`, and `test_preview_regressions.py`.

These are cosmetic individually, but 85 of them are in fresh code, which means the new
modules were never linted. That hides real defects of the same shape (unused results,
dead branches) and makes review harder.

**Resolution.**

1. Run `ruff check --fix` for the 25 auto-fixable issues (unused imports, some
   semicolons).
2. Manually expand the remaining `E701`/`E702` one-liners in the form modules into normal
   blocks. This is mechanical but should be a single dedicated commit so the diff is
   reviewable.
3. Add a `[tool.ruff]` section (see LINT-03) and a CI step that fails on new violations.

### LINT-02 (P3) — Unused imports and locals

Individually listed in LINT-01. Prefer deleting the import over `# noqa`, except where an
import exists for a side effect (none of these are).

### LINT-03 (P3) — No committed lint, format, or type configuration

There is no `pyproject.toml`, `setup.cfg`, `.flake8`, or `.ruff.toml`. `ruff` therefore
runs with defaults chosen by whichever version happens to be installed, and there is no
line-length or target-version statement matching the actual minimum runtime
(Python 3.10, per the environment and the `X | Y` type syntax already used).

**Resolution.** Add a minimal `pyproject.toml`:

```toml
[project]
name = "mdexplore"
requires-python = ">=3.10"
dynamic = ["version"]

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP"]
ignore = ["E501"]  # line length is advisory; formatter handles it

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["E501"]
```

Do not add `mypy`/`pyright` wholesale yet; the codebase has 246 broad `except` blocks and
extensive Qt dynamic typing, so a strict type gate would fail immediately. Introduce
type-checking file-by-file (`mdexplore_app/*.py` first) once the module split in ARCH-01
has begun.

---

## 5. Architecture and maintainability

### ARCH-01 (P1) — `mdexplore.py` is a 13.8k-line God module

Measured: **13,788 lines, 488 `def` statements, 5 classes**. Four of the five classes are
small (`_PreviewLogicalTextExtractor`, `MarkdownRenderer`, `MarkdownFormBridge`,
`PreviewPage`); essentially the entire application lives in `MdExploreWindow`, which
mixes:

- Qt widget construction and layout (toolbar, tree, tabs, splitter, overlays)
- Markdown rendering orchestration and diagram backends (Mermaid JS/Rust, PlantUML, SVG)
- HTML/JS asset assembly and `QWebChannel` bridging
- Search: query parsing, worker pools, progress publishing, tree pill painting
- Per-document view persistence and sidecar merging
- Copy/export: clipboard, directory copy, BASE64 image embedding, PDF print pipeline
- File-tree highlighting, notes, and context menus
- Launcher/runtime concerns (config loading, environment probing)

Consequences observed in the audit: 246 `except Exception` handlers in this file alone,
one-line statement clusters (`mdexplore.py:9623`), and test modules that must construct a
full `MdExploreWindow` to exercise a single behavior. Every new feature raises the cost
of the next one, and merge conflicts in this file are guaranteed for parallel work.

**Resolution (incremental, no rewrite).** Extract cohesive groups behind small interfaces,
one group per PR, moving code without changing behavior:

| Target module | Responsibility to move | Rough source anchor |
|---|---|---|
| `mdexplore_app/preview_pipeline.py` | `MarkdownRenderer` + special-feature detection + `setHtml`/temp-file load | lines ~284–1900, ~6940–7000 |
| `mdexplore_app/diagram_backends.py` | Mermaid JS/Rust resolution, PlantUML subprocess, SVG sanitising | ~907–1800, ~1288–1600 |
| `mdexplore_app/search_controller.py` | query parse, worker pool, progress/pill publishing | search methods in `MdExploreWindow` |
| `mdexplore_app/view_store.py` | `.mdexplore-views.json` read/merge/write and zoom/layout state | persistence methods |
| `mdexplore_app/preview_menu.py` | context-menu construction and actions | ~9400–9700 |
| `mdexplore_app/export.py` | clipboard/directory copy, BASE64 embedding, PDF print | ~11000–13600 |

Each extraction should keep `MdExploreWindow` as the composition root that wires the
collaborators, so existing tests that instantiate the window continue to pass. Target:
get `mdexplore.py` under ~5,000 lines and no single method over ~150 lines. Do not
attempt a package-wide restructure in one step.

### ARCH-02 (P1) — `pdfexplore/app.py` duplicates window logic and has drifted

`pdfexplore/app.py` is **7,506 lines / 295 functions**, and it already imports shared
helpers from `mdexplore_app` (`icons`, `file_coordination`, `constants`, `runtime`,
`search`, `tabs`, `highlight_labels`, `file_tree`). So the shared layer exists — but the
window behavior was copied rather than shared, and the copies have diverged:

- `mdexplore.py` enables `LocalContentCanAccessRemoteUrls` (True) and
  `LocalContentCanAccessFileUrls` (True).
- `pdfexplore/app.py` sets `LocalContentCanAccessRemoteUrls` to **False** and
  `LocalContentCanAccessFileUrls` to True.

That is a security-relevant divergence produced by duplication. The same pattern will
apply to recent-roots handling, highlight confirmation dialogs, tab layout, and search
pill painting unless the shared behavior is extracted.

**Resolution.** Identify the ~10 behaviors that both windows must keep identical
(WebEngine settings profile, recent-root menu, highlight/notes sidecars, tab bar layout,
status bar composition, search scope rules) and move each into `mdexplore_app` as a
small policy object or mixin with its own unit test. Then delete the copies in
`pdfexplore/app.py`. Until then, add a regression test asserting both windows apply the
same `QWebEngineSettings` profile so drift is caught immediately.

### ARCH-03 (P2) — Tree modules are layered, not duplicated

`mdexplore_app/tree.py` is only 43 lines: it subclasses `ColorizedExtensionModel` from
`file_tree.py` and specializes it for Markdown. This is **acceptable design**, not a
defect. The audit note exists only to record that `tree.py` should not be mistaken for a
competing implementation (contrast with `AGENTS.md`'s explicit warning against a
competing `.mdexplore-view.json`). No action required beyond keeping the subclass thin.

### ARCH-04 (P2) — Python/JS form-parser parity has no cross-language test

`MARKDOWN-FORMS.md` and `AGENTS.md` require that the Python and Node parsers/writers stay
in parity:

- Python: `mdexplore_app/form_fields.py`, `mdexplore_app/form_insertion.py`
- Node: `mdext/src/formFields.js`, `mdext/src/formInsertion.js`
- Browser: `assets/js/preview/form_fields.js` (copied to `mdext/media/formFields.js`)

The two implementations are line-for-line analogous (`MARKER` regex, `code_ranges`,
`parse_fields`, `apply_field`, `save_field` with `.mdform.lock`), but they are tested
separately: `tests/test_form_fields.py` on the Python side and
`mdext/test/formFields.test.js` on the Node side. Nothing asserts that the same fixture
produces the same parsed fields and the same updated bytes in both languages, so the two
will drift silently — which is exactly the failure mode the documentation warns about.

**Resolution.** Add a shared JSON fixture set (e.g. `tests/fixtures/forms/*.json`) with
input source, operation, expected value spans, and expected output. Have the Python test
load it directly; have the Node test load the same files. Add one CI job that runs both.
This is cheap and directly enforces the documented invariant.

---

## 6. Concurrency, persistence, and reliability

### CONC-01 (P0) — The form-save lock has no stale-lock recovery

`mdexplore_app/form_fields.py:105-145` implements the save transaction for Markdown form
regions. The mutual-exclusion primitive is a sentinel file created with
`O_CREAT | O_EXCL`:

```python
lock = Path(str(file) + '.mdform.lock')
fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
...
finally:
    if temporary: temporary.unlink(missing_ok=True)
    if fd is not None: os.close(fd); lock.unlink(missing_ok=True)
```

The Node implementation mirrors it (`mdext/src/formFields.js:122`, `fs.openSync(lock,
"wx")`). Both are *exclusive-create* locks, not advisory OS locks: **the lock has no
owner identity, no timestamp, and no expiry.** If the process dies between `os.open` and
the `finally` block — `SIGKILL`, OOM killer, power loss, a Qt crash inside the 20 kB
write — the lock file survives forever. Every subsequent save of that document then
raises `Another form save is in progress (.mdform.lock). Retry shortly.` and the only
recovery is a human deleting a dotfile they do not know about. This is user-visible data
loss of the *ability to edit*, and it also contradicts the project's own precedent:

- `AGENTS.md` requires stale config locks (>2 minutes) to be deleted silently.
- `mdexplore_app/file_coordination.py` already provides `advisory_file_lock()` with
  `fcntl.flock`, `timeout_seconds`, and `should_abort` — a lock that the kernel releases
  automatically on process death.
- The test suite even asserts the *stale* behavior is rejected
  (`tests/test_form_fields.py`: writes a lock, expects `ValueError`, expects the foreign
  lock content preserved) without ever testing recovery.

**Resolution.** Replace the sentinel file with the existing advisory lock, and keep the
sentinel only as an optimisation if desired:

1. Primary: wrap the transaction in
   `advisory_file_lock(Path(str(file) + '.mdform.lock'), exclusive=True, blocking=False)`.
   `flock` is released by the kernel when the fd closes or the process dies, so a crash
   self-heals. Return the same user-facing message when acquisition fails.
2. The Node/VS Code side cannot use `flock`; use
   `fs.openSync(lock, "wx")` **plus** a staleness rule: write the PID and a timestamp into
   the file, and treat an existing lock as stale when its mtime is older than a
   configurable threshold (reuse the 120 s `MDEXPLORE_CONFIG_LOCK_STALE_SECONDS` value for
   consistency) or when the recorded PID is not alive (`process.kill(pid, 0)`).
3. Add regression tests in both suites: create a lock with an old mtime and assert the
   save succeeds and removes it; create a lock with a live PID and assert the save fails
   without touching the lock.
4. Document the recovery behavior in `MARKDOWN-FORMS.md` ("a crashed save self-heals after
   N seconds; deleting `.mdform.lock` by hand is safe when no editor is saving").

Until this is fixed, the feature should be treated as not production-ready: a single
crash permanently disables editing for that file, and the failure message actively
misleads the user into retrying forever.

### CONC-02 (P2) — Writes are atomic but never durable across a power loss

`atomic_write_text()` (`file_coordination.py:87-100`) and `save_field()` both do the right
first step — write a unique temp file, `fsync` the file, then `os.replace()` — but neither
`fsync`s the **parent directory** after the rename. On ext4/xfs with default mount options
the rename is ordered by the journal, but on other filesystems (and after a hard power
loss) the rename can be lost while the temp file survives, or vice versa. The same applies
to `update_files_sidecar()` and to the form save's `os.replace(temporary, file)`.

This is a low-probability issue, but the code already pays for `fsync` on the file, so the
guarantee is half-implemented and the intent is clearly durability.

**Resolution.** Add one helper, e.g.
`def fsync_directory(path: Path) -> None` that opens `path.parent` with `os.O_RDONLY` and
calls `os.fsync()` inside a `try/except OSError` (unsupported on some platforms). Call it
after `replace()` in `atomic_write_text` and after the form `os.replace`. Add a test that
monkeypatches `os.fsync` and asserts it is called for both the file and the directory.

### CONC-03 (P2) — Broad exception handling

Counts of `except Exception`:

| File | Count |
|---|---|
| `mdexplore.py` | 246 |
| `pdfexplore/app.py` | 132 |
| `mdexplore_app/pdf.py` | 24 |
| `mdexplore_app/fast_base64.py` | 16 |
| `mdexplore_app/workers.py` | 15 |
| `mdexplore_app/file_tree.py` | 10 |
| others | 14 |

Many are legitimate best-effort guards (icon loading, clipboard helpers, optional
subprocesses) and are documented as such. The problem is volume: with 246 handlers in one
file, a genuine `AttributeError` or `TypeError` introduced by a refactor is swallowed and
the user sees a silently missing feature instead of a traceback. There is also at least one
`except BaseException` in `save_field` (correctly used for cleanup, but worth auditing).

**Resolution.** Do not remove them wholesale. Instead:

- In `mdexplore_app/*` (the extracted, testable layer) narrow to concrete exceptions where
the failure mode is known (`OSError`, `ValueError`, `json.JSONDecodeError`).
- Keep broad handlers only where the comment already states the intent, and add a
  `logger.debug(..., exc_info=True)` (or the project's debug-log helper) inside so the
  failure is at least observable when `--debug` is active.
- Add a `B902`-style review rule: a new `except Exception` in `mdexplore_app` requires a
  comment on the same line explaining why.

---

## 7. Security and supply chain

### SEC-01 (P1) — The application compiles native code at runtime

`mdexplore_app/fast_base64.py:473-495` calls `make libfastbase64.so` with a 120 s timeout
whenever the vendored shared object is missing:

```python
result = subprocess.run(
    ["make", "libfastbase64.so"],
    cwd=str(_VENDOR_FASTBASE64_DIR),
    capture_output=True, text=True, check=False, timeout=120,
)
```

This runs from the GUI/import path, inside the user's session, with the user's privileges.
Consequences:

- It fails silently on machines without a compiler or `make` (acceptable), but succeeds
  by writing a `.so` into the checkout (not acceptable for read-only installs).
- The compiler is chosen from `PATH`; a poisoned `make`/`cc` is executed implicitly.
- The project's own documentation already treats the vendored backend as optional and
  adaptive, so there is no need for a runtime build.

**Resolution.** Delete `_ensure_vendor_fastbase64_shared_lib()` from the import path. Build
the library only in `setup-mdexplore.sh` (explicit, user-visible) or ship it as a release
artifact; when absent, fall back to `pybase64`/stdlib exactly as the code already does.
If a self-build is retained, gate it behind `MDEXPLORE_BUILD_VENDOR_BASE64=1` and require
an explicit interactive confirmation.

### SEC-02 (P1) — Unpinned executable downloads

`setup-mdexplore.sh` downloads:

- `https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js` (floating major version)
- `https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js` (floating major)
- `https://github.com/plantuml/plantuml/releases/latest/download/plantuml.jar` (**`latest`**)
- `https://sh.rustup.rs` piped to a shell for the Rust toolchain

No checksum or signature is verified before the files are used, and `plantuml.jar` is
subsequently executed via `java -jar`. `latest` means the bytes change without any commit
to this repository, so a compromised upstream or CDN serves arbitrary code into the app.

**Resolution.**

1. Pin exact versions (`mathjax@3.2.x`, `mermaid@11.x.y`, a specific PlantUML release tag)
   and record SHA-256 digests in the script.
2. Verify with `sha256sum -c` after download; abort on mismatch.
3. Prefer an `integrity`/signature check where the upstream publishes one.
4. For Rust, use a pinned `rustup-init` binary with a published checksum instead of
   `curl | sh`, or document that Rust is only needed for development.

### SEC-03 (P1) — WebEngine settings diverge between the two apps

`mdexplore.py:2369-2377` sets:

```python
LocalContentCanAccessRemoteUrls, True
LocalContentCanAccessFileUrls, True
```

`pdfexplore/app.py:4410-4413` sets `LocalContentCanAccessRemoteUrls` to **False** and
`LocalContentCanAccessFileUrls` to True. The Markdown preview is the higher-risk surface:
it renders user-controlled Markdown, embedded HTML, SVG data URIs, and remote images, and
it has JavaScript enabled for Mermaid/MathJax. Allowing local content to reach remote URLs
means a malicious document can exfiltrate local file contents through image beacons or
`fetch` calls that look like ordinary remote image loads.

**Resolution.**

- Extract one `webengine_settings_profile()` into `mdexplore_app` and apply it to both
  windows. Start from the stricter `pdfexplore` profile (`remote=False`) and allow remote
  URLs only for explicitly sanitised image sources, or route remote images through a
  restricted URL scheme/interceptor.
- Add the regression test recommended in ARCH-02.
- If remote access is genuinely required for remote images, document the threat model in
  `DEVELOPERS-AGENTS.md` and sanitise `src`/`href` attributes accordingly.

### SEC-04 (P3) — Hardcoded helpers

- `mdexplore.py:13716` launches `subprocess.Popen(["/usr/bin/code", ...])`. On systems
  where VS Code lives elsewhere (Snap, Flatpak, `code-inside`, VS Codium) the action
  silently fails.
- Clipboard helpers at `mdexplore.py:11135-11146` call `wl-copy`/`xclip` unconditionally
  in sequence. That is acceptable best-effort code, but the failures are swallowed, so a
  user with no clipboard tool gets no feedback.

**Resolution.** Resolve the editor via `shutil.which("code")` with a configurable override
(environment variable or setting), and surface a one-line status message when no clipboard
backend is available instead of failing silently.

---

## 8. Testing infrastructure

### TEST-01 (P1) — No CI and no automated gate

There is no `.github/workflows/`, no `.gitlab-ci.yml`, no `Jenkinsfile`, and no
`Makefile`. Everything the audit measured was run by hand. In practice that means:

- The 454-test suite is not proven to pass end-to-end; only the 64 non-GUI tests were run
  here, because the GUI tests need a display and (for some) a real Chromium/WebEngine
  session.
- The 103 `ruff` findings accumulated unnoticed.
- The Node suite runs tests that a fresh clone would not even contain (REPO-01).
- The mid-feature working tree (REPO-06) cannot be validated as a unit.

**Resolution.** Add a minimal GitHub Actions workflow with three jobs:

```yaml
name: ci
on: [push, pull_request]
jobs:
  python:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.10"}
      - run: sudo apt-get update && sudo apt-get install -y libegl1 libxkbcommon0 libgl1
      - run: pip install -r requirements.txt ruff pytest
      - run: ruff check .
      - run: python -m py_compile mdexplore.py mdexplore_app/*.py pdfexplore/*.py
      - run: xvfb-run -a python -m pytest -q
  node:
    runs-on: ubuntu-22.04
    defaults: {run: {working-directory: mdext}}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: {node-version: "22"}
      - run: npm ci
      - run: npm run check
      - run: npm test
  shell:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get install -y shellcheck
      - run: shellcheck -S warning *.sh
      - run: bash -n *.sh
```

`xvfb-run` is the key line: it makes the GUI tests runnable headlessly without weakening
them. If WebEngine cannot initialise under Xvfb, mark those specific modules with a
`pytest.mark.gui` marker and run them in a separate, allowed-to-fail job until the
environment is sorted out.

### TEST-02 (P1) — No `conftest.py`, no headless policy, no markers

`tests/conftest.py` does not exist. There is no shared `QApplication` fixture, no
`QT_QPA_PLATFORM=offscreen` default, and no `pytest` marker configuration. Sixteen test
modules import `QApplication`/`QTest` and call `show()`, and none of them declares a skip
condition for an environment without a display. Running `pytest` on a headless machine
therefore fails in an environment-dependent way rather than skipping with a clear message.

**Resolution.** Add `tests/conftest.py` with:

- `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")` before any Qt import.
- A session-scoped `qapp` fixture that creates one `QApplication` and keeps it alive
  (multiple `QApplication` instances in one process are unsupported and are a common
  source of flaky GUI tests).
- Registration of `gui` and `slow` markers, and `addopts = "-m 'not slow'"` in
  `pyproject.toml`.
- A `requires_display` skip helper using `QGuiApplication.platformName()`.

This also gives REPO-01's Node tests a mirror: one documented way to run each suite.

### TEST-03 (P2) — Undeclared dev dependencies and no timeout plugin

`pytest 9.0.2` is installed in `.venv` but is not in `requirements.txt`, nor is `ruff`,
`pytest-timeout`, or any coverage tool. An attempted `pytest --timeout=120` failed with
`unrecognized arguments: --timeout=120`, confirming `pytest-timeout` is absent. GUI tests
that wait on `runJavaScript` callbacks (`tests/test_preview_regressions.py` and others do
this with a 10 s `QTimer`) can hang the suite indefinitely on a slow machine.

**Resolution.** Add `requirements-dev.txt`:

```text
-r requirements.txt
pytest>=9.0,<10.0
pytest-timeout>=2.3,<3.0
ruff>=0.12,<1.0
coverage[toml]>=7.6,<8.0
```

and set `timeout = 120` in `[tool.pytest.ini_options]`. CI installs
`requirements-dev.txt`; the runtime `requirements.txt` stays lean.

---

## 9. Configuration, dependencies, performance, and documentation

### CFG-01 (P3) — Malformed settings silently fall back to defaults

`mdexplore_app/constants.py` loads `mdexplore.settings.json` through `_setting()`. On a
JSON decode error or a non-dict payload it returns the built-in defaults without any
warning (the handler is a bare `except Exception: return payload`). Because this file
carries nearly every tunable in the application (search thread counts, PDF print geometry,
Mermaid cache limits, highlight colors), a single trailing comma silently reverts the whole
application to defaults. The user has no way to notice short of diffing behavior.

**Resolution.** Emit one `warnings.warn(...)` (or the project's debug-log helper) with the
file path and the decode error, and expose the failure in the `--debug` status line. Keep
the fallback so the app still starts. Optionally validate against a JSON Schema in CI so
the tracked `mdexplore.settings.json` cannot regress.

### CFG-02 (P2) — Wide version ranges, no lock file, no hashes

`requirements.txt` uses ranges such as `PySide6>=6.7.0,<7.0.0`,
`cmarkgfm>=2024.11.20,<2027.0.0`, and `pypdf>=4.2.0,<6.0.0`. Those are reasonable
compatibility bounds, but there is no `constraints.txt`, no `pip-compile`/`uv` lock, and
no hash pinning, so two machines can resolve different Qt/Chromium or cmarkgfm builds.
Given that the app's behavior depends heavily on Qt WebEngine and Chromium, that is a real
reproducibility gap. The Node side is better: `mdext/package-lock.json` exists.

**Resolution.** Generate a `constraints.txt` (or `requirements.lock`) with
`pip-compile --generate-hashes` from `requirements.txt`, use it in CI and in
`setup-mdexplore.sh` (`pip install -c constraints.txt -r requirements.txt`), and refresh
it deliberately. Keep the human-readable ranges in `requirements.txt`.

### PERF-01 (P3) — Production base64 calls carry benchmark machinery

`mdexplore_app/fast_base64.py` defaults `ADAPTIVE_VENDOR_ROUTING_ENABLED = True` and
`ADAPTIVE_VENDOR_SAMPLE_EVERY_N_CALLS = 48`. Encode/decode therefore route through bucket
statistics, may run both the vendor and fallback implementations on sampled calls, and
periodically persist a JSON state file (see REPO-02). For the dominant use cases — many
small image `data:` URIs and copy-time prefetch — the extra branching and periodic
double-work outweigh the tuning benefit, and it makes timing-sensitive tests noisier.

**Resolution.** Default adaptive routing to **off** and enable it only with
`MDEXPLORE_BASE64_ADAPTIVE=1` for benchmarking. When off, call the chosen backend
directly with no sampling, no logging, and no disk writes. Keep the benchmark script as a
separate maintenance tool rather than a runtime path.

### DOC-01 (P3) — Documentation and default drift

Observed inconsistencies:

- `README.md` documents `MDEXPLORE_MARKDOWN_ENGINE` defaulting to `cmark`, which matches
  `mdexplore.py`.
- `mdexplore.sh` force-defaults the Mermaid backend to **rust** unless the user passes
  `--mermaid-backend`, while `mdexplore_app/constants.py` defines
  `MERMAID_BACKEND_JS = "js"` as the in-process default and `mdext/package.json` defaults
  to `"rust"`. Three surfaces, two different effective defaults, depending on entry point.
- `MARKDOWN-FORMS.md` exists at the repository root and again at `mdext/MARKDOWN-FORMS.md`;
  the latter is generated by `npm run build`. The two must be kept byte-identical but
  nothing checks that.
- `AGENTS.md` says startup-generated icons persist in `~/.cache/mdexplore/icon-cache`, but
  `fastbase64` state goes to the repo root (REPO-02), so the documented cache convention is
  not applied uniformly.

**Resolution.** Pick one default per feature and state it in exactly one place
(`mdexplore_app/constants.py`), have the launcher and extension read/document that value,
and add a small documentation-consistency test (the repo already has
`tests/test_template_assets.py` and `tests/test_js_assets.py` doing structural checks).
For the duplicated forms doc, either generate both from one source or add the
"generated files are current" check from REPO-04.

### DOC-02 (P3) — No application version or changelog

`mdexplore.py` and `mdexplore_app` expose no `__version__`; `mdext/package.json` is at
`0.1.18`. There is no `CHANGELOG.md`. Bug reports therefore cannot identify which build is
running, and the two deliverables (desktop app, VS Code extension) have no shared version
story even though they share the forms feature and assets.

**Resolution.** Add `__version__ = "0.2.0"` in `mdexplore_app/__init__.py`, print it in
`--version` and in the status bar tooltip, and add a `CHANGELOG.md` with Keep-a-Changelog
sections. Bump both the Python and `mdext` versions in the same release PR.

---

## 10. Recommended remediation plan

Ordered so that each step unlocks the next and no step requires the later ones. The first
three steps are cheap and should land before any new feature work resumes.

### Step 0 — Make the tree coherent (½ day)

1. Branch: `git switch -c feature/markdown-forms`.
2. Commit the forms work as one unit: `mdexplore_app/form_fields.py`,
   `mdexplore_app/form_insertion.py`, `assets/js/preview/form_fields.js`,
   `mdext/src/formFields.js`, `mdext/src/formInsertion.js`, `mdext/media/formFields.js`,
   `MARKDOWN-FORMS.md`, `mdext/MARKDOWN-FORMS.md`, the four new Python test modules, and
   the tracked-file modifications they depend on.
3. In that commit, apply the CONC-01 lock fix and add the stale-lock tests. The feature
   should not be merged without it.
4. Run `git status` and confirm nothing is left untracked that a build depends on
   (REPO-01, REPO-06).

**Exit criteria:** a fresh `git clone` + `pip install -r requirements.txt` +
`npm --prefix mdext ci` runs the same suites the author runs.

### Step 1 — Hygiene and gates (1 day)

1. `git rm --cached fastbase64-adaptive.json`; move the adaptive/benchmark paths to
   `~/.cache/mdexplore/` (REPO-02); default adaptive routing off (PERF-01).
2. Anchor the over-broad ignore patterns (REPO-05) and fix the `mdext/test` policy
   (REPO-01).
3. Add `pyproject.toml` with `[tool.ruff]` and `[tool.pytest.ini_options]` (LINT-03,
   TEST-03).
4. Run `ruff check --fix`, then expand the `E701`/`E702` one-liners in the form modules
   (LINT-01).
5. Add `requirements-dev.txt` and `tests/conftest.py` (TEST-02, TEST-03).
6. Add the CI workflow (TEST-01). Get it green, then mark it required.

**Exit criteria:** CI enforces `ruff`, `py_compile`, Python tests under `xvfb-run`, Node
`check` + `test`, and `shellcheck`; the suite passes from a clean clone.

### Step 2 — Security and supply chain (1–2 days)

1. Remove the runtime `make` call (SEC-01); build the vendored library only in setup or
   ship it as an artifact.
2. Pin and checksum the setup downloads (SEC-02).
3. Extract a single WebEngine settings profile and apply it to both windows, defaulting
   to `LocalContentCanAccessRemoteUrls=False` (SEC-03); add the divergence regression
   test (ARCH-02).
4. Replace `/usr/bin/code` with a resolved executable and surface clipboard failures
   (SEC-04).

**Exit criteria:** no runtime compilation, no unpinned executable download, one WebEngine
profile, and a test proving both windows use it.

### Step 3 — Durability and architecture (incremental)

1. Add `fsync_directory()` and use it after every atomic rename (CONC-02).
2. Extract modules from `mdexplore.py` in the order given in ARCH-01, one PR per module,
   keeping `MdExploreWindow` as the composition root. Stop when the file is under ~5,000
   lines; do not chase a number.
3. Add the cross-language form fixtures (ARCH-04).

**Exit criteria:** each extraction is behavior-preserving (existing tests unchanged and
passing), and `mdexplore_app` gains unit tests independent of a live window.

### Step 4 — Polish (ongoing)

1. Narrow broad exceptions in `mdexplore_app` and log them under `--debug` (CONC-03).
2. Add `constraints.txt` with hashes (CFG-02).
3. Fix documentation/default drift and add a consistency test (DOC-01).
4. Add `__version__` and `CHANGELOG.md` (DOC-02).
5. Decide the vendored-artifact policy and shrink history if desired (REPO-03, REPO-04).

---

## 11. What was verified vs. what was not

**Verified by execution during this audit:**

- `py_compile` over all first-party Python modules.
- `pytest --collect-only` → 454 tests discovered.
- 64 non-GUI tests executed and passed (base64, file coordination, forms, search query,
  highlight labels, PDF layout hints).
- `npm --prefix mdext test` → 54 passed; `node --check` over all source and preview JS.
- `bash -n` and `shellcheck -S warning` over the four launcher scripts.
- Import smoke test for 10 core modules.
- `ruff check` → 103 errors, 25 auto-fixable (counts and file attribution above).
- `git check-ignore -v` for the `mdext/test` policy.
- `git ls-files | du -k` for tracked large objects; `du -sh .git` → 44 MB.
- WebEngine settings values read directly from both `mdexplore.py` and
  `pdfexplore/app.py`.
- The `.mdform.lock` acquisition/release code read in both the Python and Node
  implementations; absence of stale handling confirmed by inspection.

**Not verified (and why):**

- The full 454-test Python suite was **not** executed end-to-end. It contains 16 GUI
  modules that need a display and real Qt WebEngine; the audit environment has no
  configured display and the suite has no `conftest.py`/headless policy (TEST-02). The
  64-test non-GUI subset is the executed evidence; GUI failures cannot be ruled out.
- No runtime UI behavior was exercised (no screenshot or interaction testing). Findings
  about UI invariants are based on source and documentation, not on observed behavior.
- The Rust Mermaid renderer (`mmdr`) was not built or run; its absence affects only the
  default backend path chosen by `mdexplore.sh`.
- Severity ratings are engineering judgement based on impact and likelihood; they are
  not derived from a formal risk model.

---

## 12. Summary of the highest-value fixes

If only five changes are made, make these:

1. **CONC-01** — give `.mdform.lock` stale-lock recovery (P0: a crash currently disables
   editing permanently).
2. **TEST-01 / TEST-02** — add CI with `xvfb-run` and a `conftest.py` so the 454-test
   suite is actually a gate (P1: everything else is unenforced without it).
3. **REPO-01 / REPO-02** — make the test suite reproducible from a clone and stop writing
   runtime state into the checkout (P1: reproducibility and read-only installs).
4. **SEC-03** — unify WebEngine settings on the stricter profile (P1: the Markdown
   preview is the untrusted-content surface).
5. **SEC-01 / SEC-02** — remove runtime compilation and checksum the setup downloads
   (P1: supply chain).

The architectural work (ARCH-01/02) is the largest item but is explicitly incremental: it
should follow the gates, not block them.

<!-- AUDIT-END -->