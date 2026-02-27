# mdexplore UML

This document provides a comprehensive PlantUML view of the current `mdexplore` implementation.
All diagrams are embedded so they can be rendered directly by mdexplore (or any Markdown viewer with PlantUML support).

## 1. System Architecture

```plantuml
@startuml
skinparam componentStyle rectangle
skinparam shadowing false
skinparam ArrowColor #4b5563
skinparam defaultTextAlignment center

actor User

node "Ubuntu Desktop Session" as Desktop {
  component "mdexplore.sh\nLauncher" as Launcher
  component "mdexplore.py\nmain()" as AppEntry
  component "MdExploreWindow\n(QMainWindow)" as Window
  component "QWebEngineView\nPreview Pane" as WebView
  component "ColorizedMarkdownModel\n(QFileSystemModel)" as Model
  component "MarkdownRenderer\n(markdown-it + HTML template)" as Renderer
  component "ViewTabBar\n(multi-view tabs)" as ViewTabs
  component "PreviewRenderWorker\n(QThreadPool: render)" as PreviewWorker
  component "PlantUmlRenderWorker\n(QThreadPool: plantuml)" as PumlWorker
  component "PdfExportWorker\n(QThreadPool: pdf)" as PdfWorker
  database "Preview Cache\nmtime+size+html" as PreviewCache
  database "PlantUML Result Cache\nstatus/payload by hash" as PumlCache
  database "Mermaid SVG Cache\nauto/pdf by hash" as MermaidCache
}

folder "Markdown directories" as MdFS
file "~/.mdexplore.cfg" as UserCfg
file "<dir>/.mdexplore-colors.json" as ColorCfg
database "System Clipboard\nURLs + GNOME copied-files MIME" as Clipboard
component "VS Code CLI\ncode" as Vscode
file "vendor/plantuml/plantuml.jar" as PlantJar
node "Java Runtime" as Java
file "vendor/mathjax + vendor/mermaid" as LocalAssets
cloud "CDN fallback\n(MathJax/Mermaid)" as CDN

User --> Launcher : start app
Launcher --> AppEntry : exec .venv/bin/python mdexplore.py [PATH]
Launcher --> LocalAssets : detect local JS paths
AppEntry --> Window : construct UI + timers + pools
Window --> Model : tree model and color persistence
Window --> WebView : setHtml + JS bridge
Window --> Renderer : markdown -> HTML
Window --> ViewTabs : tab state, drag/reorder
Window --> PreviewWorker : optional background render jobs
Window --> PumlWorker : async diagram jobs
Window --> PdfWorker : async footer stamping
Window --> PreviewCache
Window --> PumlCache
Window --> MermaidCache

Model --> MdFS : list dirs + *.md
Model <--> ColorCfg : read/write highlight metadata
Window <--> UserCfg : persist effective root on close
Window --> Clipboard : copy files/text/source markdown
Window --> Vscode : Edit action
Renderer --> PlantJar : local PlantUML render request
PlantJar --> Java : java -jar
Renderer --> LocalAssets : prefer local MathJax/Mermaid
Renderer --> CDN : fallback when local assets unavailable
Window --> MdFS : read markdown files
@enduml
```

## 2. Launcher Runtime Activity (`mdexplore.sh`)

```plantuml
@startuml
start
:Resolve SCRIPT_DIR and key paths;
if (--help?) then (yes)
  :Print usage;
  stop
endif

:Parse optional PATH / file:// URI args;
:Ignore empty desktop placeholders (%u/%U/%f/%F);
if (GUI display available?) then (no)
  :Log and exit with guidance;
  stop
endif

if (.venv exists?) then (no)
  :python3 -m venv .venv;
endif

:Compute requirements hash;
if (hash changed OR fresh venv?) then (yes)
  :pip install -r requirements.txt;
else (no)
  :Skip reinstall;
endif

:Runtime import check
(markdown_it, linkify_it,
PySide6.QtWebEngineWidgets,
pypdf, reportlab...);
if (imports missing?) then (yes)
  :force-reinstall requirements;
  :recheck imports;
endif

:Configure local renderer overrides
(MDEXPLORE_MATHJAX_JS / MDEXPLORE_MERMAID_JS);
if (TARGET_PATH resolved?) then (yes)
  :Launch mdexplore.py TARGET_PATH;
else (no)
  :Launch mdexplore.py (cfg/home fallback);
endif
stop
@enduml
```

## 3. Core Class Diagram

```plantuml
@startuml
skinparam classAttributeIconSize 0
skinparam shadowing false

class ColorizedMarkdownModel {
  +COLOR_FILE_NAME: str
  -_dir_color_map: dict[str, dict[str, str]]
  -_loaded_dirs: set[str]
  -_search_match_paths: set[str]
  +data(index, role)
  +set_color_for_file(path, color_name)
  +collect_files_with_color(root, color_name): list[Path]
  +clear_all_highlights(root): int
  +set_search_match_paths(paths)
  +clear_search_match_paths()
}

class MarkdownRenderer {
  -_mathjax_local_script: Path|None
  -_mermaid_local_script: Path|None
  -_plantuml_jar_path: Path|None
  -_plantuml_setup_issue: str|None
  -_plantuml_svg_cache: dict[str, str]
  -_md: MarkdownIt
  +render_document(markdown_text, title, plantuml_resolver=None): str
  -_render_plantuml_data_uri(code): (str|None, str|None)
  -_prepare_plantuml_source(code): str
  -_prepare_mermaid_source(code): str
}

class PreviewRenderWorker {
  +path: Path
  +request_id: int
  +signals: PreviewRenderWorkerSignals
  +run()
}
class PreviewRenderWorkerSignals {
  +finished(request_id, path_key, html_doc, mtime_ns, size, error)
}

class PlantUmlRenderWorker {
  +hash_key: str
  +prepared_code: str
  +jar_path: Path|None
  +setup_issue: str|None
  +signals: PlantUmlRenderWorkerSignals
  +run()
}
class PlantUmlRenderWorkerSignals {
  +finished(hash_key, status, payload)
}

class PdfExportWorker {
  +output_path: Path
  +pdf_bytes: bytes
  +signals: PdfExportWorkerSignals
  +run()
}
class PdfExportWorkerSignals {
  +finished(output_path_text, error_text)
}

class ViewTabBar {
  +PASTEL_SEQUENCE: list[str]
  +POSITION_BAR_SEGMENTS: int
  -_drag_candidate_index: int
  -_dragging_index: int
  +paintEvent(event)
  +mousePressEvent(event)
  +mouseMoveEvent(event)
  +mouseReleaseEvent(event)
  +tabSizeHint(index): QSize
}

class MdExploreWindow {
  +MAX_DOCUMENT_VIEWS: int = 8
  +HIGHLIGHT_COLORS: list[(name,hex)]
  -root: Path
  -current_file: Path|None
  -renderer: MarkdownRenderer
  -cache: dict[path_key -> (mtime,size,html)]
  -_plantuml_results: dict[hash -> (status,payload)]
  -_mermaid_svg_cache_by_mode: dict[mode -> dict[hash->svg]]
  -_preview_scroll_positions: dict[key->y]
  -_document_view_sessions: dict[path_key->session]
  +_set_root_directory(new_root)
  +_load_preview(path)
  +_refresh_directory_view()
  +_run_match_search()
  +_export_current_preview_pdf()
  +_copy_preview_selection_as_source_markdown(...)
  +_copy_highlighted_files_to_clipboard(color)
  +_add_document_view()
  +closeEvent(event)
}

class QApplication
class QWebEngineView
class QFileSystemModel
class QThreadPool

MdExploreWindow *-- MarkdownRenderer
MdExploreWindow *-- ColorizedMarkdownModel
MdExploreWindow *-- ViewTabBar
MdExploreWindow *-- QWebEngineView
MdExploreWindow o-- PreviewRenderWorker
MdExploreWindow o-- PlantUmlRenderWorker
MdExploreWindow o-- PdfExportWorker
MdExploreWindow o-- QThreadPool
ColorizedMarkdownModel --|> QFileSystemModel
PreviewRenderWorker --> PreviewRenderWorkerSignals
PlantUmlRenderWorker --> PlantUmlRenderWorkerSignals
PdfExportWorker --> PdfExportWorkerSignals
PreviewRenderWorker ..> MarkdownRenderer : creates renderer in worker
MdExploreWindow ..> QApplication
@enduml
```

## 4. Preview Load + Progressive Diagram Restore

```plantuml
@startuml
actor User
participant "QTreeView" as Tree
participant "MdExploreWindow" as Win
participant "Preview Cache" as Cache
participant "Markdown File" as File
participant "MarkdownRenderer" as Renderer
participant "QWebEngineView" as Web
participant "PlantUmlRenderWorker" as PumlWorker
participant "java -jar plantuml.jar" as PlantJar
participant "In-page Mermaid Runtime" as MermaidJs

User -> Tree : click *.md file
Tree -> Win : _on_tree_selection_changed(path)
Win -> Win : _load_preview(path)
Win -> File : stat + read (mtime,size,text)
Win -> Cache : lookup(path_key)

alt cache hit (same mtime+size)
  Cache --> Win : html_doc (contains placeholders + hash metadata)
  Win -> Web : setHtml(cached_html)
  par progressive restore (non-blocking)
    Win -> Web : apply ready PlantUML results in small batches
    Web -> MermaidJs : mark cached Mermaid blocks as "Mermaid rendering..."
    MermaidJs -> MermaidJs : hydrate cached Mermaid SVG in small batches
  end
else cache miss/stale
  Win -> Renderer : render_document(text,title, plantuml_resolver)
  loop each PlantUML fence
    Renderer -> Win : plantuml_resolver(code, index, attrs)
    Win -> Win : register placeholder ids by hash/doc
    alt not already completed
      Win -> PumlWorker : start(hash, prepared_code)
    end
    Win --> Renderer : always emit "PlantUML rendering..." placeholder
  end
  Renderer --> Win : html_doc with placeholders
  Win -> Cache : store(path_key,mtime,size,html)
  Win -> Web : setHtml(html_doc)
end

par background diagram completion
  note over Win,PumlWorker
    PlantUML worker pool runs in parallel
    (cpu-aware cap, up to 6 workers)
  end note
  PumlWorker -> PlantJar : subprocess run -pipe -tsvg
  PlantJar --> PumlWorker : svg or stderr
  PumlWorker --> Win : finished(hash,status,payload)
  Win -> Win : update plantuml result cache
  Win -> Cache : invalidate docs using this hash
  Win -> Web : runJavaScript patch placeholder nodes in-place
end

Win -> Win : show status progress\n(ready/pending/failed counts)
@enduml
```

## 5. Search + Highlight + Apply Color Flow

```plantuml
@startuml
actor User
participant "Search QLineEdit" as Search
participant "QTimer(1s debounce)" as Debounce
participant "MdExploreWindow" as Win
participant "Filesystem Scope" as Scope
participant "ColorizedMarkdownModel" as Model
participant "QTreeView" as Tree
participant "QWebEngineView" as Web

User -> Search : type query
Search -> Win : _on_match_text_changed(text)
Win -> Debounce : start/restart

alt user presses Enter
  User -> Search : Enter
  Search -> Win : _run_match_search_now()
  Win -> Debounce : stop
end

Debounce -> Win : timeout -> _run_match_search()
Win -> Win : compile predicate\n(Boolean + implicit AND + quotes + CLOSE)
Win -> Scope : list direct *.md files (non-recursive)
loop each file
  Win -> Scope : read file name + content
  Win -> Win : predicate(name, content)
end
Win -> Model : set_search_match_paths(matches)
Model --> Tree : bold+italic matched rows
Win -> Web : highlight matches in preview (if open file matched)

alt user clicks a color button next to Search
  User -> Win : _apply_match_highlight_color(color)
  Win -> Model : set_color_for_file(file,color)
  Model -> Scope : persist .mdexplore-colors.json
  Win -> Model : clear_search_match_paths()
  Model --> Tree : remove bolding
  Win -> Web : remove search marks
end
@enduml
```

## 6. Preview Context Menu: Copy Source Markdown

```plantuml
@startuml
actor User
participant "QWebEngineView" as Web
participant "MdExploreWindow" as Win
participant "Source .md file" as Src
participant "System Clipboard" as Clip
participant "wl-copy/xclip/xsel" as CliClip

User -> Web : right-click selected preview text
Web -> Win : _show_preview_context_menu(pos)
Win -> Web : runJavaScript(selection + line metadata)
Web --> Win : selection_info
Win -> User : menu with "Copy Source Markdown"
User -> Win : choose Copy Source Markdown
Win -> Src : read all lines

alt line-range metadata exists
  Win -> Win : slice exact source line range
else direct text mapping succeeds
  Win -> Win : map selected text to source span
else fuzzy line mapping succeeds
  Win -> Win : anchor with first/last meaningful lines
else fallback
  Win -> Win : use entire source file
end

Win -> Clip : setText(Clipboard + Selection)
Win -> CliClip : optional reliability fallback
Win -> User : status message (exact/fuzzy/full fallback)
@enduml
```

## 7. PDF Export Pipeline

```plantuml
@startuml
actor User
participant "MdExploreWindow" as Win
participant "QWebEngineView" as Web
participant "Preview JS Runtime\n(MathJax/Mermaid/fonts)" as JsRuntime
participant "Qt printToPdf" as QtPdf
participant "PdfExportWorker" as PdfWorker
participant "pypdf + reportlab" as PdfLib
participant "output *.pdf" as OutPdf

User -> Win : click PDF
Win -> Win : _export_current_preview_pdf()
Win -> Win : set busy state + disable PDF button
Win -> Web : _prepare_preview_for_pdf_export()
Web -> JsRuntime : enter PDF print mode\n(force print CSS + Mermaid monochrome path)

loop precheck attempts (max 60)
  Web -> JsRuntime : check mathReady/mermaidReady/fontsReady
  JsRuntime --> Web : readiness flags
  Web --> Win : _on_pdf_precheck_result(...)
  alt all ready
    break
  else not ready and attempts remain
    Win -> Win : wait 140ms and retry
  end
end

Win -> QtPdf : printToPdf(callback)
QtPdf --> Win : raw pdf bytes

alt empty/invalid bytes
  Win -> Win : show error + clear busy
else bytes ok
  Win -> PdfWorker : start(output_path, raw_bytes)
  PdfWorker -> PdfLib : stamp "N of M" footer
  PdfLib -> OutPdf : write final numbered PDF
  PdfWorker --> Win : finished(path,error?)
  Win -> Win : restore Mermaid auto palette
  Win -> Win : clear busy + status/update dialog
end
@enduml
```

## 8. Document View/Tab Lifecycle

```plantuml
@startuml
[*] --> NoFile

state NoFile {
  [*] --> Placeholder
  Placeholder : "Select a markdown file"
}

NoFile --> SingleView : select markdown file

state SingleView {
  [*] --> ActiveView1
  ActiveView1 : one logical view
}

SingleView --> MultiView : Add View
MultiView --> MultiView : Add View (max 8)
MultiView --> SingleView : close tabs until one remains

state MultiView {
  [*] --> TabSet
  TabSet : view tabs visible\nfixed width + pastel sequence\nmanual drag reordering
  TabSet : each tab stores\nview_id, sequence, color_slot,\nscroll_y, top_line, progress
}

SingleView --> SingleView : switch files
MultiView --> MultiView : switch files

SingleView --> SessionSaved : switch to another document
MultiView --> SessionSaved : switch to another document
SessionSaved --> SingleView : return document with 1-view session
SessionSaved --> MultiView : return document with multi-view session

SingleView --> NoFile : root change / file removed
MultiView --> NoFile : root change / file removed

NoFile --> [*] : window close (persist effective root)
SingleView --> [*] : window close (persist effective root)
MultiView --> [*] : window close (persist effective root)
@enduml
```

## Notes

- Diagrams are based on current code in `mdexplore.py` and `mdexplore.sh`.
- Worker/threadpool usage is intentionally separated by concern:
  - render pool (preview HTML jobs),
  - PlantUML pool (diagram jobs),
  - PDF pool (post-processing/stamping).
- PlantUML rendering is non-blocking in UI flow: placeholders are rendered first, then patched in place as jobs finish.
- TODO (known issue): diagram zoom/pan restore (Mermaid and PlantUML) is not
  yet consistently reliable when leaving a document and returning in the same app run.
