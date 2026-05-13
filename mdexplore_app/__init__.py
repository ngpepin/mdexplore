"""Internal support package for mdexplore.

Keep this package limited to low-risk extraction boundaries:

- leaf helpers (`constants`, `runtime`, `pdf`, `icons`)
- extracted pure logic/helpers (`search`, `js`, `templates`)
- worker implementations (`workers`)
- reusable UI support classes that do not own main-window orchestration
  (`file_tree`, `tree`, `tabs`)

`mdexplore.py` remains the application entrypoint and primary orchestrator.
"""
