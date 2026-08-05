'use strict';

const vscode = require('vscode');
const { DiagramRenderers } = require('./diagramRenderers');
const { PreviewCoordinator } = require('./previewCoordinator');

function activate(context) {
  const getConfiguration = () => vscode.workspace.getConfiguration('mdExt');
  const diagramRenderers = new DiagramRenderers(context, getConfiguration);
  const coordinator = new PreviewCoordinator(context, diagramRenderers);

  const customEditorProvider = {
    resolveCustomTextEditor(document, webviewPanel) {
      coordinator.attachCustomEditor(document, webviewPanel);
    },
  };

  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider('mdExt.markdownEditor', customEditorProvider, {
      webviewOptions: { retainContextWhenHidden: true },
      supportsMultipleEditorsPerDocument: true,
    }),
    vscode.commands.registerCommand('mdExt.previewCurrent', () => coordinator.previewCurrent()),
    vscode.commands.registerCommand('mdExt.openAsEditor', () => coordinator.openAsEditor()),
    vscode.commands.registerCommand('mdExt.refreshPreview', () => coordinator.refreshVisible()),
    vscode.commands.registerCommand('mdExt.openSource', () => coordinator.openSource(coordinator.currentMarkdownUri())),
    vscode.window.onDidChangeActiveTextEditor((editor) => coordinator.onActiveEditorChanged(editor)),
    vscode.window.onDidChangeTextEditorVisibleRanges((event) => coordinator.onTextEditorVisibleRangesChanged(event)),
    vscode.workspace.onDidChangeTextDocument((event) => coordinator.onDocumentChanged(event)),
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration('mdExt')) {
        coordinator.refreshVisible();
      }
    }),
    { dispose: () => coordinator.dispose() },
  );

  coordinator.onActiveEditorChanged(vscode.window.activeTextEditor);
}

function deactivate() {
  // VS Code disposes registered providers, commands, and listeners.
}

module.exports = { activate, deactivate };
