'use strict';

const path = require('node:path');
const vscode = require('vscode');
const { SEARCH_CLOSE_WORD_GAP, extractSearchTerms, extractNearTermGroups } = require('./searchQuery');
const {
  PREVIEW_HIGHLIGHT_KIND_IMPORTANT,
  PREVIEW_HIGHLIGHT_KIND_NORMAL,
  loadHighlightEntries,
  normalizeHighlightEntries,
  replaceHighlightRange,
  saveHighlightEntries,
} = require('./highlightStore');
const { renderMarkdown } = require('./renderer');
const { PdfExporter } = require('./pdfExporter');
const { getWebviewHtml } = require('./webview');
const {
  isMarkdownPath,
  isExternalHref,
  isMarkdownPath: isMarkdownHref,
  resolveRelativeFile,
} = require('./utils');

const PREVIEW_FONT_STATE_KEY = 'mdExt.previewFontSizes.v1';
const PREVIEW_FONT_GC_KEY = 'mdExt.previewFontSizes.lastGc.v1';
const PREVIEW_FONT_MAX_AGE_MS = 180 * 24 * 60 * 60 * 1000;
const PREVIEW_FONT_GC_INTERVAL_MS = 7 * 24 * 60 * 60 * 1000;
const PREVIEW_FONT_MAX_ENTRIES = 500;

function entriesSignature(entries) {
  return JSON.stringify(normalizeHighlightEntries(entries));
}

function escapeRegexCharacter(character) {
  return '\\^$.*+?()[]{}|'.includes(character) ? `\\${character}` : character;
}

function associationPatternMatchesUri(pattern, uri) {
  let normalizedPattern = String(pattern || '').trim().replace(/\\\\/g, '/');
  const sourcePath = String(uri?.fsPath || uri?.path || '').replace(/\\\\/g, '/');
  if (!normalizedPattern || !sourcePath) {
    return false;
  }

  while (normalizedPattern.startsWith('**/')) {
    normalizedPattern = normalizedPattern.slice(3);
  }
  const target = normalizedPattern.includes('/') ? sourcePath : path.posix.basename(sourcePath);
  let regexSource = '^';
  for (let index = 0; index < normalizedPattern.length; index += 1) {
    const character = normalizedPattern[index];
    if (character === '*') {
      if (normalizedPattern[index + 1] === '*') {
        regexSource += '.*';
        index += 1;
      } else {
        regexSource += '[^/]*';
      }
      continue;
    }
    if (character === '?') {
      regexSource += '[^/]';
      continue;
    }
    if (character === '{') {
      const end = normalizedPattern.indexOf('}', index + 1);
      if (end > index) {
        const alternatives = normalizedPattern
          .slice(index + 1, end)
          .split(',')
          .map((item) => Array.from(item).map(escapeRegexCharacter).join(''))
          .filter(Boolean);
        if (alternatives.length) {
          regexSource += `(?:${alternatives.join('|')})`;
          index = end;
          continue;
        }
      }
    }
    regexSource += escapeRegexCharacter(character);
  }
  regexSource += '$';

  try {
    return new RegExp(regexSource, 'i').test(target);
  } catch {
    return false;
  }
}

class PreviewCoordinator {
  constructor(context, diagramRenderers) {
    this.context = context;
    this.diagramRenderers = diagramRenderers;
    this.activitySurface = null;
    this.customSurfaces = new Set();
    this.lastMarkdownUri = null;
    this.anchorLineByUri = new Map();
    this.renderGeneration = new WeakMap();
    this.debounceTimers = new Map();
    this.pdfExporter = new PdfExporter(context);
    this.sourceScrollSuppressions = new WeakMap();
    this.scrollLeaderByUri = new Map();
    this.splitOriginByUri = new Map();
  }

  get configuration() {
    return vscode.workspace.getConfiguration('mdExt');
  }

  configureWebview(webview, documentUri = null) {
    const roots = [this.context.extensionUri];
    if (documentUri?.scheme === 'file') {
      roots.push(vscode.Uri.file(path.dirname(documentUri.fsPath)));
    }
    for (const folder of vscode.workspace.workspaceFolders || []) {
      roots.push(folder.uri);
    }
    webview.options = {
      enableScripts: true,
      enableForms: false,
      localResourceRoots: roots,
    };
    webview.html = getWebviewHtml(webview, this.context);
  }

  attachActivityView(view) {
    this.configureWebview(view.webview, this.currentMarkdownUri());
    const surface = {
      kind: 'activity',
      webview: view.webview,
      uri: this.currentMarkdownUri(),
      disposed: false,
      searchQuery: '',
    };
    this.activitySurface = surface;
    this.attachMessages(surface);
    view.onDidDispose(() => {
      surface.disposed = true;
      if (this.activitySurface === surface) {
        this.activitySurface = null;
      }
    });
    if (surface.uri) {
      this.renderSurface(surface);
    }
  }

  attachCustomEditor(document, panel) {
    this.attachPreviewPanel(document.uri, panel, document, 'custom');
  }

  attachPreviewPanel(uri, panel, document = null, kind = 'preview') {
    this.configureWebview(panel.webview, uri);
    const surface = { kind, webview: panel.webview, uri, disposed: false, searchQuery: '' };
    this.customSurfaces.add(surface);
    this.lastMarkdownUri = uri;
    this.attachMessages(surface);
    panel.onDidDispose(() => {
      surface.disposed = true;
      this.customSurfaces.delete(surface);
    });
    this.renderSurface(surface, document);
  }

  attachMessages(surface) {
    surface.webview.onDidReceiveMessage(async (message) => {
      const type = String(message?.type || '');
      if (type === 'refresh') {
        await this.renderSurface(surface);
      } else if (type === 'edit') {
        await this.editPreviewedDocument(surface.uri);
      } else if (type === 'setPreviewFontSize') {
        await this.savePreviewFontSize(surface.uri, Number(message.size));
      } else if (type === 'setSearchQuery') {
        await this.updateSearch(surface, String(message.query || ''), Boolean(message.scrollToFirst));
      } else if (type === 'previewScroll') {
        await this.onPreviewScroll(surface, Number(message.line));
      } else if (type === 'copyRenderedText') {
        await this.copyRenderedText(surface, String(message.text || ''));
      } else if (type === 'copySourceMarkdown') {
        await this.copySourceMarkdown(surface, {
          startLine: Number(message.startLine),
          endLine: Number(message.endLine),
          selectedText: String(message.selectedText || ''),
        });
      } else if (type === 'addPersistentHighlight') {
        await this.addPersistentHighlight(surface, {
          start: Number(message.start),
          end: Number(message.end),
          kind: String(message.kind || ''),
          selectedText: String(message.selectedText || ''),
        });
      } else if (type === 'persistentHighlightsResolved') {
        await this.onPersistentHighlightsResolved(surface, message.entries);
      } else if (type === 'createPdf') {
        await this.createPdf(surface, String(message.html || ''));
      } else if (type === 'openSource') {
        await this.openSource(surface.uri);
      } else if (type === 'openLink') {
        await this.openLink(surface.uri, String(message.href || ''));
      } else if (type === 'revealLine') {
        await this.revealSourceLine(surface.uri, Number(message.line));
      }
    });
  }

  currentMarkdownUri(preferredUri = null) {
    const rememberIfMarkdown = (candidate) => {
      if (!candidate || !isMarkdownPath(candidate.fsPath || candidate.path || '')) {
        return null;
      }
      this.lastMarkdownUri = candidate;
      return candidate;
    };

    // Editor-title commands can supply the underlying resource even when the
    // active surface is a custom/webview-based editor (including VS Code's
    // Markdown Editor) and therefore has no activeTextEditor.
    const preferred = rememberIfMarkdown(preferredUri);
    if (preferred) {
      return preferred;
    }

    const editor = vscode.window.activeTextEditor;
    if (editor && editor.document.languageId === 'markdown') {
      this.lastMarkdownUri = editor.document.uri;
      return editor.document.uri;
    }

    const activeTab = vscode.window.tabGroups?.activeTabGroup?.activeTab;
    const input = activeTab?.input;
    const tabCandidates = [
      input?.uri,
      input?.modified,
      input?.original,
      input?.resource,
      input?.notebook?.uri,
      input?.modified?.uri,
      input?.original?.uri,
    ];
    for (const candidate of tabCandidates) {
      const resolved = rememberIfMarkdown(candidate);
      if (resolved) {
        return resolved;
      }
    }

    return this.lastMarkdownUri;
  }

  activeBuiltInTextEditorForUri(uri) {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'markdown' || !uri) {
      return null;
    }
    return editor.document.uri.toString() === uri.toString() ? editor : null;
  }

  activeMarkdownSurfaceCanRemainOpen(uri) {
    // mdExt deliberately interoperates only with VS Code's built-in Text Editor.
    // Any Markdown custom editor/preview surface is reopened as text before the
    // mdExt read-only preview is opened beside it.
    return Boolean(this.activeBuiltInTextEditorForUri(uri));
  }

  activeEditorColumn() {
    return vscode.window.activeTextEditor?.viewColumn
      ?? vscode.window.tabGroups?.activeTabGroup?.viewColumn
      ?? vscode.ViewColumn.Active;
  }

  previewTabForUri(uri) {
    if (!uri) {
      return null;
    }
    const key = uri.toString();
    for (const group of vscode.window.tabGroups?.all || []) {
      for (const tab of group.tabs || []) {
        const input = tab?.input;
        if (input?.viewType === 'mdExt.markdownEditor' && input?.uri?.toString() === key) {
          return tab;
        }
      }
    }
    return null;
  }

  textTabForUri(uri) {
    if (!uri) {
      return null;
    }
    const key = uri.toString();
    for (const group of vscode.window.tabGroups?.all || []) {
      for (const tab of group.tabs || []) {
        const input = tab?.input;
        if (input?.uri?.toString() !== key) {
          continue;
        }
        if (!input?.viewType || input.viewType === 'default') {
          return tab;
        }
      }
    }
    return null;
  }

  activeMdExtTabForUri(uri) {
    const tab = vscode.window.tabGroups?.activeTabGroup?.activeTab;
    return tab?.input?.viewType === 'mdExt.markdownEditor'
      && tab?.input?.uri?.toString() === uri?.toString()
      ? tab
      : null;
  }

  splitPreviewTabForUri(uri, textTab) {
    if (!uri) {
      return null;
    }
    const groups = vscode.window.tabGroups?.all || [];
    const textGroup = groups.find((group) => (group.tabs || []).includes(textTab));
    const key = uri.toString();
    for (const group of groups) {
      if (group === textGroup) {
        continue;
      }
      for (const tab of group.tabs || []) {
        const input = tab?.input;
        if (input?.viewType === 'mdExt.markdownEditor' && input?.uri?.toString() === key) {
          return tab;
        }
      }
    }
    return null;
  }

  async collapseSplitForUri(uri, previewTab, textTab) {
    const key = uri.toString();
    const origin = this.splitOriginByUri.get(key);
    this.splitOriginByUri.delete(key);

    const splitPreviewTab = this.splitPreviewTabForUri(uri, textTab);

    if (origin === 'mdext' && textTab) {
      // A split created from mdExt keeps the original mdExt tab underneath the
      // Text Editor in the left group. Remove the right-hand split preview first,
      // then close the Text Editor so the original rendered mdExt tab is revealed.
      if (splitPreviewTab) {
        await vscode.window.tabGroups.close(splitPreviewTab);
      }
      await vscode.window.tabGroups.close(textTab);
      return true;
    }
    if (origin === 'text' && previewTab) {
      await vscode.window.tabGroups.close(splitPreviewTab || previewTab);
      return true;
    }

    // If the split was not created by this coordinator instance (for example
    // after an extension-host reload), preserve whichever side the user is
    // actively invoking the toggle from.
    if (this.activeMdExtTabForUri(uri) && textTab) {
      await vscode.window.tabGroups.close(textTab);
      return true;
    }
    if (this.activeBuiltInTextEditorForUri(uri) && previewTab) {
      await vscode.window.tabGroups.close(previewTab);
      return true;
    }
    return false;
  }

  onActiveEditorChanged(editor) {
    if (!editor || editor.document.languageId !== 'markdown') {
      return;
    }
    this.lastMarkdownUri = editor.document.uri;
    const line = this.topVisibleLine(editor);
    this.rememberAnchorLine(editor.document.uri, line);
    this.syncPreviewScroll(editor.document.uri, line);
    if (this.activitySurface && !this.activitySurface.disposed) {
      this.activitySurface.uri = editor.document.uri;
      this.renderSurface(this.activitySurface, editor.document);
    }
  }

  onTextEditorVisibleRangesChanged(event) {
    const editor = event?.textEditor;
    if (!editor || editor.document.languageId !== 'markdown') {
      return;
    }
    const line = this.topVisibleLine(editor);
    const suppression = this.sourceScrollSuppressions.get(editor);
    if (suppression) {
      const nearProgrammaticTarget = Math.abs(line - suppression.line) <= 2;
      if (Date.now() < suppression.until && nearProgrammaticTarget) {
        suppression.until = Date.now() + 1200;
        this.rememberAnchorLine(editor.document.uri, line);
        return;
      }
      this.sourceScrollSuppressions.delete(editor);
    }
    if (this.scrollLeader(editor.document.uri)?.kind === 'preview') {
      return;
    }
    this.claimScrollLeadership(editor.document.uri, 'source');
    this.rememberAnchorLine(editor.document.uri, line);
    this.syncPreviewScroll(editor.document.uri, line);
  }

  onDocumentChanged(event) {
    if (event.document.languageId !== 'markdown') {
      return;
    }
    this.lastMarkdownUri = event.document.uri;
    const key = event.document.uri.toString();
    const existing = this.debounceTimers.get(key);
    if (existing) {
      clearTimeout(existing);
    }
    const delay = Math.max(50, Number(this.configuration.get('liveUpdateDelay', 180)) || 180);
    const timer = setTimeout(() => {
      this.debounceTimers.delete(key);
      this.refreshDocumentSurfaces(event.document);
    }, delay);
    this.debounceTimers.set(key, timer);
  }

  refreshDocumentSurfaces(document) {
    if (this.activitySurface && this.activitySurface.uri?.toString() === document.uri.toString()) {
      this.renderSurface(this.activitySurface, document);
    }
    for (const surface of this.customSurfaces) {
      if (surface.uri?.toString() === document.uri.toString()) {
        this.renderSurface(surface, document);
      }
    }
  }

  previewFontState() {
    const stored = this.context.workspaceState?.get?.(PREVIEW_FONT_STATE_KEY, {});
    return stored && typeof stored === 'object' && !Array.isArray(stored) ? stored : {};
  }

  previewFontSizeForUri(uri) {
    const key = uri?.toString();
    if (!key) {
      return null;
    }
    const entry = this.previewFontState()[key];
    const size = Number(entry?.size);
    return Number.isFinite(size) && size >= 8 && size <= 40 ? size : null;
  }

  async savePreviewFontSize(uri, rawSize) {
    const key = uri?.toString();
    const size = Math.round(Math.max(8, Math.min(40, Number(rawSize))) * 10) / 10;
    if (!key || !Number.isFinite(size) || !this.context.workspaceState?.update) {
      return;
    }

    const now = Date.now();
    const state = { ...this.previewFontState(), [key]: { size, touchedAt: now } };
    await this.context.workspaceState.update(PREVIEW_FONT_STATE_KEY, state);
    await this.maybeCollectPreviewFontState(now);

    for (const matchingSurface of this.matchingSurfaces(uri)) {
      if (!matchingSurface.disposed) {
        matchingSurface.webview.postMessage({ type: 'previewFontSize', size });
      }
    }
  }

  async maybeCollectPreviewFontState(now = Date.now()) {
    const workspaceState = this.context.workspaceState;
    if (!workspaceState?.get || !workspaceState?.update) {
      return;
    }
    const lastGc = Number(workspaceState.get(PREVIEW_FONT_GC_KEY, 0)) || 0;
    const state = this.previewFontState();
    if (now - lastGc < PREVIEW_FONT_GC_INTERVAL_MS && Object.keys(state).length <= PREVIEW_FONT_MAX_ENTRIES) {
      return;
    }

    const recentEntries = Object.entries(state)
      .filter(([, entry]) => now - (Number(entry?.touchedAt) || 0) <= PREVIEW_FONT_MAX_AGE_MS)
      .sort((left, right) => (Number(right[1]?.touchedAt) || 0) - (Number(left[1]?.touchedAt) || 0))
      .slice(0, PREVIEW_FONT_MAX_ENTRIES);
    await workspaceState.update(PREVIEW_FONT_STATE_KEY, Object.fromEntries(recentEntries));
    await workspaceState.update(PREVIEW_FONT_GC_KEY, now);
  }

  async renderSurface(surface, knownDocument = null) {
    if (!surface || surface.disposed || !surface.uri) {
      return;
    }
    const generation = (this.renderGeneration.get(surface) || 0) + 1;
    this.renderGeneration.set(surface, generation);
    surface.webview.postMessage({ type: 'renderStarted' });

    try {
      const document = knownDocument || await vscode.workspace.openTextDocument(surface.uri);
      const resolveAsset = (documentPath, original) => {
        const value = String(original || '').trim();
        if (!value || value.startsWith('data:') || /^https?:/i.test(value)) {
          return value;
        }
        const resolvedPath = resolveRelativeFile(documentPath, value);
        if (!resolvedPath) {
          return value;
        }
        return surface.webview.asWebviewUri(vscode.Uri.file(resolvedPath)).toString();
      };
      const result = await renderMarkdown({
        markdown: document.getText(),
        documentPath: document.uri.fsPath,
        resolveAsset,
        diagramRenderers: this.diagramRenderers,
      });
      const persistentHighlights = await loadHighlightEntries(document.uri.fsPath);
      const scrollLine = this.clampDocumentLine(document, this.anchorLineForUri(document.uri));
      if (surface.disposed || this.renderGeneration.get(surface) !== generation) {
        return;
      }
      surface.webview.postMessage({
        type: 'render',
        body: result.body,
        title: path.basename(document.uri.fsPath || document.uri.path),
        path: document.uri.fsPath || document.uri.toString(),
        persistentHighlights,
        searchState: this.searchStateForSurface(surface),
        previewFontSize: this.previewFontSizeForUri(document.uri),
        scrollLine,
        revision: document.version,
      });
    } catch (error) {
      if (!surface.disposed && this.renderGeneration.get(surface) === generation) {
        surface.webview.postMessage({
          type: 'renderError',
          message: String(error?.message || error),
        });
      }
    }
  }

  async previewCurrent(preferredUri = null) {
    const uri = this.currentMarkdownUri(preferredUri);
    if (!uri || !isMarkdownPath(uri.fsPath || uri.path)) {
      vscode.window.showInformationMessage('Open a Markdown document before starting mdExt preview.');
      return;
    }

    const key = uri.toString();
    const existingPreviewTab = this.previewTabForUri(uri);
    const existingTextTab = this.textTabForUri(uri);

    if (existingPreviewTab && existingTextTab) {
      if (await this.collapseSplitForUri(uri, existingPreviewTab, existingTextTab)) {
        return;
      }
    }

    if (existingPreviewTab && this.activeMdExtTabForUri(uri)) {
      // Keep the rendered mdExt tab in the original group, place the built-in
      // Text Editor over it, and open a second mdExt preview to the right. On the
      // next toggle collapseSplitForUri removes that temporary right preview and
      // the Text Editor, revealing the original rendered mdExt tab again.
      const sourceColumn = this.activeEditorColumn();
      this.splitOriginByUri.set(key, 'mdext');
      await vscode.commands.executeCommand('vscode.openWith', uri, 'default', sourceColumn);
      await this.openWithMdExt(uri, vscode.ViewColumn.Beside);
      return;
    }

    if (existingPreviewTab) {
      await vscode.window.tabGroups.close(existingPreviewTab);
      this.splitOriginByUri.delete(key);
      return;
    }

    if (!this.activeMarkdownSurfaceCanRemainOpen(uri)) {
      await vscode.commands.executeCommand('vscode.openWith', uri, 'default', this.activeEditorColumn());
    }

    this.splitOriginByUri.set(key, 'text');
    await this.openWithMdExt(uri, vscode.ViewColumn.Beside);
  }

  async openAsEditor() {
    const uri = this.currentMarkdownUri();
    if (!uri || !isMarkdownPath(uri.fsPath || uri.path)) {
      vscode.window.showInformationMessage('Open a Markdown document before reopening it with mdExt.');
      return;
    }
    await this.openWithMdExt(uri);
  }

  configuredEditorIdForUri(uri) {
    const associations = vscode.workspace
      .getConfiguration('workbench', uri)
      .get('editorAssociations', {});
    const entries = [];

    if (Array.isArray(associations)) {
      for (const association of associations) {
        const pattern = association?.filenamePattern || association?.pattern;
        const editorId = association?.viewType || association?.editor || association?.editorId;
        if (pattern && editorId) {
          entries.push([pattern, editorId]);
        }
      }
    } else if (associations && typeof associations === 'object') {
      entries.push(...Object.entries(associations));
    }

    let configuredEditorId = 'default';
    for (const [pattern, editorId] of entries) {
      if (associationPatternMatchesUri(pattern, uri) && editorId) {
        configuredEditorId = String(editorId);
      }
    }
    return configuredEditorId;
  }

  async editPreviewedDocument(uri) {
    if (!uri || !isMarkdownPath(uri.fsPath || uri.path)) {
      return;
    }

    const viewColumn = this.activeEditorColumn();
    // Editing from mdExt always means returning to VS Code's built-in Text
    // Editor. Do not route through Markdown Editor/Markdown Preview associations.
    await vscode.commands.executeCommand('vscode.openWith', uri, 'default', viewColumn);
  }

  async openWithMdExt(uri, viewColumn = undefined) {
    if (!uri) {
      return;
    }
    const column = viewColumn ?? vscode.window.activeTextEditor?.viewColumn ?? vscode.ViewColumn.Active;
    await vscode.commands.executeCommand('vscode.openWith', uri, 'mdExt.markdownEditor', column);
  }

  async createPdf(surface, html) {
    if (!surface?.uri || !String(html || '').trim()) {
      return;
    }
    const outputPath = this.pdfOutputPath(surface.uri);
    const sourcePath = surface.uri.fsPath || surface.uri.path;
    if (!outputPath || !sourcePath) {
      return;
    }
    try {
      const bytes = await this.pdfExporter.exportHtml({ html, sourcePath });
      await this.writePdfOutputBytes(surface, outputPath, bytes);
    } catch (error) {
      surface.webview.postMessage({
        type: 'status',
        message: `PDF export failed: ${error?.message || error}`,
        persistent: true,
      });
    }
  }

  pdfOutputPath(uri) {
    const sourcePath = uri?.fsPath || uri?.path;
    if (!sourcePath) {
      return '';
    }
    return /\.(?:md|markdown)$/i.test(sourcePath)
      ? sourcePath.replace(/\.(?:md|markdown)$/i, '.pdf')
      : `${sourcePath}.pdf`;
  }

  async writePdfOutputBytes(surface, outputPath, bytes) {
    if (!outputPath || !bytes) {
      return;
    }
    const sourcePath = surface.uri.fsPath || surface.uri.path;
    if (!sourcePath) {
      return;
    }
    try {
      await vscode.workspace.fs.writeFile(vscode.Uri.file(outputPath), bytes);
      surface.webview.postMessage({
        type: 'status',
        message: `PDF saved: ${path.basename(outputPath)}`,
      });
    } catch (error) {
      surface.webview.postMessage({
        type: 'status',
        message: `PDF export failed: ${error?.message || error}`,
        persistent: true,
      });
    }
  }

  async writeClipboard(surface, text, successMessage) {
    try {
      await vscode.env.clipboard.writeText(String(text ?? ''));
      this.postSurfaceStatus(surface, successMessage);
      return true;
    } catch (error) {
      this.postSurfaceStatus(surface, `Copy failed: ${error?.message || error}`, true);
      return false;
    }
  }

  async copyRenderedText(surface, selectedText) {
    const text = String(selectedText || '');
    if (!text.trim()) {
      this.postSurfaceStatus(surface, 'No selected rendered text to copy', true);
      return;
    }
    await this.writeClipboard(surface, text, 'Copied rendered text');
  }

  sourceLinesWithEndings(source) {
    return String(source || '').match(/[^\n]*\n|[^\n]+$/g) || [];
  }

  async copySourceMarkdown(surface, selection) {
    if (!surface?.uri) {
      return;
    }
    try {
      const document = await vscode.workspace.openTextDocument(surface.uri);
      const source = document.getText();
      if (!source) {
        await this.writeClipboard(surface, '', 'Copied source markdown (empty file)');
        return;
      }
      const lines = this.sourceLinesWithEndings(source);
      const requestedStart = Number(selection?.startLine);
      const requestedEnd = Number(selection?.endLine);
      if (Number.isFinite(requestedStart) && Number.isFinite(requestedEnd) && requestedEnd > requestedStart && lines.length) {
        const start = Math.max(0, Math.min(lines.length - 1, Math.floor(requestedStart)));
        const end = Math.max(start + 1, Math.min(lines.length, Math.floor(requestedEnd)));
        await this.writeClipboard(surface, lines.slice(start, end).join(''), `Copied source markdown lines ${start + 1}-${end}`);
        return;
      }

      const selectedText = String(selection?.selectedText || '').trim();
      const index = selectedText ? source.indexOf(selectedText) : -1;
      if (index >= 0) {
        const start = source.slice(0, index).split('\n').length - 1;
        const endOffset = index + selectedText.length;
        const end = Math.max(start + 1, source.slice(0, endOffset).split('\n').length);
        await this.writeClipboard(surface, lines.slice(start, Math.min(lines.length, end)).join(''), `Copied source markdown lines ${start + 1}-${Math.min(lines.length, end)} (fallback)`);
        return;
      }
      this.postSurfaceStatus(surface, 'Could not map the preview selection to source Markdown', true);
    } catch (error) {
      this.postSurfaceStatus(surface, `Could not read source markdown: ${error?.message || error}`, true);
    }
  }

  async openSource(uri) {
    if (!uri) {
      return;
    }
    const document = await vscode.workspace.openTextDocument(uri);
    const editor = await vscode.window.showTextDocument(document, { preview: false });
    const line = this.clampDocumentLine(document, this.anchorLineForUri(uri));
    const position = new vscode.Position(line, 0);
    editor.revealRange(new vscode.Range(position, position), vscode.TextEditorRevealType.AtTop);
  }

  async revealSourceLine(uri, line) {
    if (!uri || !Number.isFinite(line)) {
      return;
    }
    const document = await vscode.workspace.openTextDocument(uri);
    const editor = await vscode.window.showTextDocument(document, { preview: false, preserveFocus: false });
    const safeLine = this.clampDocumentLine(document, line);
    this.rememberAnchorLine(uri, safeLine);
    const position = new vscode.Position(safeLine, 0);
    editor.selection = new vscode.Selection(position, position);
    editor.revealRange(new vscode.Range(position, position), vscode.TextEditorRevealType.AtTop);
  }

  async openLink(baseUri, href) {
    if (!baseUri || !href) {
      return;
    }
    if (isExternalHref(href)) {
      await vscode.env.openExternal(vscode.Uri.parse(href));
      return;
    }
    if (href.startsWith('#')) {
      return;
    }
    const resolvedPath = resolveRelativeFile(baseUri.fsPath, href);
    if (!resolvedPath) {
      return;
    }
    const target = vscode.Uri.file(resolvedPath);
    if (isMarkdownHref(resolvedPath) && this.configuration.get('openRelativeMarkdownLinksIn', 'mdExt') === 'mdExt') {
      await this.openWithMdExt(target);
      return;
    }
    if (isMarkdownHref(resolvedPath)) {
      await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(target));
      return;
    }
    await vscode.commands.executeCommand('vscode.open', target);
  }

  refreshVisible() {
    if (this.activitySurface) {
      this.renderSurface(this.activitySurface);
    }
    for (const surface of this.customSurfaces) {
      this.renderSurface(surface);
    }
  }

  dispose() {
    for (const timer of this.debounceTimers.values()) {
      clearTimeout(timer);
    }
    this.debounceTimers.clear();
  }

  matchingSurfaces(uri) {
    const key = uri?.toString();
    if (!key) {
      return [];
    }
    const surfaces = [];
    if (this.activitySurface && !this.activitySurface.disposed && this.activitySurface.uri?.toString() === key) {
      surfaces.push(this.activitySurface);
    }
    for (const surface of this.customSurfaces) {
      if (!surface.disposed && surface.uri?.toString() === key) {
        surfaces.push(surface);
      }
    }
    return surfaces;
  }

  visibleTextEditorsForUri(uri) {
    const key = uri?.toString();
    if (!key) {
      return [];
    }
    return vscode.window.visibleTextEditors.filter(
      (editor) => editor.document.languageId === 'markdown' && editor.document.uri.toString() === key,
    );
  }

  preferredSourceEditor(uri) {
    const editors = this.visibleTextEditorsForUri(uri);
    if (!editors.length) {
      return null;
    }
    return editors.find((editor) => editor === vscode.window.activeTextEditor) || editors[0];
  }

  topVisibleLine(editor) {
    const line = editor?.visibleRanges?.[0]?.start?.line;
    return Number.isFinite(line) ? Math.max(0, Math.floor(line)) : 0;
  }

  rememberAnchorLine(uri, line) {
    if (!uri || !Number.isFinite(line)) {
      return;
    }
    this.anchorLineByUri.set(uri.toString(), Math.max(0, Math.floor(line)));
  }

  scrollLeader(uri) {
    const key = uri?.toString();
    if (!key) {
      return null;
    }
    const leader = this.scrollLeaderByUri.get(key);
    if (!leader) {
      return null;
    }
    if (Date.now() >= leader.until) {
      this.scrollLeaderByUri.delete(key);
      return null;
    }
    return leader;
  }

  claimScrollLeadership(uri, kind, durationMs = 350) {
    const key = uri?.toString();
    if (!key) {
      return;
    }
    this.scrollLeaderByUri.set(key, {
      kind,
      until: Date.now() + durationMs,
    });
  }

  anchorLineForUri(uri) {
    const editor = this.preferredSourceEditor(uri);
    if (editor) {
      const line = this.topVisibleLine(editor);
      this.rememberAnchorLine(uri, line);
      return line;
    }
    return this.anchorLineByUri.get(uri?.toString()) || 0;
  }

  clampDocumentLine(document, line) {
    const numericLine = Number.isFinite(line) ? Math.floor(line) : 0;
    const maxLine = Math.max(0, Number(document?.lineCount || 1) - 1);
    return Math.max(0, Math.min(maxLine, numericLine));
  }

  searchStateForSurface(surface, scrollToFirst = false) {
    const query = String(surface?.searchQuery || '');
    if (!query.trim()) {
      return {
        query: '',
        terms: [],
        nearTermGroups: [],
        nearWordGap: SEARCH_CLOSE_WORD_GAP,
        scrollToFirst: false,
      };
    }

    return {
      query,
      terms: extractSearchTerms(query).map(([text, caseSensitive]) => ({ text, caseSensitive })),
      nearTermGroups: extractNearTermGroups(query).map((group) => (
        group.map(([text, caseSensitive]) => ({ text, caseSensitive }))
      )),
      nearWordGap: SEARCH_CLOSE_WORD_GAP,
      scrollToFirst: Boolean(scrollToFirst),
    };
  }

  async updateSearch(surface, query, scrollToFirst = false) {
    if (!surface || surface.disposed) {
      return;
    }
    surface.searchQuery = String(query || '');
    surface.webview.postMessage({
      type: 'searchState',
      ...this.searchStateForSurface(surface, scrollToFirst),
    });
  }

  postSurfaceStatus(surface, message, persistent = false) {
    if (!surface || surface.disposed) {
      return;
    }
    surface.webview.postMessage({
      type: 'status',
      message: String(message || ''),
      persistent: Boolean(persistent),
    });
  }

  syncPersistentHighlights(uri, entries, originSurface = null) {
    if (!uri) {
      return;
    }
    const normalizedEntries = normalizeHighlightEntries(entries);
    for (const surface of this.matchingSurfaces(uri)) {
      if (surface === originSurface) {
        continue;
      }
      surface.webview.postMessage({
        type: 'persistentHighlights',
        entries: normalizedEntries,
      });
    }
  }

  async addPersistentHighlight(surface, details) {
    if (!surface?.uri?.fsPath) {
      return;
    }
    const start = Number(details?.start);
    const end = Number(details?.end);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
      this.postSurfaceStatus(surface, 'Select text to highlight', true);
      return;
    }

    try {
      const currentEntries = await loadHighlightEntries(surface.uri.fsPath);
      const updatedEntries = replaceHighlightRange(
        currentEntries,
        start,
        end,
        details?.kind || PREVIEW_HIGHLIGHT_KIND_NORMAL,
        details?.selectedText || '',
      );
      const savedEntries = await saveHighlightEntries(surface.uri.fsPath, updatedEntries);
      this.syncPersistentHighlights(surface.uri, savedEntries);
      this.postSurfaceStatus(
        surface,
        String(details?.kind || '').trim().toLowerCase() === PREVIEW_HIGHLIGHT_KIND_IMPORTANT
          ? 'Important highlight added'
          : 'Highlight added',
      );
    } catch {
      this.postSurfaceStatus(surface, 'Highlight could not be saved', true);
    }
  }

  async onPersistentHighlightsResolved(surface, entries) {
    if (!surface?.uri?.fsPath) {
      return;
    }
    const normalizedEntries = normalizeHighlightEntries(entries);
    try {
      const currentEntries = await loadHighlightEntries(surface.uri.fsPath);
      if (entriesSignature(currentEntries) === entriesSignature(normalizedEntries)) {
        return;
      }
      const savedEntries = await saveHighlightEntries(surface.uri.fsPath, normalizedEntries);
      this.syncPersistentHighlights(surface.uri, savedEntries, surface);
    } catch {
      this.postSurfaceStatus(surface, 'Highlight could not be saved', true);
    }
  }

  syncPreviewScroll(uri, line, originSurface = null) {
    if (!uri || !Number.isFinite(line)) {
      return;
    }
    for (const surface of this.matchingSurfaces(uri)) {
      if (surface === originSurface) {
        continue;
      }
      surface.webview.postMessage({ type: 'syncScroll', line });
    }
  }

  async onPreviewScroll(surface, line) {
    if (!surface?.uri || !Number.isFinite(line)) {
      return;
    }
    if (this.scrollLeader(surface.uri)?.kind === 'source') {
      return;
    }
    this.claimScrollLeadership(surface.uri, 'preview');
    this.rememberAnchorLine(surface.uri, line);
    this.syncPreviewScroll(surface.uri, line, surface);
    const editor = this.preferredSourceEditor(surface.uri);
    if (!editor) {
      return;
    }
    const safeLine = this.clampDocumentLine(editor.document, line);
    if (this.topVisibleLine(editor) === safeLine) {
      return;
    }
    const position = new vscode.Position(safeLine, 0);
    this.sourceScrollSuppressions.set(editor, {
      line: safeLine,
      until: Date.now() + 1200,
    });
    editor.revealRange(new vscode.Range(position, position), vscode.TextEditorRevealType.AtTop);
  }
}

module.exports = { PreviewCoordinator };
