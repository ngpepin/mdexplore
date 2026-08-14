'use strict';

const fs = require('node:fs');
const fsp = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { execFile } = require('node:child_process');
const { promisify } = require('node:util');
const { pathToFileURL } = require('node:url');
const vscode = require('vscode');

const execFileAsync = promisify(execFile);
const PYTHON_CHECK = 'import PySide6.QtWebEngineCore, PySide6.QtWebEngineWidgets';

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function expandHome(value) {
  const text = String(value || '').trim();
  if (!text.startsWith('~')) {
    return text;
  }
  if (text === '~') {
    return os.homedir();
  }
  if (text.startsWith(`~${path.sep}`) || text.startsWith('~/')) {
    return path.join(os.homedir(), text.slice(2));
  }
  return text;
}

function venvPython(root) {
  if (!root) {
    return [];
  }
  return process.platform === 'win32'
    ? [path.join(root, '.venv', 'Scripts', 'python.exe')]
    : [path.join(root, '.venv', 'bin', 'python')];
}

function pythonCandidates(context) {
  const configured = expandHome(vscode.workspace.getConfiguration('mdExt').get('pdfPythonPath', ''));
  const environment = expandHome(process.env.MDEXPLORE_PYTHON || '');
  const roots = [];
  for (const folder of vscode.workspace.workspaceFolders || []) {
    roots.push(folder.uri?.fsPath || '');
  }
  const extensionPath = context.extensionUri?.fsPath || context.extensionPath || '';
  if (extensionPath) {
    roots.push(path.dirname(extensionPath));
  }

  const discovered = roots.flatMap(venvPython);
  return unique([
    configured,
    environment,
    ...discovered,
    process.platform === 'win32' ? 'python.exe' : 'python3',
    process.platform === 'win32' ? 'py.exe' : 'python',
  ]);
}

async function candidateExists(candidate) {
  if (!candidate || !candidate.includes(path.sep)) {
    return true;
  }
  try {
    await fsp.access(candidate, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function resolvePdfPython(context) {
  const failures = [];
  for (const candidate of pythonCandidates(context)) {
    if (!await candidateExists(candidate)) {
      continue;
    }
    try {
      await execFileAsync(candidate, ['-c', PYTHON_CHECK], {
        timeout: 8000,
        windowsHide: true,
        maxBuffer: 256 * 1024,
      });
      return candidate;
    } catch (error) {
      failures.push(`${candidate}: ${String(error?.stderr || error?.message || error).trim()}`);
    }
  }
  const detail = failures.length ? ` Checked: ${failures.map((item) => item.split('\n')[0]).join('; ')}` : '';
  throw new Error(
    'Vector PDF export requires Python with PySide6 Qt WebEngine. Run the mdexplore install-update.sh setup or set mdExt.pdfPythonPath to a compatible Python executable.' + detail,
  );
}

function escapeHtmlAttribute(value) {
  return String(value || '')
    .replaceAll('&', '&amp;')
    .replaceAll('"', '&quot;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function injectSourceBase(html, sourcePath) {
  const sourceDirectory = path.dirname(sourcePath);
  const baseHref = pathToFileURL(`${sourceDirectory}${path.sep}`).href;
  const baseTag = `<base href="${escapeHtmlAttribute(baseHref)}">`;
  const source = String(html || '');
  if (/<head(?:\s[^>]*)?>/i.test(source)) {
    return source.replace(/<head(?:\s[^>]*)?>/i, (match) => `${match}\n  ${baseTag}`);
  }
  return `${baseTag}\n${source}`;
}

class PdfExporter {
  constructor(context) {
    this.context = context;
    this.python = null;
  }

  helperPath() {
    const extensionPath = this.context.extensionUri?.fsPath || this.context.extensionPath || '';
    return path.join(extensionPath, 'python', 'pdf_export.py');
  }

  async pythonExecutable() {
    if (this.python) {
      return this.python;
    }
    this.python = await resolvePdfPython(this.context);
    return this.python;
  }

  async exportHtml({ html, sourcePath }) {
    if (!String(html || '').trim()) {
      throw new Error('PDF export received an empty HTML snapshot');
    }
    if (!sourcePath) {
      throw new Error('PDF export source path is unavailable');
    }

    const helper = this.helperPath();
    await fsp.access(helper, fs.constants.R_OK);
    const tempDirectory = await fsp.mkdtemp(path.join(os.tmpdir(), 'mdext-pdf-'));
    const htmlPath = path.join(tempDirectory, 'preview.html');
    const pdfPath = path.join(tempDirectory, 'preview.pdf');

    try {
      await fsp.writeFile(htmlPath, injectSourceBase(html, sourcePath), 'utf8');
      const python = await this.pythonExecutable();
      const env = { ...process.env };
      if (process.platform === 'linux') {
        env.QT_QPA_PLATFORM ||= 'offscreen';
        env.QT_QUICK_BACKEND ||= 'software';
        env.QT_OPENGL ||= 'software';
        if (typeof process.getuid === 'function' && process.getuid() === 0) {
          env.QTWEBENGINE_DISABLE_SANDBOX ||= '1';
        }
      }
      await execFileAsync(python, [helper, '--html', htmlPath, '--output', pdfPath], {
        timeout: 90000,
        windowsHide: true,
        maxBuffer: 1024 * 1024,
        env,
      });
      const bytes = await fsp.readFile(pdfPath);
      if (!bytes.length) {
        throw new Error('Native PDF renderer produced an empty file');
      }
      return bytes;
    } finally {
      await fsp.rm(tempDirectory, { recursive: true, force: true });
    }
  }
}

module.exports = { PdfExporter, injectSourceBase, pythonCandidates, resolvePdfPython };
