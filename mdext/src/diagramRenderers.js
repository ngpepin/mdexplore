'use strict';

const fs = require('node:fs');
const fsp = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { execFile, spawn } = require('node:child_process');
const { promisify } = require('node:util');
const { sha1 } = require('./utils');

const execFileAsync = promisify(execFile);

function normalizeEmbeddedSvgSource(source) {
  return String(source ?? '').replaceAll('\r\n', '\n').replaceAll('\r', '\n').trim();
}

function sanitizeEmbeddedSvg(source) {
  const normalized = normalizeEmbeddedSvgSource(source);
  if (!normalized) {
    return { svg: '', error: 'Embedded SVG is empty.' };
  }

  const svg = normalized
    .replace(/^\s*<\?xml[\s\S]*?\?>\s*/i, '')
    .replace(/^\s*<!doctype[\s\S]*?>\s*/i, '');
  if (!/^<svg\b/i.test(svg) || !/<\/svg\s*>\s*$/i.test(svg)) {
    return { svg: '', error: 'Embedded SVG must have an <svg> root element.' };
  }

  // Embedded SVG is ultimately loaded through an <img> data URI rather than
  // injected as live SVG. Keep a defensive cleanup pass as well so source
  // markup cannot carry executable SVG constructs into the generated image.
  const cleaned = svg
    .replace(/<(?:script|foreignObject|iframe|object|embed)\b[^>]*>[\s\S]*?<\/(?:script|foreignObject|iframe|object|embed)\s*>/gi, '')
    .replace(/\s+on[a-z0-9:_-]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)/gi, '')
    .replace(/\s+(?:href|xlink:href|src)\s*=\s*(["'])\s*javascript:[\s\S]*?\1/gi, '')
    .replace(/\s+(?:href|xlink:href|src)\s*=\s*javascript:[^\s>]+/gi, '');
  return { svg: cleaned, error: '' };
}

function runProcessWithInput(command, args, input, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
    });
    const stdout = [];
    const stderr = [];
    let settled = false;
    const timeoutMs = Math.max(1000, Number(options.timeout) || 25000);
    const maxBuffer = Math.max(1024, Number(options.maxBuffer) || 16 * 1024 * 1024);
    let capturedBytes = 0;

    const finish = (error, result) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      if (error) {
        reject(error);
      } else {
        resolve(result);
      }
    };

    const append = (target, chunk) => {
      capturedBytes += chunk.length;
      if (capturedBytes > maxBuffer) {
        child.kill('SIGKILL');
        finish(new Error(`Process output exceeded ${maxBuffer} bytes.`));
        return;
      }
      target.push(chunk);
    };

    child.stdout.on('data', (chunk) => append(stdout, chunk));
    child.stderr.on('data', (chunk) => append(stderr, chunk));
    child.once('error', (error) => finish(error));
    child.once('close', (code, signal) => {
      const result = {
        code: Number(code ?? -1),
        signal: signal || '',
        stdout: Buffer.concat(stdout).toString('utf8'),
        stderr: Buffer.concat(stderr).toString('utf8'),
      };
      if (code === 0) {
        finish(null, result);
      } else {
        const error = new Error(
          result.stderr.trim() || result.stdout.trim() || `${command} exited with code ${code}`,
        );
        Object.assign(error, result);
        finish(error);
      }
    });

    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      finish(new Error(`${command} timed out after ${timeoutMs} ms.`));
    }, timeoutMs);

    child.stdin.once('error', (error) => finish(error));
    child.stdin.end(Buffer.from(String(input ?? ''), 'utf8'));
  });
}

const RUST_THEME = {
  theme: 'base',
  themeVariables: {
    background: '#0f172a',
    primaryColor: '#1e293b',
    primaryBorderColor: '#93c5fd',
    primaryTextColor: '#e5e7eb',
    secondaryColor: '#172554',
    tertiaryColor: '#111827',
    lineColor: '#d1d5db',
    textColor: '#e5e7eb',
    edgeLabelBackground: '#0f172a',
    clusterBkg: '#1f2937',
    clusterBorder: '#94a3b8',
    actorBkg: '#1e293b',
    actorBorder: '#93c5fd',
    actorLine: '#d1d5db',
    noteBkg: '#1f2937',
    noteBorderColor: '#93c5fd',
    fontFamily: 'Noto Sans, DejaVu Sans, sans-serif',
  },
};

class DiagramRenderers {
  constructor(context, getConfiguration) {
    this.context = context;
    this.getConfiguration = getConfiguration;
    this.mermaidCache = new Map();
    this.plantUmlCache = new Map();
    this.embeddedSvgCache = new Map();
    this.mmdrPath = context.asAbsolutePath(path.join('bin', 'linux-x64', 'mmdr'));
    this.plantUmlJar = context.asAbsolutePath(path.join('vendor', 'plantuml', 'plantuml.jar'));
    this.ensureExecutable();
  }

  ensureExecutable() {
    if (process.platform !== 'linux') {
      return;
    }
    try {
      fs.chmodSync(this.mmdrPath, 0o755);
    } catch {
      // JavaScript Mermaid remains available as the portable fallback.
    }
  }

  async renderMermaid(source) {
    const configured = this.getConfiguration().get('mermaidBackend', 'rust');
    if (configured !== 'rust') {
      return { renderer: 'javascript', svg: '', error: '' };
    }
    if (process.platform !== 'linux' || process.arch !== 'x64' || !fs.existsSync(this.mmdrPath)) {
      return {
        renderer: 'javascript',
        svg: '',
        error: 'Bundled mmdr is available only on Linux x64; using Mermaid JavaScript.',
      };
    }

    const normalized = String(source ?? '').replaceAll('\r\n', '\n').trim();
    const cacheKey = sha1(`${JSON.stringify(RUST_THEME)}\n${normalized}`);
    const cached = this.mermaidCache.get(cacheKey);
    if (cached) {
      return { renderer: 'rust', svg: cached, error: '' };
    }

    let tempDir = '';
    try {
      tempDir = await fsp.mkdtemp(path.join(os.tmpdir(), 'mdext-mermaid-'));
      const input = path.join(tempDir, 'diagram.mmd');
      const output = path.join(tempDir, 'diagram.svg');
      const config = path.join(tempDir, 'config.json');
      await Promise.all([
        fsp.writeFile(input, normalized, 'utf8'),
        fsp.writeFile(config, JSON.stringify(RUST_THEME), 'utf8'),
      ]);

      const candidates = [
        ['-i', input, '-o', output, '-e', 'svg', '-c', config],
        ['-i', input, '-o', output, '-e', 'svg'],
        [input, output, '--output-format', 'svg'],
      ];
      let lastError = '';
      for (const args of candidates) {
        try {
          await execFileAsync(this.mmdrPath, args, {
            timeout: 20000,
            maxBuffer: 8 * 1024 * 1024,
          });
          const svg = (await fsp.readFile(output, 'utf8')).trim();
          if (svg.toLowerCase().includes('<svg')) {
            this.mermaidCache.set(cacheKey, svg);
            return { renderer: 'rust', svg, error: '' };
          }
          lastError = 'mmdr completed without producing SVG output.';
        } catch (error) {
          lastError = String(error?.stderr || error?.message || error);
        }
      }
      return { renderer: 'javascript', svg: '', error: lastError };
    } catch (error) {
      return { renderer: 'javascript', svg: '', error: String(error?.message || error) };
    } finally {
      if (tempDir) {
        await fsp.rm(tempDir, { recursive: true, force: true }).catch(() => undefined);
      }
    }
  }

  renderEmbeddedSvg(source) {
    const normalized = normalizeEmbeddedSvgSource(source);
    const cacheKey = sha1(normalized);
    const cached = this.embeddedSvgCache.get(cacheKey);
    if (cached) {
      return { dataUri: cached, error: '' };
    }

    const sanitized = sanitizeEmbeddedSvg(normalized);
    if (!sanitized.svg) {
      return { dataUri: '', error: sanitized.error || 'Embedded SVG rendering failed.' };
    }

    const dataUri = `data:image/svg+xml;base64,${Buffer.from(sanitized.svg, 'utf8').toString('base64')}`;
    this.embeddedSvgCache.set(cacheKey, dataUri);
    return { dataUri, error: '' };
  }

  async renderPlantUml(source) {
    if (!this.getConfiguration().get('plantUml.enabled', true)) {
      return { svg: '', error: 'PlantUML rendering is disabled in mdExt settings.' };
    }
    if (!fs.existsSync(this.plantUmlJar)) {
      return { svg: '', error: 'The bundled PlantUML jar is unavailable.' };
    }
    const normalized = this.preparePlantUml(source);
    const cacheKey = sha1(normalized);
    const cached = this.plantUmlCache.get(cacheKey);
    if (cached) {
      return { svg: cached, error: '' };
    }
    try {
      const { stdout, stderr } = await runProcessWithInput(
        'java',
        ['-Djava.awt.headless=true', '-jar', this.plantUmlJar, '-pipe', '-tsvg', '-charset', 'UTF-8'],
        normalized,
        {
          timeout: 25000,
          maxBuffer: 16 * 1024 * 1024,
        },
      );
      const svg = String(stdout ?? '').trim();
      if (!svg.toLowerCase().includes('<svg')) {
        return { svg: '', error: String(stderr || 'PlantUML did not produce SVG output.') };
      }
      this.plantUmlCache.set(cacheKey, svg);
      return { svg, error: '' };
    } catch (error) {
      return { svg: '', error: String(error?.stderr || error?.message || error) };
    }
  }

  preparePlantUml(source) {
    const text = String(source ?? '').trim();
    if (/^@start\w+/i.test(text)) {
      return text;
    }
    return `@startuml\n${text}\n@enduml\n`;
  }
}

module.exports = { DiagramRenderers, sanitizeEmbeddedSvg };
