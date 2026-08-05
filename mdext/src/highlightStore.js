'use strict';

const fsp = require('node:fs/promises');
const path = require('node:path');

const HIGHLIGHTING_FILE_NAME = '.mdexplore-highlighting.json';
const PREVIEW_HIGHLIGHT_KIND_NORMAL = 'normal';
const PREVIEW_HIGHLIGHT_KIND_IMPORTANT = 'important';
const PREVIEW_HIGHLIGHT_OFFSET_SPACE_PREVIEW = 'preview_text_v2';
const PREVIEW_HIGHLIGHT_OFFSET_SPACE_SOURCE = 'markdown_source_v1';
const PREVIEW_PERSISTENT_HIGHLIGHT_COLOR = 'rgba(102, 86, 178, 0.36)';
const PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_COLOR = 'rgba(225, 214, 255, 0.76)';
const PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_TEXT_COLOR = '#170534';
const PREVIEW_PERSISTENT_HIGHLIGHT_MARKER_COLOR = 'rgba(112, 90, 188, 0.92)';
const PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_MARKER_COLOR = 'rgba(154, 132, 220, 0.96)';

let nextHighlightIdToken = 0;

function normalizeHighlightKind(kind) {
  return String(kind || '').trim().toLowerCase() === PREVIEW_HIGHLIGHT_KIND_IMPORTANT
    ? PREVIEW_HIGHLIGHT_KIND_IMPORTANT
    : PREVIEW_HIGHLIGHT_KIND_NORMAL;
}

function normalizeAnchorText(value) {
  const text = String(value || '').replace(/\s+/g, ' ').trim();
  return text.length >= 3 ? text.slice(0, 480) : '';
}

function newHighlightId() {
  const token = nextHighlightIdToken;
  nextHighlightIdToken += 1;
  return `h${Date.now().toString(16)}-${token.toString(16)}`;
}

function pathDirectoryAndName(documentPath) {
  const rawPath = String(documentPath || '').trim();
  if (!rawPath) {
    return null;
  }
  const fileName = path.basename(rawPath);
  if (!fileName) {
    return null;
  }
  return {
    directory: path.resolve(path.dirname(rawPath)),
    fileName,
  };
}

function highlightingFilePath(documentPath) {
  const resolved = pathDirectoryAndName(documentPath);
  if (!resolved) {
    return null;
  }
  return path.join(resolved.directory, HIGHLIGHTING_FILE_NAME);
}

function normalizeHighlightEntries(rawEntries) {
  if (!Array.isArray(rawEntries)) {
    return [];
  }

  const sanitized = [];
  for (const item of rawEntries) {
    if (!item || typeof item !== 'object') {
      continue;
    }
    const rawId = typeof item.id === 'string' ? item.id.trim() : '';
    if (!rawId) {
      continue;
    }
    const start = Number.parseInt(item.start, 10);
    const end = Number.parseInt(item.end, 10);
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start) {
      continue;
    }
    const entry = {
      id: rawId,
      start,
      end,
      kind: normalizeHighlightKind(item.kind),
    };
    const anchorText = normalizeAnchorText(item.anchor_text);
    if (anchorText) {
      entry.anchor_text = anchorText;
    }
    const offsetSpace = typeof item.offset_space === 'string'
      ? item.offset_space.trim().toLowerCase()
      : '';
    if (offsetSpace === PREVIEW_HIGHLIGHT_OFFSET_SPACE_PREVIEW || offsetSpace === PREVIEW_HIGHLIGHT_OFFSET_SPACE_SOURCE) {
      entry.offset_space = offsetSpace;
    }
    sanitized.push(entry);
  }

  if (!sanitized.length) {
    return [];
  }

  sanitized.sort((left, right) => {
    if (left.start !== right.start) {
      return left.start - right.start;
    }
    if (left.end !== right.end) {
      return left.end - right.end;
    }
    return String(left.kind).localeCompare(String(right.kind));
  });

  const merged = [];
  for (const entry of sanitized) {
    const previous = merged[merged.length - 1];
    if (!previous) {
      merged.push(entry);
      continue;
    }
    if (previous.kind === entry.kind && entry.start <= previous.end) {
      previous.start = Math.min(previous.start, entry.start);
      previous.end = Math.max(previous.end, entry.end);
      continue;
    }
    merged.push(entry);
  }

  return merged;
}

function cloneEntries(entries) {
  return normalizeHighlightEntries(entries).map((entry) => ({ ...entry }));
}

async function readSidecarPayload(filePath) {
  if (!filePath) {
    return {};
  }
  try {
    const raw = await fsp.readFile(filePath, 'utf8');
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') {
      return {};
    }
    return parsed;
  } catch {
    return {};
  }
}

function payloadFilesMap(payload) {
  if (!payload || typeof payload !== 'object') {
    return {};
  }
  const files = payload.files && typeof payload.files === 'object' ? payload.files : payload;
  return files && typeof files === 'object' ? files : {};
}

function tagLegacyUntypedEntries(rawEntries, entries) {
  if (!Array.isArray(rawEntries) || !entries.length) {
    return cloneEntries(entries);
  }
  const implicitKeys = new Set();
  for (const rawEntry of rawEntries) {
    if (!rawEntry || typeof rawEntry !== 'object') {
      continue;
    }
    const rawId = typeof rawEntry.id === 'string' ? rawEntry.id.trim() : '';
    const start = Number.parseInt(rawEntry.start, 10);
    const end = Number.parseInt(rawEntry.end, 10);
    if (!rawId || !Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start) {
      continue;
    }
    const offsetSpace = typeof rawEntry.offset_space === 'string' ? rawEntry.offset_space.trim() : '';
    if (offsetSpace) {
      continue;
    }
    implicitKeys.add(`${rawId}\u0000${start}\u0000${end}\u0000${normalizeHighlightKind(rawEntry.kind)}`);
  }

  return entries.map((entry) => {
    const tagged = { ...entry };
    const key = `${tagged.id}\u0000${tagged.start}\u0000${tagged.end}\u0000${tagged.kind}`;
    if (!Object.prototype.hasOwnProperty.call(tagged, 'offset_space') && implicitKeys.has(key)) {
      tagged.legacy_offset_untyped = 1;
    }
    return tagged;
  });
}

async function loadHighlightEntries(documentPath) {
  const resolved = pathDirectoryAndName(documentPath);
  if (!resolved) {
    return [];
  }
  const payload = await readSidecarPayload(path.join(resolved.directory, HIGHLIGHTING_FILE_NAME));
  const files = payloadFilesMap(payload);
  let rawEntries = [];
  for (const [rawName, value] of Object.entries(files)) {
    if (path.basename(String(rawName || '')) !== resolved.fileName) {
      continue;
    }
    rawEntries = Array.isArray(value) ? value : [];
    break;
  }
  return tagLegacyUntypedEntries(rawEntries, normalizeHighlightEntries(rawEntries));
}

async function atomicWriteText(filePath, text) {
  const directory = path.dirname(filePath);
  await fsp.mkdir(directory, { recursive: true });
  const temporaryPath = path.join(
    directory,
    `.${path.basename(filePath)}.tmp.${process.pid}.${Date.now()}`,
  );
  try {
    await fsp.writeFile(temporaryPath, text, 'utf8');
    await fsp.rename(temporaryPath, filePath);
  } finally {
    await fsp.unlink(temporaryPath).catch(() => undefined);
  }
}

async function saveHighlightEntries(documentPath, entries) {
  const resolved = pathDirectoryAndName(documentPath);
  if (!resolved) {
    return [];
  }
  const normalized = cloneEntries(entries);
  const filePath = path.join(resolved.directory, HIGHLIGHTING_FILE_NAME);
  const payload = await readSidecarPayload(filePath);
  const files = payloadFilesMap(payload);
  const nextFiles = {};

  for (const [rawName, rawEntries] of Object.entries(files)) {
    const rawBaseName = path.basename(String(rawName || ''));
    if (!rawBaseName || rawBaseName === resolved.fileName) {
      continue;
    }
    const normalizedEntries = normalizeHighlightEntries(rawEntries);
    if (normalizedEntries.length) {
      nextFiles[rawBaseName] = normalizedEntries;
    }
  }

  if (normalized.length) {
    nextFiles[resolved.fileName] = normalized;
  }

  const orderedFiles = Object.fromEntries(
    Object.keys(nextFiles)
      .sort((left, right) => left.localeCompare(right))
      .map((fileName) => [fileName, nextFiles[fileName]]),
  );

  if (!Object.keys(orderedFiles).length) {
    await fsp.unlink(filePath).catch(() => undefined);
    return [];
  }

  await atomicWriteText(
    filePath,
    `${JSON.stringify({ files: orderedFiles }, null, 2)}\n`,
  );
  return normalized;
}

function replaceHighlightRange(entries, start, end, kind, anchorText = '') {
  const safeStart = Math.max(0, Math.floor(Number(start) || 0));
  const safeEnd = Math.max(safeStart + 1, Math.floor(Number(end) || 0));
  const normalizedKind = normalizeHighlightKind(kind);
  const normalizedAnchorText = normalizeAnchorText(anchorText);
  const existingEntries = cloneEntries(entries);
  const updated = [];

  for (const entry of existingEntries) {
    const entryStart = entry.start;
    const entryEnd = entry.end;
    const entryKind = normalizeHighlightKind(entry.kind);
    const entryAnchorText = normalizeAnchorText(entry.anchor_text);
    if (entryEnd <= safeStart || entryStart >= safeEnd) {
      const preserved = {
        id: entry.id || newHighlightId(),
        start: entryStart,
        end: entryEnd,
        kind: entryKind,
        offset_space: typeof entry.offset_space === 'string' && entry.offset_space.trim()
          ? entry.offset_space.trim().toLowerCase()
          : PREVIEW_HIGHLIGHT_OFFSET_SPACE_PREVIEW,
      };
      if (entryAnchorText) {
        preserved.anchor_text = entryAnchorText;
      }
      updated.push(preserved);
      continue;
    }
    if (entryStart < safeStart) {
      const leftEntry = {
        id: newHighlightId(),
        start: entryStart,
        end: safeStart,
        kind: entryKind,
        offset_space: PREVIEW_HIGHLIGHT_OFFSET_SPACE_PREVIEW,
      };
      if (entryAnchorText) {
        leftEntry.anchor_text = entryAnchorText;
      }
      updated.push(leftEntry);
    }
    if (entryEnd > safeEnd) {
      const rightEntry = {
        id: newHighlightId(),
        start: safeEnd,
        end: entryEnd,
        kind: entryKind,
        offset_space: PREVIEW_HIGHLIGHT_OFFSET_SPACE_PREVIEW,
      };
      if (entryAnchorText) {
        rightEntry.anchor_text = entryAnchorText;
      }
      updated.push(rightEntry);
    }
  }

  const nextEntry = {
    id: newHighlightId(),
    start: safeStart,
    end: safeEnd,
    kind: normalizedKind,
    offset_space: PREVIEW_HIGHLIGHT_OFFSET_SPACE_PREVIEW,
  };
  if (normalizedAnchorText) {
    nextEntry.anchor_text = normalizedAnchorText;
  }
  updated.push(nextEntry);
  return normalizeHighlightEntries(updated);
}

module.exports = {
  HIGHLIGHTING_FILE_NAME,
  PREVIEW_HIGHLIGHT_KIND_NORMAL,
  PREVIEW_HIGHLIGHT_KIND_IMPORTANT,
  PREVIEW_HIGHLIGHT_OFFSET_SPACE_PREVIEW,
  PREVIEW_HIGHLIGHT_OFFSET_SPACE_SOURCE,
  PREVIEW_PERSISTENT_HIGHLIGHT_COLOR,
  PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_COLOR,
  PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_TEXT_COLOR,
  PREVIEW_PERSISTENT_HIGHLIGHT_MARKER_COLOR,
  PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_MARKER_COLOR,
  normalizeHighlightKind,
  normalizeHighlightEntries,
  pathDirectoryAndName,
  highlightingFilePath,
  loadHighlightEntries,
  saveHighlightEntries,
  replaceHighlightRange,
  newHighlightId,
};