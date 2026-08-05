'use strict';

const crypto = require('node:crypto');
const path = require('node:path');

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function sha1(value) {
  return crypto.createHash('sha1').update(String(value ?? ''), 'utf8').digest('hex');
}

function nonce() {
  return crypto.randomBytes(18).toString('base64url');
}

function isMarkdownPath(value) {
  return /\.(?:md|markdown)$/i.test(String(value ?? '').split(/[?#]/, 1)[0]);
}

function isExternalHref(value) {
  return /^(?:https?:|mailto:|tel:)/i.test(String(value ?? '').trim());
}

function isAnchorHref(value) {
  return String(value ?? '').trim().startsWith('#');
}

function decodePathComponentSafely(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function stripQueryAndFragment(value) {
  return String(value ?? '').split('#', 1)[0].split('?', 1)[0];
}

function resolveRelativeFile(documentPath, href) {
  const raw = decodePathComponentSafely(stripQueryAndFragment(String(href ?? '').trim()));
  if (!raw || isExternalHref(raw) || isAnchorHref(raw) || raw.startsWith('data:')) {
    return null;
  }
  if (raw.startsWith('file:')) {
    try {
      return new URL(raw).pathname;
    } catch {
      return null;
    }
  }
  return path.resolve(path.dirname(documentPath), raw);
}

function textToBase64(value) {
  return Buffer.from(String(value ?? ''), 'utf8').toString('base64');
}

module.exports = {
  escapeHtml,
  sha1,
  nonce,
  isMarkdownPath,
  isExternalHref,
  isAnchorHref,
  resolveRelativeFile,
  textToBase64,
};
