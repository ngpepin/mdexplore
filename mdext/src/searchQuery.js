'use strict';

const SEARCH_CLOSE_WORD_GAP = 50;

function tokenizeMatchQuery(query) {
  const source = String(query || '');
  const tokens = [];
  let index = 0;

  while (index < source.length) {
    const character = source[index];
    if (/\s/.test(character)) {
      index += 1;
      continue;
    }
    if (character === '(') {
      tokens.push(['LPAREN', character, false]);
      index += 1;
      continue;
    }
    if (character === ')') {
      tokens.push(['RPAREN', character, false]);
      index += 1;
      continue;
    }
    if (character === ',') {
      tokens.push(['COMMA', character, false]);
      index += 1;
      continue;
    }
    if (character === '"' || character === "'") {
      const quoteCharacter = character;
      const isCaseSensitive = quoteCharacter === "'";
      index += 1;
      const buffer = [];
      while (index < source.length) {
        const current = source[index];
        if (current === '\\' && index + 1 < source.length) {
          const nextCharacter = source[index + 1];
          if (nextCharacter === quoteCharacter || nextCharacter === '\\') {
            buffer.push(nextCharacter);
            index += 2;
            continue;
          }
        }
        if (current === quoteCharacter) {
          index += 1;
          break;
        }
        buffer.push(current);
        index += 1;
      }
      tokens.push(['TERM', buffer.join(''), isCaseSensitive]);
      continue;
    }

    const start = index;
    while (index < source.length && !/\s/.test(source[index]) && !'(),'.includes(source[index])) {
      index += 1;
    }
    const rawToken = source.slice(start, index);
    if (!rawToken) {
      continue;
    }
    const upperToken = rawToken.toUpperCase();
    if (upperToken === 'AND' || upperToken === 'OR' || upperToken === 'NOT') {
      tokens.push(['OP', upperToken, false]);
      continue;
    }
    if (upperToken === 'NEAR' || upperToken === 'CLOSE') {
      tokens.push(['NEAR', 'NEAR', false]);
      continue;
    }
    tokens.push(['TERM', rawToken, false]);
  }

  return tokens;
}

function extractSearchTerms(query) {
  if (!query) {
    return [];
  }

  const terms = [];
  const seen = new Set();
  for (const [tokenType, tokenValue, isCaseSensitive] of tokenizeMatchQuery(query)) {
    if (tokenType !== 'TERM') {
      continue;
    }
    if (!String(tokenValue || '').trim()) {
      continue;
    }
    const key = isCaseSensitive
      ? `S:${tokenValue}`
      : `I:${String(tokenValue).toLocaleLowerCase()}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    terms.push([tokenValue, Boolean(isCaseSensitive)]);
  }

  terms.sort((left, right) => String(right[0]).length - String(left[0]).length);
  return terms;
}

function extractNearTermGroups(query) {
  if (!query) {
    return [];
  }

  const tokens = tokenizeMatchQuery(query);
  const groups = [];
  let index = 0;

  while (index < tokens.length) {
    const [tokenType] = tokens[index];
    if (tokenType !== 'NEAR') {
      index += 1;
      continue;
    }
    if (index + 1 >= tokens.length || tokens[index + 1][0] !== 'LPAREN') {
      index += 1;
      continue;
    }

    let probe = index + 2;
    let isValid = false;
    const group = [];
    while (probe < tokens.length) {
      const [partType, partValue, isCaseSensitive] = tokens[probe];
      if (partType === 'RPAREN') {
        isValid = true;
        break;
      }
      if (partType === 'COMMA') {
        probe += 1;
        continue;
      }
      if (partType === 'TERM') {
        if (String(partValue || '').trim()) {
          group.push([partValue, Boolean(isCaseSensitive)]);
        }
        probe += 1;
        continue;
      }
      isValid = false;
      break;
    }

    if (isValid && group.length >= 2) {
      groups.push(group);
    }
    index = probe > index ? probe + 1 : index + 1;
  }

  return groups;
}

module.exports = {
  SEARCH_CLOSE_WORD_GAP,
  tokenizeMatchQuery,
  extractSearchTerms,
  extractNearTermGroups,
};