"""Pure search-query parsing and NEAR-window helpers."""

from __future__ import annotations

import re
from bisect import bisect_right

from .constants import SEARCH_CLOSE_WORD_GAP

SearchToken = tuple[str, str, bool]
SearchTerm = tuple[str, bool]
NearTermGroup = list[SearchTerm]
NearFocusWindow = dict[str, object]


def tokenize_match_query(query: str) -> list[SearchToken]:
    """Tokenize query supporting operators plus single/double-quoted terms."""
    tokens: list[SearchToken] = []
    i = 0
    length = len(query)

    while i < length:
        ch = query[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append(("LPAREN", ch, False))
            i += 1
            continue
        if ch == ")":
            tokens.append(("RPAREN", ch, False))
            i += 1
            continue
        if ch == ",":
            tokens.append(("COMMA", ch, False))
            i += 1
            continue
        if ch in {'"', "'"}:
            quote_char = ch
            is_case_sensitive = quote_char == "'"
            i += 1
            buffer: list[str] = []
            while i < length:
                current = query[i]
                if current == "\\" and i + 1 < length:
                    next_char = query[i + 1]
                    if next_char in {quote_char, "\\"}:
                        buffer.append(next_char)
                        i += 2
                        continue
                if current == quote_char:
                    i += 1
                    break
                buffer.append(current)
                i += 1
            tokens.append(("TERM", "".join(buffer), is_case_sensitive))
            continue

        start = i
        while i < length and not query[i].isspace() and query[i] not in "(),":
            i += 1
        token_value = query[start:i]
        if not token_value:
            continue
        upper = token_value.upper()
        if upper in {"AND", "OR", "NOT"}:
            tokens.append(("OP", upper, False))
        elif upper in {"NEAR", "CLOSE"}:
            # `CLOSE(...)` remains accepted as a legacy alias, but the parser
            # and AST normalize both spellings to the canonical `NEAR` form.
            tokens.append(("NEAR", "NEAR", False))
        else:
            tokens.append(("TERM", token_value, False))

    return tokens


def extract_search_terms(query: str) -> list[SearchTerm]:
    """Extract unique searchable terms with case mode from a query string."""
    if not query:
        return []

    terms: list[SearchTerm] = []
    seen: set[str] = set()
    for token_type, token_value, is_case_sensitive in tokenize_match_query(query):
        if token_type != "TERM":
            continue
        if not token_value.strip():
            continue
        key = (
            f"S:{token_value}"
            if is_case_sensitive
            else f"I:{token_value.casefold()}"
        )
        if key in seen:
            continue
        seen.add(key)
        terms.append((token_value, bool(is_case_sensitive)))

    terms.sort(key=lambda item: len(item[0]), reverse=True)
    return terms


def extract_near_term_groups(query: str) -> list[NearTermGroup]:
    """Extract NEAR(...) argument groups from an arbitrary search query."""
    if not query:
        return []

    tokens = tokenize_match_query(query)
    groups: list[NearTermGroup] = []
    i = 0
    token_count = len(tokens)

    while i < token_count:
        token_type, _token_value, _token_case_sensitive = tokens[i]
        if token_type != "NEAR":
            i += 1
            continue

        if i + 1 >= token_count or tokens[i + 1][0] != "LPAREN":
            i += 1
            continue

        j = i + 2
        group: NearTermGroup = []
        is_valid = False

        while j < token_count:
            part_type, part_value, part_case_sensitive = tokens[j]
            if part_type == "RPAREN":
                is_valid = True
                break
            if part_type == "COMMA":
                j += 1
                continue
            if part_type == "TERM":
                if part_value.strip():
                    group.append((part_value, part_case_sensitive))
                j += 1
                continue
            is_valid = False
            break

        if is_valid and len(group) >= 2:
            groups.append(group)
        i = j + 1 if j > i else i + 1

    return groups


def should_use_near_word_boundaries(term_text: str) -> bool:
    """Return whether NEAR() proximity should treat a term as a whole word."""
    return bool(term_text) and (not any(ch.isspace() for ch in term_text)) and bool(
        re.fullmatch(r"\w+", term_text, flags=re.UNICODE)
    )


def compile_near_term_pattern(
    term_text: str, is_case_sensitive: bool
) -> re.Pattern[str]:
    """Compile one NEAR() term matcher, adding word boundaries when needed."""
    escaped = re.escape(term_text)
    if should_use_near_word_boundaries(term_text):
        escaped = rf"(?<!\w){escaped}(?!\w)"
    flags = 0 if is_case_sensitive else re.IGNORECASE
    return re.compile(escaped, flags)


def collect_near_focus_windows(
    content: str, groups: list[NearTermGroup]
) -> list[NearFocusWindow]:
    """Return qualifying non-overlapping NEAR() windows in document order."""
    if not content or not groups:
        return []

    word_matches = list(re.finditer(r"\S+", content))
    if not word_matches:
        return []
    word_starts = [match.start() for match in word_matches]

    def earliest_window_for_group(
        group: NearTermGroup, min_start_char: int = 0
    ) -> NearFocusWindow | None:
        occurrences_by_term: list[list[tuple[int, int, int, int]]] = [
            [] for _ in range(len(group))
        ]

        for term_index, (term_text, is_case_sensitive) in enumerate(group):
            if not term_text:
                return None
            pattern = compile_near_term_pattern(term_text, bool(is_case_sensitive))
            for match in pattern.finditer(content):
                start_char, end_char = match.span()
                if start_char < min_start_char:
                    continue
                start_word = bisect_right(word_starts, start_char) - 1
                if start_word < 0:
                    continue
                end_probe = end_char - 1 if end_char > start_char else start_char
                end_word = bisect_right(word_starts, end_probe) - 1
                if end_word < start_word:
                    end_word = start_word
                occurrences_by_term[term_index].append(
                    (start_word, end_word, start_char, end_char)
                )

        if any(not occurrences for occurrences in occurrences_by_term):
            return None

        for occurrences in occurrences_by_term:
            occurrences.sort(key=lambda item: (item[2], item[0], item[1], item[3]))

        ordered_terms = sorted(
            range(len(group)), key=lambda idx: len(occurrences_by_term[idx])
        )
        best_assignment: dict[str, int] | None = None

        def search(
            order_index: int,
            used_starts: set[int],
            min_start_word: int | None,
            max_end_word: int | None,
            min_start_char_value: int | None,
            max_end_char: int | None,
        ) -> None:
            nonlocal best_assignment
            if order_index >= len(ordered_terms):
                if (
                    min_start_word is None
                    or max_end_word is None
                    or min_start_char_value is None
                    or max_end_char is None
                ):
                    return
                candidate = {
                    "span": max_end_word - min_start_word,
                    "start_word": min_start_word,
                    "end_word": max_end_word,
                    "start_char": min_start_char_value,
                    "end_char": max_end_char,
                }
                if best_assignment is None:
                    best_assignment = candidate
                    return
                if candidate["start_char"] < best_assignment["start_char"] or (
                    candidate["start_char"] == best_assignment["start_char"]
                    and (
                        candidate["span"] < best_assignment["span"]
                        or (
                            candidate["span"] == best_assignment["span"]
                            and candidate["end_char"] < best_assignment["end_char"]
                        )
                    )
                ):
                    best_assignment = candidate
                return

            term_index = ordered_terms[order_index]
            for start_word, end_word, start_char, end_char in occurrences_by_term[
                term_index
            ]:
                if start_char in used_starts:
                    continue
                next_min_start_word = (
                    start_word
                    if min_start_word is None
                    else min(min_start_word, start_word)
                )
                next_max_end_word = (
                    end_word if max_end_word is None else max(max_end_word, end_word)
                )
                if next_max_end_word - next_min_start_word > SEARCH_CLOSE_WORD_GAP:
                    continue
                next_min_start_char = (
                    start_char
                    if min_start_char_value is None
                    else min(min_start_char_value, start_char)
                )
                next_max_end_char = (
                    end_char if max_end_char is None else max(max_end_char, end_char)
                )
                if best_assignment is not None:
                    if next_min_start_char > best_assignment["start_char"]:
                        continue
                    candidate_span = next_max_end_word - next_min_start_word
                    if (
                        next_min_start_char == best_assignment["start_char"]
                        and candidate_span > best_assignment["span"]
                    ):
                        continue
                used_starts.add(start_char)
                search(
                    order_index + 1,
                    used_starts,
                    next_min_start_word,
                    next_max_end_word,
                    next_min_start_char,
                    next_max_end_char,
                )
                used_starts.remove(start_char)

        search(0, set(), None, None, None, None)
        if best_assignment is None:
            return None

        return {
            "span": best_assignment["span"],
            "start_char": best_assignment["start_char"],
            "end_char": best_assignment["end_char"],
            "terms": group,
        }

    windows: list[NearFocusWindow] = []
    for group in groups:
        next_min_start_char = 0
        while True:
            candidate = earliest_window_for_group(group, next_min_start_char)
            if candidate is None:
                break
            windows.append(candidate)
            next_min_start_char = max(
                next_min_start_char + 1, int(candidate["end_char"])
            )

    windows.sort(
        key=lambda item: (
            int(item.get("start_char", 10**9)),
            int(item.get("span", 10**9)),
            int(item.get("end_char", 10**9)),
        )
    )
    return windows


def best_near_focus_window(
    content: str, groups: list[NearTermGroup]
) -> NearFocusWindow | None:
    """Return the first qualifying NEAR() window used for preview focus."""
    windows = collect_near_focus_windows(content, groups)
    return windows[0] if windows else None


def count_highlighted_term_ranges(
    content: str,
    terms: list[SearchTerm],
    *,
    near_focus_range: tuple[int, int] | None = None,
    enforce_near_boundaries: bool = False,
) -> int:
    """Count non-overlapping term highlight ranges in raw file content."""
    if not content or not terms:
        return 0

    ranges: list[tuple[int, int]] = []
    for term_text, is_case_sensitive in terms:
        if not term_text.strip():
            continue
        if enforce_near_boundaries and should_use_near_word_boundaries(term_text):
            pattern = compile_near_term_pattern(term_text, bool(is_case_sensitive))
        else:
            flags = 0 if is_case_sensitive else re.IGNORECASE
            pattern = re.compile(re.escape(term_text), flags)
        for match in pattern.finditer(content):
            start, end = match.span()
            if near_focus_range is not None:
                focus_start, focus_end = near_focus_range
                if start < focus_start or end > focus_end:
                    continue
            ranges.append((start, end))

    if not ranges:
        return 0

    ranges.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    deduped: list[tuple[int, int]] = []
    last_end = -1
    for start, end in ranges:
        if start < last_end:
            continue
        deduped.append((start, end))
        last_end = end
    return len(deduped)


def compile_match_hit_counter(query: str):
    """Compile a lightweight per-file hit counter from query terms."""
    tokens = tokenize_match_query(query)
    near_term_groups = extract_near_term_groups(query)
    compiled_patterns: list[re.Pattern[str]] = []
    seen: set[str] = set()

    for token_type, token_value, is_case_sensitive in tokens:
        if token_type != "TERM":
            continue
        if not token_value.strip():
            continue
        key = (
            f"S:{token_value}"
            if is_case_sensitive
            else f"I:{token_value.casefold()}"
        )
        if key in seen:
            continue
        seen.add(key)
        flags = 0 if is_case_sensitive else re.IGNORECASE
        compiled_patterns.append(re.compile(re.escape(token_value), flags))

    if not compiled_patterns:
        return lambda _name, _content: 0

    def counter(file_name: str, content: str) -> int:
        if near_term_groups:
            return len(collect_near_focus_windows(content or "", near_term_groups))
        haystack = f"{file_name}\n{content}"
        total = 0
        for pattern in compiled_patterns:
            total += len(pattern.findall(haystack))
        return total

    return counter


def _compile_simple_match_predicate(tokens: list[SearchToken]):
    """Fallback matcher: all terms must appear in filename or content."""
    terms = [
        (value, is_case_sensitive)
        for token_type, value, is_case_sensitive in tokens
        if token_type == "TERM" and value.strip()
    ]
    if not terms:
        return lambda _file_name, _file_content: True

    def predicate(file_name: str, file_content: str) -> bool:
        name_text = file_name or ""
        content_text = file_content or ""
        name_folded = name_text.casefold()
        content_folded = content_text.casefold()
        for term_text, is_case_sensitive in terms:
            if is_case_sensitive:
                if term_text not in name_text and term_text not in content_text:
                    return False
            else:
                folded = term_text.casefold()
                if folded not in name_folded and folded not in content_folded:
                    return False
        return True

    return predicate


def compile_match_predicate(query: str):
    """Compile boolean query with implicit AND, quotes, and NEAR(...)."""
    tokens = tokenize_match_query(query)
    if not tokens:
        return lambda _name, _content: True

    class QueryParseError(Exception):
        pass

    idx = 0

    def peek(offset: int = 0) -> SearchToken | None:
        token_index = idx + offset
        if 0 <= token_index < len(tokens):
            return tokens[token_index]
        return None

    def token_starts_expression(token_index: int) -> bool:
        if token_index < 0 or token_index >= len(tokens):
            return False
        token_type, token_value, _token_case_sensitive = tokens[token_index]
        if token_type in {"TERM", "LPAREN", "NEAR"}:
            return True
        if token_type == "OP" and token_value == "NOT":
            return True
        if token_type == "OP" and token_value in {"AND", "OR"}:
            next_token = tokens[token_index + 1] if token_index + 1 < len(tokens) else None
            return bool(next_token and next_token[0] == "LPAREN")
        return False

    def consume(
        expected_type: str | None = None, expected_value: str | None = None
    ) -> SearchToken:
        nonlocal idx
        token = peek()
        if token is None:
            raise QueryParseError("Unexpected end of query")
        token_type, token_value, token_case_sensitive = token
        if expected_type is not None and token_type != expected_type:
            raise QueryParseError(f"Expected {expected_type} but found {token_type}")
        if expected_value is not None and token_value != expected_value:
            raise QueryParseError(f"Expected {expected_value} but found {token_value}")
        idx += 1
        return token_type, token_value, token_case_sensitive

    def parse_expression(allow_implicit_and: bool = True):
        return parse_or(allow_implicit_and)

    def parse_or(allow_implicit_and: bool = True):
        node = parse_and(allow_implicit_and)
        while True:
            token = peek()
            if token is None or token[0] != "OP" or token[1] != "OR":
                break
            consume("OP", "OR")
            right = parse_and(allow_implicit_and)
            node = ("OR", node, right)
        return node

    def parse_and(allow_implicit_and: bool = True):
        node = parse_not(allow_implicit_and)
        while True:
            token = peek()
            if token is not None and token[0] == "OP" and token[1] == "AND":
                consume("OP", "AND")
                right = parse_not(allow_implicit_and)
                node = ("AND", node, right)
                continue
            if allow_implicit_and and token_starts_expression(idx):
                right = parse_not(allow_implicit_and)
                node = ("AND", node, right)
                continue
            break
        return node

    def parse_not(allow_implicit_and: bool = True):
        token = peek()
        if token is not None and token[0] == "OP" and token[1] == "NOT":
            consume("OP", "NOT")
            return ("NOT", parse_not(allow_implicit_and))
        return parse_primary(allow_implicit_and)

    def parse_logic_call(operator_name: str):
        consume("OP", operator_name)
        consume("LPAREN")
        while True:
            token = peek()
            if token is None:
                raise QueryParseError(f"Unterminated {operator_name}(...)")
            if token[0] == "RPAREN":
                break
            if token[0] == "COMMA":
                consume("COMMA")
                continue
            if not token_starts_expression(idx):
                raise QueryParseError(
                    f"{operator_name}() accepts expression arguments only"
                )
            break

        items: list[tuple] = []
        while True:
            token = peek()
            if token is None:
                raise QueryParseError(f"Unterminated {operator_name}(...)")
            if token[0] == "RPAREN":
                break
            if token[0] == "COMMA":
                consume("COMMA")
                continue

            items.append(parse_expression(allow_implicit_and=False))
            token = peek()
            if token is None:
                raise QueryParseError(f"Unterminated {operator_name}(...)")
            if token[0] == "COMMA":
                consume("COMMA")
                continue
            if token[0] == "RPAREN":
                break
            if token_starts_expression(idx):
                continue
            raise QueryParseError(f"Unexpected token in {operator_name}(...)")

        consume("RPAREN")
        if not items:
            raise QueryParseError(f"{operator_name}() requires at least one argument")
        combined = items[0]
        for item in items[1:]:
            combined = (operator_name, combined, item)
        return combined

    def parse_near_call():
        consume("NEAR", "NEAR")
        consume("LPAREN")
        terms: list[SearchTerm] = []
        while True:
            token = peek()
            if token is None:
                raise QueryParseError("Unterminated NEAR(...)")
            token_type, token_value, token_case_sensitive = token
            if token_type == "RPAREN":
                break
            if token_type == "COMMA":
                consume("COMMA")
                continue
            if token_type == "TERM":
                consume("TERM")
                if token_value.strip():
                    terms.append((token_value, token_case_sensitive))
                continue
            raise QueryParseError("NEAR(...) accepts terms only")
        consume("RPAREN")
        if len(terms) < 2:
            raise QueryParseError("NEAR(...) requires at least two terms")
        return ("NEAR", terms)

    def parse_primary(_allow_implicit_and: bool = True):
        token = peek()
        if token is None:
            raise QueryParseError("Missing query operand")
        token_type, token_value, token_case_sensitive = token
        if token_type == "TERM":
            consume("TERM")
            return ("TERM", token_value, token_case_sensitive)
        if token_type == "NEAR":
            return parse_near_call()
        if (
            token_type == "OP"
            and token_value in {"AND", "OR"}
            and peek(1) is not None
            and peek(1)[0] == "LPAREN"
        ):
            return parse_logic_call(token_value)
        if token_type == "LPAREN":
            consume("LPAREN")
            node = parse_expression()
            consume("RPAREN")
            return node
        raise QueryParseError(f"Unexpected token: {token_type}")

    def term_matches(
        term: str,
        is_case_sensitive: bool,
        file_name: str,
        file_content: str,
        file_name_folded: str,
        file_content_folded: str,
    ) -> bool:
        if not term:
            return False
        if is_case_sensitive:
            return term in file_name or term in file_content
        term_folded = term.casefold()
        return term_folded in file_name_folded or term_folded in file_content_folded

    def near_terms_match(terms: list[SearchTerm], file_content: str) -> bool:
        return best_near_focus_window(file_content or "", [terms]) is not None

    def evaluate(
        node,
        file_name: str,
        file_content: str,
        file_name_folded: str,
        file_content_folded: str,
    ) -> bool:
        node_type = node[0]
        if node_type == "TERM":
            _kind, term_text, is_case_sensitive = node
            return term_matches(
                term_text,
                bool(is_case_sensitive),
                file_name,
                file_content,
                file_name_folded,
                file_content_folded,
            )
        if node_type == "NEAR":
            _kind, near_terms = node
            return near_terms_match(near_terms, file_content)
        if node_type == "NOT":
            _kind, operand = node
            return not evaluate(
                operand,
                file_name,
                file_content,
                file_name_folded,
                file_content_folded,
            )
        if node_type == "AND":
            _kind, left_node, right_node = node
            return evaluate(
                left_node,
                file_name,
                file_content,
                file_name_folded,
                file_content_folded,
            ) and evaluate(
                right_node,
                file_name,
                file_content,
                file_name_folded,
                file_content_folded,
            )
        if node_type == "OR":
            _kind, left_node, right_node = node
            return evaluate(
                left_node,
                file_name,
                file_content,
                file_name_folded,
                file_content_folded,
            ) or evaluate(
                right_node,
                file_name,
                file_content,
                file_name_folded,
                file_content_folded,
            )
        return False

    try:
        expression = parse_expression()
        if idx != len(tokens):
            raise QueryParseError("Unexpected trailing query tokens")
    except QueryParseError:
        return _compile_simple_match_predicate(tokens)

    def predicate(file_name: str, file_content: str) -> bool:
        name_text = file_name or ""
        content_text = file_content or ""
        return evaluate(
            expression,
            name_text,
            content_text,
            name_text.casefold(),
            content_text.casefold(),
        )

    return predicate
