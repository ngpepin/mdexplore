import unittest

import mdexplore
from mdexplore_app import search as search_query


class SearchQuerySyntaxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.window = mdexplore.MdExploreWindow.__new__(mdexplore.MdExploreWindow)

    class _FakeLineEdit:
        def __init__(self, text: str) -> None:
            self._text = text

        def text(self) -> str:
            return self._text

    def test_tokenizer_distinguishes_single_and_double_quoted_terms(self) -> None:
        tokens = self.window._tokenize_match_query(
            r"""alpha 'Beta Gamma' "Delta Epsilon" 'it\'s fine'"""
        )
        self.assertEqual(
            tokens,
            [
                ("TERM", "alpha", False),
                ("TERM", "Beta Gamma", True),
                ("TERM", "Delta Epsilon", False),
                ("TERM", "it's fine", True),
            ],
        )

    def test_tokenizer_normalizes_legacy_close_keyword_to_near(self) -> None:
        tokens = self.window._tokenize_match_query("""CLOSE(alpha,beta)""")
        self.assertEqual(
            tokens,
            [
                ("NEAR", "NEAR", False),
                ("LPAREN", "(", False),
                ("TERM", "alpha", False),
                ("COMMA", ",", False),
                ("TERM", "beta", False),
                ("RPAREN", ")", False),
            ],
        )

    def test_single_quotes_inside_double_quotes_are_literal(self) -> None:
        tokens = self.window._tokenize_match_query(
            r'''"Program Director's RAG"'''
        )
        self.assertEqual(tokens, [("TERM", "Program Director's RAG", False)])

    def test_unterminated_single_quote_remains_one_case_sensitive_term(self) -> None:
        tokens = self.window._tokenize_match_query("'Program Director")
        self.assertEqual(tokens, [("TERM", "Program Director", True)])

    def test_current_search_terms_preserve_quoted_trailing_space(self) -> None:
        self.window.match_input = self._FakeLineEdit("'The '")
        self.assertEqual(self.window._current_search_terms(), [("The ", True)])

    def test_current_near_groups_preserve_quoted_trailing_space(self) -> None:
        self.window.match_input = self._FakeLineEdit("""NEAR('The ' "quick brown")""")
        self.assertEqual(
            self.window._current_near_term_groups(),
            [[("The ", True), ("quick brown", False)]],
        )

    def test_single_quoted_phrase_is_case_sensitive(self) -> None:
        predicate = self.window._compile_match_predicate("'Program Director RAG'")
        self.assertTrue(predicate("example.md", "Program Director RAG"))
        self.assertFalse(predicate("example.md", "program director rag"))

    def test_double_quoted_phrase_is_case_insensitive(self) -> None:
        predicate = self.window._compile_match_predicate('"Program Director RAG"')
        self.assertTrue(predicate("example.md", "PROGRAM DIRECTOR RAG"))
        self.assertTrue(predicate("example.md", "program director rag"))

    def test_double_quoted_phrase_with_apostrophe_matches_insensitively(self) -> None:
        predicate = self.window._compile_match_predicate(
            r'''"Program Director's RAG"'''
        )
        self.assertTrue(predicate("example.md", "program director's rag"))
        self.assertTrue(predicate("example.md", "PROGRAM DIRECTOR'S RAG"))

    def test_single_quoted_trailing_space_is_not_trimmed(self) -> None:
        predicate = self.window._compile_match_predicate("'The '")
        self.assertTrue(predicate("example.md", "The quick brown fox"))
        self.assertFalse(predicate("example.md", "There is no trailing separator"))

    def test_double_quoted_trailing_space_is_not_trimmed(self) -> None:
        predicate = self.window._compile_match_predicate('"The "')
        self.assertTrue(predicate("example.md", "the quick brown fox"))
        self.assertFalse(predicate("example.md", "There is no trailing separator"))

    def test_near_query_respects_mixed_case_modes(self) -> None:
        predicate = self.window._compile_match_predicate(
            """NEAR('Exact Case' "other phrase")"""
        )
        self.assertTrue(predicate("example.md", "Exact Case and OTHER PHRASE"))
        self.assertFalse(predicate("example.md", "exact case and OTHER PHRASE"))

    def test_near_query_requires_distinct_occurrences_for_overlapping_terms(self) -> None:
        predicate = self.window._compile_match_predicate("""NEAR('The ', the)""")
        self.assertFalse(predicate("example.md", "The quick brown fox"))
        self.assertTrue(predicate("example.md", "The quick brown the fox"))

    def test_legacy_close_alias_matches_like_near(self) -> None:
        predicate = self.window._compile_match_predicate("""CLOSE(alpha,beta)""")
        self.assertTrue(predicate("example.md", "alpha goes with beta"))
        self.assertFalse(predicate("example.md", "alpha only"))

    def test_hit_counter_counts_single_quoted_terms_separately(self) -> None:
        counter = self.window._compile_match_hit_counter("'Alpha' alpha")
        self.assertEqual(counter("example.md", "Alpha alpha ALPHA"), 4)

    def test_hit_counter_for_near_query_matches_near_window_highlights(self) -> None:
        counter = self.window._compile_match_hit_counter("""NEAR('The ',the)""")
        self.assertEqual(counter("example.md", "The quick brown fox.\nThey said hello.\nA later the appears here.\n"), 1)

    def test_hit_counter_for_near_query_counts_repeated_windows(self) -> None:
        counter = self.window._compile_match_hit_counter(
            """NEAR('Nicolas','Pepin')"""
        )
        self.assertEqual(
            counter("example.md", "Nicolas Pepin\nOther text\nNicolas Pepin\n"),
            2,
        )

    def test_best_near_focus_window_covers_full_multiword_terms(self) -> None:
        content = (
            "Exact Case and other phrase appear here.\n"
            "Program Director's RAG pipeline.\n"
            "Joe met Anne Smith yesterday.\n"
        )
        cases = [
            (
                [("other phrase", False), ("Exact Case", False)],
                "Exact Case and other phrase",
            ),
            (
                [("Program Director's RAG", False), ("pipeline", False)],
                "Program Director's RAG pipeline",
            ),
            (
                [("Joe", False), ("Anne Smith", False)],
                "Joe met Anne Smith",
            ),
        ]

        for group, expected_window_text in cases:
            with self.subTest(group=group):
                chosen = self.window._best_near_focus_window(content, [group])
                self.assertIsNotNone(chosen)
                assert chosen is not None
                window_text = content[chosen["start_char"] : chosen["end_char"]]
                self.assertEqual(window_text, expected_window_text)
                self.assertEqual(
                    search_query.count_highlighted_term_ranges(
                        content,
                        group,
                        near_focus_range=(chosen["start_char"], chosen["end_char"]),
                        enforce_near_boundaries=True,
                    ),
                    2,
                )


if __name__ == "__main__":
    unittest.main()
