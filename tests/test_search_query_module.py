import unittest

from mdexplore_app import search as search_query


class SearchQueryModuleTests(unittest.TestCase):
    def test_tokenizer_preserves_case_modes_and_escaping(self) -> None:
        tokens = search_query.tokenize_match_query(
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

    def test_tokenizer_normalizes_near_and_close_to_near(self) -> None:
        tokens = search_query.tokenize_match_query(
            """NEAR(alpha,beta) CLOSE(gamma,delta)"""
        )
        self.assertEqual(
            tokens,
            [
                ("NEAR", "NEAR", False),
                ("LPAREN", "(", False),
                ("TERM", "alpha", False),
                ("COMMA", ",", False),
                ("TERM", "beta", False),
                ("RPAREN", ")", False),
                ("NEAR", "NEAR", False),
                ("LPAREN", "(", False),
                ("TERM", "gamma", False),
                ("COMMA", ",", False),
                ("TERM", "delta", False),
                ("RPAREN", ")", False),
            ],
        )

    def test_extract_search_terms_dedupes_and_preserves_trailing_space(self) -> None:
        self.assertEqual(
            search_query.extract_search_terms("""'The ' "alpha beta" alpha ALPHA"""),
            [("alpha beta", False), ("alpha", False), ("The ", True)],
        )

    def test_extract_near_term_groups_preserves_trailing_space(self) -> None:
        self.assertEqual(
            search_query.extract_near_term_groups("""NEAR('The ' "quick brown")"""),
            [[("The ", True), ("quick brown", False)]],
        )

    def test_compile_term_pattern_single_space_uses_boundary_mode(self) -> None:
        pattern = search_query.compile_term_pattern("Nico ", True)
        self.assertIsNotNone(pattern.search("Nico."))
        self.assertIsNotNone(pattern.search("Nico)"))
        self.assertIsNotNone(pattern.search("Nico\n"))

    def test_compile_term_pattern_double_space_remains_literal(self) -> None:
        pattern = search_query.compile_term_pattern("  Nico", True)
        self.assertIsNotNone(pattern.search("  Nico"))
        self.assertIsNone(pattern.search("\nNico"))

    def test_compile_match_predicate_falls_back_on_invalid_boolean_query(self) -> None:
        predicate = search_query.compile_match_predicate("alpha OR")
        self.assertTrue(predicate("example.md", "alpha appears here"))
        self.assertFalse(predicate("example.md", "beta appears here"))

    def test_best_near_focus_window_covers_full_multiword_terms(self) -> None:
        content = (
            "Exact Case and other phrase appear here.\n"
            "Program Director's RAG pipeline.\n"
            "Joe met Anne Smith yesterday.\n"
        )
        chosen = search_query.best_near_focus_window(
            content, [[("Program Director's RAG", False), ("pipeline", False)]]
        )
        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertEqual(
            content[chosen["start_char"] : chosen["end_char"]],
            "Program Director's RAG pipeline",
        )

    def test_match_hit_counter_counts_repeated_near_windows(self) -> None:
        counter = search_query.compile_match_hit_counter("""NEAR('Nicolas','Pepin')""")
        self.assertEqual(
            counter("example.md", "Nicolas Pepin\nOther text\nNicolas Pepin\n"),
            2,
        )

    def test_legacy_close_alias_remains_accepted(self) -> None:
        predicate = search_query.compile_match_predicate("""CLOSE(alpha,beta)""")
        self.assertTrue(predicate("example.md", "alpha goes with beta"))
        self.assertFalse(predicate("example.md", "alpha only"))


if __name__ == "__main__":
    unittest.main()
