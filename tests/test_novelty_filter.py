"""
Tests for the AST-based novelty filter.

Validates identical code -> similarity = 1.0, completely different code ->
low similarity, variable renaming invariance, and that the filter rejects
hypotheses with AST-sim > tau_nov.
"""

import json

import numpy as np
import pytest

from aqaa.core.types import HypothesisSpec
from aqaa.hypothesis_engine.novelty_filter import (
    FilterResult,
    NoveltyFilter,
    compute_ast_similarity,
    _normalize_ast,
    _ast_to_label_sequence,
    _levenshtein,
)


# ---------------------------------------------------------------------------
# Test: Identical code -> similarity = 1.0
# ---------------------------------------------------------------------------

class TestIdenticalCode:
    """Identical code snippets should have similarity 1.0."""

    def test_exact_same_code(self):
        code = "def alpha(close, volume):\n    return close * volume\n"
        assert compute_ast_similarity(code, code) == 1.0

    def test_same_logic_same_names(self):
        code = """
def factor(prices, volumes):
    ma = sum(prices) / len(prices)
    return ma * volumes
"""
        assert compute_ast_similarity(code, code) == 1.0

    def test_empty_code(self):
        """Two empty strings should yield similarity 1.0 (both trivial)."""
        # Empty string parses to an AST with just a Module node
        sim = compute_ast_similarity("", "")
        assert sim == 1.0


# ---------------------------------------------------------------------------
# Test: Completely different code -> low similarity
# ---------------------------------------------------------------------------

class TestDifferentCode:
    """Structurally different code should have low similarity."""

    def test_very_different_functions(self):
        code1 = """
def alpha(close):
    return close[-1] / close[-2] - 1
"""
        code2 = """
def factor(prices, volumes, highs, lows):
    ma20 = sum(prices[-20:]) / 20
    ma50 = sum(prices[-50:]) / 50
    spread = ma20 - ma50
    volatility = max(highs) - min(lows)
    if volatility > 0:
        signal = spread / volatility
    else:
        signal = 0
    return signal * volumes[-1]
"""
        sim = compute_ast_similarity(code1, code2)
        assert sim < 0.50  # Should be notably dissimilar

    def test_expression_vs_class(self):
        code1 = "x = 1 + 2"
        code2 = """
class MyClass:
    def __init__(self, a, b, c, d, e):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
    def compute(self):
        return self.a * self.b + self.c - self.d / self.e
"""
        sim = compute_ast_similarity(code1, code2)
        assert sim < 0.30


# ---------------------------------------------------------------------------
# Test: Variable renaming doesn't affect similarity
# ---------------------------------------------------------------------------

class TestVariableRenaming:
    """Renaming variables should produce similarity close to 1.0."""

    def test_renamed_variables(self):
        code1 = """
def alpha(close, volume):
    fast_ma = sum(close[-5:]) / 5
    slow_ma = sum(close[-20:]) / 20
    return (fast_ma - slow_ma) * volume[-1]
"""
        code2 = """
def factor(prices, vol):
    short_avg = sum(prices[-5:]) / 5
    long_avg = sum(prices[-20:]) / 20
    return (short_avg - long_avg) * vol[-1]
"""
        sim = compute_ast_similarity(code1, code2)
        assert sim > 0.90  # Structural identity up to names

    def test_renamed_function_and_args(self):
        code1 = "def foo(x, y):\n    return x + y\n"
        code2 = "def bar(a, b):\n    return a + b\n"
        sim = compute_ast_similarity(code1, code2)
        assert sim == 1.0  # Exactly the same structure

    def test_renamed_local_variables(self):
        code1 = """
temp = 42
result = temp * 2
"""
        code2 = """
value = 42
output = value * 2
"""
        sim = compute_ast_similarity(code1, code2)
        assert sim == 1.0


# ---------------------------------------------------------------------------
# Test: Intermediate similarity
# ---------------------------------------------------------------------------

class TestIntermediateSimilarity:
    """Partially similar code should have moderate similarity."""

    def test_added_line(self):
        code1 = """
def alpha(close):
    return close[-1] - close[-2]
"""
        code2 = """
def alpha(close):
    diff = close[-1] - close[-2]
    return diff
"""
        sim = compute_ast_similarity(code1, code2)
        # Similar but not identical structure
        assert 0.50 < sim < 1.0


# ---------------------------------------------------------------------------
# Test: NoveltyFilter rejects similar hypotheses
# ---------------------------------------------------------------------------

class TestNoveltyFilterRejects:
    """Filter should reject candidates with AST-sim > tau_nov."""

    def _make_hypothesis(self, code: str, quality: float = 0.8) -> HypothesisSpec:
        """Create a HypothesisSpec with code embedded in raw_prompt."""
        return HypothesisSpec(
            quality_score=quality,
            raw_prompt=json.dumps({"code_template": code}),
        )

    def _make_kb(self, factors):
        """Create a minimal knowledge base mock."""
        class FakeKB:
            def __init__(self, items):
                self._items = items

            def get_all_factors(self):
                return self._items

        return FakeKB(factors)

    def test_novel_hypothesis_passes(self):
        nf = NoveltyFilter(tau_nov=0.75, theta_q=0.3)

        candidate = self._make_hypothesis(
            "def alpha(close):\n    return close[-1] / close[-2] - 1\n"
        )
        kb = self._make_kb([
            {"id": "existing_1", "code": (
                "def factor(prices, volumes, highs, lows):\n"
                "    ma20 = sum(prices[-20:]) / 20\n"
                "    ma50 = sum(prices[-50:]) / 50\n"
                "    return (ma20 - ma50) * volumes[-1]\n"
            )},
        ])

        results = nf.filter_hypotheses([candidate], kb)
        assert len(results) == 1

    def test_duplicate_hypothesis_rejected(self):
        nf = NoveltyFilter(tau_nov=0.75, theta_q=0.3)

        code = "def alpha(close):\n    return close[-1] / close[-2] - 1\n"
        candidate = self._make_hypothesis(code)

        # Same code in the KB (with different variable names)
        kb = self._make_kb([
            {"id": "existing_1", "code": (
                "def factor(prices):\n    return prices[-1] / prices[-2] - 1\n"
            )},
        ])

        results = nf.filter_hypotheses([candidate], kb)
        # Should be rejected because AST similarity >= 0.75
        assert len(results) == 0

    def test_low_quality_rejected(self):
        """Even if novel, low quality hypotheses should be rejected."""
        nf = NoveltyFilter(tau_nov=0.75, theta_q=0.5)

        candidate = self._make_hypothesis(
            "def alpha(x):\n    return x\n",
            quality=0.2,  # Below theta_q = 0.5
        )
        kb = self._make_kb([])

        results = nf.filter_hypotheses([candidate], kb)
        assert len(results) == 0

    def test_empty_knowledge_base(self):
        """With no existing factors, all quality-passing candidates pass."""
        nf = NoveltyFilter(tau_nov=0.75, theta_q=0.3)

        candidate = self._make_hypothesis(
            "def alpha(close):\n    return close[-1]\n",
            quality=0.8,
        )
        results = nf.filter_hypotheses([candidate], None)
        assert len(results) == 1

    def test_filter_detailed_results(self):
        nf = NoveltyFilter(tau_nov=0.75, theta_q=0.3)

        code = "def alpha(x):\n    return x * 2\n"
        candidate = self._make_hypothesis(code, quality=0.9)

        kb = self._make_kb([
            {"id": "existing", "code": "def alpha(y):\n    return y * 2\n"},
        ])

        detailed = nf.filter_hypotheses_detailed([candidate], kb)
        assert len(detailed) == 1
        result = detailed[0]
        assert isinstance(result, FilterResult)
        # Similarity should be ~1.0 (same structure, different name)
        assert result.max_similarity > 0.75
        assert result.passed is False
        assert result.most_similar_id == "existing"

    def test_tau_nov_override(self):
        """Override tau_nov per call."""
        nf = NoveltyFilter(tau_nov=0.50, theta_q=0.3)

        code = "def alpha(x):\n    return x * 2\n"
        candidate = self._make_hypothesis(code, quality=0.9)

        kb = self._make_kb([
            {"id": "existing", "code": "def factor(y):\n    return y * 3\n"},
        ])

        # With high tau_nov, the candidate should pass
        results_high = nf.filter_hypotheses(
            [candidate], kb, tau_nov=0.99
        )
        # With low tau_nov, the candidate may be rejected
        results_low = nf.filter_hypotheses(
            [candidate], kb, tau_nov=0.10
        )

        assert len(results_high) >= len(results_low)


# ---------------------------------------------------------------------------
# Test: compute_ast_similarity edge cases
# ---------------------------------------------------------------------------

class TestASTSimilarityEdgeCases:

    def test_syntax_error_returns_zero(self):
        """Invalid Python code should return 0.0 similarity."""
        sim = compute_ast_similarity("def bad(:", "x = 1")
        assert sim == 0.0

    def test_both_syntax_errors_return_zero(self):
        sim = compute_ast_similarity("def bad(:", "class [invalid")
        assert sim == 0.0

    def test_similarity_in_range(self):
        """Similarity should always be in [0, 1]."""
        snippets = [
            "x = 1",
            "y = x + 2",
            "def f(a): return a",
            "for i in range(10): print(i)",
            "class A:\n    pass",
        ]
        for c1 in snippets:
            for c2 in snippets:
                sim = compute_ast_similarity(c1, c2)
                assert 0.0 <= sim <= 1.0

    def test_convenience_method(self):
        """NoveltyFilter.compute_ast_similarity should match module-level."""
        nf = NoveltyFilter()
        code1 = "x = 1 + 2"
        code2 = "y = 3 + 4"
        assert nf.compute_ast_similarity(code1, code2) == compute_ast_similarity(code1, code2)


# ---------------------------------------------------------------------------
# Test: Internal helpers
# ---------------------------------------------------------------------------

class TestInternalHelpers:

    def test_levenshtein_identical(self):
        assert _levenshtein(["a", "b", "c"], ["a", "b", "c"]) == 0

    def test_levenshtein_completely_different(self):
        assert _levenshtein(["a", "b"], ["c", "d"]) == 2

    def test_levenshtein_insertion(self):
        assert _levenshtein(["a", "b"], ["a", "x", "b"]) == 1

    def test_levenshtein_empty(self):
        assert _levenshtein([], ["a", "b"]) == 2
        assert _levenshtein(["a"], []) == 1
        assert _levenshtein([], []) == 0

    def test_normalize_ast_strips_names(self):
        """Variable names should be replaced with canonical names."""
        import ast
        tree = _normalize_ast("x = 42\ny = x + 1\n")
        # The names 'x' and 'y' should be replaced with '_v0', '_v1', etc.
        names = [
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        ]
        assert all(n.startswith("_v") for n in names)

    def test_label_sequence_nonempty(self):
        import ast
        tree = ast.parse("x = 1")
        labels = _ast_to_label_sequence(tree)
        assert len(labels) > 0
        assert "Module" in labels
