"""
AST-based novelty filter for alpha-factor deduplication.

Implements the filtering rule from the KDD 2026 paper (Eq. 2):

    H = { h | AST-sim(h, QFKB) < tau_nov  AND  quality(h) > theta_q }

where ``AST-sim`` is a normalised structural similarity metric computed on
the Python abstract syntax tree of the factor code, and ``tau_nov`` is the
novelty threshold (default 0.75).

The filter works by:
  1. Parsing both the candidate and existing factor code into Python ASTs.
  2. Normalising variable names so that only the *structure* matters.
  3. Computing a simplified tree edit distance.
  4. Converting the distance to a similarity score in [0, 1].

Usage
-----
>>> from aqaa.hypothesis_engine import NoveltyFilter
>>> nf = NoveltyFilter()
>>> filtered = nf.filter_hypotheses(candidates, knowledge_base, tau_nov=0.75)
"""

from __future__ import annotations

import ast
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aqaa.core.types import HypothesisSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AST normalisation
# ---------------------------------------------------------------------------

class _ASTNormaliser(ast.NodeTransformer):
    """Rename all user-defined identifiers to canonical placeholders.

    This makes the structural comparison insensitive to superficial naming
    differences (e.g. ``fast_ma`` vs ``short_sma``).  Built-in names and
    attribute accesses are preserved to maintain semantic meaning.
    """

    _BUILTIN_NAMES = frozenset({
        "True", "False", "None", "print", "len", "range", "int", "float",
        "str", "list", "dict", "set", "tuple", "abs", "max", "min", "sum",
        "round", "sorted", "enumerate", "zip", "map", "filter", "isinstance",
        "type", "hasattr", "getattr", "setattr",
    })

    def __init__(self) -> None:
        super().__init__()
        self._name_map: Dict[str, str] = {}
        self._counter: int = 0

    def _canonical(self, original: str) -> str:
        """Map *original* to a canonical name like ``_v0``, ``_v1``, ..."""
        if original in self._BUILTIN_NAMES:
            return original
        if original not in self._name_map:
            self._name_map[original] = f"_v{self._counter}"
            self._counter += 1
        return self._name_map[original]

    def visit_Name(self, node: ast.Name) -> ast.Name:  # noqa: N802
        node.id = self._canonical(node.id)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:  # noqa: N802
        node.name = self._canonical(node.name)
        for arg in node.args.args:
            arg.arg = self._canonical(arg.arg)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._canonical(node.arg)
        return self.generic_visit(node)


def _normalize_ast(code: str) -> ast.AST:
    """Parse *code* and return a variable-name-normalised AST.

    Parameters
    ----------
    code : str
        Valid Python source code.

    Returns
    -------
    ast.AST
        Normalised abstract syntax tree.

    Raises
    ------
    SyntaxError
        If *code* is not valid Python.
    """
    tree = ast.parse(code)
    normaliser = _ASTNormaliser()
    tree = normaliser.visit(tree)
    ast.fix_missing_locations(tree)
    return tree


# ---------------------------------------------------------------------------
# Simplified tree edit distance
# ---------------------------------------------------------------------------

def _ast_to_label_sequence(tree: ast.AST) -> List[str]:
    """Flatten an AST into a pre-order label sequence.

    Each node is represented by its class name (e.g. ``'FunctionDef'``,
    ``'BinOp'``, ``'Return'``).  This provides a lightweight structural
    fingerprint suitable for rapid comparison.
    """
    labels: List[str] = []
    queue: deque[ast.AST] = deque([tree])
    while queue:
        node = queue.popleft()
        labels.append(type(node).__name__)
        for child in ast.iter_child_nodes(node):
            queue.append(child)
    return labels


def _tree_edit_distance(tree1: ast.AST, tree2: ast.AST) -> int:
    """Compute a simplified tree edit distance between two ASTs.

    This uses the label-sequence approach: convert each tree to its
    pre-order label sequence and compute the Levenshtein edit distance
    on the resulting strings.  This is an O(n*m) approximation that
    captures structural similarity without the full Zhang-Shasha
    complexity.

    Parameters
    ----------
    tree1, tree2 : ast.AST
        Normalised abstract syntax trees.

    Returns
    -------
    int
        Edit distance (lower = more similar).
    """
    seq1 = _ast_to_label_sequence(tree1)
    seq2 = _ast_to_label_sequence(tree2)
    return _levenshtein(seq1, seq2)


def _levenshtein(s: Sequence[str], t: Sequence[str]) -> int:
    """Levenshtein edit distance between two sequences of labels.

    Uses the standard dynamic-programming algorithm with O(min(n,m))
    space via the two-row optimisation.
    """
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n

    # Ensure s is the shorter sequence for space efficiency.
    if n > m:
        s, t = t, s
        n, m = m, n

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for j in range(1, m + 1):
        curr[0] = j
        for i in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[i] = min(
                curr[i - 1] + 1,       # insertion
                prev[i] + 1,           # deletion
                prev[i - 1] + cost,    # substitution
            )
        prev, curr = curr, prev

    return prev[n]


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------

def compute_ast_similarity(code1: str, code2: str) -> float:
    """Compute structural similarity between two Python code snippets.

    The similarity is defined as:

        sim = 1 - edit_distance / max(|seq1|, |seq2|)

    where the edit distance operates on the normalised AST label sequences.
    A similarity of 1.0 means the two code snippets are structurally
    identical (up to variable renaming); 0.0 means completely different.

    Parameters
    ----------
    code1, code2 : str
        Python source code strings.

    Returns
    -------
    float
        Similarity score in [0, 1].
    """
    try:
        tree1 = _normalize_ast(code1)
        tree2 = _normalize_ast(code2)
    except SyntaxError as exc:
        logger.warning("SyntaxError during AST parsing: %s", exc)
        return 0.0

    seq1 = _ast_to_label_sequence(tree1)
    seq2 = _ast_to_label_sequence(tree2)

    if not seq1 and not seq2:
        return 1.0
    if not seq1 or not seq2:
        return 0.0

    dist = _levenshtein(seq1, seq2)
    max_len = max(len(seq1), len(seq2))
    similarity = 1.0 - dist / max_len

    return float(np.clip(similarity, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Novelty Filter
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """Result of filtering a single hypothesis.

    Attributes:
        hypothesis: The original HypothesisSpec.
        passed: Whether the hypothesis survived the filter.
        max_similarity: Highest AST similarity to any existing factor.
        most_similar_id: ID of the most similar existing factor.
        quality_ok: Whether quality exceeds the threshold.
    """

    hypothesis: HypothesisSpec
    passed: bool = False
    max_similarity: float = 0.0
    most_similar_id: str = ""
    quality_ok: bool = True


class NoveltyFilter:
    """AST-based novelty filter for deduplicating alpha hypotheses.

    Implements the filtering rule:

        H = { h | AST-sim(h, QFKB) < tau_nov  AND  quality(h) > theta_q }

    Parameters
    ----------
    tau_nov : float
        Maximum allowed AST similarity to any existing factor.
        Default 0.75 per the KDD 2026 paper.
    theta_q : float
        Minimum quality score for admission.  Default 0.3.
    """

    def __init__(
        self,
        tau_nov: float = 0.75,
        theta_q: float = 0.3,
    ) -> None:
        self.tau_nov = tau_nov
        self.theta_q = theta_q

    def filter_hypotheses(
        self,
        candidates: List[HypothesisSpec],
        knowledge_base: Any,
        tau_nov: Optional[float] = None,
        theta_q: Optional[float] = None,
    ) -> List[HypothesisSpec]:
        """Filter a list of candidate hypotheses for novelty and quality.

        Parameters
        ----------
        candidates : list[HypothesisSpec]
            Candidate hypotheses to filter.
        knowledge_base : QuantFinanceKnowledgeBase
            The QFKB containing existing factor profiles.
        tau_nov : float | None
            Override the instance-level novelty threshold.
        theta_q : float | None
            Override the instance-level quality threshold.

        Returns
        -------
        list[HypothesisSpec]
            Hypotheses that pass both novelty and quality gates.
        """
        tau = tau_nov if tau_nov is not None else self.tau_nov
        theta = theta_q if theta_q is not None else self.theta_q

        # Extract existing factor code from the knowledge base.
        existing_code = self._extract_existing_code(knowledge_base)

        accepted: List[HypothesisSpec] = []
        for candidate in candidates:
            result = self._evaluate_single(candidate, existing_code, tau, theta)
            if result.passed:
                accepted.append(candidate)
                logger.debug(
                    "Hypothesis %s ACCEPTED (max_sim=%.3f, quality=%.3f)",
                    candidate.id,
                    result.max_similarity,
                    candidate.quality_score,
                )
            else:
                logger.debug(
                    "Hypothesis %s REJECTED (max_sim=%.3f, quality=%.3f, "
                    "similar_to=%s)",
                    candidate.id,
                    result.max_similarity,
                    candidate.quality_score,
                    result.most_similar_id,
                )

        logger.info(
            "NoveltyFilter: %d / %d candidates passed (tau_nov=%.2f, theta_q=%.2f)",
            len(accepted),
            len(candidates),
            tau,
            theta,
        )
        return accepted

    def filter_hypotheses_detailed(
        self,
        candidates: List[HypothesisSpec],
        knowledge_base: Any,
        tau_nov: Optional[float] = None,
        theta_q: Optional[float] = None,
    ) -> List[FilterResult]:
        """Like ``filter_hypotheses`` but returns detailed ``FilterResult`` objects.

        Useful for analysis and debugging.
        """
        tau = tau_nov if tau_nov is not None else self.tau_nov
        theta = theta_q if theta_q is not None else self.theta_q
        existing_code = self._extract_existing_code(knowledge_base)
        return [
            self._evaluate_single(candidate, existing_code, tau, theta)
            for candidate in candidates
        ]

    def compute_ast_similarity(self, code1: str, code2: str) -> float:
        """Convenience wrapper around the module-level function."""
        return compute_ast_similarity(code1, code2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_single(
        self,
        candidate: HypothesisSpec,
        existing_code: List[Tuple[str, str]],
        tau: float,
        theta: float,
    ) -> FilterResult:
        """Evaluate a single hypothesis against all existing factors."""
        result = FilterResult(hypothesis=candidate)

        # Quality gate.
        if candidate.quality_score < theta:
            result.quality_ok = False
            result.passed = False
            return result

        result.quality_ok = True

        # Extract candidate code from raw_prompt (JSON with code_template).
        candidate_code = self._extract_code(candidate)
        if not candidate_code:
            # No code to compare -- pass by default.
            result.passed = True
            return result

        # Novelty gate: check against all existing factors.
        max_sim = 0.0
        most_similar = ""
        for factor_id, factor_code in existing_code:
            if not factor_code:
                continue
            sim = compute_ast_similarity(candidate_code, factor_code)
            if sim > max_sim:
                max_sim = sim
                most_similar = factor_id

        result.max_similarity = max_sim
        result.most_similar_id = most_similar
        result.passed = max_sim < tau

        return result

    @staticmethod
    def _extract_code(spec: HypothesisSpec) -> str:
        """Extract the code_template from a HypothesisSpec's raw_prompt."""
        import json as _json

        try:
            data = _json.loads(spec.raw_prompt)
            return data.get("code_template", "")
        except (ValueError, TypeError):
            return spec.raw_prompt

    @staticmethod
    def _extract_existing_code(knowledge_base: Any) -> List[Tuple[str, str]]:
        """Extract (id, code) pairs from the knowledge base."""
        pairs: List[Tuple[str, str]] = []
        if knowledge_base is None:
            return pairs

        if hasattr(knowledge_base, "get_all_factors"):
            for factor in knowledge_base.get_all_factors():
                fid = factor.get("id", "") if isinstance(factor, dict) else getattr(factor, "id", "")
                code = factor.get("code", "") if isinstance(factor, dict) else getattr(factor, "code", "")
                if code:
                    pairs.append((fid, code))

        return pairs
