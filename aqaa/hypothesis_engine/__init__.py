"""
Hypothesis Engine (HE) -- Module 1 of the AQAA pipeline.

The Hypothesis Engine is responsible for the upstream generation and filtering
of candidate alpha-factor ideas.  It comprises three cooperating components:

1. **HypothesisProposer** -- Template-based (and optionally LLM-driven)
   generation of structured hypothesis specifications across five alpha
   categories: momentum, mean-reversion, volatility, fundamental, and
   sentiment signals.
2. **NoveltyFilter** -- AST-based deduplication that rejects hypotheses whose
   normalised abstract syntax tree is too similar to any factor already in
   the Quantitative Finance Knowledge Base (QFKB).
3. **QuantFinanceKnowledgeBase** (QFKB) -- An in-memory store of all past
   and present factor profiles, supporting similarity queries, retirement
   tracking, and JSON-based persistence.

Filtering rule (KDD 2026, Eq. 2):

    H = { h | AST-sim(h, QFKB) < tau_nov  AND  quality(h) > theta_q }

Submodules
----------
proposer
    HypothesisProposer for structured hypothesis generation.
novelty_filter
    AST-similarity-based deduplication.
knowledge_base
    Quantitative Finance Knowledge Base (QFKB).
"""

from aqaa.hypothesis_engine.proposer import HypothesisProposer
from aqaa.hypothesis_engine.novelty_filter import NoveltyFilter
from aqaa.hypothesis_engine.knowledge_base import QuantFinanceKnowledgeBase

__all__ = [
    "HypothesisProposer",
    "NoveltyFilter",
    "QuantFinanceKnowledgeBase",
]
