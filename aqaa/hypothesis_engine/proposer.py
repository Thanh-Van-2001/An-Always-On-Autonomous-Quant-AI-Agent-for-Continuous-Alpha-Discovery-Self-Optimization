"""
Hypothesis Proposer -- structured generation of alpha-factor candidates.

Generates candidate alpha hypotheses using structured templates across five
categories of quantitative signals.  Each hypothesis is emitted as a
``HypothesisSpec`` (from ``aqaa.core.types``) together with a JSON
specification containing the factor name, formula, data requirements, and
expected economic mechanism.

In production the proposer may delegate to an LLM for creative
diversification; the template-based path ensures deterministic baseline
coverage even without an LLM endpoint.

Usage
-----
>>> from aqaa.hypothesis_engine import HypothesisProposer
>>> proposer = HypothesisProposer()
>>> specs = proposer.generate_hypotheses(knowledge_base=kb, market_data={}, n=10)
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from aqaa.core.types import HypothesisSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alpha-category templates
# ---------------------------------------------------------------------------

# Each template is a dict with keys: name, category, formula, code_template,
# data_requirements, expected_mechanism, and optional parameter_ranges.

_MOMENTUM_TEMPLATES: List[Dict[str, Any]] = [
    {
        "name": "cross_sectional_momentum_{lookback}d",
        "category": "momentum",
        "formula": "rank(return_{lookback}d)",
        "code_template": (
            "def factor(df):\n"
            "    ret = df['close'].pct_change({lookback})\n"
            "    return ret.rank(axis=1, pct=True)"
        ),
        "data_requirements": ["close"],
        "expected_mechanism": (
            "Cross-sectional momentum: stocks with higher past returns "
            "tend to outperform in the near term due to slow information "
            "diffusion and investor under-reaction."
        ),
        "parameter_ranges": {"lookback": [5, 10, 20, 60, 120]},
    },
    {
        "name": "ts_momentum_{fast}_{slow}",
        "category": "momentum",
        "formula": "SMA(close, {fast}) / SMA(close, {slow}) - 1",
        "code_template": (
            "def factor(df):\n"
            "    fast = df['close'].rolling({fast}).mean()\n"
            "    slow = df['close'].rolling({slow}).mean()\n"
            "    return fast / slow - 1"
        ),
        "data_requirements": ["close"],
        "expected_mechanism": (
            "Time-series momentum via dual moving-average crossover; "
            "captures trend-following signals from behavioural persistence."
        ),
        "parameter_ranges": {"fast": [5, 10], "slow": [20, 60]},
    },
]

_MEAN_REVERSION_TEMPLATES: List[Dict[str, Any]] = [
    {
        "name": "mean_reversion_zscore_{window}d",
        "category": "mean_reversion",
        "formula": "(close - SMA(close, {window})) / std(close, {window})",
        "code_template": (
            "def factor(df):\n"
            "    mu = df['close'].rolling({window}).mean()\n"
            "    sigma = df['close'].rolling({window}).std()\n"
            "    return -(df['close'] - mu) / sigma.clip(lower=1e-8)"
        ),
        "data_requirements": ["close"],
        "expected_mechanism": (
            "Short-term mean reversion: prices that deviate significantly "
            "from their rolling mean tend to revert, capturing liquidity "
            "provision and over-reaction effects."
        ),
        "parameter_ranges": {"window": [5, 10, 20]},
    },
]

_VOLATILITY_TEMPLATES: List[Dict[str, Any]] = [
    {
        "name": "vol_ratio_{short}_{long}",
        "category": "volatility",
        "formula": "std(return, {short}) / std(return, {long})",
        "code_template": (
            "def factor(df):\n"
            "    ret = df['close'].pct_change(1)\n"
            "    short_vol = ret.rolling({short}).std()\n"
            "    long_vol = ret.rolling({long}).std()\n"
            "    return short_vol / long_vol.clip(lower=1e-8)"
        ),
        "data_requirements": ["close"],
        "expected_mechanism": (
            "Volatility regime indicator: a rising short/long vol ratio "
            "signals increasing uncertainty and can predict regime shifts."
        ),
        "parameter_ranges": {"short": [5, 10], "long": [20, 60]},
    },
    {
        "name": "realised_vol_rank_{window}d",
        "category": "volatility",
        "formula": "-rank(std(return, {window}))",
        "code_template": (
            "def factor(df):\n"
            "    ret = df['close'].pct_change(1)\n"
            "    vol = ret.rolling({window}).std()\n"
            "    return -vol.rank(axis=1, pct=True)"
        ),
        "data_requirements": ["close"],
        "expected_mechanism": (
            "Low-volatility anomaly: stocks with lower realised volatility "
            "deliver higher risk-adjusted returns than CAPM predicts."
        ),
        "parameter_ranges": {"window": [20, 60]},
    },
]

_FUNDAMENTAL_TEMPLATES: List[Dict[str, Any]] = [
    {
        "name": "earnings_yield",
        "category": "fundamental",
        "formula": "rank(eps / price)",
        "code_template": (
            "def factor(df):\n"
            "    return (df['eps_ttm'] / df['close'].clip(lower=1e-8))"
            ".rank(axis=1, pct=True)"
        ),
        "data_requirements": ["close", "eps_ttm"],
        "expected_mechanism": (
            "Value factor: stocks with higher earnings yield are "
            "undervalued relative to their fundamental earnings power."
        ),
        "parameter_ranges": {},
    },
    {
        "name": "book_to_market",
        "category": "fundamental",
        "formula": "rank(book_value / market_cap)",
        "code_template": (
            "def factor(df):\n"
            "    return (df['book_value'] / df['market_cap'].clip(lower=1e-8))"
            ".rank(axis=1, pct=True)"
        ),
        "data_requirements": ["book_value", "market_cap"],
        "expected_mechanism": (
            "Classic Fama-French value factor: high book-to-market stocks "
            "earn a premium for bearing distress risk."
        ),
        "parameter_ranges": {},
    },
]

_SENTIMENT_TEMPLATES: List[Dict[str, Any]] = [
    {
        "name": "news_sentiment_ma_{window}d",
        "category": "sentiment",
        "formula": "SMA(news_sentiment, {window})",
        "code_template": (
            "def factor(df):\n"
            "    return df['news_sentiment'].rolling({window}).mean()"
        ),
        "data_requirements": ["news_sentiment"],
        "expected_mechanism": (
            "Smoothed news sentiment captures persistent shifts in market "
            "narrative; positive sentiment drift precedes positive returns "
            "due to attention-driven buying."
        ),
        "parameter_ranges": {"window": [3, 5, 10]},
    },
    {
        "name": "social_volume_zscore_{window}d",
        "category": "sentiment",
        "formula": "(social_volume - SMA(social_volume, {window})) / std(social_volume, {window})",
        "code_template": (
            "def factor(df):\n"
            "    mu = df['social_volume'].rolling({window}).mean()\n"
            "    sigma = df['social_volume'].rolling({window}).std()\n"
            "    return (df['social_volume'] - mu) / sigma.clip(lower=1e-8)"
        ),
        "data_requirements": ["social_volume"],
        "expected_mechanism": (
            "Abnormal social-media attention spikes predict short-term "
            "return continuation due to retail order flow."
        ),
        "parameter_ranges": {"window": [5, 10, 20]},
    },
]

# Aggregate all templates.
_ALL_TEMPLATES: List[Dict[str, Any]] = list(
    itertools.chain(
        _MOMENTUM_TEMPLATES,
        _MEAN_REVERSION_TEMPLATES,
        _VOLATILITY_TEMPLATES,
        _FUNDAMENTAL_TEMPLATES,
        _SENTIMENT_TEMPLATES,
    )
)


# ---------------------------------------------------------------------------
# Hypothesis Proposer
# ---------------------------------------------------------------------------

class HypothesisProposer:
    """Generate structured alpha-factor hypotheses from templates.

    The proposer expands parameterised templates to create a pool of
    candidate hypotheses, then samples *n* from this pool (optionally
    weighted by category diversity).

    Parameters
    ----------
    templates : list[dict] | None
        Override the built-in template library.
    seed : int | None
        Random seed for reproducible hypothesis sampling.
    """

    def __init__(
        self,
        templates: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.templates = templates if templates is not None else _ALL_TEMPLATES
        self.rng = np.random.default_rng(seed)
        self._expanded_cache: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_hypotheses(
        self,
        knowledge_base: Any = None,
        market_data: Any = None,
        n: int = 10,
    ) -> List[HypothesisSpec]:
        """Generate up to *n* candidate ``HypothesisSpec`` objects.

        Parameters
        ----------
        knowledge_base : object | None
            A ``QuantFinanceKnowledgeBase`` instance.  If provided, the
            proposer avoids re-proposing factors already in the KB (by
            AST hash).  Actual novelty filtering should be done downstream
            by the ``NoveltyFilter``.
        market_data : object | None
            Reserved for future use (e.g. regime-conditioned generation).
        n : int
            Maximum number of hypotheses to return.

        Returns
        -------
        list[HypothesisSpec]
        """
        expanded = self._expand_templates()

        # Optionally exclude hypotheses whose AST hash already exists in KB.
        if knowledge_base is not None and hasattr(knowledge_base, "get_all_hashes"):
            existing_hashes = set(knowledge_base.get_all_hashes())
            expanded = [
                e for e in expanded
                if e.get("ast_hash", "") not in existing_hashes
            ]

        # Sample n hypotheses, ensuring category diversity.
        sampled = self._diverse_sample(expanded, n)

        specs: List[HypothesisSpec] = []
        for item in sampled:
            spec = HypothesisSpec(
                id=uuid.uuid4().hex[:12],
                description=item["expected_mechanism"],
                source_type="template",
                ast_hash=item.get("ast_hash", ""),
                novelty_score=1.0,  # Novelty not yet computed; default to max.
                quality_score=0.5,  # Neutral prior quality.
                raw_prompt=json.dumps(item, indent=2),
                created_at=datetime.now(UTC),
            )
            # Attach the full JSON spec as metadata on the raw_prompt.
            specs.append(spec)

        logger.info(
            "Generated %d hypotheses across categories: %s",
            len(specs),
            _count_categories(sampled),
        )
        return specs

    def get_expanded_templates(self) -> List[Dict[str, Any]]:
        """Return the full list of parameter-expanded templates.

        Useful for inspection and testing.
        """
        return self._expand_templates()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expand_templates(self) -> List[Dict[str, Any]]:
        """Expand parameterised templates into concrete instances."""
        if self._expanded_cache is not None:
            return self._expanded_cache

        expanded: List[Dict[str, Any]] = []
        for template in self.templates:
            param_ranges = template.get("parameter_ranges", {})
            if not param_ranges:
                # No parameters to expand.
                instance = deepcopy(template)
                instance["ast_hash"] = _hash_code(
                    instance.get("code_template", "")
                )
                expanded.append(instance)
                continue

            # Cartesian product of parameter values.
            keys = sorted(param_ranges.keys())
            for combo in itertools.product(*(param_ranges[k] for k in keys)):
                params = dict(zip(keys, combo))
                instance = deepcopy(template)
                instance["name"] = template["name"].format(**params)
                instance["formula"] = template["formula"].format(**params)
                instance["code_template"] = template["code_template"].format(
                    **params
                )
                instance["parameters"] = params
                instance["ast_hash"] = _hash_code(instance["code_template"])
                expanded.append(instance)

        self._expanded_cache = expanded
        logger.debug("Expanded %d templates into %d instances.", len(self.templates), len(expanded))
        return expanded

    def _diverse_sample(
        self,
        candidates: List[Dict[str, Any]],
        n: int,
    ) -> List[Dict[str, Any]]:
        """Sample *n* candidates with category-balanced diversity.

        Cycles through categories to pick one hypothesis from each in turn,
        until *n* are selected or the pool is exhausted.
        """
        if len(candidates) <= n:
            return list(candidates)

        by_cat: Dict[str, List[Dict[str, Any]]] = {}
        for c in candidates:
            cat = c.get("category", "other")
            by_cat.setdefault(cat, []).append(c)

        # Shuffle within each category.
        for cat_list in by_cat.values():
            self.rng.shuffle(cat_list)

        selected: List[Dict[str, Any]] = []
        cat_names = sorted(by_cat.keys())
        idx = 0
        while len(selected) < n:
            cat = cat_names[idx % len(cat_names)]
            if by_cat[cat]:
                selected.append(by_cat[cat].pop(0))
            idx += 1
            # Safety: break if all pools are empty.
            if all(len(v) == 0 for v in by_cat.values()):
                break

        return selected


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# json is imported lazily to keep module-level imports lightweight, but we
# use it inside generate_hypotheses via raw_prompt serialisation.
import json  # noqa: E402


def _hash_code(code: str) -> str:
    """SHA-256 hash of a code string for deduplication."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _count_categories(items: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Count items per category."""
    counts: Dict[str, int] = {}
    for item in items:
        cat = item.get("category", "other")
        counts[cat] = counts.get(cat, 0) + 1
    return counts
