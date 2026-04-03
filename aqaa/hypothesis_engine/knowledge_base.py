"""
Quantitative Finance Knowledge Base (QFKB).

An in-memory store of all past and present alpha-factor profiles, providing:

* **Storage** -- full lifecycle tracking from hypothesis through retirement.
* **Similarity queries** -- retrieve the *k* factors most similar to a given
  hypothesis (by AST hash or description embedding).
* **Retirement tracking** -- identify retired/archived factors for mutation
  and evolutionary re-use by the Hypothesis Engine.
* **Persistence** -- JSON serialisation / deserialisation for checkpointing.

The QFKB is the long-term memory of the AQAA system; every module reads
from it and the Feedback Loop Orchestrator writes to it at the end of each
24-hour cycle.

Usage
-----
>>> from aqaa.hypothesis_engine import QuantFinanceKnowledgeBase
>>> kb = QuantFinanceKnowledgeBase()
>>> kb.add_factor({"id": "abc123", "name": "momentum_20d", "code": "...", "state": "LIVE_DEPLOYED"})
>>> kb.query_similar(hypothesis, top_k=5)
"""

from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from aqaa.core.types import AlphaFactor, AlphaState, HypothesisSpec

logger = logging.getLogger(__name__)


class QuantFinanceKnowledgeBase:
    """In-memory knowledge base for quantitative factor profiles.

    Parameters
    ----------
    persist_path : str | Path | None
        If provided, the knowledge base will be saved to / loaded from
        this JSON file.
    """

    def __init__(self, persist_path: Optional[str | Path] = None) -> None:
        self.persist_path = Path(persist_path) if persist_path else None

        # Core storage: factor_id -> factor_profile (dict).
        self._factors: Dict[str, Dict[str, Any]] = {}

        # Index: ast_hash -> set of factor_ids for fast dedup lookups.
        self._hash_index: Dict[str, Set[str]] = {}

        # Load from disk if available.
        if self.persist_path and self.persist_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Public API -- mutation
    # ------------------------------------------------------------------

    def add_factor(self, factor_profile: Dict[str, Any]) -> str:
        """Add or update a factor profile in the knowledge base.

        Parameters
        ----------
        factor_profile : dict
            Must contain at minimum an ``"id"`` key.  Recommended keys:
            ``name``, ``code``, ``state``, ``ic_mean``, ``icir``, ``sharpe``,
            ``mdd``, ``created_at``, ``hypothesis``.

        Returns
        -------
        str
            The factor ID.
        """
        fid = factor_profile.get("id")
        if not fid:
            fid = hashlib.sha256(
                json.dumps(factor_profile, sort_keys=True, default=str).encode()
            ).hexdigest()[:12]
            factor_profile["id"] = fid

        # Track timestamps.
        if fid not in self._factors:
            factor_profile.setdefault("added_at", datetime.now(UTC).isoformat())
        factor_profile["updated_at"] = datetime.now(UTC).isoformat()

        self._factors[fid] = factor_profile

        # Update hash index.
        ast_hash = factor_profile.get("ast_hash", "")
        if ast_hash:
            self._hash_index.setdefault(ast_hash, set()).add(fid)

        logger.debug("Factor %s added/updated in QFKB.", fid)
        return fid

    def remove_factor(self, factor_id: str) -> bool:
        """Remove a factor from the knowledge base.

        Returns True if the factor existed and was removed.
        """
        profile = self._factors.pop(factor_id, None)
        if profile is None:
            return False

        # Clean hash index.
        ast_hash = profile.get("ast_hash", "")
        if ast_hash and ast_hash in self._hash_index:
            self._hash_index[ast_hash].discard(factor_id)
            if not self._hash_index[ast_hash]:
                del self._hash_index[ast_hash]

        logger.debug("Factor %s removed from QFKB.", factor_id)
        return True

    # ------------------------------------------------------------------
    # Public API -- queries
    # ------------------------------------------------------------------

    def get_factor(self, factor_id: str) -> Optional[Dict[str, Any]]:
        """Return a deep copy of the factor profile, or None."""
        profile = self._factors.get(factor_id)
        return deepcopy(profile) if profile else None

    def get_all_factors(self) -> List[Dict[str, Any]]:
        """Return deep copies of all factor profiles."""
        return [deepcopy(f) for f in self._factors.values()]

    def get_all_hashes(self) -> List[str]:
        """Return all AST hashes in the knowledge base.

        Used by the HypothesisProposer to avoid re-proposing existing
        factor structures.
        """
        return list(self._hash_index.keys())

    def query_similar(
        self,
        hypothesis: HypothesisSpec | Dict[str, Any],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve the *top_k* factors most similar to *hypothesis*.

        Similarity is determined by:
          1. Exact AST-hash match (highest priority).
          2. Keyword overlap between hypothesis description and factor name
             / hypothesis fields (lightweight text similarity).

        Parameters
        ----------
        hypothesis : HypothesisSpec | dict
            The query hypothesis.
        top_k : int
            Maximum number of results.

        Returns
        -------
        list[dict]
            Factor profiles sorted by descending similarity.
        """
        if isinstance(hypothesis, HypothesisSpec):
            query_hash = hypothesis.ast_hash
            query_text = hypothesis.description.lower()
        else:
            query_hash = hypothesis.get("ast_hash", "")
            query_text = hypothesis.get("description", "").lower()

        scored: List[tuple] = []  # (score, factor_profile)

        for fid, profile in self._factors.items():
            score = 0.0

            # Hash match bonus.
            if query_hash and profile.get("ast_hash") == query_hash:
                score += 1.0

            # Keyword overlap.
            factor_text = " ".join([
                profile.get("name", ""),
                profile.get("hypothesis", ""),
                profile.get("description", ""),
            ]).lower()

            query_tokens = set(query_text.split())
            factor_tokens = set(factor_text.split())
            if query_tokens and factor_tokens:
                overlap = len(query_tokens & factor_tokens)
                union = len(query_tokens | factor_tokens)
                score += overlap / union if union > 0 else 0.0

            scored.append((score, profile))

        # Sort descending by score.
        scored.sort(key=lambda x: x[0], reverse=True)

        return [deepcopy(profile) for _, profile in scored[:top_k]]

    def get_retired_factors(self) -> List[Dict[str, Any]]:
        """Return all factors in RETIRED or ARCHIVED state.

        These are candidates for mutation by the Hypothesis Engine.
        """
        retired_states = {
            AlphaState.RETIRED.name,
            AlphaState.ARCHIVED.name,
            "RETIRED",
            "ARCHIVED",
        }
        return [
            deepcopy(f)
            for f in self._factors.values()
            if f.get("state") in retired_states
        ]

    def get_active_factors(self) -> List[Dict[str, Any]]:
        """Return factors in active deployment states."""
        active_states = {
            AlphaState.LIVE_DEPLOYED.name,
            AlphaState.PAPER_TRADING.name,
            "LIVE_DEPLOYED",
            "PAPER_TRADING",
        }
        return [
            deepcopy(f)
            for f in self._factors.values()
            if f.get("state") in active_states
        ]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics about the knowledge base.

        Returns
        -------
        dict
            Keys: ``total_factors``, ``active``, ``retired``, ``archived``,
            ``avg_lifespan_days``, ``avg_ic``, ``unique_hashes``.
        """
        total = len(self._factors)
        active = len(self.get_active_factors())
        retired_states = {
            AlphaState.RETIRED.name, "RETIRED",
        }
        archived_states = {
            AlphaState.ARCHIVED.name, "ARCHIVED",
        }
        retired = sum(
            1 for f in self._factors.values()
            if f.get("state") in retired_states
        )
        archived = sum(
            1 for f in self._factors.values()
            if f.get("state") in archived_states
        )

        # Average lifespan of retired factors.
        lifespans: List[float] = []
        for f in self._factors.values():
            ld = f.get("lifespan_days")
            if ld is not None and ld > 0:
                lifespans.append(float(ld))

        avg_lifespan = float(np.mean(lifespans)) if lifespans else 0.0

        # Average IC across all factors.
        ics = [
            f.get("ic_mean", 0.0)
            for f in self._factors.values()
            if f.get("ic_mean") is not None
        ]
        avg_ic = float(np.mean(ics)) if ics else 0.0

        return {
            "total_factors": total,
            "active": active,
            "retired": retired,
            "archived": archived,
            "avg_lifespan_days": round(avg_lifespan, 1),
            "avg_ic": round(avg_ic, 6),
            "unique_hashes": len(self._hash_index),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Serialise the knowledge base to a JSON file.

        Parameters
        ----------
        path : str | Path | None
            Override the default persist path.

        Returns
        -------
        Path
            The file that was written.
        """
        target = Path(path) if path else self.persist_path
        if target is None:
            raise ValueError("No persist_path configured and no path given.")

        target.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "saved_at": datetime.now(UTC).isoformat(),
            "factors": self._factors,
        }
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

        logger.info("QFKB saved to %s (%d factors).", target, len(self._factors))
        return target

    def load(self, path: Optional[str | Path] = None) -> int:
        """Load the knowledge base from a JSON file.

        Parameters
        ----------
        path : str | Path | None
            Override the default persist path.

        Returns
        -------
        int
            Number of factors loaded.
        """
        source = Path(path) if path else self.persist_path
        if source is None:
            raise ValueError("No persist_path configured and no path given.")
        if not source.exists():
            logger.warning("QFKB file not found: %s", source)
            return 0

        with open(source, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        self._factors = data.get("factors", {})

        # Rebuild hash index.
        self._hash_index.clear()
        for fid, profile in self._factors.items():
            ast_hash = profile.get("ast_hash", "")
            if ast_hash:
                self._hash_index.setdefault(ast_hash, set()).add(fid)

        logger.info("QFKB loaded from %s (%d factors).", source, len(self._factors))
        return len(self._factors)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._factors)

    def __contains__(self, factor_id: str) -> bool:
        return factor_id in self._factors

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"QuantFinanceKnowledgeBase("
            f"total={stats['total_factors']}, "
            f"active={stats['active']}, "
            f"retired={stats['retired']})"
        )
