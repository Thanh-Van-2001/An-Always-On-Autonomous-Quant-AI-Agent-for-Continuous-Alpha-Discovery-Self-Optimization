"""
Feedback Loop Orchestrator.

Coordinates the 24-hour autonomous cycle of the AQAA system.  Every cycle:

1. The UCB1 bandit allocates a compute budget across the five modules.
2. Each module executes its workload within its allocated GPU-hours.
3. Live performance feedback is collected and fed back to the knowledge base.
4. The bandit is updated with the marginal IC gains achieved per module.

This implements Algorithm 1 (Outer Loop) from the KDD 2026 paper, Section 4.

Usage
-----
>>> from aqaa.flo import FeedbackLoopOrchestrator
>>> orch = FeedbackLoopOrchestrator()
>>> orch.run_outer_loop()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from aqaa.core.types import AlphaFactor, AlphaState, BacktestResult
from aqaa.flo.ucb1 import UCB1Bandit

logger = logging.getLogger(__name__)

# Module names must match the UCB1 arm names exactly.
MODULE_NAMES: List[str] = ["HE", "CSBA", "APO", "LEMA", "KB"]


# ---------------------------------------------------------------------------
# Budget Allocator
# ---------------------------------------------------------------------------

@dataclass
class ComputeAllocation:
    """A concrete compute allocation for a single cycle.

    Attributes:
        module: Module name (e.g. 'HE', 'CSBA').
        gpu_hours: Allocated GPU-hours for this module.
        fraction: Fraction of total budget.
        timestamp: When the allocation was made.
    """

    module: str = ""
    gpu_hours: float = 0.0
    fraction: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class BudgetAllocator:
    """Translates UCB1 arm fractions into concrete GPU-hour allocations.

    The allocator enforces a minimum allocation per module (``min_fraction``)
    to ensure that no module is completely starved, which is critical for
    maintaining the always-on property of the system.

    Parameters
    ----------
    min_fraction : float
        Minimum compute fraction per module.  Default 0.05 (5%).
    """

    def __init__(self, min_fraction: float = 0.05) -> None:
        self.min_fraction = min_fraction

    def allocate(
        self,
        bandit: UCB1Bandit,
        total_gpu_hours: float,
    ) -> List[ComputeAllocation]:
        """Produce a list of ``ComputeAllocation`` from UCB1 fractions.

        The raw fractions from the bandit are clipped so that every module
        receives at least ``min_fraction`` of the budget, then re-normalised
        to sum to 1.

        Parameters
        ----------
        bandit : UCB1Bandit
            The bandit whose allocation to materialise.
        total_gpu_hours : float
            Total GPU-hours available this cycle.

        Returns
        -------
        list[ComputeAllocation]
            One entry per module.
        """
        raw = bandit.get_allocation()
        n_modules = len(raw)
        min_total = self.min_fraction * n_modules

        if min_total > 1.0:
            raise ValueError(
                f"min_fraction={self.min_fraction} * {n_modules} modules "
                f"= {min_total} > 1.0; reduce min_fraction"
            )

        # Clip to minimum and re-normalise.
        fractions = {
            name: max(frac, self.min_fraction)
            for name, frac in raw.items()
        }
        total = sum(fractions.values())
        fractions = {name: f / total for name, f in fractions.items()}

        now = datetime.now(UTC)
        allocations = [
            ComputeAllocation(
                module=name,
                gpu_hours=round(frac * total_gpu_hours, 4),
                fraction=round(frac, 6),
                timestamp=now,
            )
            for name, frac in fractions.items()
        ]

        logger.info(
            "Budget allocated: %s",
            {a.module: f"{a.gpu_hours:.2f}h ({a.fraction:.1%})" for a in allocations},
        )
        return allocations


# ---------------------------------------------------------------------------
# Feedback Loop Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class CycleRecord:
    """Immutable record of a single outer-loop cycle for auditability.

    Attributes:
        cycle_id: Sequential cycle number.
        started_at: Cycle start timestamp.
        ended_at: Cycle end timestamp (None if still running).
        allocations: Compute allocations for this cycle.
        rewards: Observed reward per module.
        ic_before: Mean portfolio IC at the start of the cycle.
        ic_after: Mean portfolio IC at the end of the cycle.
        notes: Free-form notes / warnings.
    """

    cycle_id: int = 0
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    allocations: List[ComputeAllocation] = field(default_factory=list)
    rewards: Dict[str, float] = field(default_factory=dict)
    ic_before: float = 0.0
    ic_after: float = 0.0
    notes: str = ""


class FeedbackLoopOrchestrator:
    """Central coordinator for the AQAA 24-hour autonomous cycle.

    This is the top-level class that drives the outer loop described in
    Algorithm 1 of the KDD 2026 paper.

    Parameters
    ----------
    budget_gpu_hours : float
        Total GPU-hours available per cycle.  Default 10.
    ucb_c : float
        UCB1 exploration constant.  Default 1.5.
    min_budget_fraction : float
        Minimum per-module compute fraction.  Default 0.05.
    log_dir : str | Path | None
        Directory to persist cycle logs as JSON.  If None, logging is
        in-memory only.
    seed : int | None
        Random seed for the UCB1 bandit.
    """

    def __init__(
        self,
        budget_gpu_hours: float = 10.0,
        ucb_c: float = 1.5,
        min_budget_fraction: float = 0.05,
        log_dir: Optional[str | Path] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.budget_gpu_hours = budget_gpu_hours
        self.bandit = UCB1Bandit(c=ucb_c, seed=seed)
        self.allocator = BudgetAllocator(min_fraction=min_budget_fraction)
        self.log_dir = Path(log_dir) if log_dir else None
        self.cycle_count: int = 0
        self.history: List[CycleRecord] = []

        # Knowledge-base state (lightweight in-memory representation).
        self._factor_archive: List[Dict[str, Any]] = []
        self._last_ic: float = 0.0

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_outer_loop(
        self,
        n_cycles: int = 1,
        live_results: Optional[Dict[str, Any]] = None,
    ) -> List[CycleRecord]:
        """Execute *n_cycles* of the 24-hour outer loop.

        Each cycle:
          1. ``allocate_compute`` -- UCB1 distributes GPU-hours.
          2. (Module execution is external; this method simulates rewards.)
          3. ``collect_feedback`` -- record live performance.
          4. ``update_knowledge_base`` -- archive / update factor profiles.

        Parameters
        ----------
        n_cycles : int
            Number of cycles to run.
        live_results : dict | None
            Optional external live-performance data to feed into the loop.
            If None, synthetic rewards are generated for demonstration.

        Returns
        -------
        list[CycleRecord]
            Records for every cycle executed.
        """
        records: List[CycleRecord] = []

        for _ in range(n_cycles):
            self.cycle_count += 1
            record = CycleRecord(
                cycle_id=self.cycle_count,
                started_at=datetime.now(UTC),
                ic_before=self._last_ic,
            )

            # Step 1: Allocate compute budget.
            allocations = self.allocate_compute(self.budget_gpu_hours)
            record.allocations = allocations

            # Step 2: Simulate module execution and collect per-module rewards.
            rewards = self._execute_modules(allocations, live_results)
            record.rewards = rewards

            # Step 3: Update the UCB1 bandit with observed rewards.
            for alloc in allocations:
                arm_idx = self.bandit.arm_names.index(alloc.module)
                self.bandit.update(arm_idx, rewards.get(alloc.module, 0.0))

            # Step 4: Collect feedback and update knowledge base.
            feedback = live_results or self._synthetic_feedback(rewards)
            self.collect_feedback(feedback)

            record.ic_after = self._last_ic
            record.ended_at = datetime.now(UTC)
            records.append(record)
            self.history.append(record)

            self._persist_cycle(record)

            logger.info(
                "Cycle %d complete: IC %.4f -> %.4f  |  allocation=%s",
                record.cycle_id,
                record.ic_before,
                record.ic_after,
                {a.module: f"{a.gpu_hours:.2f}h" for a in allocations},
            )

        return records

    def allocate_compute(
        self,
        budget_gpu_hours: Optional[float] = None,
    ) -> List[ComputeAllocation]:
        """Use the UCB1 bandit to distribute the compute budget.

        Parameters
        ----------
        budget_gpu_hours : float | None
            Override the default budget for this cycle.

        Returns
        -------
        list[ComputeAllocation]
        """
        budget = budget_gpu_hours if budget_gpu_hours is not None else self.budget_gpu_hours
        return self.allocator.allocate(self.bandit, budget)

    def collect_feedback(self, live_results: Dict[str, Any]) -> None:
        """Feed live-performance data back into the system.

        Parameters
        ----------
        live_results : dict
            Must contain at minimum an ``"ic"`` key with the current mean
            portfolio information coefficient.  May optionally contain
            ``"factor_profiles"`` for knowledge-base updates.
        """
        if "ic" in live_results:
            self._last_ic = float(live_results["ic"])

        factor_profiles = live_results.get("factor_profiles", [])
        if factor_profiles:
            self.update_knowledge_base(factor_profiles)

        logger.info(
            "Feedback collected: IC=%.4f, %d factor profiles",
            self._last_ic,
            len(factor_profiles),
        )

    def update_knowledge_base(
        self,
        factor_profiles: List[Dict[str, Any]],
    ) -> None:
        """Archive retired factors and update the QFKB.

        Parameters
        ----------
        factor_profiles : list[dict]
            Each dict describes a factor's current state.  Factors with
            ``state == 'RETIRED'`` are moved to the archive.
        """
        for profile in factor_profiles:
            state = profile.get("state", "")
            factor_id = profile.get("id", "unknown")

            if state == AlphaState.RETIRED.name or state == "RETIRED":
                profile["archived_at"] = datetime.now(UTC).isoformat()
                self._factor_archive.append(profile)
                logger.info("Factor %s archived (retired).", factor_id)
            else:
                # Update existing entry or append.
                existing = [
                    f for f in self._factor_archive if f.get("id") == factor_id
                ]
                if existing:
                    existing[0].update(profile)
                else:
                    self._factor_archive.append(profile)

        logger.info(
            "Knowledge base updated: %d total entries, %d newly archived.",
            len(self._factor_archive),
            sum(
                1 for f in factor_profiles
                if f.get("state") in (AlphaState.RETIRED.name, "RETIRED")
            ),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a JSON-serialisable status snapshot."""
        return {
            "cycle_count": self.cycle_count,
            "budget_gpu_hours": self.budget_gpu_hours,
            "current_allocation": self.bandit.get_allocation(),
            "bandit_summary": self.bandit.summary(),
            "last_ic": self._last_ic,
            "archive_size": len(self._factor_archive),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_modules(
        self,
        allocations: List[ComputeAllocation],
        live_results: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Simulate module execution and return per-module rewards.

        In a production system this would dispatch real compute jobs.  Here
        we generate synthetic rewards proportional to each module's
        allocation, with the expectation that external callers will provide
        real results via ``live_results``.
        """
        if live_results and "rewards" in live_results:
            return live_results["rewards"]

        # Synthetic reward model: base rate * sqrt(gpu_hours) + noise.
        base_rates = {"HE": 0.010, "CSBA": 0.015, "APO": 0.008, "LEMA": 0.012, "KB": 0.006}
        rng = self.bandit.rng
        rewards: Dict[str, float] = {}
        for alloc in allocations:
            base = base_rates.get(alloc.module, 0.01)
            reward = base * np.sqrt(max(alloc.gpu_hours, 0.01)) + float(
                rng.normal(0, 0.002)
            )
            rewards[alloc.module] = round(max(reward, 0.0), 6)

        return rewards

    def _synthetic_feedback(self, rewards: Dict[str, float]) -> Dict[str, Any]:
        """Build a synthetic feedback dict from simulated rewards."""
        ic_delta = sum(rewards.values())
        return {
            "ic": self._last_ic + ic_delta,
            "factor_profiles": [],
        }

    def _persist_cycle(self, record: CycleRecord) -> None:
        """Write a cycle record to disk as JSON if ``log_dir`` is set."""
        if self.log_dir is None:
            return

        path = self.log_dir / f"cycle_{record.cycle_id:05d}.json"
        data = {
            "cycle_id": record.cycle_id,
            "started_at": record.started_at.isoformat() if record.started_at else None,
            "ended_at": record.ended_at.isoformat() if record.ended_at else None,
            "ic_before": record.ic_before,
            "ic_after": record.ic_after,
            "rewards": record.rewards,
            "allocations": [
                {
                    "module": a.module,
                    "gpu_hours": a.gpu_hours,
                    "fraction": a.fraction,
                }
                for a in record.allocations
            ],
            "notes": record.notes,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

        logger.debug("Cycle record persisted to %s", path)

    def __repr__(self) -> str:
        return (
            f"FeedbackLoopOrchestrator(cycles={self.cycle_count}, "
            f"budget={self.budget_gpu_hours}h, ic={self._last_ic:.4f})"
        )
