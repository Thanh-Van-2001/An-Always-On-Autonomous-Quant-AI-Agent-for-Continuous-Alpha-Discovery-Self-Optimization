"""
Alpha Lifecycle Management (ALM) -- 8-state machine.

Implements the lifecycle FSM depicted in Figure 3 of the KDD 2026 paper.
Each :class:`AlphaFactor` moves through the following states:

    HYPOTHESIS_GENERATED -> BACKTESTING -> PAPER_TRADING -> LIVE_DEPLOYED
        -> UNDER_REVIEW -> RETIRED -> ARCHIVED -> MUTATED
        (MUTATED loops back to HYPOTHESIS_GENERATED)

Transition guards enforce the statistical quality gates described in
Section 4.3 (Algorithmic Details -- Live Execution & Monitoring Agent).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from aqaa.core.types import AlphaFactor, AlphaState, BacktestResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed transitions and their human-readable labels
# ---------------------------------------------------------------------------

_ALLOWED_TRANSITIONS: Dict[AlphaState, Set[AlphaState]] = {
    AlphaState.HYPOTHESIS_GENERATED: {AlphaState.BACKTESTING},
    AlphaState.BACKTESTING: {AlphaState.PAPER_TRADING},
    AlphaState.PAPER_TRADING: {AlphaState.LIVE_DEPLOYED},
    AlphaState.LIVE_DEPLOYED: {AlphaState.UNDER_REVIEW},
    AlphaState.UNDER_REVIEW: {AlphaState.RETIRED, AlphaState.LIVE_DEPLOYED},
    AlphaState.RETIRED: {AlphaState.ARCHIVED},
    AlphaState.ARCHIVED: {AlphaState.MUTATED},
    AlphaState.MUTATED: {AlphaState.HYPOTHESIS_GENERATED},
}


# ---------------------------------------------------------------------------
# Transition-gate parameters (defaults)
# ---------------------------------------------------------------------------

@dataclass
class TransitionThresholds:
    """Quality-gate thresholds that govern state transitions.

    Attributes:
        novelty_min: Minimum novelty score to enter BACKTESTING.
        ic_min: Minimum mean IC to pass backtesting.
        sharpe_min: Minimum Sharpe ratio to pass backtesting.
        live_ic_ratio: Live IC must exceed backtest IC * this ratio
                       to be promoted from PAPER_TRADING.
        grace_period_days: Days of grace in UNDER_REVIEW before retirement.
    """

    novelty_min: float = 0.30
    ic_min: float = 0.03
    sharpe_min: float = 1.0
    live_ic_ratio: float = 0.80
    grace_period_days: int = 7


# ---------------------------------------------------------------------------
# Transition history record
# ---------------------------------------------------------------------------

@dataclass
class TransitionRecord:
    """An immutable record of a single state transition."""

    factor_id: str
    from_state: AlphaState
    to_state: AlphaState
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reason: str = ""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AlphaLifecycleManager:
    """Manages the lifecycle of every alpha factor within the AQAA system.

    The manager stores references to :class:`AlphaFactor` objects, enforces
    transition guards, and maintains a full audit trail of state changes.

    Parameters
    ----------
    thresholds : TransitionThresholds, optional
        Quality-gate thresholds.  Defaults are used if not supplied.
    """

    def __init__(
        self,
        thresholds: Optional[TransitionThresholds] = None,
    ) -> None:
        self.thresholds = thresholds or TransitionThresholds()
        # factor_id -> AlphaFactor
        self._factors: Dict[str, AlphaFactor] = {}
        # full history of transitions
        self._history: List[TransitionRecord] = []
        # factor_id -> timestamp when UNDER_REVIEW was entered
        self._review_entered: Dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, factor: AlphaFactor) -> None:
        """Register a factor with the lifecycle manager.

        The factor is stored by its ``id`` and must start in the
        ``HYPOTHESIS_GENERATED`` state.

        Raises
        ------
        ValueError
            If a factor with the same id is already registered.
        """
        if factor.id in self._factors:
            raise ValueError(
                f"Factor {factor.id!r} is already registered."
            )
        self._factors[factor.id] = factor
        logger.info(
            "Registered factor %s in state %s",
            factor.id,
            factor.state.name,
        )

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------

    def transition(
        self,
        factor: AlphaFactor,
        new_state: AlphaState,
        *,
        reason: str = "",
        backtest_result: Optional[BacktestResult] = None,
        live_ic: Optional[float] = None,
        backtest_ic: Optional[float] = None,
        novelty_score: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        """Attempt to transition *factor* to *new_state*.

        Returns ``True`` if the transition succeeds, ``False`` if a guard
        rejects it.

        Parameters
        ----------
        factor : AlphaFactor
            The factor to transition (must be registered).
        new_state : AlphaState
            Target state.
        reason : str, optional
            Free-text reason recorded in the audit trail.
        backtest_result : BacktestResult, optional
            Required when transitioning BACKTESTING -> PAPER_TRADING.
        live_ic : float, optional
            Required when transitioning PAPER_TRADING -> LIVE_DEPLOYED.
        backtest_ic : float, optional
            Required when transitioning PAPER_TRADING -> LIVE_DEPLOYED.
        novelty_score : float, optional
            Required when transitioning HYPOTHESIS_GENERATED -> BACKTESTING.
        now : datetime, optional
            Override for the current time (useful for testing).

        Returns
        -------
        bool
            ``True`` on success, ``False`` if the guard rejected the move.

        Raises
        ------
        ValueError
            If the factor is unregistered or the transition is structurally
            invalid (not in the FSM edge set).
        """
        now = now or datetime.now(UTC)

        if factor.id not in self._factors:
            raise ValueError(
                f"Factor {factor.id!r} is not registered with this manager."
            )

        old_state = factor.state

        # --- structural validity ---
        allowed = _ALLOWED_TRANSITIONS.get(old_state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Transition {old_state.name} -> {new_state.name} is not "
                f"allowed.  Valid targets: "
                f"{', '.join(s.name for s in allowed) if allowed else 'none'}."
            )

        # --- quality gates ---
        if not self._check_guard(
            factor,
            old_state,
            new_state,
            backtest_result=backtest_result,
            live_ic=live_ic,
            backtest_ic=backtest_ic,
            novelty_score=novelty_score,
            now=now,
        ):
            logger.warning(
                "Guard rejected transition %s -> %s for factor %s",
                old_state.name,
                new_state.name,
                factor.id,
            )
            return False

        # --- apply ---
        factor.transition(new_state)
        record = TransitionRecord(
            factor_id=factor.id,
            from_state=old_state,
            to_state=new_state,
            timestamp=now,
            reason=reason,
        )
        self._history.append(record)

        # Track UNDER_REVIEW entry for grace period
        if new_state == AlphaState.UNDER_REVIEW:
            self._review_entered[factor.id] = now
        elif factor.id in self._review_entered:
            del self._review_entered[factor.id]

        logger.info(
            "Factor %s: %s -> %s (%s)",
            factor.id,
            old_state.name,
            new_state.name,
            reason or "no reason given",
        )
        return True

    # ------------------------------------------------------------------
    # Guard logic
    # ------------------------------------------------------------------

    def _check_guard(
        self,
        factor: AlphaFactor,
        old: AlphaState,
        new: AlphaState,
        *,
        backtest_result: Optional[BacktestResult] = None,
        live_ic: Optional[float] = None,
        backtest_ic: Optional[float] = None,
        novelty_score: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        """Evaluate the transition-specific quality gate.

        Returns ``True`` if the guard is satisfied.
        """
        t = self.thresholds

        # HYPOTHESIS_GENERATED -> BACKTESTING: novelty filter
        if old == AlphaState.HYPOTHESIS_GENERATED and new == AlphaState.BACKTESTING:
            if novelty_score is None:
                logger.warning("novelty_score is required for HG -> BT gate.")
                return False
            return novelty_score >= t.novelty_min

        # BACKTESTING -> PAPER_TRADING: IC > 0.03 and Sharpe > 1.0
        if old == AlphaState.BACKTESTING and new == AlphaState.PAPER_TRADING:
            if backtest_result is None:
                logger.warning("backtest_result is required for BT -> PT gate.")
                return False
            return (
                backtest_result.ic >= t.ic_min
                and backtest_result.sharpe >= t.sharpe_min
            )

        # PAPER_TRADING -> LIVE_DEPLOYED: Live IC > backtest IC * 0.8
        if old == AlphaState.PAPER_TRADING and new == AlphaState.LIVE_DEPLOYED:
            if live_ic is None or backtest_ic is None:
                logger.warning(
                    "live_ic and backtest_ic are required for PT -> LD gate."
                )
                return False
            return live_ic >= backtest_ic * t.live_ic_ratio

        # UNDER_REVIEW -> RETIRED: grace period expired
        if old == AlphaState.UNDER_REVIEW and new == AlphaState.RETIRED:
            entered = self._review_entered.get(factor.id)
            if entered is None:
                # No record of when review started; allow conservatively.
                return True
            now = now or datetime.now(UTC)
            return (now - entered) >= timedelta(days=t.grace_period_days)

        # UNDER_REVIEW -> LIVE_DEPLOYED: recovery (re-promotion)
        if old == AlphaState.UNDER_REVIEW and new == AlphaState.LIVE_DEPLOYED:
            # Allowed if live IC has recovered above threshold.
            if live_ic is None or backtest_ic is None:
                logger.warning(
                    "live_ic and backtest_ic are required for UR -> LD gate."
                )
                return False
            return live_ic >= backtest_ic * t.live_ic_ratio

        # RETIRED -> ARCHIVED, ARCHIVED -> MUTATED, MUTATED -> HG:
        # no quantitative gate -- always allowed.
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_factors(self) -> List[AlphaFactor]:
        """Return factors in deployment-relevant states.

        Active states are: PAPER_TRADING, LIVE_DEPLOYED, UNDER_REVIEW.
        """
        active_states = {
            AlphaState.PAPER_TRADING,
            AlphaState.LIVE_DEPLOYED,
            AlphaState.UNDER_REVIEW,
        }
        return [
            f for f in self._factors.values() if f.state in active_states
        ]

    def get_factors_by_state(
        self, state: AlphaState
    ) -> List[AlphaFactor]:
        """Return all factors currently in *state*."""
        return [
            f for f in self._factors.values() if f.state == state
        ]

    def get_factor(self, factor_id: str) -> Optional[AlphaFactor]:
        """Look up a factor by id, or return ``None``."""
        return self._factors.get(factor_id)

    def get_history(
        self,
        factor_id: Optional[str] = None,
    ) -> List[TransitionRecord]:
        """Return transition history, optionally filtered by factor id."""
        if factor_id is None:
            return list(self._history)
        return [r for r in self._history if r.factor_id == factor_id]

    @property
    def factor_count(self) -> int:
        """Total number of registered factors."""
        return len(self._factors)

    def summary(self) -> Dict[str, int]:
        """Return a mapping ``{state_name: count}`` across all factors."""
        counts: Dict[str, int] = {s.name: 0 for s in AlphaState}
        for f in self._factors.values():
            counts[f.state.name] += 1
        return counts
