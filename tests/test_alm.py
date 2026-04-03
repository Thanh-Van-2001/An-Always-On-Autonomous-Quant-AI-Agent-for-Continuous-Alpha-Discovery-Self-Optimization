"""
Tests for the Alpha Lifecycle Management (ALM) module.

Validates the 8-state FSM, transition guards, the 7-day grace period for
UNDER_REVIEW -> RETIRED, and factor registration/querying.
"""

from datetime import datetime, timedelta

import pytest

from aqaa.core.types import AlphaFactor, AlphaState, BacktestResult
from aqaa.lema.alm import (
    AlphaLifecycleManager,
    TransitionThresholds,
    TransitionRecord,
    _ALLOWED_TRANSITIONS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manager():
    return AlphaLifecycleManager()


@pytest.fixture
def factor():
    return AlphaFactor(id="test_factor_001", name="Test Factor")


@pytest.fixture
def registered(manager, factor):
    """Return (manager, factor) with the factor already registered."""
    manager.register(factor)
    return manager, factor


# ---------------------------------------------------------------------------
# Test: Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    """Verify factor registration and duplicate detection."""

    def test_register_factor(self, manager, factor):
        manager.register(factor)
        assert manager.factor_count == 1
        assert manager.get_factor(factor.id) is factor

    def test_register_duplicate_raises(self, manager, factor):
        manager.register(factor)
        with pytest.raises(ValueError, match="already registered"):
            manager.register(factor)

    def test_get_factor_unknown(self, manager):
        assert manager.get_factor("nonexistent") is None


# ---------------------------------------------------------------------------
# Test: Valid state transitions (full happy path)
# ---------------------------------------------------------------------------

class TestHappyPath:
    """Walk a factor through the complete lifecycle."""

    def test_full_lifecycle(self, manager):
        f = AlphaFactor(id="lifecycle_001", name="Full Lifecycle")
        manager.register(f)

        # 1. HYPOTHESIS_GENERATED -> BACKTESTING (novelty gate)
        assert manager.transition(
            f, AlphaState.BACKTESTING, novelty_score=0.50
        ) is True
        assert f.state == AlphaState.BACKTESTING

        # 2. BACKTESTING -> PAPER_TRADING (IC + Sharpe gate)
        bt_result = BacktestResult(ic=0.05, sharpe=1.5)
        assert manager.transition(
            f, AlphaState.PAPER_TRADING, backtest_result=bt_result
        ) is True
        assert f.state == AlphaState.PAPER_TRADING

        # 3. PAPER_TRADING -> LIVE_DEPLOYED (live IC / backtest IC gate)
        assert manager.transition(
            f, AlphaState.LIVE_DEPLOYED,
            live_ic=0.045, backtest_ic=0.05,  # 0.045 >= 0.05*0.8 = 0.04
        ) is True
        assert f.state == AlphaState.LIVE_DEPLOYED

        # 4. LIVE_DEPLOYED -> UNDER_REVIEW (no quantitative gate needed
        #    beyond structural validity -- guard returns True for LD->UR)
        now = datetime(2025, 6, 1, 12, 0)
        assert manager.transition(
            f, AlphaState.UNDER_REVIEW, now=now
        ) is True
        assert f.state == AlphaState.UNDER_REVIEW

        # 5. UNDER_REVIEW -> RETIRED (grace period must have elapsed)
        after_grace = now + timedelta(days=7)
        assert manager.transition(
            f, AlphaState.RETIRED, now=after_grace
        ) is True
        assert f.state == AlphaState.RETIRED

        # 6. RETIRED -> ARCHIVED
        assert manager.transition(f, AlphaState.ARCHIVED) is True
        assert f.state == AlphaState.ARCHIVED

        # 7. ARCHIVED -> MUTATED
        assert manager.transition(f, AlphaState.MUTATED) is True
        assert f.state == AlphaState.MUTATED

        # 8. MUTATED -> HYPOTHESIS_GENERATED (loop)
        assert manager.transition(
            f, AlphaState.HYPOTHESIS_GENERATED
        ) is True
        assert f.state == AlphaState.HYPOTHESIS_GENERATED

        # Verify full history
        history = manager.get_history(f.id)
        assert len(history) == 8
        assert all(isinstance(r, TransitionRecord) for r in history)


# ---------------------------------------------------------------------------
# Test: Invalid transitions are rejected
# ---------------------------------------------------------------------------

class TestInvalidTransitions:
    """Transitions not in the FSM edge set must raise ValueError."""

    def test_hg_to_live_rejected(self, registered):
        manager, factor = registered
        with pytest.raises(ValueError, match="not allowed"):
            manager.transition(factor, AlphaState.LIVE_DEPLOYED)

    def test_backtesting_to_live_rejected(self, registered):
        manager, factor = registered
        manager.transition(factor, AlphaState.BACKTESTING, novelty_score=0.5)
        with pytest.raises(ValueError, match="not allowed"):
            manager.transition(factor, AlphaState.LIVE_DEPLOYED)

    def test_retired_to_live_rejected(self, registered):
        """Cannot jump from RETIRED directly to LIVE_DEPLOYED."""
        manager, factor = registered
        # Walk factor to RETIRED
        manager.transition(factor, AlphaState.BACKTESTING, novelty_score=0.5)
        manager.transition(
            factor, AlphaState.PAPER_TRADING,
            backtest_result=BacktestResult(ic=0.05, sharpe=1.5),
        )
        manager.transition(
            factor, AlphaState.LIVE_DEPLOYED,
            live_ic=0.05, backtest_ic=0.05,
        )
        now = datetime(2025, 1, 1)
        manager.transition(factor, AlphaState.UNDER_REVIEW, now=now)
        manager.transition(
            factor, AlphaState.RETIRED,
            now=now + timedelta(days=7),
        )
        with pytest.raises(ValueError, match="not allowed"):
            manager.transition(factor, AlphaState.LIVE_DEPLOYED)

    def test_unregistered_factor_raises(self, manager):
        unregistered = AlphaFactor(id="ghost_factor")
        with pytest.raises(ValueError, match="not registered"):
            manager.transition(unregistered, AlphaState.BACKTESTING)


# ---------------------------------------------------------------------------
# Test: Quality gate rejections (guard failures)
# ---------------------------------------------------------------------------

class TestGuardRejections:
    """Guard failures should return False (not raise)."""

    def test_low_novelty_rejects_hg_to_bt(self, registered):
        manager, factor = registered
        assert manager.transition(
            factor, AlphaState.BACKTESTING, novelty_score=0.10
        ) is False
        assert factor.state == AlphaState.HYPOTHESIS_GENERATED

    def test_missing_novelty_rejects_hg_to_bt(self, registered):
        manager, factor = registered
        # novelty_score not provided
        assert manager.transition(
            factor, AlphaState.BACKTESTING
        ) is False

    def test_low_ic_rejects_bt_to_pt(self, registered):
        manager, factor = registered
        manager.transition(factor, AlphaState.BACKTESTING, novelty_score=0.5)
        bt_result = BacktestResult(ic=0.02, sharpe=1.5)  # IC < 0.03
        assert manager.transition(
            factor, AlphaState.PAPER_TRADING, backtest_result=bt_result
        ) is False

    def test_low_sharpe_rejects_bt_to_pt(self, registered):
        manager, factor = registered
        manager.transition(factor, AlphaState.BACKTESTING, novelty_score=0.5)
        bt_result = BacktestResult(ic=0.05, sharpe=0.8)  # Sharpe < 1.0
        assert manager.transition(
            factor, AlphaState.PAPER_TRADING, backtest_result=bt_result
        ) is False

    def test_low_live_ic_rejects_pt_to_ld(self, registered):
        manager, factor = registered
        manager.transition(factor, AlphaState.BACKTESTING, novelty_score=0.5)
        manager.transition(
            factor, AlphaState.PAPER_TRADING,
            backtest_result=BacktestResult(ic=0.05, sharpe=1.5),
        )
        # live_ic=0.03 < backtest_ic=0.05 * 0.80 = 0.04
        assert manager.transition(
            factor, AlphaState.LIVE_DEPLOYED,
            live_ic=0.03, backtest_ic=0.05,
        ) is False


# ---------------------------------------------------------------------------
# Test: 7-day grace period for UNDER_REVIEW -> RETIRED
# ---------------------------------------------------------------------------

class TestGracePeriod:
    """UNDER_REVIEW -> RETIRED requires 7-day grace period to elapse."""

    def _walk_to_under_review(self, manager, factor, now):
        """Helper to bring a factor to UNDER_REVIEW state."""
        manager.transition(factor, AlphaState.BACKTESTING, novelty_score=0.5)
        manager.transition(
            factor, AlphaState.PAPER_TRADING,
            backtest_result=BacktestResult(ic=0.05, sharpe=1.5),
        )
        manager.transition(
            factor, AlphaState.LIVE_DEPLOYED,
            live_ic=0.05, backtest_ic=0.05,
        )
        manager.transition(factor, AlphaState.UNDER_REVIEW, now=now)

    def test_before_grace_period_rejects(self, registered):
        manager, factor = registered
        now = datetime(2025, 6, 1, 12, 0)
        self._walk_to_under_review(manager, factor, now)

        # Only 3 days later -- grace period not yet expired
        assert manager.transition(
            factor, AlphaState.RETIRED,
            now=now + timedelta(days=3),
        ) is False
        assert factor.state == AlphaState.UNDER_REVIEW

    def test_exactly_at_grace_period_passes(self, registered):
        manager, factor = registered
        now = datetime(2025, 6, 1, 12, 0)
        self._walk_to_under_review(manager, factor, now)

        # Exactly 7 days -> should pass (>=)
        assert manager.transition(
            factor, AlphaState.RETIRED,
            now=now + timedelta(days=7),
        ) is True
        assert factor.state == AlphaState.RETIRED

    def test_after_grace_period_passes(self, registered):
        manager, factor = registered
        now = datetime(2025, 6, 1, 12, 0)
        self._walk_to_under_review(manager, factor, now)

        assert manager.transition(
            factor, AlphaState.RETIRED,
            now=now + timedelta(days=14),
        ) is True

    def test_under_review_to_live_recovery(self, registered):
        """Factor can recover from UNDER_REVIEW back to LIVE_DEPLOYED."""
        manager, factor = registered
        now = datetime(2025, 6, 1)
        self._walk_to_under_review(manager, factor, now)

        # Recovery: live IC has improved
        assert manager.transition(
            factor, AlphaState.LIVE_DEPLOYED,
            live_ic=0.05, backtest_ic=0.05,
        ) is True
        assert factor.state == AlphaState.LIVE_DEPLOYED


# ---------------------------------------------------------------------------
# Test: Querying
# ---------------------------------------------------------------------------

class TestQuerying:
    """Test query methods on the lifecycle manager."""

    def test_get_active_factors(self, manager):
        f1 = AlphaFactor(id="f1")
        f2 = AlphaFactor(id="f2")
        f3 = AlphaFactor(id="f3")
        manager.register(f1)
        manager.register(f2)
        manager.register(f3)

        # f1 -> BACKTESTING (not active)
        manager.transition(f1, AlphaState.BACKTESTING, novelty_score=0.5)
        # f2 -> PAPER_TRADING (active)
        manager.transition(f2, AlphaState.BACKTESTING, novelty_score=0.5)
        manager.transition(
            f2, AlphaState.PAPER_TRADING,
            backtest_result=BacktestResult(ic=0.05, sharpe=1.5),
        )

        active = manager.get_active_factors()
        assert len(active) == 1
        assert active[0].id == "f2"

    def test_get_factors_by_state(self, manager):
        f1 = AlphaFactor(id="f1")
        f2 = AlphaFactor(id="f2")
        manager.register(f1)
        manager.register(f2)

        manager.transition(f1, AlphaState.BACKTESTING, novelty_score=0.5)

        bt = manager.get_factors_by_state(AlphaState.BACKTESTING)
        hg = manager.get_factors_by_state(AlphaState.HYPOTHESIS_GENERATED)

        assert len(bt) == 1
        assert bt[0].id == "f1"
        assert len(hg) == 1
        assert hg[0].id == "f2"

    def test_summary(self, manager):
        f1 = AlphaFactor(id="f1")
        manager.register(f1)
        summary = manager.summary()
        assert summary["HYPOTHESIS_GENERATED"] == 1
        assert sum(summary.values()) == 1

    def test_history(self, manager):
        f = AlphaFactor(id="hist_factor")
        manager.register(f)
        manager.transition(f, AlphaState.BACKTESTING, novelty_score=0.5)

        all_history = manager.get_history()
        factor_history = manager.get_history("hist_factor")
        assert len(all_history) == 1
        assert len(factor_history) == 1
        assert factor_history[0].from_state == AlphaState.HYPOTHESIS_GENERATED
        assert factor_history[0].to_state == AlphaState.BACKTESTING


# ---------------------------------------------------------------------------
# Test: Allowed transitions structure
# ---------------------------------------------------------------------------

class TestAllowedTransitionsTable:
    """Validate the FSM edge set itself."""

    def test_all_states_have_entries(self):
        for state in AlphaState:
            assert state in _ALLOWED_TRANSITIONS

    def test_hg_only_goes_to_bt(self):
        assert _ALLOWED_TRANSITIONS[AlphaState.HYPOTHESIS_GENERATED] == {
            AlphaState.BACKTESTING
        }

    def test_under_review_has_two_targets(self):
        targets = _ALLOWED_TRANSITIONS[AlphaState.UNDER_REVIEW]
        assert AlphaState.RETIRED in targets
        assert AlphaState.LIVE_DEPLOYED in targets
        assert len(targets) == 2

    def test_mutated_loops_to_hg(self):
        assert _ALLOWED_TRANSITIONS[AlphaState.MUTATED] == {
            AlphaState.HYPOTHESIS_GENERATED
        }
