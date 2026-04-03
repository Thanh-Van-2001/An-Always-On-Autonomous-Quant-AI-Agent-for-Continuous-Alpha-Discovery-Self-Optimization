"""
Tests for the vectorised backtester module.

Validates IC computation (Spearman rank correlation), ICIR, Sharpe ratio
annualization, MDD computation, and acceptance thresholds.
"""

import numpy as np
import pytest
from scipy import stats as sp_stats

from aqaa.csba.backtester import (
    DEFAULT_IC_THRESHOLD,
    DEFAULT_ICIR_THRESHOLD,
    DEFAULT_MDD_THRESHOLD,
    DEFAULT_SHARPE_THRESHOLD,
    TRADING_DAYS_PER_YEAR,
    BacktestMetrics,
    Backtester,
)
from aqaa.core.types import BacktestResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backtester():
    return Backtester()


# ---------------------------------------------------------------------------
# Test: IC computation (Spearman rank correlation)
# ---------------------------------------------------------------------------

class TestComputeIC:
    """Verify IC is Spearman rank correlation per date."""

    def test_perfect_positive_correlation(self, backtester):
        """Predictions identical to returns -> IC = 1.0."""
        rng = np.random.default_rng(42)
        values = rng.standard_normal((5, 20))
        ic = Backtester.compute_ic(values, values)
        np.testing.assert_allclose(ic, 1.0, atol=1e-10)

    def test_perfect_negative_correlation(self, backtester):
        """Predictions = -returns -> IC = -1.0."""
        rng = np.random.default_rng(42)
        values = rng.standard_normal((5, 20))
        ic = Backtester.compute_ic(values, -values)
        np.testing.assert_allclose(ic, -1.0, atol=1e-10)

    def test_matches_scipy_spearmanr(self):
        """IC for each date should match scipy.stats.spearmanr."""
        rng = np.random.default_rng(77)
        predictions = rng.standard_normal((10, 50))
        returns = rng.standard_normal((10, 50))

        ic = Backtester.compute_ic(predictions, returns)
        for t in range(10):
            expected_corr, _ = sp_stats.spearmanr(
                predictions[t], returns[t]
            )
            assert abs(ic[t] - expected_corr) < 1e-10

    def test_1d_input(self):
        """1-D input (single date) should work."""
        rng = np.random.default_rng(88)
        pred = rng.standard_normal(30)
        ret = rng.standard_normal(30)

        ic = Backtester.compute_ic(pred, ret)
        assert ic.shape == (1,)
        expected, _ = sp_stats.spearmanr(pred, ret)
        assert abs(ic[0] - expected) < 1e-10

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            Backtester.compute_ic(
                np.zeros((5, 10)), np.zeros((5, 11))
            )

    def test_too_few_assets_raises(self):
        with pytest.raises(ValueError, match="at least 3 assets"):
            Backtester.compute_ic(
                np.zeros((5, 2)), np.zeros((5, 2))
            )

    def test_nan_handling(self):
        """NaN positions should be masked out."""
        rng = np.random.default_rng(55)
        pred = rng.standard_normal((1, 20))
        ret = rng.standard_normal((1, 20))
        pred[0, 0] = np.nan
        ret[0, 5] = np.nan

        ic = Backtester.compute_ic(pred, ret)
        assert not np.isnan(ic[0])


# ---------------------------------------------------------------------------
# Test: ICIR = mean(IC) / std(IC)
# ---------------------------------------------------------------------------

class TestComputeICIR:
    """Verify ICIR computation."""

    def test_known_values(self):
        ic = np.array([0.04, 0.06, 0.05, 0.03, 0.07])
        expected = np.mean(ic) / np.std(ic, ddof=1)
        assert abs(Backtester.compute_icir(ic) - expected) < 1e-10

    def test_constant_ic_returns_zero(self):
        """If all IC values are identical, std=0 -> ICIR=0."""
        ic = np.full(10, 0.05)
        assert Backtester.compute_icir(ic) == 0.0

    def test_single_observation_returns_zero(self):
        assert Backtester.compute_icir(np.array([0.05])) == 0.0

    def test_empty_after_nan_removal(self):
        ic = np.array([np.nan, np.nan])
        assert Backtester.compute_icir(ic) == 0.0

    def test_nan_values_ignored(self):
        ic = np.array([0.04, np.nan, 0.06, 0.05, np.nan, 0.03])
        clean = np.array([0.04, 0.06, 0.05, 0.03])
        expected = np.mean(clean) / np.std(clean, ddof=1)
        assert abs(Backtester.compute_icir(ic) - expected) < 1e-10


# ---------------------------------------------------------------------------
# Test: Sharpe ratio annualization
# ---------------------------------------------------------------------------

class TestComputeSharpe:
    """Verify annualized Sharpe = daily_sharpe * sqrt(252)."""

    def test_known_sharpe(self):
        rng = np.random.default_rng(42)
        daily_returns = rng.normal(0.001, 0.01, size=252)

        sharpe = Backtester.compute_sharpe(daily_returns)

        # Manual calculation
        mean_excess = np.mean(daily_returns)
        std_excess = np.std(daily_returns, ddof=1)
        expected = (mean_excess / std_excess) * np.sqrt(TRADING_DAYS_PER_YEAR)
        assert abs(sharpe - expected) < 1e-10

    def test_zero_returns(self):
        """All-zero returns -> Sharpe = 0."""
        assert Backtester.compute_sharpe(np.zeros(100)) == 0.0

    def test_constant_positive_returns(self):
        """Constant positive returns -> std=0 -> Sharpe=0."""
        assert Backtester.compute_sharpe(np.full(100, 0.001)) == 0.0

    def test_with_risk_free_rate(self):
        rng = np.random.default_rng(99)
        daily = rng.normal(0.002, 0.01, size=252)
        rf = 0.0001

        sharpe = Backtester.compute_sharpe(daily, rf=rf)
        excess = daily - rf
        expected = (np.mean(excess) / np.std(excess, ddof=1)) * np.sqrt(252)
        assert abs(sharpe - expected) < 1e-10

    def test_insufficient_data(self):
        assert Backtester.compute_sharpe(np.array([0.01])) == 0.0


# ---------------------------------------------------------------------------
# Test: MDD computation
# ---------------------------------------------------------------------------

class TestComputeMDD:
    """Verify maximum drawdown on known sequences."""

    def test_known_drawdown(self):
        """Price goes 1.0 -> 1.2 -> 0.9 -> 1.1.  MDD = (1.2-0.9)/1.2 = 25%."""
        cum = np.array([1.0, 1.2, 0.9, 1.1])
        mdd = Backtester.compute_mdd(cum)
        expected = (1.2 - 0.9) / 1.2  # = 0.25
        assert abs(mdd - expected) < 1e-10

    def test_monotonically_increasing(self):
        """No drawdown at all -> MDD = 0."""
        cum = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        assert Backtester.compute_mdd(cum) == 0.0

    def test_monotonically_decreasing(self):
        """Continuous decline: MDD = total decline / peak."""
        cum = np.array([1.0, 0.8, 0.6, 0.4])
        mdd = Backtester.compute_mdd(cum)
        expected = (1.0 - 0.4) / 1.0  # = 0.6
        assert abs(mdd - expected) < 1e-10

    def test_multiple_drawdowns(self):
        """Multiple drawdowns, return the max."""
        cum = np.array([1.0, 1.5, 1.2, 1.8, 1.0])
        # Drawdown 1: 1.5 -> 1.2 = 20%
        # Drawdown 2: 1.8 -> 1.0 = 44.4%
        mdd = Backtester.compute_mdd(cum)
        expected = (1.8 - 1.0) / 1.8
        assert abs(mdd - expected) < 1e-10

    def test_single_point(self):
        assert Backtester.compute_mdd(np.array([1.0])) == 0.0

    def test_empty(self):
        assert Backtester.compute_mdd(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# Test: Acceptance thresholds
# ---------------------------------------------------------------------------

class TestAcceptanceThresholds:
    """Verify that the acceptance logic correctly gates on all four metrics."""

    def test_passing_result(self, backtester):
        result = BacktestResult(
            ic=0.05, icir=1.2, sharpe=1.5, mdd=0.15
        )
        assert backtester.check_acceptance(result) is True

    def test_ic_too_low(self, backtester):
        result = BacktestResult(
            ic=0.02, icir=1.2, sharpe=1.5, mdd=0.15
        )
        assert backtester.check_acceptance(result) is False

    def test_icir_too_low(self, backtester):
        result = BacktestResult(
            ic=0.05, icir=0.5, sharpe=1.5, mdd=0.15
        )
        assert backtester.check_acceptance(result) is False

    def test_sharpe_too_low(self, backtester):
        result = BacktestResult(
            ic=0.05, icir=1.2, sharpe=0.8, mdd=0.15
        )
        assert backtester.check_acceptance(result) is False

    def test_mdd_too_high(self, backtester):
        result = BacktestResult(
            ic=0.05, icir=1.2, sharpe=1.5, mdd=0.30
        )
        assert backtester.check_acceptance(result) is False

    def test_boundary_values_pass(self, backtester):
        """Exact threshold values should pass (>=, <=)."""
        result = BacktestResult(
            ic=DEFAULT_IC_THRESHOLD,
            icir=DEFAULT_ICIR_THRESHOLD,
            sharpe=DEFAULT_SHARPE_THRESHOLD,
            mdd=DEFAULT_MDD_THRESHOLD,
        )
        assert backtester.check_acceptance(result) is True

    def test_custom_thresholds(self):
        bt = Backtester(
            ic_threshold=0.05,
            icir_threshold=1.0,
            sharpe_threshold=1.5,
            mdd_threshold=0.10,
        )
        # Passes custom thresholds
        result = BacktestResult(ic=0.06, icir=1.1, sharpe=1.6, mdd=0.08)
        assert bt.check_acceptance(result) is True

        # Fails custom MDD threshold
        result_bad = BacktestResult(ic=0.06, icir=1.1, sharpe=1.6, mdd=0.12)
        assert bt.check_acceptance(result_bad) is False


# ---------------------------------------------------------------------------
# Test: BacktestMetrics meets_thresholds
# ---------------------------------------------------------------------------

class TestBacktestMetrics:
    """Test the lightweight BacktestMetrics container."""

    def test_meets_thresholds_all_passing(self):
        m = BacktestMetrics(ic=0.05, icir=1.0, sharpe=1.5, mdd=0.15)
        assert m.meets_thresholds() is True

    def test_meets_thresholds_one_failing(self):
        m = BacktestMetrics(ic=0.01, icir=1.0, sharpe=1.5, mdd=0.15)
        assert m.meets_thresholds() is False

    def test_meets_thresholds_custom(self):
        m = BacktestMetrics(ic=0.10, icir=2.0, sharpe=3.0, mdd=0.05)
        assert m.meets_thresholds(
            ic_threshold=0.08, icir_threshold=1.5,
            sharpe_threshold=2.5, mdd_threshold=0.10,
        ) is True


# ---------------------------------------------------------------------------
# Test: ARR and Calmar
# ---------------------------------------------------------------------------

class TestARRAndCalmar:
    """Test annualised return and Calmar ratio."""

    def test_compute_arr(self):
        """Known daily return -> known annualised return."""
        # 252 days of 0.1% daily return
        daily = np.full(252, 0.001)
        arr = Backtester.compute_arr(daily)
        # Geometric: (1.001)^252 - 1
        expected = (1.001 ** 252) - 1.0
        assert abs(arr - expected) < 1e-6

    def test_compute_arr_empty(self):
        assert Backtester.compute_arr(np.array([])) == 0.0

    def test_calmar_normal(self):
        assert abs(Backtester.compute_calmar(0.20, 0.10) - 2.0) < 1e-10

    def test_calmar_zero_mdd(self):
        assert Backtester.compute_calmar(0.20, 0.0) == 0.0
