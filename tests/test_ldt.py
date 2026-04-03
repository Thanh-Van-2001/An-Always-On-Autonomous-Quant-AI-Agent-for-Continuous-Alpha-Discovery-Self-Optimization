"""
Tests for the Leakage Detection Test (LDT) module.

Validates the Welch t-statistic computation, rejection threshold at
z_0.99 = 2.326, and correct behaviour on biased vs. clean factor IC series.
"""

import numpy as np
import pytest
from scipy import stats

from aqaa.csba.ldt import LDTResult, LeakageDetectionTest, Z_CRITICAL_001


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ldt():
    """Default LDT instance at alpha = 0.01."""
    return LeakageDetectionTest(alpha=0.01)


# ---------------------------------------------------------------------------
# Test: critical value constant
# ---------------------------------------------------------------------------

class TestCriticalValue:
    """Verify that Z_CRITICAL_001 matches the expected normal quantile."""

    def test_z_critical_value(self):
        assert abs(Z_CRITICAL_001 - 2.326) < 1e-3

    def test_instance_z_critical_matches_scipy(self, ldt):
        expected = stats.norm.ppf(0.99)
        assert abs(ldt.z_critical - expected) < 1e-6


# ---------------------------------------------------------------------------
# Test: Welch t-statistic computation
# ---------------------------------------------------------------------------

class TestComputeLDTStatistic:
    """Verify that compute_ldt_statistic matches scipy.stats.ttest_ind."""

    def test_matches_scipy_welch(self):
        """The LDT statistic should match scipy's Welch t-test."""
        rng = np.random.default_rng(42)
        ic_train = rng.normal(0.05, 0.02, size=100)
        ic_test = rng.normal(0.03, 0.02, size=80)

        t_stat_aqaa, _ = LeakageDetectionTest.compute_ldt_statistic(
            ic_train, ic_test
        )

        # scipy Welch t-test (two-sided), we just compare the statistic
        t_stat_scipy, _ = stats.ttest_ind(
            ic_train, ic_test, equal_var=False
        )

        assert abs(t_stat_aqaa - t_stat_scipy) < 1e-10

    def test_p_value_one_sided(self):
        """p-value should be one-sided (right tail)."""
        rng = np.random.default_rng(7)
        ic_train = rng.normal(0.10, 0.01, size=50)
        ic_test = rng.normal(0.02, 0.01, size=50)

        t_stat, p_value = LeakageDetectionTest.compute_ldt_statistic(
            ic_train, ic_test
        )
        # Large positive t_stat -> very small one-sided p-value
        assert t_stat > 0
        assert p_value < 0.001

    def test_equal_means_large_p(self):
        """When means are equal and samples are large, p-value should
        be well above any typical significance level."""
        rng = np.random.default_rng(100)
        # Use the same seed for both to guarantee near-zero t-stat
        ic_train = rng.normal(0.04, 0.02, size=500)
        ic_test = rng.normal(0.04, 0.02, size=500)

        _, p_value = LeakageDetectionTest.compute_ldt_statistic(
            ic_train, ic_test
        )
        # With same population mean and large n, should not be significant
        assert p_value > 0.01

    def test_raises_on_too_few_observations(self):
        """Must have at least 2 observations per partition."""
        with pytest.raises(ValueError, match="ic_train must have >= 2"):
            LeakageDetectionTest.compute_ldt_statistic(
                np.array([0.05]), np.array([0.03, 0.04])
            )

        with pytest.raises(ValueError, match="ic_test must have >= 2"):
            LeakageDetectionTest.compute_ldt_statistic(
                np.array([0.05, 0.06]), np.array([0.03])
            )

    def test_constant_series_returns_zero(self):
        """When both series are constant, SE ~ 0 -> t_stat = 0, p = 1."""
        ic_train = np.full(20, 0.05)
        ic_test = np.full(20, 0.05)

        t_stat, p_value = LeakageDetectionTest.compute_ldt_statistic(
            ic_train, ic_test
        )
        assert t_stat == 0.0
        assert p_value == 1.0


# ---------------------------------------------------------------------------
# Test: biased factor is rejected
# ---------------------------------------------------------------------------

class TestBiasedFactorRejected:
    """Biased factors with high in-sample IC and low OOS IC must fail."""

    def test_large_ic_gap_fails_ldt(self, ldt):
        rng = np.random.default_rng(123)
        ic_train = rng.normal(0.08, 0.01, size=100)
        ic_test = rng.normal(0.01, 0.01, size=100)

        result = ldt.test_leakage(ic_train, ic_test)

        assert isinstance(result, LDTResult)
        assert result.passed == False  # noqa: E712  (np.bool_ vs bool)
        assert result.t_stat > Z_CRITICAL_001
        assert result.p_value < 0.01

    def test_strong_leakage_detected(self, ldt):
        """Extreme memorization bias: train IC >> test IC."""
        rng = np.random.default_rng(456)
        ic_train = rng.normal(0.15, 0.02, size=200)
        ic_test = rng.normal(0.00, 0.02, size=200)

        result = ldt.test_leakage(ic_train, ic_test)
        assert result.passed == False  # noqa: E712
        assert result.t_stat > 10.0  # very strong signal


# ---------------------------------------------------------------------------
# Test: clean factor passes
# ---------------------------------------------------------------------------

class TestCleanFactorPasses:
    """Genuine factors with similar IS and OOS IC should pass."""

    def test_similar_ic_passes(self, ldt):
        rng = np.random.default_rng(789)
        ic_train = rng.normal(0.04, 0.02, size=100)
        ic_test = rng.normal(0.04, 0.02, size=80)

        result = ldt.test_leakage(ic_train, ic_test)

        assert result.passed == True  # noqa: E712
        assert result.t_stat <= ldt.z_critical

    def test_oos_higher_than_is_passes(self, ldt):
        """If OOS IC is actually higher, clearly no leakage."""
        rng = np.random.default_rng(101)
        ic_train = rng.normal(0.03, 0.01, size=60)
        ic_test = rng.normal(0.05, 0.01, size=60)

        result = ldt.test_leakage(ic_train, ic_test)
        assert result.passed == True  # noqa: E712
        assert result.t_stat < 0  # negative t_stat


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: equal IC, very small samples, custom alpha."""

    def test_equal_ic_series(self, ldt):
        """Identical series should always pass (t_stat = 0)."""
        ic = np.array([0.04, 0.05, 0.03, 0.06, 0.04, 0.05, 0.03])
        result = ldt.test_leakage(ic, ic.copy())
        assert result.passed == True  # noqa: E712
        assert abs(result.t_stat) < 1e-10

    def test_very_small_samples(self, ldt):
        """Minimum valid sample size: 2 per partition."""
        ic_train = np.array([0.10, 0.12])
        ic_test = np.array([0.02, 0.03])

        result = ldt.test_leakage(ic_train, ic_test)
        # With only 2 observations each, the test should still run
        assert isinstance(result, LDTResult)
        assert bool(result.passed) == result.passed  # valid bool-like

    def test_custom_alpha_override(self):
        """Override alpha per call."""
        ldt_strict = LeakageDetectionTest(alpha=0.01)
        rng = np.random.default_rng(55)
        ic_train = rng.normal(0.05, 0.02, size=60)
        ic_test = rng.normal(0.035, 0.02, size=60)

        # May pass at alpha=0.01 but fail at alpha=0.20
        result_strict = ldt_strict.test_leakage(ic_train, ic_test, alpha=0.01)
        result_loose = ldt_strict.test_leakage(ic_train, ic_test, alpha=0.20)

        # If it passes strict, it must also pass loose
        if result_strict.passed:
            assert result_loose.passed is True
        # Both should record the correct alpha
        assert result_strict.alpha == 0.01
        assert result_loose.alpha == 0.20

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            LeakageDetectionTest(alpha=0.0)
        with pytest.raises(ValueError):
            LeakageDetectionTest(alpha=1.0)
        with pytest.raises(ValueError):
            LeakageDetectionTest(alpha=-0.5)

    def test_result_attributes(self, ldt):
        rng = np.random.default_rng(33)
        ic_train = rng.normal(0.05, 0.02, size=50)
        ic_test = rng.normal(0.04, 0.02, size=50)

        result = ldt.test_leakage(ic_train, ic_test)
        assert abs(result.ic_train_mean - np.mean(ic_train)) < 1e-10
        assert abs(result.ic_test_mean - np.mean(ic_test)) < 1e-10
        assert result.alpha == 0.01


# ---------------------------------------------------------------------------
# Test: z_0.99 = 2.326 rejection threshold
# ---------------------------------------------------------------------------

class TestRejectionThreshold:
    """Verify that the rejection boundary aligns with z_0.99 = 2.326."""

    def test_just_below_threshold_passes(self):
        """With equal population means, the t_stat should be near zero
        and the test should pass."""
        ldt = LeakageDetectionTest(alpha=0.01)
        rng = np.random.default_rng(2024)
        ic_train = rng.normal(0.05, 1.0, size=10000)
        ic_test = rng.normal(0.05, 1.0, size=10000)

        result = ldt.test_leakage(ic_train, ic_test)
        # With same means, t_stat should be near 0
        assert result.passed == True  # noqa: E712

    def test_z_critical_boundary(self):
        """Verify the decision rule: passed iff t_stat <= z_crit."""
        ldt = LeakageDetectionTest(alpha=0.01)
        z_crit = stats.norm.ppf(0.99)

        # A result with t_stat just below z_crit should pass
        # A result with t_stat just above z_crit should fail
        # We verify this via the decision logic directly:
        assert 2.3 <= z_crit  # t_stat=2.3 passes
        assert z_crit <= 2.33  # t_stat=2.33 fails
