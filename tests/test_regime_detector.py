"""
Tests for the HMM regime detector.

Validates regime detection on synthetic data with clear regimes and
the online update mechanism.
"""

import numpy as np
import pytest

from aqaa.apo.regime_detector import RegimeDetector
from aqaa.core.types import MarketRegime


# ---------------------------------------------------------------------------
# Helper: generate synthetic returns with clear regimes
# ---------------------------------------------------------------------------

def _make_regime_returns(seed: int = 42) -> np.ndarray:
    """Generate synthetic daily returns with 3 distinct regimes.

    - Bull:     mean = +0.002, std = 0.005  (positive drift, low vol)
    - Bear:     mean = -0.003, std = 0.015  (negative drift, high vol)
    - Sideways: mean =  0.000, std = 0.003  (no drift, very low vol)
    """
    rng = np.random.default_rng(seed)
    bull = rng.normal(0.002, 0.005, size=100)
    bear = rng.normal(-0.003, 0.015, size=100)
    sideways = rng.normal(0.000, 0.003, size=100)
    return np.concatenate([bull, bear, sideways])


# ---------------------------------------------------------------------------
# Test: Fitting on synthetic data
# ---------------------------------------------------------------------------

class TestFit:
    """Verify that the HMM fits and learns meaningful parameters."""

    def test_fit_returns_self(self):
        detector = RegimeDetector(n_iter=50, random_seed=42)
        returns = _make_regime_returns()
        result = detector.fit(returns)
        assert result is detector
        assert detector.is_fitted is True

    def test_learned_means_distinguish_regimes(self):
        """The learned emission means should separate bull, bear, sideways."""
        detector = RegimeDetector(n_iter=100, random_seed=42)
        returns = _make_regime_returns()
        detector.fit(returns)

        means = detector.means_
        assert means is not None
        # There should be a positive mean (bull), a negative mean (bear),
        # and one near zero (sideways).
        assert np.max(means) > 0
        assert np.min(means) < 0

    def test_transition_matrix_is_stochastic(self):
        """Rows of the transition matrix should sum to 1."""
        detector = RegimeDetector(n_iter=50, random_seed=42)
        returns = _make_regime_returns()
        detector.fit(returns)

        tm = detector.transition_matrix
        for row in tm:
            assert abs(row.sum() - 1.0) < 1e-6
            assert np.all(row >= 0)

    def test_stds_are_positive(self):
        detector = RegimeDetector(n_iter=50, random_seed=42)
        returns = _make_regime_returns()
        detector.fit(returns)
        assert np.all(detector.stds_ > 0)

    def test_regime_history_length(self):
        detector = RegimeDetector(n_iter=50, random_seed=42)
        returns = _make_regime_returns()
        detector.fit(returns)
        assert len(detector.regime_history) == len(returns)

    def test_regime_history_contains_valid_regimes(self):
        detector = RegimeDetector(n_iter=50, random_seed=42)
        returns = _make_regime_returns()
        detector.fit(returns)
        valid = {MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS}
        for regime in detector.regime_history:
            assert regime in valid

    def test_too_few_observations_raises(self):
        detector = RegimeDetector(n_iter=10, random_seed=42)
        with pytest.raises(ValueError, match="at least"):
            detector.fit(np.array([0.01, 0.02]))

    def test_not_fitted_raises(self):
        detector = RegimeDetector()
        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.predict_regime(np.array([0.01, 0.02]))


# ---------------------------------------------------------------------------
# Test: Predict regime on a window of returns
# ---------------------------------------------------------------------------

class TestPredictRegime:
    """Verify predict_regime on strongly typed windows."""

    def test_predict_bull_window(self):
        """A window of strongly positive returns should be classified
        as BULL (or at least not BEAR)."""
        detector = RegimeDetector(n_iter=100, random_seed=42)
        returns = _make_regime_returns()
        detector.fit(returns)

        # Feed a window of clearly positive returns
        rng = np.random.default_rng(100)
        bull_window = rng.normal(0.003, 0.003, size=30)
        regime = detector.predict_regime(bull_window)

        assert isinstance(regime, MarketRegime)
        # The regime should NOT be BEAR for very positive returns
        assert regime != MarketRegime.BEAR

    def test_predict_bear_window(self):
        """A window of strongly negative returns should be BEAR."""
        detector = RegimeDetector(n_iter=100, random_seed=42)
        returns = _make_regime_returns()
        detector.fit(returns)

        rng = np.random.default_rng(200)
        bear_window = rng.normal(-0.005, 0.010, size=30)
        regime = detector.predict_regime(bear_window)

        assert isinstance(regime, MarketRegime)
        # Should identify the strong negative drift
        assert regime != MarketRegime.BULL


# ---------------------------------------------------------------------------
# Test: Online update
# ---------------------------------------------------------------------------

class TestOnlineUpdate:
    """Verify single-step filtering update."""

    def test_update_online_returns_regime(self):
        detector = RegimeDetector(n_iter=50, random_seed=42)
        returns = _make_regime_returns()
        detector.fit(returns)

        regime = detector.update_online(0.003)
        assert isinstance(regime, MarketRegime)

    def test_update_online_appends_to_history(self):
        detector = RegimeDetector(n_iter=50, random_seed=42)
        returns = _make_regime_returns()
        detector.fit(returns)

        n_before = len(detector.regime_history)
        detector.update_online(0.001)
        detector.update_online(-0.002)
        assert len(detector.regime_history) == n_before + 2

    def test_state_probs_sum_to_one(self):
        detector = RegimeDetector(n_iter=50, random_seed=42)
        returns = _make_regime_returns()
        detector.fit(returns)

        detector.update_online(0.001)
        probs = detector.state_probabilities
        assert abs(probs.sum() - 1.0) < 1e-6
        assert np.all(probs >= 0)

    def test_multiple_online_updates_stable(self):
        """Running many online updates should not cause numerical issues."""
        detector = RegimeDetector(n_iter=50, random_seed=42)
        returns = _make_regime_returns()
        detector.fit(returns)

        rng = np.random.default_rng(77)
        for _ in range(200):
            ret = rng.normal(0.001, 0.01)
            regime = detector.update_online(ret)
            assert isinstance(regime, MarketRegime)

        probs = detector.state_probabilities
        assert abs(probs.sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test: State probabilities accessor
# ---------------------------------------------------------------------------

class TestAccessors:

    def test_state_probabilities_shape(self):
        detector = RegimeDetector(n_iter=50, random_seed=42)
        returns = _make_regime_returns()
        detector.fit(returns)

        probs = detector.state_probabilities
        assert probs.shape == (3,)

    def test_transition_matrix_shape(self):
        detector = RegimeDetector(n_iter=50, random_seed=42)
        returns = _make_regime_returns()
        detector.fit(returns)

        tm = detector.transition_matrix
        assert tm.shape == (3, 3)

    def test_state_probabilities_is_copy(self):
        """Modifying the returned array should not affect internal state."""
        detector = RegimeDetector(n_iter=50, random_seed=42)
        returns = _make_regime_returns()
        detector.fit(returns)

        probs = detector.state_probabilities
        probs[:] = 0.0  # mutate
        # Internal state should be unchanged
        internal = detector.state_probabilities
        assert internal.sum() > 0
