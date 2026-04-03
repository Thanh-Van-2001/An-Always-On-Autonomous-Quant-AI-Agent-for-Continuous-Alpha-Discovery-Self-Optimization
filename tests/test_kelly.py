"""
Tests for the Kelly Criterion position sizing module.

Validates the full Kelly formula, fractional Kelly, gross exposure cap at 2x,
and regime-adjusted sizing.
"""

import pytest

from aqaa.apo.kelly_sizing import KellySizer, _DEFAULT_REGIME_KELLY_SCALE
from aqaa.core.types import MarketRegime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sizer():
    return KellySizer(fraction=0.5, max_gross_exposure=2.0)


# ---------------------------------------------------------------------------
# Test: Full Kelly formula
# ---------------------------------------------------------------------------

class TestFullKelly:
    """Verify f* = p - q / b."""

    def test_classic_kelly_example(self):
        """p=0.55, b=1.0 -> f* = 0.55 - 0.45/1.0 = 0.10."""
        f = KellySizer.compute_kelly_fraction(win_rate=0.55, win_loss_ratio=1.0)
        assert abs(f - 0.10) < 1e-10

    def test_high_edge(self):
        """p=0.60, b=1.5 -> f* = 0.60 - 0.40/1.5 = 0.3333."""
        f = KellySizer.compute_kelly_fraction(win_rate=0.60, win_loss_ratio=1.5)
        expected = 0.60 - 0.40 / 1.5
        assert abs(f - expected) < 1e-10

    def test_no_edge(self):
        """p=0.50, b=1.0 -> f* = 0.50 - 0.50/1.0 = 0.0."""
        f = KellySizer.compute_kelly_fraction(win_rate=0.50, win_loss_ratio=1.0)
        assert abs(f - 0.0) < 1e-10

    def test_negative_edge(self):
        """p=0.40, b=1.0 -> f* = 0.40 - 0.60/1.0 = -0.20."""
        f = KellySizer.compute_kelly_fraction(win_rate=0.40, win_loss_ratio=1.0)
        assert abs(f - (-0.20)) < 1e-10

    def test_high_win_loss_ratio(self):
        """p=0.45, b=3.0 -> f* = 0.45 - 0.55/3.0 = 0.2667."""
        f = KellySizer.compute_kelly_fraction(win_rate=0.45, win_loss_ratio=3.0)
        expected = 0.45 - 0.55 / 3.0
        assert abs(f - expected) < 1e-10

    def test_invalid_win_rate_raises(self):
        with pytest.raises(ValueError, match="win_rate must be in"):
            KellySizer.compute_kelly_fraction(win_rate=0.0, win_loss_ratio=1.0)
        with pytest.raises(ValueError, match="win_rate must be in"):
            KellySizer.compute_kelly_fraction(win_rate=1.0, win_loss_ratio=1.0)
        with pytest.raises(ValueError, match="win_rate must be in"):
            KellySizer.compute_kelly_fraction(win_rate=-0.5, win_loss_ratio=1.0)

    def test_invalid_win_loss_ratio_raises(self):
        with pytest.raises(ValueError, match="win_loss_ratio must be > 0"):
            KellySizer.compute_kelly_fraction(win_rate=0.55, win_loss_ratio=0.0)
        with pytest.raises(ValueError, match="win_loss_ratio must be > 0"):
            KellySizer.compute_kelly_fraction(win_rate=0.55, win_loss_ratio=-1.0)


# ---------------------------------------------------------------------------
# Test: Fractional Kelly
# ---------------------------------------------------------------------------

class TestFractionalKelly:
    """Verify fractional Kelly multiplier."""

    def test_half_kelly(self, sizer):
        """Half-Kelly: 0.10 * 0.5 = 0.05."""
        full = 0.10
        frac = sizer.fractional_kelly(full)
        assert abs(frac - 0.05) < 1e-10

    def test_quarter_kelly(self):
        sizer = KellySizer(fraction=0.25)
        full = 0.20
        frac = sizer.fractional_kelly(full)
        assert abs(frac - 0.05) < 1e-10

    def test_full_kelly_fraction_one(self):
        sizer = KellySizer(fraction=1.0)
        frac = sizer.fractional_kelly(0.15)
        assert abs(frac - 0.15) < 1e-10

    def test_negative_kelly_floored_to_zero(self, sizer):
        """Negative full Kelly -> fractional Kelly = 0.0."""
        frac = sizer.fractional_kelly(-0.20)
        assert frac == 0.0

    def test_fraction_override(self, sizer):
        """Override fraction per call."""
        full = 0.10
        frac = sizer.fractional_kelly(full, fraction=0.75)
        assert abs(frac - 0.075) < 1e-10

    def test_zero_full_kelly(self, sizer):
        frac = sizer.fractional_kelly(0.0)
        assert frac == 0.0


# ---------------------------------------------------------------------------
# Test: Gross exposure cap at 2x
# ---------------------------------------------------------------------------

class TestGrossExposureCap:
    """Verify the hard cap on gross exposure."""

    def test_within_cap_no_rescale(self, sizer):
        weights = {"A": 0.5, "B": -0.3, "C": 0.2}
        kelly_frac = 1.0  # no additional scaling
        sized = sizer.apply_kelly_sizing(weights, kelly_frac)

        gross = sum(abs(v) for v in sized.values())
        # 0.5 + 0.3 + 0.2 = 1.0, well within 2.0 cap
        assert abs(gross - 1.0) < 1e-10

    def test_exceeds_cap_rescaled(self, sizer):
        """Weights that exceed 2x gross exposure must be scaled down."""
        weights = {"A": 1.5, "B": -1.0, "C": 0.8}
        kelly_frac = 2.0
        # After Kelly: A=3.0, B=-2.0, C=1.6 -> gross = 6.6
        sized = sizer.apply_kelly_sizing(weights, kelly_frac)

        gross = sum(abs(v) for v in sized.values())
        assert gross <= 2.0 + 1e-10

    def test_cap_at_exactly_2x(self, sizer):
        """Gross exposure = 2.0 should not be rescaled."""
        weights = {"A": 1.0, "B": -0.5, "C": 0.5}
        kelly_frac = 1.0
        # gross = 1.0 + 0.5 + 0.5 = 2.0, exactly at cap
        sized = sizer.apply_kelly_sizing(weights, kelly_frac)
        gross = sum(abs(v) for v in sized.values())
        assert abs(gross - 2.0) < 1e-10

    def test_custom_cap_override(self):
        sizer = KellySizer(fraction=0.5, max_gross_exposure=2.0)
        weights = {"A": 1.0, "B": -0.5}
        kelly_frac = 3.0
        # After Kelly: A=3.0, B=-1.5 -> gross = 4.5
        # Custom cap of 3.0
        sized = sizer.apply_kelly_sizing(
            weights, kelly_frac, max_gross_exposure=3.0
        )
        gross = sum(abs(v) for v in sized.values())
        assert gross <= 3.0 + 1e-10

    def test_proportions_preserved_after_cap(self, sizer):
        """After rescaling, relative proportions should be maintained."""
        weights = {"A": 2.0, "B": -1.0}
        kelly_frac = 2.0
        # After Kelly: A=4.0, B=-2.0 -> gross = 6.0 -> rescale to 2.0
        sized = sizer.apply_kelly_sizing(weights, kelly_frac)

        # A should be twice B in absolute terms
        ratio = abs(sized["A"]) / abs(sized["B"])
        assert abs(ratio - 2.0) < 1e-10

    def test_empty_weights(self, sizer):
        sized = sizer.apply_kelly_sizing({}, 1.0)
        assert sized == {}

    def test_zero_kelly_fraction(self, sizer):
        """Kelly fraction = 0 -> all weights become 0."""
        weights = {"A": 1.0, "B": -0.5}
        sized = sizer.apply_kelly_sizing(weights, 0.0)
        for v in sized.values():
            assert v == 0.0


# ---------------------------------------------------------------------------
# Test: Regime-adjusted Kelly
# ---------------------------------------------------------------------------

class TestRegimeAdjustedKelly:
    """Verify regime scaling factors."""

    def test_bull_regime_no_reduction(self, sizer):
        kelly = 0.10
        adjusted = sizer.regime_adjusted_kelly(kelly, MarketRegime.BULL)
        assert abs(adjusted - 0.10) < 1e-10

    def test_sideways_regime_reduction(self, sizer):
        kelly = 0.10
        adjusted = sizer.regime_adjusted_kelly(kelly, MarketRegime.SIDEWAYS)
        expected = 0.10 * 0.7
        assert abs(adjusted - expected) < 1e-10

    def test_bear_regime_reduction(self, sizer):
        kelly = 0.10
        adjusted = sizer.regime_adjusted_kelly(kelly, MarketRegime.BEAR)
        expected = 0.10 * 0.4
        assert abs(adjusted - expected) < 1e-10

    def test_negative_kelly_floored(self, sizer):
        """Even after regime scaling, result is floored at 0."""
        adjusted = sizer.regime_adjusted_kelly(-0.10, MarketRegime.BULL)
        assert adjusted == 0.0

    def test_custom_regime_scale(self):
        custom = {
            MarketRegime.BULL: 0.8,
            MarketRegime.SIDEWAYS: 0.5,
            MarketRegime.BEAR: 0.2,
        }
        sizer = KellySizer(regime_kelly_scale=custom)
        adjusted = sizer.regime_adjusted_kelly(0.10, MarketRegime.BULL)
        assert abs(adjusted - 0.08) < 1e-10


# ---------------------------------------------------------------------------
# Test: Default regime scales
# ---------------------------------------------------------------------------

class TestDefaultRegimeScales:

    def test_default_scale_values(self):
        assert _DEFAULT_REGIME_KELLY_SCALE[MarketRegime.BULL] == 1.0
        assert _DEFAULT_REGIME_KELLY_SCALE[MarketRegime.SIDEWAYS] == 0.7
        assert _DEFAULT_REGIME_KELLY_SCALE[MarketRegime.BEAR] == 0.4

    def test_sizer_uses_defaults(self):
        sizer = KellySizer()
        assert sizer.regime_kelly_scale[MarketRegime.BULL] == 1.0


# ---------------------------------------------------------------------------
# Test: End-to-end Kelly sizing workflow
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Simulate the full Kelly sizing pipeline."""

    def test_full_pipeline(self):
        sizer = KellySizer(fraction=0.5, max_gross_exposure=2.0)

        # Step 1: Compute full Kelly
        full_kelly = sizer.compute_kelly_fraction(
            win_rate=0.55, win_loss_ratio=1.2
        )
        assert full_kelly > 0

        # Step 2: Apply fractional Kelly
        frac_kelly = sizer.fractional_kelly(full_kelly)
        assert frac_kelly < full_kelly
        assert frac_kelly > 0

        # Step 3: Regime adjustment (bear market)
        adj_kelly = sizer.regime_adjusted_kelly(frac_kelly, MarketRegime.BEAR)
        assert adj_kelly < frac_kelly
        assert adj_kelly > 0

        # Step 4: Apply to portfolio weights
        weights = {"AAPL": 0.8, "MSFT": 0.6, "TSLA": -0.4}
        sized = sizer.apply_kelly_sizing(weights, adj_kelly)

        # Gross exposure should be within cap
        gross = sum(abs(v) for v in sized.values())
        assert gross <= 2.0 + 1e-10
