"""
Kelly Criterion position sizing.

Provides full and fractional Kelly fraction computation with
regime-adaptive scaling and a hard cap on gross exposure (default 2x per
the KDD 2026 paper specification).

Typical usage::

    sizer = KellySizer(fraction=0.5, max_gross_exposure=2.0)
    kelly_f = sizer.compute_kelly_fraction(win_rate=0.55, win_loss_ratio=1.2)
    frac = sizer.fractional_kelly(kelly_f)
    sized = sizer.apply_kelly_sizing(weights, frac)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

from aqaa.core.types import MarketRegime

logger = logging.getLogger(__name__)

# Regime-dependent scaling factors for Kelly fraction.
# Bear and sideways markets reduce the Kelly allocation to limit drawdowns.
_DEFAULT_REGIME_KELLY_SCALE: Dict[MarketRegime, float] = {
    MarketRegime.BULL: 1.0,
    MarketRegime.SIDEWAYS: 0.7,
    MarketRegime.BEAR: 0.4,
}


@dataclass
class KellySizer:
    """Fractional Kelly position sizer with regime adjustment.

    Parameters
    ----------
    fraction : float
        Default fractional Kelly multiplier (0.5 = half-Kelly).
    max_gross_exposure : float
        Hard cap on the sum of absolute weights after Kelly sizing.
        The paper mandates 2.0.
    regime_kelly_scale : dict | None
        Override mapping from :class:`MarketRegime` to a multiplicative
        scale applied to the Kelly fraction.  Falls back to built-in
        defaults.
    """

    fraction: float = 0.5
    max_gross_exposure: float = 2.0
    regime_kelly_scale: Dict[MarketRegime, float] | None = None

    def __post_init__(self) -> None:
        if self.regime_kelly_scale is None:
            self.regime_kelly_scale = dict(_DEFAULT_REGIME_KELLY_SCALE)

    # ------------------------------------------------------------------
    # Core Kelly computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
        """Compute the full Kelly fraction.

        The classic Kelly formula for a simple bet is::

            f* = p - q / b

        where *p* = probability of winning, *q* = 1 - p, and
        *b* = ratio of average win to average loss.

        Parameters
        ----------
        win_rate : float
            Estimated probability of a winning trade, in (0, 1).
        win_loss_ratio : float
            Average gain on wins divided by average loss on losses (> 0).

        Returns
        -------
        float
            Full Kelly fraction.  May be negative if the edge is
            insufficient, in which case the caller should avoid the bet.

        Raises
        ------
        ValueError
            If *win_rate* is not in (0, 1) or *win_loss_ratio* <= 0.
        """
        if not (0.0 < win_rate < 1.0):
            raise ValueError(f"win_rate must be in (0, 1), got {win_rate}")
        if win_loss_ratio <= 0.0:
            raise ValueError(
                f"win_loss_ratio must be > 0, got {win_loss_ratio}"
            )

        q = 1.0 - win_rate
        full_kelly = win_rate - q / win_loss_ratio
        return float(full_kelly)

    def fractional_kelly(
        self,
        full_kelly: float,
        fraction: float | None = None,
    ) -> float:
        """Apply a fractional multiplier to the full Kelly fraction.

        Parameters
        ----------
        full_kelly : float
            The full Kelly fraction (output of :meth:`compute_kelly_fraction`).
        fraction : float | None
            Override for the instance-level ``self.fraction``.

        Returns
        -------
        float
            Fractional Kelly value, floored at 0.0 (no negative sizing).
        """
        frac = fraction if fraction is not None else self.fraction
        return float(max(full_kelly * frac, 0.0))

    # ------------------------------------------------------------------
    # Sizing application
    # ------------------------------------------------------------------

    def apply_kelly_sizing(
        self,
        weights: Dict[str, float],
        kelly_fraction: float,
        max_gross_exposure: float | None = None,
    ) -> Dict[str, float]:
        """Scale portfolio weights by the Kelly fraction and enforce the
        gross-exposure cap.

        Parameters
        ----------
        weights : dict[str, float]
            Factor-name -> raw optimised weight.
        kelly_fraction : float
            The (fractional) Kelly sizing scalar to apply.
        max_gross_exposure : float | None
            Override for the instance-level ``self.max_gross_exposure``.

        Returns
        -------
        dict[str, float]
            Weights after Kelly scaling and exposure capping.
        """
        cap = max_gross_exposure if max_gross_exposure is not None else self.max_gross_exposure

        scaled = {k: v * kelly_fraction for k, v in weights.items()}

        gross = sum(abs(v) for v in scaled.values())
        if gross > cap and gross > 0.0:
            scale_down = cap / gross
            scaled = {k: v * scale_down for k, v in scaled.items()}
            logger.info(
                "Gross exposure %.4f exceeded cap %.4f; rescaled by %.4f.",
                gross,
                cap,
                scale_down,
            )

        return scaled

    # ------------------------------------------------------------------
    # Regime-adjusted sizing
    # ------------------------------------------------------------------

    def regime_adjusted_kelly(
        self,
        kelly: float,
        regime: MarketRegime,
    ) -> float:
        """Adjust a Kelly fraction for the current market regime.

        In bear or volatile (sideways) regimes the Kelly allocation is
        reduced to limit drawdown risk.

        Parameters
        ----------
        kelly : float
            Fractional Kelly value (e.g. from :meth:`fractional_kelly`).
        regime : MarketRegime
            Current market regime.

        Returns
        -------
        float
            Regime-scaled Kelly fraction, floored at 0.0.
        """
        scale = self.regime_kelly_scale.get(regime, 1.0)
        adjusted = kelly * scale
        return float(max(adjusted, 0.0))
