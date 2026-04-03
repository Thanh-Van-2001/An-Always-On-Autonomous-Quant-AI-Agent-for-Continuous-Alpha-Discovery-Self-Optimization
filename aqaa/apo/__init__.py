"""
Adaptive Portfolio Optimizer (APO) module.

Implements regime-aware portfolio construction as described in the KDD 2026
paper.  The module comprises three cooperating components:

1. **RegimeDetector** -- 3-state HMM (bull / bear / sideways) fitted via the
   Baum-Welch (EM) algorithm with online regime-probability updates.
2. **PortfolioOptimizer** -- Mean-variance optimizer solved with SLSQP that
   adjusts the risk-aversion parameter lambda according to the current
   market regime and penalises correlated factor exposures to limit crowding.
3. **KellySizer** -- Fractional Kelly position sizing capped at 2x gross
   exposure, with regime-adaptive scaling.
"""

from aqaa.apo.regime_detector import RegimeDetector
from aqaa.apo.optimizer import PortfolioOptimizer
from aqaa.apo.kelly_sizing import KellySizer

__all__ = [
    "RegimeDetector",
    "PortfolioOptimizer",
    "KellySizer",
]
