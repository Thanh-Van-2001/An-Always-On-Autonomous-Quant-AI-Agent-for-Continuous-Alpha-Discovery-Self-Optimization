"""
AQAA - Always-On Autonomous Quant AI Agent

A closed-loop system that continuously generates, validates, deploys,
and retires alpha factors with minimal human intervention.

Reference: KDD 2026 paper.
"""

__version__ = "1.0.0"
__author__ = "BeeTrade Technologies Inc"

from aqaa.core.types import (
    AlphaState,
    MarketRegime,
    AlphaFactor,
    BacktestResult,
    HypothesisSpec,
    PortfolioWeights,
    TradeOrder,
    ICTimeSeries,
)

__all__ = [
    "AlphaState",
    "MarketRegime",
    "AlphaFactor",
    "BacktestResult",
    "HypothesisSpec",
    "PortfolioWeights",
    "TradeOrder",
    "ICTimeSeries",
]
