"""
AQAA core module.

Contains foundational types, configuration management, and shared
utilities used across all AQAA subsystems.
"""

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
from aqaa.core.config import AQAAConfig, load_config

__all__ = [
    "AlphaState",
    "MarketRegime",
    "AlphaFactor",
    "BacktestResult",
    "HypothesisSpec",
    "PortfolioWeights",
    "TradeOrder",
    "ICTimeSeries",
    "AQAAConfig",
    "load_config",
]
