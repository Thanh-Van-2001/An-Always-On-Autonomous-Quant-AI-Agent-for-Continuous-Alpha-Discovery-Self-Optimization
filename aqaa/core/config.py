"""
Configuration management for the AQAA system.

Provides a hierarchical, YAML-backed configuration with sensible defaults
matching the parameter values specified in the KDD 2026 paper.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HypothesisEngineConfig:
    """Settings for the Hypothesis Engine (Module 1)."""

    tau_nov: float = 0.75          # Novelty threshold for deduplication
    max_code_iterations: int = 10  # Max LLM refinement loops per hypothesis
    source_types: List[str] = field(default_factory=lambda: ["academic", "news", "data"])
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o"
    batch_size: int = 16


@dataclass
class BacktesterConfig:
    """Settings for the Vectorised Backtester (Module 2)."""

    ic_threshold: float = 0.03    # Minimum mean IC to pass
    icir_threshold: float = 0.8   # Minimum ICIR to pass
    sharpe_threshold: float = 1.0 # Minimum annualised Sharpe to pass
    mdd_threshold: float = 0.25   # Maximum drawdown to pass (25%)
    ldt_alpha: float = 0.01       # Significance level for Leakage Detection Test
    buffer_months: int = 6        # Buffer zone around LLM cutoff (6 months default; 12 for GPT-4-class)
    buffer_months_gpt4: int = 12  # Extended buffer for GPT-4-class models (Section 4.3)
    paper_trade_days: int = 30    # Forward paper-trading gate duration (days)
    llm_cutoff_date: str = "2024-04-01"  # GPT-4o training cutoff (Section 5.1)
    lookback_years: int = 5       # Backtest history length
    cost_bps: float = 5.0         # Default round-trip cost in basis points
    slippage_bps: float = 2.0     # Default slippage in basis points


@dataclass
class PortfolioConfig:
    """Settings for the Portfolio Constructor (Module 3)."""

    kelly_fraction: float = 0.5   # Fractional Kelly scalar
    max_gross_exposure: float = 1.0
    max_single_weight: float = 0.05
    rho_crowd: float = 0.65       # Crowding-penalty correlation threshold
    rebalance_freq: str = "daily"
    risk_model: str = "ledoit_wolf"


@dataclass
class ExecutionConfig:
    """Settings for the Execution Engine (Module 4)."""

    default_algo: str = "TWAP"
    urgency: str = "medium"       # low | medium | high
    max_participation_rate: float = 0.05
    broker: str = "simulated"


@dataclass
class WatchdogConfig:
    """Settings for the IC Watchdog / Self-Critique Loop (Module 5)."""

    theta_min: float = 0.015      # IC-decay threshold triggering review
    cusum_drift: float = 0.005    # CUSUM drift parameter
    review_window_days: int = 60
    ucb_c: float = 1.5            # UCB exploration constant for factor selection


@dataclass
class MarketConfig:
    """Market-specific overrides."""

    name: str = "us_equities"
    universe: str = "sp500"
    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    frequency: str = "daily"
    currency: str = "USD"
    trading_hours: str = "09:30-16:00 ET"


@dataclass
class LoggingConfig:
    """Logging and observability settings."""

    level: str = "INFO"
    log_dir: str = "logs"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    enable_wandb: bool = False


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class AQAAConfig:
    """Root configuration object for the entire AQAA system.

    Constructed from a YAML file via ``load_config()`` or instantiated
    directly with default values matching the KDD 2026 paper.
    """

    hypothesis_engine: HypothesisEngineConfig = field(default_factory=HypothesisEngineConfig)
    backtester: BacktesterConfig = field(default_factory=BacktesterConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # -- helpers -------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Recursively serialise the config to a plain dict."""
        import dataclasses
        return _dataclass_to_dict(self)

    def merge(self, overrides: Dict[str, Any]) -> None:
        """Merge a (possibly nested) dict of overrides into this config."""
        _merge_into_dataclass(self, overrides)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> AQAAConfig:
    """Load an ``AQAAConfig`` from a YAML file.

    Unknown top-level keys are silently ignored so that market-overlay
    files can be loaded on top of the base config.

    Parameters
    ----------
    path:
        Filesystem path to the YAML configuration file.

    Returns
    -------
    AQAAConfig
        A fully-populated configuration object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    cfg = AQAAConfig()
    cfg.merge(raw)
    return cfg


def load_config_with_overlay(
    base_path: str | Path,
    overlay_path: Optional[str | Path] = None,
) -> AQAAConfig:
    """Load a base config and optionally merge a market-specific overlay.

    Parameters
    ----------
    base_path:
        Path to the default/base YAML config.
    overlay_path:
        Optional path to a market-specific overlay YAML.

    Returns
    -------
    AQAAConfig
    """
    cfg = load_config(base_path)
    if overlay_path is not None:
        overlay_path = Path(overlay_path)
        if not overlay_path.exists():
            raise FileNotFoundError(f"Overlay config not found: {overlay_path}")
        with open(overlay_path, "r", encoding="utf-8") as fh:
            overlay: Dict[str, Any] = yaml.safe_load(fh) or {}
        cfg.merge(overlay)
    return cfg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SECTION_MAP = {
    "hypothesis_engine": HypothesisEngineConfig,
    "backtester": BacktesterConfig,
    "portfolio": PortfolioConfig,
    "execution": ExecutionConfig,
    "watchdog": WatchdogConfig,
    "market": MarketConfig,
    "logging": LoggingConfig,
}


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass instance to a plain dict."""
    import dataclasses

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _dataclass_to_dict(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    elif isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def _merge_into_dataclass(dc: Any, overrides: Dict[str, Any]) -> None:
    """Recursively merge *overrides* into an existing dataclass instance."""
    import dataclasses

    if not dataclasses.is_dataclass(dc) or isinstance(dc, type):
        return

    field_names = {f.name for f in dataclasses.fields(dc)}
    for key, value in overrides.items():
        if key not in field_names:
            continue
        current = getattr(dc, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge_into_dataclass(current, value)
        else:
            setattr(dc, key, value)
