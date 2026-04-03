#!/usr/bin/env python3
"""
Synthetic market data generator for the AQAA backtesting pipeline.

Generates realistic market data with the following properties so that
experiments can run without any external data dependencies:

- Fat-tailed return distributions (Student-t innovations)
- Volatility clustering via GARCH(1,1) dynamics
- Cross-sectional correlations via a factor model
- Regime structure: bull, bear, sideways, recovery
- COVID crash (March 2020)
- Fundamental features: PE ratio, book-to-market, market cap,
  earnings surprise
- Realistic transaction cost simulation

All generators use seeded numpy RNGs for full reproducibility.

Usage (CLI)::

    python scripts/generate_data.py --market us --output data/
    python scripts/generate_data.py --market china --output data/
    python scripts/generate_data.py --market crypto --output data/
    python scripts/generate_data.py --market factors --output data/
    python scripts/generate_data.py --market all --output data/

Usage (Python)::

    from scripts.generate_data import SyntheticMarketGenerator
    gen = SyntheticMarketGenerator(seed=42)
    df = gen.generate_us_equities(n_stocks=100)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252

# Regime date boundaries (inclusive).  Used for US equities and China A-shares.
# These define the "ground truth" regime schedule baked into the synthetic data
# so that experiments can evaluate regime-detection accuracy.
REGIME_SCHEDULE: List[Tuple[str, str, str]] = [
    # (start, end, regime_name)
    ("2020-01-01", "2020-02-19", "bull"),        # pre-COVID bull
    ("2020-02-20", "2020-03-23", "covid_crash"),  # COVID crash
    ("2020-03-24", "2021-12-31", "bull"),         # post-COVID bull
    ("2022-01-01", "2022-05-31", "sideways"),     # sideways / chop
    ("2022-06-01", "2022-10-31", "bear"),         # bear market
    ("2022-11-01", "2022-12-31", "sideways"),     # transition
    ("2023-01-01", "2025-12-31", "recovery"),     # steady recovery
]

# Default ticker prefixes
US_TICKER_PREFIX = "SYN_US"
CN_TICKER_PREFIX = "SYN_CN"


# ======================================================================
# Helper: GARCH(1,1) volatility path
# ======================================================================

def _garch_volatility(
    n_steps: int,
    omega: float = 1e-6,
    alpha: float = 0.09,
    beta: float = 0.90,
    initial_var: float = 0.0002,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a GARCH(1,1) conditional variance path.

    sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

    Parameters
    ----------
    n_steps : int
        Number of time steps to generate.
    omega, alpha, beta : float
        GARCH(1,1) parameters.  The persistence alpha + beta should
        be < 1 for stationarity.
    initial_var : float
        Initial conditional variance.
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        1-D array of conditional standard deviations (length *n_steps*).
    """
    if rng is None:
        rng = np.random.default_rng()

    var = np.empty(n_steps, dtype=np.float64)
    var[0] = initial_var

    # Draw standardised innovations (Student-t with 5 df for fat tails).
    z = rng.standard_t(df=5, size=n_steps)

    for t in range(1, n_steps):
        eps_prev = z[t - 1] * np.sqrt(var[t - 1])
        var[t] = omega + alpha * eps_prev ** 2 + beta * var[t - 1]
        # Floor to avoid numerical issues.
        var[t] = max(var[t], 1e-10)

    return np.sqrt(var)


# ======================================================================
# Helper: regime drift schedule
# ======================================================================

def _build_regime_drifts(
    dates: pd.DatetimeIndex,
    base_annual_drift: float = 0.08,
    crash_daily_drift: float = -0.04,
    bear_annual_drift: float = -0.15,
    sideways_annual_drift: float = 0.0,
    recovery_annual_drift: float = 0.12,
) -> np.ndarray:
    """Build a per-day drift array based on the regime schedule.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Trading dates.
    base_annual_drift : float
        Annual drift during normal bull regimes.
    crash_daily_drift : float
        Daily drift during the COVID crash (extremely negative).
    bear_annual_drift : float
        Annual drift during bear regimes.
    sideways_annual_drift : float
        Annual drift during sideways regimes.
    recovery_annual_drift : float
        Annual drift during recovery regimes.

    Returns
    -------
    np.ndarray
        1-D array of daily drifts (same length as *dates*).
    """
    daily_drifts = np.full(len(dates), base_annual_drift / TRADING_DAYS_PER_YEAR)

    for start_str, end_str, regime in REGIME_SCHEDULE:
        start_dt = pd.Timestamp(start_str)
        end_dt = pd.Timestamp(end_str)
        mask = (dates >= start_dt) & (dates <= end_dt)

        if regime == "covid_crash":
            daily_drifts[mask] = crash_daily_drift
        elif regime == "bear":
            daily_drifts[mask] = bear_annual_drift / TRADING_DAYS_PER_YEAR
        elif regime == "sideways":
            daily_drifts[mask] = sideways_annual_drift / TRADING_DAYS_PER_YEAR
        elif regime == "recovery":
            daily_drifts[mask] = recovery_annual_drift / TRADING_DAYS_PER_YEAR
        # "bull" uses the base drift -- no override needed.

    return daily_drifts


# ======================================================================
# Helper: cross-sectional factor model
# ======================================================================

def _factor_model_returns(
    n_dates: int,
    n_stocks: int,
    n_factors: int = 5,
    factor_vol: float = 0.01,
    idio_vol: float = 0.015,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate correlated returns via a linear factor model.

    r_{i,t} = sum_k beta_{i,k} * f_{k,t}  +  eps_{i,t}

    where f_{k,t} ~ N(0, factor_vol^2) and eps_{i,t} ~ N(0, idio_vol^2).
    Factor loadings beta are drawn once from N(0, 1) and held constant.

    Parameters
    ----------
    n_dates, n_stocks, n_factors : int
        Dimensions.
    factor_vol : float
        Daily standard deviation of each common factor.
    idio_vol : float
        Daily standard deviation of idiosyncratic noise.
    rng : numpy Generator, optional

    Returns
    -------
    (systematic, idiosyncratic) : tuple of np.ndarray
        Both arrays have shape (n_dates, n_stocks).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Factor loadings: (n_stocks, n_factors)
    betas = rng.standard_normal((n_stocks, n_factors))

    # Factor returns: (n_dates, n_factors)
    factor_rets = rng.normal(0, factor_vol, size=(n_dates, n_factors))

    # Systematic component: (n_dates, n_stocks)
    systematic = factor_rets @ betas.T

    # Idiosyncratic: (n_dates, n_stocks)
    idiosyncratic = rng.normal(0, idio_vol, size=(n_dates, n_stocks))

    return systematic, idiosyncratic


# ======================================================================
# Helper: generate fundamental features
# ======================================================================

def _generate_fundamentals(
    dates: pd.DatetimeIndex,
    tickers: List[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate realistic fundamental features for each (date, ticker).

    Features
    --------
    pe_ratio : float
        Price/earnings ratio.  Cross-sectional distribution is log-normal
        with sector-like clustering.
    book_to_market : float
        Book value / market cap.  Inverse-ish correlated with PE.
    market_cap_bn : float
        Market capitalisation in billions (USD).  Log-normal with drift.
    earnings_surprise : float
        Quarterly earnings surprise (actual - estimate) / |estimate|.
        Mostly zero between earnings dates, nonzero on announcement days
        (approximately every 63 trading days per stock, staggered).

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        ``[date, ticker, pe_ratio, book_to_market, market_cap_bn,
        earnings_surprise]``.
    """
    n_dates = len(dates)
    n_stocks = len(tickers)

    # --- PE ratio ---
    # Base PE drawn once per stock (sector effect), then random-walked.
    base_pe = rng.lognormal(mean=2.9, sigma=0.4, size=n_stocks)  # ~18x median
    pe_noise = rng.normal(0, 0.3, size=(n_dates, n_stocks))
    pe_walk = np.cumsum(pe_noise, axis=0) * 0.05
    pe_matrix = np.clip(base_pe[None, :] + pe_walk, 3.0, 200.0)

    # --- Book-to-market ---
    # Roughly inverse of PE with noise.
    btm_base = 1.0 / pe_matrix + rng.normal(0, 0.005, size=(n_dates, n_stocks))
    btm_matrix = np.clip(btm_base, 0.01, 2.0)

    # --- Market cap ---
    # Log-normal base with small daily drift + noise.
    log_mcap_base = rng.normal(loc=2.5, scale=1.0, size=n_stocks)  # log(bn)
    mcap_drift = rng.normal(0.0001, 0.005, size=(n_dates, n_stocks))
    log_mcap = log_mcap_base[None, :] + np.cumsum(mcap_drift, axis=0)
    mcap_matrix = np.exp(log_mcap)
    mcap_matrix = np.clip(mcap_matrix, 0.1, 3000.0)

    # --- Earnings surprise ---
    # Each stock has a staggered quarterly cycle.
    earnings_matrix = np.zeros((n_dates, n_stocks), dtype=np.float64)
    for j in range(n_stocks):
        # Stagger start by 0-62 days so not all stocks report together.
        offset = rng.integers(0, 63)
        for t in range(offset, n_dates, 63):
            # Surprise ~ N(0.01, 0.05) -- slight positive bias (typical).
            earnings_matrix[t, j] = rng.normal(0.01, 0.05)

    # Assemble into long-format DataFrame.
    records = []
    for i, dt in enumerate(dates):
        for j, ticker in enumerate(tickers):
            records.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "pe_ratio": round(float(pe_matrix[i, j]), 2),
                    "book_to_market": round(float(btm_matrix[i, j]), 4),
                    "market_cap_bn": round(float(mcap_matrix[i, j]), 2),
                    "earnings_surprise": round(float(earnings_matrix[i, j]), 4),
                }
            )

    return pd.DataFrame(records)


# ======================================================================
# Helper: transaction costs model
# ======================================================================

def _transaction_cost_matrix(
    n_dates: int,
    n_stocks: int,
    base_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate per-stock per-day round-trip transaction costs.

    The cost for each (date, stock) is:

        cost = base_cost_bps + slippage_bps * (1 + spread_noise)

    where spread_noise ~ Exp(1) to capture occasional wide-spread events
    (illiquidity spikes).

    Parameters
    ----------
    n_dates, n_stocks : int
        Dimensions.
    base_cost_bps : float
        Commission component in basis points.
    slippage_bps : float
        Average slippage component in basis points.
    rng : numpy Generator

    Returns
    -------
    np.ndarray
        Shape (n_dates, n_stocks), values in basis points.
    """
    if rng is None:
        rng = np.random.default_rng()

    spread_noise = rng.exponential(1.0, size=(n_dates, n_stocks))
    cost = base_cost_bps + slippage_bps * (1.0 + spread_noise)
    return cost


# ======================================================================
# Main generator class
# ======================================================================

class SyntheticMarketGenerator:
    """Generate reproducible synthetic market data for AQAA experiments.

    All methods return pandas DataFrames in a standardised long format
    (date, ticker, open, high, low, close, volume, ...) that can be
    consumed directly by the AQAA backtester and portfolio optimiser.

    Parameters
    ----------
    seed : int
        Master seed for all random number generation.  Sub-generators
        for each market use derived seeds (seed + offset) to ensure
        independence across markets while maintaining reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    # ------------------------------------------------------------------
    # US Equities
    # ------------------------------------------------------------------

    def generate_us_equities(
        self,
        n_stocks: int = 100,
        start: str = "2020-01-01",
        end: str = "2025-12-31",
    ) -> pd.DataFrame:
        """Generate S&P 500-like daily OHLCV data with fundamentals.

        Realistic properties:
        - Fat-tailed returns (Student-t innovations with df=5).
        - Volatility clustering via GARCH(1,1).
        - Cross-sectional correlations via a 5-factor model.
        - Regime structure: bull, COVID crash, sideways, bear, recovery.
        - Fundamental features: PE, book-to-market, market cap,
          earnings surprise.
        - Per-stock transaction cost estimates.

        Parameters
        ----------
        n_stocks : int
            Number of synthetic stocks to generate.
        start, end : str
            Date range in ``YYYY-MM-DD`` format.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ``[date, ticker, open, high, low, close, volume,
            returns, regime, pe_ratio, book_to_market, market_cap_bn,
            earnings_surprise, transaction_cost_bps]``.
        """
        rng = np.random.default_rng(self.seed)
        logger.info(
            "Generating US equities: n_stocks=%d, %s to %s",
            n_stocks, start, end,
        )

        # Trading dates (US business days).
        dates = pd.bdate_range(start=start, end=end, freq="B")
        n_dates = len(dates)
        tickers = [f"{US_TICKER_PREFIX}_{i:04d}" for i in range(n_stocks)]

        # --- Regime-based drifts ---
        daily_drifts = _build_regime_drifts(dates)

        # --- GARCH volatility per stock ---
        vols = np.empty((n_dates, n_stocks), dtype=np.float64)
        for j in range(n_stocks):
            # Vary GARCH parameters slightly across stocks.
            omega = rng.uniform(5e-7, 2e-6)
            alpha = rng.uniform(0.05, 0.12)
            beta = rng.uniform(0.85, 0.93)
            # Ensure stationarity: alpha + beta < 1
            if alpha + beta >= 0.99:
                beta = 0.99 - alpha - 0.01
            vols[:, j] = _garch_volatility(
                n_dates, omega=omega, alpha=alpha, beta=beta,
                initial_var=rng.uniform(1e-4, 3e-4), rng=rng,
            )

        # --- Factor model for cross-sectional correlation ---
        systematic, idiosyncratic = _factor_model_returns(
            n_dates, n_stocks, n_factors=5,
            factor_vol=0.008, idio_vol=0.012, rng=rng,
        )

        # --- Combine into returns ---
        # r_{i,t} = drift_t + vol_{i,t} * z_{i,t} + systematic_{i,t}
        z = rng.standard_t(df=5, size=(n_dates, n_stocks))
        returns = (
            daily_drifts[:, None]
            + vols * z * 0.5          # GARCH-scaled idiosyncratic
            + systematic              # factor-model systematic
            + idiosyncratic * 0.3     # additional idiosyncratic
        )

        # Clip extreme returns to stay realistic (daily limit ~+/-15%).
        returns = np.clip(returns, -0.15, 0.15)

        # --- Price paths from returns ---
        initial_prices = rng.uniform(20.0, 300.0, size=n_stocks)
        cum_returns = np.cumprod(1.0 + returns, axis=0)
        close_prices = initial_prices[None, :] * cum_returns

        # OHLCV from close prices.
        intraday_noise = rng.uniform(0.002, 0.015, size=(n_dates, n_stocks))
        open_prices = close_prices * (1.0 + rng.normal(0, 0.003, size=(n_dates, n_stocks)))
        high_prices = np.maximum(open_prices, close_prices) * (1.0 + intraday_noise)
        low_prices = np.minimum(open_prices, close_prices) * (1.0 - intraday_noise)

        # Volume: log-normal with regime-dependent scaling.
        base_volume = rng.lognormal(mean=14.0, sigma=1.0, size=n_stocks)
        vol_multiplier = np.ones(n_dates)
        for s, e, regime in REGIME_SCHEDULE:
            mask = (dates >= pd.Timestamp(s)) & (dates <= pd.Timestamp(e))
            if regime == "covid_crash":
                vol_multiplier[mask] = 3.0  # huge volume during crash
            elif regime == "bear":
                vol_multiplier[mask] = 1.5
            elif regime == "sideways":
                vol_multiplier[mask] = 0.8
        volume_noise = rng.lognormal(0, 0.3, size=(n_dates, n_stocks))
        volume = (base_volume[None, :] * vol_multiplier[:, None] * volume_noise).astype(np.int64)

        # --- Transaction costs ---
        tc = _transaction_cost_matrix(n_dates, n_stocks, base_cost_bps=5.0, slippage_bps=2.0, rng=rng)

        # --- Regime labels ---
        regime_labels = np.full(n_dates, "unknown", dtype=object)
        for s, e, regime in REGIME_SCHEDULE:
            mask = (dates >= pd.Timestamp(s)) & (dates <= pd.Timestamp(e))
            regime_labels[mask] = regime

        # --- Fundamental features ---
        logger.info("Generating fundamental features ...")
        fundamentals = _generate_fundamentals(dates, tickers, rng)

        # --- Assemble long-format DataFrame ---
        logger.info("Assembling DataFrame ...")
        records = []
        for i in range(n_dates):
            for j in range(n_stocks):
                records.append(
                    {
                        "date": dates[i],
                        "ticker": tickers[j],
                        "open": round(float(open_prices[i, j]), 2),
                        "high": round(float(high_prices[i, j]), 2),
                        "low": round(float(low_prices[i, j]), 2),
                        "close": round(float(close_prices[i, j]), 2),
                        "volume": int(volume[i, j]),
                        "returns": round(float(returns[i, j]), 6),
                        "regime": regime_labels[i],
                        "transaction_cost_bps": round(float(tc[i, j]), 2),
                    }
                )

        df = pd.DataFrame(records)

        # Merge fundamentals.
        df = df.merge(fundamentals, on=["date", "ticker"], how="left")

        logger.info(
            "US equities generated: %d rows, %d stocks, %d dates",
            len(df), n_stocks, n_dates,
        )
        return df

    # ------------------------------------------------------------------
    # China A-Shares
    # ------------------------------------------------------------------

    def generate_china_ashares(
        self,
        n_stocks: int = 100,
        start: str = "2020-01-01",
        end: str = "2025-12-31",
    ) -> pd.DataFrame:
        """Generate CSI 500-like daily OHLCV data.

        Similar to US equities but with:
        - Higher volatility (China A-shares are ~30-40% more volatile).
        - Daily price limits of +/-10%.
        - Higher transaction costs (10 bps + 5 bps slippage).
        - T+1 settlement constraint (noted in metadata, not enforced here).
        - Different regime timing (China-specific).

        Parameters
        ----------
        n_stocks : int
            Number of synthetic stocks.
        start, end : str
            Date range.

        Returns
        -------
        pd.DataFrame
            Same schema as ``generate_us_equities`` but with CN tickers
            and China-specific properties.
        """
        rng = np.random.default_rng(self.seed + 1000)
        logger.info(
            "Generating China A-shares: n_stocks=%d, %s to %s",
            n_stocks, start, end,
        )

        # Use Chinese trading calendar approximation (pandas business days
        # is a reasonable proxy -- exact SSE calendar would require a
        # third-party library).
        dates = pd.bdate_range(start=start, end=end, freq="B")
        n_dates = len(dates)
        tickers = [f"{CN_TICKER_PREFIX}_{i:04d}" for i in range(n_stocks)]

        # --- Regime drifts (scaled up for higher vol market) ---
        daily_drifts = _build_regime_drifts(
            dates,
            base_annual_drift=0.06,
            crash_daily_drift=-0.05,      # sharper crash
            bear_annual_drift=-0.20,
            sideways_annual_drift=-0.02,
            recovery_annual_drift=0.10,
        )

        # --- GARCH volatility (higher params for China) ---
        vols = np.empty((n_dates, n_stocks), dtype=np.float64)
        for j in range(n_stocks):
            omega = rng.uniform(1e-6, 4e-6)
            alpha = rng.uniform(0.08, 0.15)
            beta = rng.uniform(0.82, 0.90)
            if alpha + beta >= 0.99:
                beta = 0.99 - alpha - 0.01
            vols[:, j] = _garch_volatility(
                n_dates, omega=omega, alpha=alpha, beta=beta,
                initial_var=rng.uniform(2e-4, 5e-4), rng=rng,
            )

        # --- Factor model ---
        systematic, idiosyncratic = _factor_model_returns(
            n_dates, n_stocks, n_factors=5,
            factor_vol=0.012, idio_vol=0.018, rng=rng,
        )

        # --- Returns ---
        z = rng.standard_t(df=4, size=(n_dates, n_stocks))  # fatter tails
        returns = (
            daily_drifts[:, None]
            + vols * z * 0.6
            + systematic
            + idiosyncratic * 0.4
        )

        # China A-share daily price limits: +/-10%
        returns = np.clip(returns, -0.10, 0.10)

        # --- Price paths ---
        initial_prices = rng.uniform(5.0, 100.0, size=n_stocks)
        cum_returns = np.cumprod(1.0 + returns, axis=0)
        close_prices = initial_prices[None, :] * cum_returns

        intraday_noise = rng.uniform(0.003, 0.020, size=(n_dates, n_stocks))
        open_prices = close_prices * (1.0 + rng.normal(0, 0.004, size=(n_dates, n_stocks)))
        high_prices = np.maximum(open_prices, close_prices) * (1.0 + intraday_noise)
        low_prices = np.minimum(open_prices, close_prices) * (1.0 - intraday_noise)

        # Volume
        base_volume = rng.lognormal(mean=15.0, sigma=1.2, size=n_stocks)
        vol_multiplier = np.ones(n_dates)
        for s, e, regime in REGIME_SCHEDULE:
            mask = (dates >= pd.Timestamp(s)) & (dates <= pd.Timestamp(e))
            if regime == "covid_crash":
                vol_multiplier[mask] = 2.5
            elif regime == "bear":
                vol_multiplier[mask] = 1.8
        volume_noise = rng.lognormal(0, 0.4, size=(n_dates, n_stocks))
        volume = (base_volume[None, :] * vol_multiplier[:, None] * volume_noise).astype(np.int64)

        # Transaction costs (higher for China)
        tc = _transaction_cost_matrix(n_dates, n_stocks, base_cost_bps=10.0, slippage_bps=5.0, rng=rng)

        # Regime labels
        regime_labels = np.full(n_dates, "unknown", dtype=object)
        for s, e, regime in REGIME_SCHEDULE:
            mask = (dates >= pd.Timestamp(s)) & (dates <= pd.Timestamp(e))
            regime_labels[mask] = regime

        # Fundamentals
        fundamentals = _generate_fundamentals(dates, tickers, rng)

        # Assemble
        logger.info("Assembling China A-shares DataFrame ...")
        records = []
        for i in range(n_dates):
            for j in range(n_stocks):
                records.append(
                    {
                        "date": dates[i],
                        "ticker": tickers[j],
                        "open": round(float(open_prices[i, j]), 2),
                        "high": round(float(high_prices[i, j]), 2),
                        "low": round(float(low_prices[i, j]), 2),
                        "close": round(float(close_prices[i, j]), 2),
                        "volume": int(volume[i, j]),
                        "returns": round(float(returns[i, j]), 6),
                        "regime": regime_labels[i],
                        "transaction_cost_bps": round(float(tc[i, j]), 2),
                    }
                )

        df = pd.DataFrame(records)
        df = df.merge(fundamentals, on=["date", "ticker"], how="left")

        logger.info(
            "China A-shares generated: %d rows, %d stocks, %d dates",
            len(df), n_stocks, n_dates,
        )
        return df

    # ------------------------------------------------------------------
    # Crypto
    # ------------------------------------------------------------------

    def generate_crypto(
        self,
        pairs: Optional[List[str]] = None,
        start: str = "2020-01-01",
        end: str = "2025-12-31",
        freq: str = "1h",
    ) -> pd.DataFrame:
        """Generate hourly OHLCV data for cryptocurrency pairs.

        Crypto-specific properties:
        - 24/7 trading (no weekday filter).
        - Much higher volatility (~3-5x equities).
        - Leverage effects (vol increases more on down moves).
        - BTC-dominance correlation structure.
        - Regime-dependent dynamics (halving cycles, etc.).

        Parameters
        ----------
        pairs : list of str, optional
            Crypto pairs to generate.  Defaults to
            ``["BTC/USDT", "ETH/USDT"]``.
        start, end : str
            Date range.
        freq : str
            Frequency string for pandas (default ``"1h"`` = hourly).

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ``[datetime, pair, open, high, low, close, volume,
            returns, transaction_cost_bps]``.
        """
        if pairs is None:
            pairs = ["BTC/USDT", "ETH/USDT"]

        rng = np.random.default_rng(self.seed + 2000)
        logger.info(
            "Generating crypto data: pairs=%s, %s to %s, freq=%s",
            pairs, start, end, freq,
        )

        # 24/7 datetime index.
        timestamps = pd.date_range(start=start, end=end, freq=freq)
        n_steps = len(timestamps)
        n_pairs = len(pairs)

        # Pair-specific base parameters.
        # BTC: lower vol than alts; ETH: higher vol, correlated with BTC.
        pair_params = {}
        for p in pairs:
            if "BTC" in p.upper():
                pair_params[p] = {
                    "annual_vol": 0.70,   # ~70% annualised vol
                    "annual_drift": 0.30, # long-term upward bias
                    "initial_price": 7200.0,  # Jan 2020 BTC price
                    "garch_alpha": 0.10,
                    "garch_beta": 0.88,
                    "leverage_coeff": 0.3,  # asymmetric vol
                }
            elif "ETH" in p.upper():
                pair_params[p] = {
                    "annual_vol": 0.90,
                    "annual_drift": 0.40,
                    "initial_price": 130.0,  # Jan 2020 ETH price
                    "garch_alpha": 0.12,
                    "garch_beta": 0.86,
                    "leverage_coeff": 0.35,
                }
            else:
                pair_params[p] = {
                    "annual_vol": 1.00,
                    "annual_drift": 0.10,
                    "initial_price": rng.uniform(0.5, 50.0),
                    "garch_alpha": 0.12,
                    "garch_beta": 0.85,
                    "leverage_coeff": 0.25,
                }

        # Hours per year for annualisation.
        hours_per_year = 365.25 * 24

        # BTC is the "market factor"; alts are correlated with BTC.
        btc_idx = None
        for i, p in enumerate(pairs):
            if "BTC" in p.upper():
                btc_idx = i
                break

        all_returns = np.empty((n_steps, n_pairs), dtype=np.float64)

        for j, pair in enumerate(pairs):
            params = pair_params[pair]
            hourly_vol = params["annual_vol"] / np.sqrt(hours_per_year)
            hourly_drift = params["annual_drift"] / hours_per_year

            # GARCH volatility path.
            omega = hourly_vol ** 2 * (1 - params["garch_alpha"] - params["garch_beta"])
            omega = max(omega, 1e-10)
            vol_path = _garch_volatility(
                n_steps,
                omega=omega,
                alpha=params["garch_alpha"],
                beta=params["garch_beta"],
                initial_var=hourly_vol ** 2,
                rng=rng,
            )

            # Innovations with fat tails.
            z = rng.standard_t(df=4, size=n_steps)

            # Build returns.
            rets = hourly_drift + vol_path * z

            # Leverage effect: amplify vol on negative returns.
            leverage = params["leverage_coeff"]
            for t in range(1, n_steps):
                if rets[t - 1] < 0:
                    rets[t] *= (1.0 + leverage * abs(rets[t - 1]) / hourly_vol)

            all_returns[:, j] = rets

        # Cross-pair correlation: alts follow BTC with beta > 1.
        if btc_idx is not None:
            btc_rets = all_returns[:, btc_idx].copy()
            for j, pair in enumerate(pairs):
                if j == btc_idx:
                    continue
                # Alt = beta * BTC + idiosyncratic
                beta_to_btc = rng.uniform(1.1, 1.5)
                corr_weight = rng.uniform(0.4, 0.7)
                all_returns[:, j] = (
                    corr_weight * beta_to_btc * btc_rets
                    + (1.0 - corr_weight) * all_returns[:, j]
                )

        # Crypto regime overlays (BTC halving cycles, etc.)
        # Bull: 2020-10 to 2021-11 (post-halving run)
        # Crash: 2021-11 to 2022-06 (crypto winter starts)
        # Bear: 2022-06 to 2022-12
        # Recovery: 2023-01 onwards
        crypto_regimes = [
            ("2020-01-01", "2020-03-12", -0.03 / 24),   # pre-COVID + crash
            ("2020-03-13", "2020-09-30", 0.002),          # recovery
            ("2020-10-01", "2021-11-10", 0.003),          # bull run
            ("2021-11-11", "2022-06-15", -0.001),         # crypto winter entry
            ("2022-06-16", "2022-12-31", -0.0005),        # bear
            ("2023-01-01", "2025-12-31", 0.001),          # recovery
        ]
        for s, e, extra_drift in crypto_regimes:
            mask = (timestamps >= pd.Timestamp(s)) & (timestamps <= pd.Timestamp(e))
            all_returns[mask, :] += extra_drift

        # Clip to realistic hourly bounds.
        all_returns = np.clip(all_returns, -0.15, 0.15)

        # --- Price paths ---
        records = []
        for j, pair in enumerate(pairs):
            params = pair_params[pair]
            cum_ret = np.cumprod(1.0 + all_returns[:, j])
            close_price = params["initial_price"] * cum_ret

            # OHLV from close.
            hourly_noise = rng.uniform(0.001, 0.008, size=n_steps)
            open_price = close_price * (1.0 + rng.normal(0, 0.002, size=n_steps))
            high_price = np.maximum(open_price, close_price) * (1.0 + hourly_noise)
            low_price = np.minimum(open_price, close_price) * (1.0 - hourly_noise)

            # Volume (in base currency units).
            base_vol = 500.0 if "BTC" in pair.upper() else 5000.0
            vol_noise = rng.lognormal(0, 0.5, size=n_steps)
            volume = (base_vol * vol_noise).astype(np.float64)

            # Transaction costs.
            tc = 8.0 + 3.0 * rng.exponential(1.0, size=n_steps)

            for i in range(n_steps):
                records.append(
                    {
                        "datetime": timestamps[i],
                        "pair": pair,
                        "open": round(float(open_price[i]), 2),
                        "high": round(float(high_price[i]), 2),
                        "low": round(float(low_price[i]), 2),
                        "close": round(float(close_price[i]), 2),
                        "volume": round(float(volume[i]), 4),
                        "returns": round(float(all_returns[i, j]), 8),
                        "transaction_cost_bps": round(float(tc[i]), 2),
                    }
                )

        df = pd.DataFrame(records)
        logger.info(
            "Crypto data generated: %d rows, %d pairs, %d timestamps",
            len(df), n_pairs, n_steps,
        )
        return df

    # ------------------------------------------------------------------
    # Factor Returns / IC time series
    # ------------------------------------------------------------------

    def generate_factor_returns(
        self,
        n_factors: int = 50,
        n_days: int = 1260,
        alpha_decay_schedule: Optional[Dict[int, int]] = None,
    ) -> pd.DataFrame:
        """Generate simulated factor IC time series with realistic properties.

        The generated factors fall into three categories:

        1. **Stable factors** (~40%): IC is drawn from a stationary process
           with a positive mean.  These represent robust alpha sources.
        2. **Decaying factors** (~30%): IC starts positive but drifts toward
           zero (or below) after a specified decay onset day.  These
           simulate alpha decay / crowding.
        3. **Spurious factors** (~30%): IC is essentially noise around zero,
           simulating factors discovered by overfitting or LLM memorisation
           that do not generalise.

        The ``alpha_decay_schedule`` parameter controls which factors
        decay and when.  If not provided, a default schedule is generated
        where decaying factors begin losing efficacy between day 200 and
        day 800.

        Parameters
        ----------
        n_factors : int
            Total number of factors.
        n_days : int
            Number of trading days (default 1260 = 5 years).
        alpha_decay_schedule : dict, optional
            Mapping from factor index (int) to the day (int) at which
            decay begins.  Only factors in this dict will decay; all
            others keep their default category assignment.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``[day, factor_id, ic, category]``
            where ``day`` ranges from 0 to ``n_days - 1``,
            ``factor_id`` is ``"F_0000"`` .. ``"F_{n_factors-1}"``, and
            ``category`` is one of ``"stable"``, ``"decaying"``,
            ``"spurious"``.
        """
        rng = np.random.default_rng(self.seed + 3000)
        logger.info(
            "Generating factor IC series: n_factors=%d, n_days=%d",
            n_factors, n_days,
        )

        # Assign categories.
        n_stable = int(n_factors * 0.4)
        n_decaying = int(n_factors * 0.3)
        # Remaining are spurious.

        categories = (
            ["stable"] * n_stable
            + ["decaying"] * n_decaying
            + ["spurious"] * (n_factors - n_stable - n_decaying)
        )
        rng.shuffle(categories)

        # Default decay schedule for decaying factors.
        if alpha_decay_schedule is None:
            alpha_decay_schedule = {}
            for idx, cat in enumerate(categories):
                if cat == "decaying":
                    alpha_decay_schedule[idx] = int(rng.integers(200, 800))

        factor_ids = [f"F_{i:04d}" for i in range(n_factors)]

        records = []

        for idx in range(n_factors):
            cat = categories[idx]
            fid = factor_ids[idx]

            if cat == "stable":
                # Stationary IC with positive mean.
                ic_mean = rng.uniform(0.02, 0.06)
                ic_std = rng.uniform(0.02, 0.04)
                # AR(1) process for autocorrelation.
                ar_coeff = rng.uniform(0.3, 0.7)
                ic_series = np.empty(n_days, dtype=np.float64)
                ic_series[0] = ic_mean + rng.normal(0, ic_std)
                for t in range(1, n_days):
                    ic_series[t] = (
                        ic_mean * (1 - ar_coeff)
                        + ar_coeff * ic_series[t - 1]
                        + rng.normal(0, ic_std * np.sqrt(1 - ar_coeff ** 2))
                    )

            elif cat == "decaying":
                # IC starts positive, then decays after onset.
                ic_mean_initial = rng.uniform(0.03, 0.07)
                ic_std = rng.uniform(0.02, 0.04)
                decay_onset = alpha_decay_schedule.get(idx, 500)
                # Decay rate: IC loses ~0.5-2 bps per day after onset.
                decay_rate = rng.uniform(0.00005, 0.0002)

                ar_coeff = rng.uniform(0.3, 0.6)
                ic_series = np.empty(n_days, dtype=np.float64)
                ic_series[0] = ic_mean_initial + rng.normal(0, ic_std)
                for t in range(1, n_days):
                    if t < decay_onset:
                        target = ic_mean_initial
                    else:
                        # Linear decay in the mean.
                        target = max(
                            ic_mean_initial - decay_rate * (t - decay_onset),
                            -0.01,
                        )
                    ic_series[t] = (
                        target * (1 - ar_coeff)
                        + ar_coeff * ic_series[t - 1]
                        + rng.normal(0, ic_std * np.sqrt(1 - ar_coeff ** 2))
                    )

            else:  # spurious
                # IC is noise around zero -- these factors are not predictive.
                ic_std = rng.uniform(0.03, 0.06)
                ic_series = rng.normal(0, ic_std, size=n_days)
                # Add a tiny positive mean to simulate in-sample overfitting.
                ic_series += rng.uniform(-0.005, 0.005)

            for t in range(n_days):
                records.append(
                    {
                        "day": t,
                        "factor_id": fid,
                        "ic": round(float(ic_series[t]), 6),
                        "category": cat,
                    }
                )

        df = pd.DataFrame(records)
        logger.info(
            "Factor IC series generated: %d rows, %d factors, %d days",
            len(df), n_factors, n_days,
        )
        return df


# ======================================================================
# CLI entry point
# ======================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic market data for AQAA experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_data.py --market us --output data/
  python scripts/generate_data.py --market china --output data/
  python scripts/generate_data.py --market crypto --output data/
  python scripts/generate_data.py --market factors --output data/
  python scripts/generate_data.py --market all --output data/
        """,
    )
    parser.add_argument(
        "--market",
        type=str,
        choices=["us", "china", "crypto", "factors", "all"],
        default="all",
        help="Which market to generate (default: all).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/",
        help="Output directory (default: data/).",
    )
    parser.add_argument(
        "--n-stocks",
        type=int,
        default=100,
        help="Number of stocks for equity markets (default: 100).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date (default: 2020-01-01).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-12-31",
        help="End date (default: 2025-12-31).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "csv"],
        default="parquet",
        help="Output file format (default: parquet).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for synthetic data generation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = SyntheticMarketGenerator(seed=args.seed)

    markets_to_generate = (
        ["us", "china", "crypto", "factors"]
        if args.market == "all"
        else [args.market]
    )

    def _save(df: pd.DataFrame, name: str) -> None:
        if args.format == "parquet":
            path = output_dir / f"{name}.parquet"
            df.to_parquet(path, index=False, engine="pyarrow")
        else:
            path = output_dir / f"{name}.csv"
            df.to_csv(path, index=False)
        logger.info("Saved %s (%d rows)", path, len(df))

    for market in markets_to_generate:
        if market == "us":
            df = gen.generate_us_equities(
                n_stocks=args.n_stocks, start=args.start, end=args.end,
            )
            _save(df, "synthetic_us_equities")

        elif market == "china":
            df = gen.generate_china_ashares(
                n_stocks=args.n_stocks, start=args.start, end=args.end,
            )
            _save(df, "synthetic_china_ashares")

        elif market == "crypto":
            df = gen.generate_crypto(start=args.start, end=args.end)
            _save(df, "synthetic_crypto")

        elif market == "factors":
            df = gen.generate_factor_returns()
            _save(df, "synthetic_factor_ic")

    logger.info("Done. All data written to %s", output_dir)


if __name__ == "__main__":
    main()
