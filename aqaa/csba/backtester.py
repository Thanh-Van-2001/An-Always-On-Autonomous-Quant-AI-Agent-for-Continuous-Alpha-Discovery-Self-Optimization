"""
Vectorised backtesting engine for alpha factor evaluation.

Computes the standard quantitative performance metrics used throughout the
AQAA pipeline and referenced in Table 2 of the KDD 2026 paper:

- **IC**     -- Information Coefficient (Spearman rank correlation between
                factor predictions and subsequent returns).
- **ICIR**   -- IC Information Ratio (mean IC / std IC).  A measure of the
                consistency of the factor signal.
- **Sharpe** -- Annualised Sharpe ratio of the factor's long-short return
                stream.
- **MDD**    -- Maximum Drawdown of the cumulative return curve.
- **ARR**    -- Annualised Rate of Return.
- **Calmar** -- Calmar ratio (ARR / MDD).

Acceptance thresholds (from the paper):
    IC   > 0.03
    ICIR > 0.8
    Sharpe > 1.0
    MDD  < 25%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
from scipy import stats as sp_stats

from aqaa.core.types import BacktestResult

logger = logging.getLogger(__name__)

# Number of trading days in a calendar year (standard convention).
TRADING_DAYS_PER_YEAR = 252

# Default acceptance thresholds as specified in Table 2 of the paper.
DEFAULT_IC_THRESHOLD = 0.03
DEFAULT_ICIR_THRESHOLD = 0.8
DEFAULT_SHARPE_THRESHOLD = 1.0
DEFAULT_MDD_THRESHOLD = 0.25  # 25%


@dataclass
class BacktestMetrics:
    """Lightweight container for the six core performance metrics.

    This is used internally by the ``Backtester``; the canonical
    ``BacktestResult`` from ``aqaa.core.types`` is returned to callers.
    """

    ic: float = 0.0
    icir: float = 0.0
    sharpe: float = 0.0
    mdd: float = 0.0
    arr: float = 0.0
    calmar: float = 0.0

    def meets_thresholds(
        self,
        ic_threshold: float = DEFAULT_IC_THRESHOLD,
        icir_threshold: float = DEFAULT_ICIR_THRESHOLD,
        sharpe_threshold: float = DEFAULT_SHARPE_THRESHOLD,
        mdd_threshold: float = DEFAULT_MDD_THRESHOLD,
    ) -> bool:
        """Return ``True`` if all acceptance gates are satisfied."""
        return (
            self.ic >= ic_threshold
            and self.icir >= icir_threshold
            and self.sharpe >= sharpe_threshold
            and self.mdd <= mdd_threshold
        )


class Backtester:
    """Vectorised cross-sectional backtester for single alpha factors.

    The backtester operates on two aligned arrays:

    * **predictions** -- the factor's cross-sectional signal for each
      (date, asset) pair, flattened into a 2-D array of shape
      ``(n_dates, n_assets)``.
    * **returns** -- the *next-period* asset returns with the same shape.

    For each date the Information Coefficient is computed as the Spearman
    rank correlation between the prediction cross-section and the return
    cross-section.  Portfolio returns are derived from a simple long-short
    quintile strategy on the ranked predictions.

    Parameters
    ----------
    ic_threshold : float
        Minimum mean IC required to pass the backtest gate.
    icir_threshold : float
        Minimum ICIR required.
    sharpe_threshold : float
        Minimum annualised Sharpe required.
    mdd_threshold : float
        Maximum drawdown allowed (as a positive fraction, e.g. 0.25 = 25%).
    cost_bps : float
        Round-trip transaction cost in basis points.
    """

    def __init__(
        self,
        ic_threshold: float = DEFAULT_IC_THRESHOLD,
        icir_threshold: float = DEFAULT_ICIR_THRESHOLD,
        sharpe_threshold: float = DEFAULT_SHARPE_THRESHOLD,
        mdd_threshold: float = DEFAULT_MDD_THRESHOLD,
        cost_bps: float = 5.0,
    ) -> None:
        self.ic_threshold = ic_threshold
        self.icir_threshold = icir_threshold
        self.sharpe_threshold = sharpe_threshold
        self.mdd_threshold = mdd_threshold
        self.cost_bps = cost_bps

    # ------------------------------------------------------------------
    # Core metric computations
    # ------------------------------------------------------------------

    @staticmethod
    def compute_ic(
        predictions: np.ndarray,
        returns: np.ndarray,
    ) -> np.ndarray:
        """Compute per-date Information Coefficient (rank IC).

        For each row (date) the Spearman rank correlation between
        ``predictions[t]`` and ``returns[t]`` is computed.

        Parameters
        ----------
        predictions : np.ndarray
            2-D array of shape ``(n_dates, n_assets)`` with factor values.
        returns : np.ndarray
            2-D array of shape ``(n_dates, n_assets)`` with next-period returns.

        Returns
        -------
        np.ndarray
            1-D array of length ``n_dates`` with IC values.

        Raises
        ------
        ValueError
            If the shapes of *predictions* and *returns* do not match or if
            there are fewer than 3 assets.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        returns = np.asarray(returns, dtype=np.float64)

        if predictions.shape != returns.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} "
                f"vs returns {returns.shape}"
            )

        # Handle 1-D input (single date).
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)
            returns = returns.reshape(1, -1)

        n_dates, n_assets = predictions.shape
        if n_assets < 3:
            raise ValueError(
                f"Need at least 3 assets for rank correlation, got {n_assets}"
            )

        ic_values = np.empty(n_dates, dtype=np.float64)
        for t in range(n_dates):
            pred_t = predictions[t]
            ret_t = returns[t]

            # Mask out NaN positions.
            valid = ~(np.isnan(pred_t) | np.isnan(ret_t))
            if valid.sum() < 3:
                ic_values[t] = np.nan
                continue

            corr, _ = sp_stats.spearmanr(pred_t[valid], ret_t[valid])
            ic_values[t] = corr

        return ic_values

    @staticmethod
    def compute_icir(ic_series: np.ndarray) -> float:
        """Compute the IC Information Ratio: mean(IC) / std(IC).

        Parameters
        ----------
        ic_series : np.ndarray
            1-D array of IC observations (NaN entries are ignored).

        Returns
        -------
        float
            ICIR value, or 0.0 if the series has insufficient data.
        """
        ic = np.asarray(ic_series, dtype=np.float64)
        ic = ic[~np.isnan(ic)]
        if len(ic) < 2:
            return 0.0
        std = np.std(ic, ddof=1)
        if std < 1e-15:
            return 0.0
        return float(np.mean(ic) / std)

    @staticmethod
    def compute_sharpe(
        returns: np.ndarray,
        rf: float = 0.0,
    ) -> float:
        """Compute annualised Sharpe ratio.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of *daily* portfolio returns.
        rf : float
            Daily risk-free rate (default 0).

        Returns
        -------
        float
            Annualised Sharpe ratio.
        """
        ret = np.asarray(returns, dtype=np.float64)
        ret = ret[~np.isnan(ret)]
        if len(ret) < 2:
            return 0.0
        excess = ret - rf
        std = np.std(excess, ddof=1)
        if std < 1e-15:
            return 0.0
        daily_sharpe = np.mean(excess) / std
        return float(daily_sharpe * np.sqrt(TRADING_DAYS_PER_YEAR))

    @staticmethod
    def compute_mdd(cumulative_returns: np.ndarray) -> float:
        """Compute maximum drawdown from a cumulative return series.

        Parameters
        ----------
        cumulative_returns : np.ndarray
            1-D array of cumulative returns (e.g. starting at 1.0).

        Returns
        -------
        float
            Maximum drawdown as a positive fraction (e.g. 0.15 = 15%).
        """
        cum = np.asarray(cumulative_returns, dtype=np.float64)
        if len(cum) < 2:
            return 0.0
        running_max = np.maximum.accumulate(cum)
        drawdowns = (running_max - cum) / np.where(
            running_max > 0, running_max, 1.0
        )
        return float(np.nanmax(drawdowns))

    @staticmethod
    def compute_arr(returns: np.ndarray) -> float:
        """Compute annualised rate of return.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of daily portfolio returns.

        Returns
        -------
        float
            Annualised return (geometric).
        """
        ret = np.asarray(returns, dtype=np.float64)
        ret = ret[~np.isnan(ret)]
        if len(ret) == 0:
            return 0.0
        cumulative = np.prod(1.0 + ret)
        n_years = len(ret) / TRADING_DAYS_PER_YEAR
        if n_years < 1e-6:
            return 0.0
        arr = cumulative ** (1.0 / n_years) - 1.0
        return float(arr)

    @staticmethod
    def compute_calmar(arr: float, mdd: float) -> float:
        """Compute Calmar ratio = ARR / MDD.

        Parameters
        ----------
        arr : float
            Annualised rate of return.
        mdd : float
            Maximum drawdown (positive fraction).

        Returns
        -------
        float
            Calmar ratio, or 0.0 if MDD is near zero.
        """
        if abs(mdd) < 1e-10:
            return 0.0
        return arr / mdd

    # ------------------------------------------------------------------
    # High-level backtest runner
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        factor_code: str,
        data: Dict[str, Any],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> BacktestResult:
        """Execute a full backtest for a single factor.

        This method expects ``data`` to contain pre-computed arrays so that
        the backtester itself remains purely vectorised and deterministic.
        Factor code execution (which may involve Qlib expressions or
        arbitrary Python) is handled upstream by the ``CodeSynthesizer``.

        Parameters
        ----------
        factor_code : str
            The factor's source code (stored in the result for provenance).
        data : dict
            Must contain:
            - ``"predictions"`` : np.ndarray of shape (n_dates, n_assets)
            - ``"returns"``     : np.ndarray of shape (n_dates, n_assets)
            Optionally:
            - ``"dates"`` : list/array of date objects for each row
        start_date, end_date : date, optional
            Restrict the evaluation window.

        Returns
        -------
        BacktestResult
            Populated with all six core metrics and the IC time-series.

        Raises
        ------
        KeyError
            If required keys are missing from *data*.
        """
        predictions = np.asarray(data["predictions"], dtype=np.float64)
        returns = np.asarray(data["returns"], dtype=np.float64)

        if predictions.shape != returns.shape:
            raise ValueError(
                f"predictions shape {predictions.shape} != "
                f"returns shape {returns.shape}"
            )

        # Date filtering.
        dates = data.get("dates")
        if dates is not None and (start_date or end_date):
            dates_arr = np.array(dates)
            mask = np.ones(len(dates_arr), dtype=bool)
            if start_date is not None:
                mask &= dates_arr >= np.datetime64(start_date)
            if end_date is not None:
                mask &= dates_arr <= np.datetime64(end_date)
            predictions = predictions[mask]
            returns = returns[mask]
            dates = dates_arr[mask]

        # --- IC series ---
        ic_series = self.compute_ic(predictions, returns)
        ic_mean = float(np.nanmean(ic_series))
        icir = self.compute_icir(ic_series)

        # --- Portfolio returns (simple long-short quintile) ---
        port_returns = self._long_short_returns(predictions, returns)

        # Apply transaction costs.
        if self.cost_bps > 0:
            daily_cost = self.cost_bps / 10_000.0
            port_returns = port_returns - daily_cost

        sharpe = self.compute_sharpe(port_returns)
        arr = self.compute_arr(port_returns)

        cum_returns = np.cumprod(1.0 + port_returns)
        mdd = self.compute_mdd(cum_returns)
        calmar = self.compute_calmar(arr, mdd)

        result = BacktestResult(
            ic=ic_mean,
            icir=icir,
            sharpe=sharpe,
            mdd=mdd,
            arr=arr,
            calmar=calmar,
            passed_ldt=False,  # filled in later by TIF
            paper_trade_passed=False,
        )

        logger.info(
            "Backtest complete: IC=%.4f  ICIR=%.4f  Sharpe=%.2f  "
            "MDD=%.2f%%  ARR=%.2f%%  Calmar=%.2f",
            ic_mean,
            icir,
            sharpe,
            mdd * 100,
            arr * 100,
            calmar,
        )

        return result

    def check_acceptance(self, result: BacktestResult) -> bool:
        """Check whether a ``BacktestResult`` meets all acceptance thresholds.

        Returns
        -------
        bool
            ``True`` if the factor passes all gates.
        """
        passes = (
            result.ic >= self.ic_threshold
            and result.icir >= self.icir_threshold
            and result.sharpe >= self.sharpe_threshold
            and result.mdd <= self.mdd_threshold
        )
        if not passes:
            reasons = []
            if result.ic < self.ic_threshold:
                reasons.append(
                    f"IC {result.ic:.4f} < {self.ic_threshold}"
                )
            if result.icir < self.icir_threshold:
                reasons.append(
                    f"ICIR {result.icir:.4f} < {self.icir_threshold}"
                )
            if result.sharpe < self.sharpe_threshold:
                reasons.append(
                    f"Sharpe {result.sharpe:.2f} < {self.sharpe_threshold}"
                )
            if result.mdd > self.mdd_threshold:
                reasons.append(
                    f"MDD {result.mdd:.2%} > {self.mdd_threshold:.0%}"
                )
            logger.info(
                "Factor REJECTED: %s", "; ".join(reasons)
            )
        else:
            logger.info("Factor ACCEPTED on all metric gates.")
        return passes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _long_short_returns(
        predictions: np.ndarray,
        returns: np.ndarray,
    ) -> np.ndarray:
        """Construct a simple long-top/short-bottom quintile portfolio.

        For each date, assets in the top quintile of predictions are held
        long with equal weight and assets in the bottom quintile are held
        short.  The portfolio return is (long_avg - short_avg) / 2.

        Parameters
        ----------
        predictions : np.ndarray  (n_dates, n_assets)
        returns : np.ndarray      (n_dates, n_assets)

        Returns
        -------
        np.ndarray
            1-D array of daily long-short portfolio returns.
        """
        n_dates, n_assets = predictions.shape
        port_ret = np.zeros(n_dates, dtype=np.float64)

        for t in range(n_dates):
            pred_t = predictions[t]
            ret_t = returns[t]

            valid = ~(np.isnan(pred_t) | np.isnan(ret_t))
            n_valid = valid.sum()
            if n_valid < 5:
                port_ret[t] = 0.0
                continue

            # Quintile cutoffs.
            pred_valid = pred_t[valid]
            ret_valid = ret_t[valid]
            q80 = np.percentile(pred_valid, 80)
            q20 = np.percentile(pred_valid, 20)

            long_mask = pred_valid >= q80
            short_mask = pred_valid <= q20

            long_ret = np.mean(ret_valid[long_mask]) if long_mask.any() else 0.0
            short_ret = np.mean(ret_valid[short_mask]) if short_mask.any() else 0.0

            port_ret[t] = (long_ret - short_ret) / 2.0

        return port_ret
