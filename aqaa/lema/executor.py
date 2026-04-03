"""
Order Execution and Pre-Trade Risk Management.

Provides a simulated TWAP (Time-Weighted Average Price) execution
engine and a pre-trade risk-check layer that enforces position limits,
sector-concentration caps, and VaR (Value at Risk) budgets.

This is a *simulation-first* implementation: actual broker connectivity
is abstracted behind the order interface so that the same logic can run
in backtest, paper-trading, and live modes without modification.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aqaa.core.types import TradeOrder

logger = logging.getLogger(__name__)


# ======================================================================
# Risk limits
# ======================================================================

@dataclass
class RiskLimits:
    """Pre-trade risk limits enforced before order submission.

    Attributes:
        max_position: Maximum absolute position size per instrument
                      (in notional or share terms).
        max_sector_pct: Maximum portfolio weight in any single sector
                        (e.g. 0.25 = 25 %).
        var_budget: Maximum incremental 1-day VaR (in the same units as
                    P&L) that a single order may add to the portfolio.
        max_order_notional: Maximum notional value per single order.
        max_portfolio_gross: Maximum gross portfolio exposure.
    """

    max_position: float = 1_000_000.0
    max_sector_pct: float = 0.25
    var_budget: float = 50_000.0
    max_order_notional: float = 500_000.0
    max_portfolio_gross: float = 10_000_000.0


# ======================================================================
# Portfolio snapshot (for risk checks)
# ======================================================================

@dataclass
class PortfolioSnapshot:
    """Lightweight view of the current portfolio state.

    Attributes:
        positions: Mapping ``{symbol: signed_quantity}``.
        sector_map: Mapping ``{symbol: sector_label}``.
        prices: Mapping ``{symbol: last_price}``.
        total_nav: Total net asset value of the portfolio.
    """

    positions: Dict[str, float] = field(default_factory=dict)
    sector_map: Dict[str, str] = field(default_factory=dict)
    prices: Dict[str, float] = field(default_factory=dict)
    total_nav: float = 10_000_000.0


# ======================================================================
# Risk-check result
# ======================================================================

@dataclass
class RiskCheckResult:
    """Outcome of a single pre-trade risk check.

    Attributes:
        passed: ``True`` if the order is within all limits.
        violations: List of human-readable violation descriptions.
    """

    passed: bool = True
    violations: List[str] = field(default_factory=list)


# ======================================================================
# TWAP execution result
# ======================================================================

@dataclass
class TWAPResult:
    """Result of a simulated TWAP execution.

    Attributes:
        order: The original order.
        child_orders: List of time-sliced child orders.
        avg_price: Volume-weighted average execution price.
        total_quantity: Total quantity filled.
        slices: Number of TWAP slices used.
        duration_minutes: Requested execution duration.
    """

    order: TradeOrder
    child_orders: List[TradeOrder] = field(default_factory=list)
    avg_price: float = 0.0
    total_quantity: float = 0.0
    slices: int = 0
    duration_minutes: int = 0


# ======================================================================
# Order Executor
# ======================================================================

class OrderExecutor:
    """Simulated order execution engine with pre-trade risk checks.

    Parameters
    ----------
    risk_limits : RiskLimits, optional
        Default risk limits.  Can be overridden per call.
    default_slippage_bps : float
        Default slippage model in basis points per slice.
        Used in TWAP simulation to perturb fill prices.
    random_seed : int or None
        Seed for the slippage RNG.  ``None`` for non-deterministic.
    """

    def __init__(
        self,
        risk_limits: Optional[RiskLimits] = None,
        default_slippage_bps: float = 1.0,
        random_seed: Optional[int] = None,
    ) -> None:
        self.risk_limits = risk_limits or RiskLimits()
        self.default_slippage_bps = default_slippage_bps
        self._rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # TWAP execution
    # ------------------------------------------------------------------

    def execute_twap(
        self,
        orders: List[TradeOrder],
        duration_minutes: int = 30,
        num_slices: Optional[int] = None,
    ) -> List[TWAPResult]:
        """Simulate TWAP execution for a batch of parent orders.

        Each parent order is split into *num_slices* equal-sized child
        orders spaced evenly over *duration_minutes*.  A random
        slippage is applied to each slice to simulate market impact.

        Parameters
        ----------
        orders : list of TradeOrder
            Parent orders to execute.
        duration_minutes : int
            Total execution window in minutes (default 30).
        num_slices : int, optional
            Number of time slices.  Defaults to ``max(1, duration_minutes // 5)``.

        Returns
        -------
        list of TWAPResult
            One result per parent order.
        """
        if num_slices is None:
            num_slices = max(1, duration_minutes // 5)

        results: List[TWAPResult] = []

        for order in orders:
            child_orders: List[TradeOrder] = []
            slice_qty = order.quantity / num_slices

            total_cost = 0.0
            total_filled = 0.0

            base_price = order.price if order.price > 0 else 100.0

            interval = timedelta(minutes=duration_minutes / num_slices)
            slice_time = order.timestamp

            for s in range(num_slices):
                # Simulate slippage: uniform in [-slippage, +slippage]
                slippage_frac = (
                    self._rng.uniform(-1, 1) * self.default_slippage_bps * 1e-4
                )
                # For buys, slippage tends upward; for sells, downward
                direction = 1.0 if order.side == "BUY" else -1.0
                fill_price = base_price * (1.0 + direction * abs(slippage_frac))

                child = TradeOrder(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=slice_qty,
                    price=fill_price,
                    order_type="LIMIT",
                    timestamp=slice_time,
                    execution_algo="TWAP",
                )
                child_orders.append(child)

                total_cost += fill_price * slice_qty
                total_filled += slice_qty
                slice_time = slice_time + interval

            avg_price = total_cost / total_filled if total_filled > 0 else 0.0

            results.append(
                TWAPResult(
                    order=order,
                    child_orders=child_orders,
                    avg_price=avg_price,
                    total_quantity=total_filled,
                    slices=num_slices,
                    duration_minutes=duration_minutes,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    def risk_check(
        self,
        order: TradeOrder,
        portfolio: PortfolioSnapshot,
        limits: Optional[RiskLimits] = None,
    ) -> RiskCheckResult:
        """Run pre-trade risk checks on a single order.

        Enforces:
        1. **Position limit** -- post-trade position must not exceed
           ``max_position``.
        2. **Sector concentration** -- post-trade sector weight must
           not exceed ``max_sector_pct``.
        3. **VaR budget** -- incremental notional must not exceed
           ``var_budget``.
        4. **Single-order notional** -- order notional must not exceed
           ``max_order_notional``.
        5. **Gross exposure** -- portfolio gross must not exceed
           ``max_portfolio_gross``.

        Parameters
        ----------
        order : TradeOrder
            The order to check.
        portfolio : PortfolioSnapshot
            Current portfolio state.
        limits : RiskLimits, optional
            Override limits.

        Returns
        -------
        RiskCheckResult
        """
        lim = limits or self.risk_limits
        violations: List[str] = []

        price = order.price if order.price > 0 else portfolio.prices.get(order.symbol, 100.0)
        order_notional = abs(order.quantity * price)
        signed_qty = order.quantity if order.side == "BUY" else -order.quantity

        # 1. Position limit
        current_pos = portfolio.positions.get(order.symbol, 0.0)
        new_pos = current_pos + signed_qty
        if abs(new_pos) > lim.max_position:
            violations.append(
                f"Position limit: post-trade |position| "
                f"{abs(new_pos):.2f} > max {lim.max_position:.2f}"
            )

        # 2. Sector concentration
        sector = portfolio.sector_map.get(order.symbol)
        if sector is not None and portfolio.total_nav > 0:
            # Sum notional in the same sector
            sector_notional = 0.0
            for sym, pos in portfolio.positions.items():
                if portfolio.sector_map.get(sym) == sector:
                    sym_price = portfolio.prices.get(sym, 100.0)
                    sector_notional += abs(pos * sym_price)
            # Add this order's contribution
            sector_notional += order_notional
            sector_pct = sector_notional / portfolio.total_nav
            if sector_pct > lim.max_sector_pct:
                violations.append(
                    f"Sector concentration: {sector} at "
                    f"{sector_pct:.2%} > max {lim.max_sector_pct:.2%}"
                )

        # 3. VaR budget (simplified: treat order notional as a proxy
        #    for incremental VaR; a real implementation would use the
        #    portfolio covariance matrix).
        #    Here we use: incremental VaR ~ notional * 2.33 * daily_vol
        #    With a conservative daily vol of 2%.
        daily_vol = 0.02
        z_99 = 2.326  # 99 % quantile
        incremental_var = order_notional * z_99 * daily_vol
        if incremental_var > lim.var_budget:
            violations.append(
                f"VaR budget: incremental VaR "
                f"{incremental_var:,.2f} > budget {lim.var_budget:,.2f}"
            )

        # 4. Single-order notional
        if order_notional > lim.max_order_notional:
            violations.append(
                f"Order notional: {order_notional:,.2f} > "
                f"max {lim.max_order_notional:,.2f}"
            )

        # 5. Gross exposure
        current_gross = sum(
            abs(pos * portfolio.prices.get(sym, 100.0))
            for sym, pos in portfolio.positions.items()
        )
        new_gross = current_gross + order_notional
        if new_gross > lim.max_portfolio_gross:
            violations.append(
                f"Gross exposure: {new_gross:,.2f} > "
                f"max {lim.max_portfolio_gross:,.2f}"
            )

        passed = len(violations) == 0

        if not passed:
            logger.warning(
                "Risk check FAILED for %s %s %.0f %s: %s",
                order.side,
                order.symbol,
                order.quantity,
                order.order_type,
                "; ".join(violations),
            )

        return RiskCheckResult(passed=passed, violations=violations)

    def batch_risk_check(
        self,
        orders: List[TradeOrder],
        portfolio: PortfolioSnapshot,
        limits: Optional[RiskLimits] = None,
    ) -> List[RiskCheckResult]:
        """Run risk checks on a batch of orders sequentially.

        Each order is checked against the *original* portfolio snapshot
        (not the hypothetical post-trade state of previous orders).
        For a cumulative check, callers should update the snapshot
        between calls.

        Parameters
        ----------
        orders : list of TradeOrder
        portfolio : PortfolioSnapshot
        limits : RiskLimits, optional

        Returns
        -------
        list of RiskCheckResult
            One result per order, in the same order.
        """
        return [
            self.risk_check(order, portfolio, limits) for order in orders
        ]
