"""
Regime-aware mean-variance portfolio optimiser.

Solves the constrained optimisation problem:

    w* = argmin_w  w^T Sigma w  -  lambda * mu^T w
    subject to:  ||w||_1 <= C,   w >= 0

where *lambda* is a risk-aversion parameter that the optimiser adjusts
automatically based on the current market regime (more conservative in
bear markets, more aggressive in bull markets) and Sigma is optionally
penalised for correlated factors to reduce crowding risk.

The solver delegates to :func:`scipy.optimize.minimize` with the SLSQP
method.

Typical usage::

    opt = PortfolioOptimizer(base_lambda=1.0, gross_exposure_limit=1.5)
    weights = opt.optimize_weights(
        factor_names=["mom", "val", "qual"],
        covariance_matrix=cov,
        expected_returns=mu,
        regime=MarketRegime.BULL,
    )
    risk = opt.compute_risk_metrics(weights, cov)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as sopt

from aqaa.core.types import MarketRegime

logger = logging.getLogger(__name__)

# Default regime multipliers for the risk-aversion parameter lambda.
# Higher lambda => more weight on expected returns vs. risk.
_DEFAULT_REGIME_LAMBDA_MULT: Dict[MarketRegime, float] = {
    MarketRegime.BULL: 1.5,      # more aggressive
    MarketRegime.SIDEWAYS: 1.0,  # baseline
    MarketRegime.BEAR: 0.4,      # much more conservative
}


@dataclass
class PortfolioOptimizer:
    """Regime-aware constrained mean-variance optimiser.

    Parameters
    ----------
    base_lambda : float
        Baseline risk-aversion parameter.
    gross_exposure_limit : float
        Maximum sum of absolute weights (L1 constraint *C*).
    crowding_penalty : float
        Coefficient [0, 1] that blends the raw covariance matrix with a
        correlation-penalised version to discourage correlated factor
        loadings.  0 = no penalty, 1 = full correlation penalty.
    regime_lambda_mult : dict | None
        Override mapping from :class:`MarketRegime` to the lambda
        multiplier.  Falls back to the built-in defaults.
    """

    base_lambda: float = 1.0
    gross_exposure_limit: float = 1.5
    crowding_penalty: float = 0.3
    regime_lambda_mult: Optional[Dict[MarketRegime, float]] = None

    def __post_init__(self) -> None:
        if self.regime_lambda_mult is None:
            self.regime_lambda_mult = dict(_DEFAULT_REGIME_LAMBDA_MULT)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize_weights(
        self,
        factor_names: list[str],
        covariance_matrix: NDArray,
        expected_returns: NDArray,
        regime: MarketRegime = MarketRegime.SIDEWAYS,
    ) -> Dict[str, float]:
        """Compute optimal long-only factor weights.

        Parameters
        ----------
        factor_names : list[str]
            Names corresponding to each row / column of the matrices.
        covariance_matrix : (N, N) array
            Factor covariance matrix Sigma.
        expected_returns : (N,) array
            Vector of expected returns mu.
        regime : MarketRegime
            Current market regime for lambda adjustment.

        Returns
        -------
        dict[str, float]
            Mapping from factor name to optimised weight.
        """
        cov = np.asarray(covariance_matrix, dtype=np.float64)
        mu = np.asarray(expected_returns, dtype=np.float64).ravel()
        n = len(mu)

        if cov.shape != (n, n):
            raise ValueError(
                f"Covariance shape {cov.shape} does not match "
                f"expected_returns length {n}."
            )
        if len(factor_names) != n:
            raise ValueError(
                f"factor_names length {len(factor_names)} != {n}."
            )

        # Regime-adjusted lambda.
        lam = self.base_lambda * self.regime_lambda_mult.get(regime, 1.0)

        # Optionally penalise correlated factors.
        sigma = self._apply_crowding_penalty(cov)

        # --- Objective: min  w^T Sigma w  -  lambda * mu^T w ---
        def objective(w: NDArray) -> float:
            return float(w @ sigma @ w - lam * mu @ w)

        def gradient(w: NDArray) -> NDArray:
            return 2.0 * sigma @ w - lam * mu

        # --- Constraints ---
        # L1 norm <= gross_exposure_limit  (since w >= 0, ||w||_1 = sum(w)).
        constraints = [
            {
                "type": "ineq",
                "fun": lambda w: self.gross_exposure_limit - np.sum(w),
            },
        ]

        # --- Bounds: w_i >= 0 ---
        bounds = [(0.0, None)] * n

        # Initial guess: equal weight within the exposure budget.
        w0 = np.full(n, self.gross_exposure_limit / n)

        result = sopt.minimize(
            objective,
            w0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )

        if not result.success:
            logger.warning("SLSQP did not converge: %s", result.message)

        weights = np.maximum(result.x, 0.0)  # clip numerical noise

        # Normalise if the solver slightly exceeded the exposure limit.
        total = weights.sum()
        if total > self.gross_exposure_limit:
            weights *= self.gross_exposure_limit / total

        return dict(zip(factor_names, weights.tolist()))

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    def compute_risk_metrics(
        self,
        weights: Dict[str, float],
        covariance_matrix: NDArray,
        *,
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """Compute portfolio-level risk statistics.

        Parameters
        ----------
        weights : dict[str, float]
            Factor-name -> weight mapping (must match covariance ordering).
        covariance_matrix : (N, N) array
            Covariance matrix aligned with the weight vector.
        confidence_level : float
            VaR / ES quantile (default 95%).

        Returns
        -------
        dict with keys:
            portfolio_variance, portfolio_vol, VaR_parametric,
            expected_shortfall.
        """
        w = np.array(list(weights.values()), dtype=np.float64)
        cov = np.asarray(covariance_matrix, dtype=np.float64)

        port_var = float(w @ cov @ w)
        port_vol = float(np.sqrt(max(port_var, 0.0)))

        # Parametric VaR assuming normal returns.
        from scipy.stats import norm

        z = norm.ppf(confidence_level)
        var = z * port_vol  # positive number = loss at CL
        # Expected shortfall (CVaR) for normal distribution.
        es = port_vol * norm.pdf(z) / (1.0 - confidence_level)

        return {
            "portfolio_variance": port_var,
            "portfolio_vol": port_vol,
            "VaR_parametric": float(var),
            "expected_shortfall": float(es),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_crowding_penalty(self, cov: NDArray) -> NDArray:
        """Blend the covariance with a correlation-inflated version.

        When ``crowding_penalty > 0``, off-diagonal elements of the
        covariance matrix are *increased* proportionally to the absolute
        correlation, discouraging the optimiser from loading on highly
        correlated factors.
        """
        if self.crowding_penalty <= 0.0:
            return cov

        n = cov.shape[0]
        vols = np.sqrt(np.diag(cov))
        # Avoid division by zero.
        vols = np.where(vols > 0, vols, 1e-10)

        corr = cov / np.outer(vols, vols)
        np.fill_diagonal(corr, 1.0)

        # Penalised covariance: inflate off-diagonal by |rho|.
        penalty = np.abs(corr)
        np.fill_diagonal(penalty, 0.0)

        penalised_cov = cov + self.crowding_penalty * penalty * np.outer(vols, vols)

        # Ensure symmetry.
        penalised_cov = 0.5 * (penalised_cov + penalised_cov.T)

        return penalised_cov
