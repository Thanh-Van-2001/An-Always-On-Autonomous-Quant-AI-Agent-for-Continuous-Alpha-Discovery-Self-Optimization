"""
Factor Crowding Detection.

Monitors pairwise correlation among live alpha-factor return streams
to detect *crowding* -- the condition where multiple factors in the
portfolio are effectively trading the same signal.  When crowding is
detected, the module suggests rebalancing weights to restore diversity.

Section 4.3 of the KDD 2026 paper specifies:
- A factor pool is crowded when the mean off-diagonal Pearson
  correlation exceeds rho_crowd (default 0.65).
- The rebalancing heuristic targets a user-specified correlation
  ceiling by down-weighting the most correlated factors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CrowdingReport:
    """Summary of a crowding analysis.

    Attributes:
        is_crowded: Whether the mean pairwise correlation exceeds the
                    threshold.
        mean_correlation: Mean of the upper-triangle off-diagonal
                         entries of the Pearson correlation matrix.
        max_correlation: Maximum off-diagonal correlation.
        correlation_matrix: Full N x N Pearson correlation matrix.
        crowded_pairs: List of (i, j, rho) tuples for pairs exceeding
                       the threshold individually.
    """

    is_crowded: bool = False
    mean_correlation: float = 0.0
    max_correlation: float = 0.0
    correlation_matrix: Optional[np.ndarray] = None
    crowded_pairs: List[Tuple[int, int, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.crowded_pairs is None:
            self.crowded_pairs = []


class CrowdingDetector:
    """Detects factor crowding via pairwise return correlation.

    Parameters
    ----------
    rho_crowd : float
        Mean pairwise correlation threshold above which the factor pool
        is considered crowded.  Default 0.65.
    """

    def __init__(self, rho_crowd: float = 0.65) -> None:
        if not 0.0 < rho_crowd <= 1.0:
            raise ValueError("rho_crowd must be in (0, 1]")
        self.rho_crowd = rho_crowd

    # ------------------------------------------------------------------
    # Correlation computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_pairwise_correlation(
        factor_returns: np.ndarray,
    ) -> np.ndarray:
        """Compute the Pearson correlation matrix of factor returns.

        Parameters
        ----------
        factor_returns : np.ndarray
            2-D array of shape ``(T, N)`` where *T* is the number of
            time steps and *N* is the number of factors.

        Returns
        -------
        np.ndarray
            Symmetric ``(N, N)`` Pearson correlation matrix.

        Raises
        ------
        ValueError
            If the input has fewer than 2 time steps or fewer than 2
            factors.
        """
        X = np.asarray(factor_returns, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("factor_returns must be a 2-D array")
        T, N = X.shape
        if T < 2:
            raise ValueError("Need at least 2 time steps for correlation")
        if N < 2:
            raise ValueError("Need at least 2 factors for pairwise correlation")

        # np.corrcoef expects rows = variables, so transpose.
        corr = np.corrcoef(X, rowvar=False)

        # Handle NaN from constant columns (zero-variance factors).
        corr = np.nan_to_num(corr, nan=0.0)

        return corr

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_crowding(
        self,
        factor_returns: np.ndarray,
        rho_crowd: Optional[float] = None,
    ) -> CrowdingReport:
        """Analyse factor returns for crowding.

        Parameters
        ----------
        factor_returns : np.ndarray
            Shape ``(T, N)``.
        rho_crowd : float, optional
            Override for the crowding threshold.

        Returns
        -------
        CrowdingReport
        """
        threshold = rho_crowd if rho_crowd is not None else self.rho_crowd
        corr = self.compute_pairwise_correlation(factor_returns)
        N = corr.shape[0]

        # Extract upper triangle (excluding diagonal)
        iu = np.triu_indices(N, k=1)
        upper_vals = corr[iu]

        mean_corr = float(np.mean(np.abs(upper_vals))) if len(upper_vals) > 0 else 0.0
        max_corr = float(np.max(np.abs(upper_vals))) if len(upper_vals) > 0 else 0.0

        # Identify individually crowded pairs
        crowded_pairs: List[Tuple[int, int, float]] = []
        for idx in range(len(iu[0])):
            i, j = int(iu[0][idx]), int(iu[1][idx])
            rho = float(corr[i, j])
            if abs(rho) >= threshold:
                crowded_pairs.append((i, j, rho))

        is_crowded = mean_corr >= threshold

        if is_crowded:
            logger.warning(
                "Factor crowding detected: mean |rho| = %.4f "
                "(threshold %.4f), %d crowded pairs",
                mean_corr,
                threshold,
                len(crowded_pairs),
            )

        return CrowdingReport(
            is_crowded=is_crowded,
            mean_correlation=mean_corr,
            max_correlation=max_corr,
            correlation_matrix=corr,
            crowded_pairs=crowded_pairs,
        )

    # ------------------------------------------------------------------
    # Rebalancing suggestion
    # ------------------------------------------------------------------

    def suggest_rebalancing(
        self,
        factor_returns: np.ndarray,
        target_correlation: float = 0.4,
        current_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Suggest portfolio weights that reduce factor crowding.

        The heuristic iteratively down-weights the factor that
        contributes most to the weighted average correlation, stopping
        when the weighted mean correlation drops below
        *target_correlation* or a minimum weight floor is reached.

        Parameters
        ----------
        factor_returns : np.ndarray
            Shape ``(T, N)``.
        target_correlation : float
            Desired ceiling for mean weighted pairwise correlation.
        current_weights : np.ndarray, optional
            Starting weights, shape ``(N,)``.  If ``None``, equal
            weights are used.

        Returns
        -------
        np.ndarray
            Suggested weights of shape ``(N,)``, summing to 1.0.
        """
        corr = self.compute_pairwise_correlation(factor_returns)
        N = corr.shape[0]

        if current_weights is not None:
            w = np.asarray(current_weights, dtype=np.float64).copy()
            if len(w) != N:
                raise ValueError(
                    f"current_weights length ({len(w)}) != number of "
                    f"factors ({N})"
                )
        else:
            w = np.ones(N, dtype=np.float64) / N

        abs_corr = np.abs(corr)
        np.fill_diagonal(abs_corr, 0.0)

        min_weight = 1e-4
        max_iterations = 100

        for _ in range(max_iterations):
            # Weighted mean correlation
            # For each factor i, its "crowding contribution" is:
            #   c_i = sum_j (w_j * |rho_{i,j}|) / (1 - w_i)
            # We weight by w_i to get the portfolio-level crowding.
            crowding_contrib = abs_corr @ w  # shape (N,)
            weighted_mean = float(w @ crowding_contrib)

            if weighted_mean <= target_correlation:
                break

            # Down-weight the worst offender
            worst = int(np.argmax(crowding_contrib))
            w[worst] = max(w[worst] * 0.8, min_weight)

            # Re-normalise
            w_sum = w.sum()
            if w_sum > 0:
                w /= w_sum

        return w
