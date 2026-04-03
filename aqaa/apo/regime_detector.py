"""
Hidden Markov Model regime detector.

Implements a 3-state Gaussian HMM (bull / bear / sideways) using the
Baum-Welch (EM) algorithm for parameter estimation and the forward
algorithm for online regime-probability updates.  The implementation is
self-contained (numpy only, no external HMM library required).

Typical usage::

    detector = RegimeDetector(n_iter=50, tol=1e-4)
    detector.fit(daily_returns)
    current_regime = detector.predict_regime(recent_window)
    detector.update_online(new_return)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from aqaa.core.types import MarketRegime

logger = logging.getLogger(__name__)

# Canonical ordering of hidden states.
_STATE_TO_REGIME = {
    0: MarketRegime.BULL,
    1: MarketRegime.BEAR,
    2: MarketRegime.SIDEWAYS,
}


# ---------------------------------------------------------------------------
# Helper: univariate Gaussian PDF (log-space for numerical stability)
# ---------------------------------------------------------------------------

def _log_normal_pdf(x: NDArray, mu: float, sigma: float) -> NDArray:
    """Compute log N(x | mu, sigma) element-wise."""
    var = sigma ** 2
    return -0.5 * (np.log(2.0 * np.pi * var) + (x - mu) ** 2 / var)


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------

@dataclass
class RegimeDetector:
    """3-state Gaussian Hidden Markov Model for market-regime detection.

    Parameters
    ----------
    n_iter : int
        Maximum number of Baum-Welch iterations.
    tol : float
        Convergence tolerance on the log-likelihood improvement.
    random_seed : int | None
        Seed for reproducible parameter initialisation.
    """

    n_iter: int = 100
    tol: float = 1e-4
    random_seed: Optional[int] = 42
    n_states: int = field(default=3, init=False)

    # --- Learned parameters (populated by fit) ---
    initial_probs_: Optional[NDArray] = field(default=None, init=False, repr=False)
    transition_matrix_: Optional[NDArray] = field(default=None, init=False, repr=False)
    means_: Optional[NDArray] = field(default=None, init=False, repr=False)
    stds_: Optional[NDArray] = field(default=None, init=False, repr=False)

    # --- Runtime state ---
    _state_probs: Optional[NDArray] = field(default=None, init=False, repr=False)
    regime_history: List[MarketRegime] = field(default_factory=list, init=False, repr=False)
    is_fitted: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_params(self, returns: NDArray) -> None:
        """Heuristic initialisation of HMM parameters from data."""
        rng = np.random.default_rng(self.random_seed)
        k = self.n_states

        # Uniform initial state probabilities.
        self.initial_probs_ = np.full(k, 1.0 / k)

        # Slightly sticky transition matrix (prefer staying in same state).
        off_diag = 0.1
        self.transition_matrix_ = np.full((k, k), off_diag / (k - 1))
        np.fill_diagonal(self.transition_matrix_, 1.0 - off_diag)

        # K-means-style initialisation for emission means/stds.
        sorted_returns = np.sort(returns)
        n = len(sorted_returns)
        terciles = np.array_split(sorted_returns, k)

        # Order: bull (highest returns), bear (lowest), sideways (middle).
        self.means_ = np.array([
            terciles[2].mean(),   # bull
            terciles[0].mean(),   # bear
            terciles[1].mean(),   # sideways
        ])
        self.stds_ = np.array([
            max(terciles[2].std(), 1e-6),
            max(terciles[0].std(), 1e-6),
            max(terciles[1].std(), 1e-6),
        ])

    # ------------------------------------------------------------------
    # Forward-backward (Baum-Welch E-step helpers)
    # ------------------------------------------------------------------

    def _emission_log_probs(self, returns: NDArray) -> NDArray:
        """Compute log B[j, t] = log p(o_t | state j)."""
        T = len(returns)
        log_B = np.zeros((self.n_states, T))
        for j in range(self.n_states):
            log_B[j] = _log_normal_pdf(returns, self.means_[j], self.stds_[j])
        return log_B

    def _forward(self, log_B: NDArray) -> Tuple[NDArray, NDArray]:
        """Scaled forward pass.

        Returns
        -------
        alpha : (K, T) array of forward probabilities (scaled).
        scales : (T,) normalisation constants.
        """
        K, T = log_B.shape
        alpha = np.zeros((K, T))
        scales = np.zeros(T)

        # t = 0
        alpha[:, 0] = self.initial_probs_ * np.exp(log_B[:, 0])
        scales[0] = alpha[:, 0].sum()
        if scales[0] == 0.0:
            scales[0] = 1e-300
        alpha[:, 0] /= scales[0]

        A = self.transition_matrix_  # (K, K)
        for t in range(1, T):
            alpha[:, t] = (A.T @ alpha[:, t - 1]) * np.exp(log_B[:, t])
            scales[t] = alpha[:, t].sum()
            if scales[t] == 0.0:
                scales[t] = 1e-300
            alpha[:, t] /= scales[t]

        return alpha, scales

    def _backward(self, log_B: NDArray, scales: NDArray) -> NDArray:
        """Scaled backward pass."""
        K, T = log_B.shape
        beta = np.zeros((K, T))
        beta[:, T - 1] = 1.0

        A = self.transition_matrix_
        for t in range(T - 2, -1, -1):
            beta[:, t] = A @ (np.exp(log_B[:, t + 1]) * beta[:, t + 1])
            if scales[t + 1] > 0.0:
                beta[:, t] /= scales[t + 1]

        return beta

    # ------------------------------------------------------------------
    # Baum-Welch (EM)
    # ------------------------------------------------------------------

    def fit(self, returns_series: NDArray) -> "RegimeDetector":
        """Fit the HMM to an array of return observations via Baum-Welch.

        Parameters
        ----------
        returns_series : 1-D array-like
            Sequence of daily (or intra-day) returns.

        Returns
        -------
        self
        """
        returns = np.asarray(returns_series, dtype=np.float64).ravel()
        if len(returns) < self.n_states:
            raise ValueError(
                f"Need at least {self.n_states} observations, got {len(returns)}."
            )

        self._init_params(returns)

        T = len(returns)
        K = self.n_states
        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            log_B = self._emission_log_probs(returns)
            alpha, scales = self._forward(log_B)
            beta = self._backward(log_B, scales)

            # Log-likelihood from forward scales.
            log_likelihood = np.sum(np.log(scales + 1e-300))

            # Convergence check.
            if abs(log_likelihood - prev_ll) < self.tol:
                logger.info(
                    "Baum-Welch converged at iteration %d (LL=%.6f)",
                    iteration,
                    log_likelihood,
                )
                break
            prev_ll = log_likelihood

            # --- E-step: posterior state probabilities ---
            gamma = alpha * beta  # (K, T)
            gamma_sum = gamma.sum(axis=0, keepdims=True)
            gamma_sum[gamma_sum == 0.0] = 1e-300
            gamma /= gamma_sum

            # --- E-step: pairwise state posteriors xi(i, j, t) ---
            A = self.transition_matrix_
            xi_sum = np.zeros((K, K))
            for t in range(T - 1):
                denom = 0.0
                tmp = np.zeros((K, K))
                for i in range(K):
                    for j in range(K):
                        tmp[i, j] = (
                            alpha[i, t]
                            * A[i, j]
                            * np.exp(log_B[j, t + 1])
                            * beta[j, t + 1]
                        )
                denom = tmp.sum()
                if denom == 0.0:
                    denom = 1e-300
                xi_sum += tmp / denom

            # --- M-step ---
            self.initial_probs_ = gamma[:, 0] / gamma[:, 0].sum()

            for i in range(K):
                denom = xi_sum[i].sum()
                if denom > 0.0:
                    self.transition_matrix_[i] = xi_sum[i] / denom

            for j in range(K):
                weight = gamma[j]
                w_sum = weight.sum()
                if w_sum > 0.0:
                    self.means_[j] = (weight * returns).sum() / w_sum
                    diff = returns - self.means_[j]
                    self.stds_[j] = max(
                        np.sqrt((weight * diff ** 2).sum() / w_sum), 1e-6
                    )
        else:
            logger.info(
                "Baum-Welch did not converge after %d iterations (LL=%.6f)",
                self.n_iter,
                prev_ll,
            )

        # Initial filtering state.
        self._state_probs = gamma[:, -1].copy()
        self.is_fitted = True

        # Build regime history for the training sample.
        self.regime_history = [
            _STATE_TO_REGIME[int(np.argmax(gamma[:, t]))] for t in range(T)
        ]

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_regime(self, recent_returns: NDArray) -> MarketRegime:
        """Run the forward algorithm on *recent_returns* and return the MAP regime.

        Parameters
        ----------
        recent_returns : 1-D array-like
            A window of recent returns (e.g. last 20 trading days).

        Returns
        -------
        MarketRegime
            Most probable current regime.
        """
        self._check_fitted()
        recent = np.asarray(recent_returns, dtype=np.float64).ravel()
        log_B = self._emission_log_probs(recent)
        alpha, _ = self._forward(log_B)

        # Normalised filtering distribution at final time step.
        probs = alpha[:, -1]
        s = probs.sum()
        if s > 0.0:
            probs = probs / s
        self._state_probs = probs

        best = int(np.argmax(probs))
        regime = _STATE_TO_REGIME[best]
        self.regime_history.append(regime)
        return regime

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update_online(self, new_return: float) -> MarketRegime:
        """Perform a single-step online (filtering) update.

        Uses the current state distribution, the transition matrix, and
        the new observation to produce an updated regime estimate without
        re-running the full forward pass.

        Parameters
        ----------
        new_return : float
            The latest observed return.

        Returns
        -------
        MarketRegime
            Updated regime estimate.
        """
        self._check_fitted()

        # Predict step: p(s_t | o_{1:t-1}) = A^T @ p(s_{t-1} | o_{1:t-1}).
        predicted = self.transition_matrix_.T @ self._state_probs

        # Update step: multiply by emission likelihood and normalise.
        log_emissions = np.array([
            _log_normal_pdf(np.array([new_return]), self.means_[j], self.stds_[j])[0]
            for j in range(self.n_states)
        ])
        # Shift for numerical stability before exp.
        log_emissions -= log_emissions.max()
        emissions = np.exp(log_emissions)

        updated = predicted * emissions
        total = updated.sum()
        if total > 0.0:
            updated /= total
        self._state_probs = updated

        best = int(np.argmax(updated))
        regime = _STATE_TO_REGIME[best]
        self.regime_history.append(regime)
        return regime

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def state_probabilities(self) -> NDArray:
        """Current filtering distribution over regimes."""
        self._check_fitted()
        return self._state_probs.copy()

    @property
    def transition_matrix(self) -> NDArray:
        """Learned transition matrix (K x K)."""
        self._check_fitted()
        return self.transition_matrix_.copy()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("RegimeDetector has not been fitted yet. Call fit() first.")
