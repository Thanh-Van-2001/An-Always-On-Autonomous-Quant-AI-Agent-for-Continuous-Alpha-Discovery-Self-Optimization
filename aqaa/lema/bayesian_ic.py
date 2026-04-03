"""
Bayesian IC Tracking via a Local-Level (Kalman Filter) Model.

This module implements the Bayesian information-coefficient tracker
described in Section 4.3 of the KDD 2026 paper.  The generative model is:

    IC_t = IC_{t-1} + d_t + epsilon_t          (observation equation)
    d_t  ~ N(0, sigma2_drift)                   (state drift)
    epsilon_t ~ N(0, sigma2_obs)                 (observation noise)

Equivalently, this is a *local-level model* (also known as the random-walk-
plus-noise model) solved by the standard Kalman filter recursions.

The posterior predictive at each step provides:

    E[IC_t | IC_{1:t-1}]   -- point forecast
    Var[IC_t | IC_{1:t-1}] -- forecast uncertainty

From the posterior we compute a *decay probability*:

    P(IC_t < theta_min | IC_{1:t-1})

When this probability exceeds a threshold (default 0.90), the factor is
flagged for review (UNDER_REVIEW transition in the ALM).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """Internal state of the Kalman filter at a single time step.

    Attributes:
        mu: Posterior mean of the latent IC level.
        var: Posterior variance of the latent IC level.
        t: Time index (number of observations processed).
    """

    mu: float = 0.0
    var: float = 1.0
    t: int = 0


class BayesianICTracker:
    """Tracks the latent IC level of a factor using a Kalman filter.

    Parameters
    ----------
    sigma2_obs : float
        Observation-noise variance.  Controls how much a single IC
        measurement can deviate from the latent level.  A typical
        starting value for daily cross-sectional IC is 0.001.
    sigma2_drift : float
        State-drift variance per time step.  Controls how quickly the
        latent IC level is allowed to change.  A typical starting value
        is 1e-5 (slow drift).
    prior_mu : float
        Prior mean for the latent IC at t = 0.
    prior_var : float
        Prior variance for the latent IC at t = 0.  Use a diffuse prior
        (e.g. 1.0) when little is known beforehand.

    Notes
    -----
    The Kalman filter recursions are:

    **Predict step** (time update):

    .. math::
        \\mu_{t|t-1} = \\mu_{t-1|t-1}
        P_{t|t-1}   = P_{t-1|t-1} + \\sigma^2_{\\text{drift}}

    **Update step** (measurement update):

    .. math::
        K_t         = P_{t|t-1} / (P_{t|t-1} + \\sigma^2_{\\text{obs}})
        \\mu_{t|t}  = \\mu_{t|t-1} + K_t (y_t - \\mu_{t|t-1})
        P_{t|t}     = (1 - K_t) P_{t|t-1}

    where :math:`y_t` is the observed IC value.
    """

    def __init__(
        self,
        sigma2_obs: float = 0.001,
        sigma2_drift: float = 1e-5,
        prior_mu: float = 0.0,
        prior_var: float = 1.0,
    ) -> None:
        if sigma2_obs <= 0:
            raise ValueError("sigma2_obs must be positive")
        if sigma2_drift < 0:
            raise ValueError("sigma2_drift must be non-negative")
        if prior_var <= 0:
            raise ValueError("prior_var must be positive")

        self.sigma2_obs = sigma2_obs
        self.sigma2_drift = sigma2_drift

        # Current Kalman state
        self._state = KalmanState(mu=prior_mu, var=prior_var, t=0)

        # Full history for diagnostics
        self._observations: List[float] = []
        self._posterior_means: List[float] = [prior_mu]
        self._posterior_vars: List[float] = [prior_var]
        self._kalman_gains: List[float] = []
        self._prediction_errors: List[float] = []

    # ------------------------------------------------------------------
    # Core recursion
    # ------------------------------------------------------------------

    def update(self, y: float) -> Tuple[float, float]:
        """Incorporate a new IC observation and return the posterior.

        Parameters
        ----------
        y : float
            Observed IC value at the current time step.

        Returns
        -------
        (mu_post, var_post) : tuple of float
            Posterior mean and variance of the latent IC after the update.
        """
        # Predict step
        mu_pred = self._state.mu
        var_pred = self._state.var + self.sigma2_drift

        # Innovation (prediction error)
        innovation = y - mu_pred
        innovation_var = var_pred + self.sigma2_obs

        # Kalman gain
        K = var_pred / innovation_var

        # Update step
        mu_post = mu_pred + K * innovation
        var_post = (1.0 - K) * var_pred

        # Joseph form for numerical stability:
        # P = (I - K*H) P_pred (I - K*H)' + K R K'
        # For scalar case this simplifies to:
        var_post_joseph = (
            (1.0 - K) ** 2 * var_pred + K ** 2 * self.sigma2_obs
        )
        # Use the Joseph form when the simple form would be negative
        # (can happen with extreme floating-point values).
        if var_post < 0:
            var_post = var_post_joseph

        # Store
        self._state = KalmanState(
            mu=mu_post, var=var_post, t=self._state.t + 1
        )
        self._observations.append(y)
        self._posterior_means.append(mu_post)
        self._posterior_vars.append(var_post)
        self._kalman_gains.append(K)
        self._prediction_errors.append(innovation)

        return mu_post, var_post

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_ic(
        self, steps_ahead: int = 1
    ) -> Tuple[float, float]:
        """Return the posterior predictive for *steps_ahead* steps.

        For a local-level model the h-step-ahead predictive is:

        .. math::
            \\mu_{t+h|t} = \\mu_{t|t}
            P_{t+h|t}   = P_{t|t} + h \\sigma^2_{\\text{drift}}
                           + \\sigma^2_{\\text{obs}}

        Parameters
        ----------
        steps_ahead : int
            Forecast horizon (>= 1).

        Returns
        -------
        (mu_pred, var_pred) : tuple of float
            Predictive mean and variance of the *observed* IC at
            t + steps_ahead.
        """
        if steps_ahead < 1:
            raise ValueError("steps_ahead must be >= 1")

        mu_pred = self._state.mu
        var_pred = (
            self._state.var
            + steps_ahead * self.sigma2_drift
            + self.sigma2_obs
        )
        return mu_pred, var_pred

    # ------------------------------------------------------------------
    # Decay probability
    # ------------------------------------------------------------------

    def decay_probability(
        self,
        theta_min: float = 0.015,
        steps_ahead: int = 1,
    ) -> float:
        """Compute P(IC_{t+h} < theta_min | IC_{1:t}).

        Under the Gaussian posterior predictive this is simply the CDF
        of the normal distribution evaluated at *theta_min*.

        Parameters
        ----------
        theta_min : float
            Minimum acceptable IC level.
        steps_ahead : int
            Forecast horizon.

        Returns
        -------
        float
            Probability that the future IC falls below *theta_min*.
        """
        mu, var = self.predict_ic(steps_ahead=steps_ahead)
        if var <= 0:
            # Degenerate -- return 0 or 1 depending on the mean.
            return 0.0 if mu >= theta_min else 1.0

        sigma = math.sqrt(var)
        return float(stats.norm.cdf(theta_min, loc=mu, scale=sigma))

    def is_decaying(
        self,
        theta_min: float = 0.015,
        p_threshold: float = 0.90,
        steps_ahead: int = 1,
    ) -> bool:
        """Check whether the decay trigger fires.

        Returns ``True`` when P(IC_{t+h} < theta_min) > p_threshold.
        """
        return self.decay_probability(
            theta_min=theta_min, steps_ahead=steps_ahead
        ) > p_threshold

    # ------------------------------------------------------------------
    # Log-likelihood (for model selection / diagnostics)
    # ------------------------------------------------------------------

    def log_likelihood(self) -> float:
        """Compute the marginal log-likelihood of the observations.

        This uses the prediction-error decomposition:

        .. math::
            \\log p(y_{1:T}) = \\sum_{t=1}^{T} \\log p(y_t | y_{1:t-1})

        where each term is Gaussian with mean ``mu_{t|t-1}`` and variance
        ``P_{t|t-1} + sigma2_obs``.

        Returns
        -------
        float
            Log-likelihood value (can be -inf for very unlikely data).
        """
        if len(self._observations) == 0:
            return 0.0

        ll = 0.0
        # Reconstruct the one-step-ahead predictive variances.
        # posterior_vars[i] = P_{i|i}, so P_{i+1|i} = P_{i|i} + sigma2_drift
        for i, (y, innov) in enumerate(
            zip(self._observations, self._prediction_errors)
        ):
            # P_{t|t-1}
            var_pred = self._posterior_vars[i] + self.sigma2_drift
            # Innovation variance
            S = var_pred + self.sigma2_obs
            ll += -0.5 * (math.log(2 * math.pi * S) + innov ** 2 / S)
        return ll

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def current_mu(self) -> float:
        """Posterior mean of the latent IC."""
        return self._state.mu

    @property
    def current_var(self) -> float:
        """Posterior variance of the latent IC."""
        return self._state.var

    @property
    def n_observations(self) -> int:
        """Number of observations incorporated so far."""
        return self._state.t

    @property
    def posterior_means(self) -> np.ndarray:
        """Full history of posterior means (length T+1, starting at the prior)."""
        return np.asarray(self._posterior_means)

    @property
    def posterior_vars(self) -> np.ndarray:
        """Full history of posterior variances."""
        return np.asarray(self._posterior_vars)

    @property
    def observations(self) -> np.ndarray:
        """Array of all IC observations."""
        return np.asarray(self._observations)

    def reset(
        self,
        prior_mu: float = 0.0,
        prior_var: float = 1.0,
    ) -> None:
        """Reset the filter to a fresh prior."""
        self._state = KalmanState(mu=prior_mu, var=prior_var, t=0)
        self._observations.clear()
        self._posterior_means.clear()
        self._posterior_means.append(prior_mu)
        self._posterior_vars.clear()
        self._posterior_vars.append(prior_var)
        self._kalman_gains.clear()
        self._prediction_errors.clear()
