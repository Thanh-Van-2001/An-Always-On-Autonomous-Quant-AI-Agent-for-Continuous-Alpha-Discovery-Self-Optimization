"""
Leakage Detection Test (LDT) -- TIF Step 3.

When a large language model is used to *generate* trading strategies, there is
a subtle but critical risk: the LLM may have memorized historical price
patterns during pre-training.  A factor that looks profitable in-sample may
simply be regurgitating patterns the model has already "seen", rather than
capturing a genuine economic signal.  This is the **memorization bias**
problem.

The Leakage Detection Test addresses this by comparing the Information
Coefficient (IC) of a factor on a training period (which falls within the
LLM's pre-training window) against its IC on a strictly out-of-sample test
period (after the LLM's knowledge cutoff plus a safety buffer).  If the
in-sample IC is *significantly* higher than the out-of-sample IC, the factor
is likely exploiting memorized information rather than a true alpha signal.

Formally, the test statistic is a Welch two-sample t-statistic:

    LDT  =  (IC_train - IC_test) / sqrt(s^2_train / n_train  +  s^2_test / n_test)

Under the null hypothesis H_0: IC_train = IC_test (no leakage), LDT follows
a t-distribution that is well-approximated by a standard normal for the
sample sizes typical in daily factor evaluation.

We reject H_0 (declare leakage) when LDT > z_{1-alpha}, with alpha = 0.01
giving a critical value of z_0.99 = 2.326.

Reference
---------
KDD 2026, Section 3.2 -- Temporal Integrity Framework, Step 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# One-sided critical value at alpha = 0.01 (normal approximation).
Z_CRITICAL_001 = 2.326


@dataclass
class LDTResult:
    """Outcome of a single Leakage Detection Test.

    Attributes:
        passed: ``True`` if the factor is *not* flagged for leakage
                (i.e. we fail to reject H_0).
        t_stat: The computed LDT / Welch t-statistic.
        p_value: One-sided p-value (right tail).
        ic_train_mean: Mean IC on the training partition.
        ic_test_mean: Mean IC on the test partition.
        alpha: Significance level used.
    """

    passed: bool
    t_stat: float
    p_value: float
    ic_train_mean: float
    ic_test_mean: float
    alpha: float


class LeakageDetectionTest:
    """Statistical test for LLM memorization bias in alpha factors.

    Parameters
    ----------
    alpha : float, optional
        Significance level for the one-sided test (default 0.01).

    Examples
    --------
    >>> ldt = LeakageDetectionTest(alpha=0.01)
    >>> ic_train = np.array([0.04, 0.05, 0.03, 0.06, 0.04])
    >>> ic_test  = np.array([0.03, 0.04, 0.02, 0.05, 0.03])
    >>> result = ldt.test_leakage(ic_train, ic_test)
    >>> result.passed
    True
    """

    def __init__(self, alpha: float = 0.01) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        # Precompute critical value from normal approximation.
        self.z_critical = stats.norm.ppf(1.0 - alpha)
        logger.debug(
            "LeakageDetectionTest initialised: alpha=%.4f, z_critical=%.4f",
            alpha,
            self.z_critical,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def compute_ldt_statistic(
        ic_train: np.ndarray,
        ic_test: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute the Welch t-statistic for the leakage detection test.

        Parameters
        ----------
        ic_train : np.ndarray
            1-D array of daily IC values from the training partition.
        ic_test : np.ndarray
            1-D array of daily IC values from the test partition.

        Returns
        -------
        t_stat : float
            LDT = (IC_train_mean - IC_test_mean)
                  / sqrt(s2_train/n_train + s2_test/n_test)
        p_value : float
            One-sided p-value (right tail) under a normal approximation.

        Raises
        ------
        ValueError
            If either array has fewer than 2 observations.
        """
        ic_train = np.asarray(ic_train, dtype=np.float64)
        ic_test = np.asarray(ic_test, dtype=np.float64)

        n_train = len(ic_train)
        n_test = len(ic_test)

        if n_train < 2:
            raise ValueError(
                f"ic_train must have >= 2 observations, got {n_train}"
            )
        if n_test < 2:
            raise ValueError(
                f"ic_test must have >= 2 observations, got {n_test}"
            )

        mean_train = np.mean(ic_train)
        mean_test = np.mean(ic_test)
        var_train = np.var(ic_train, ddof=1)
        var_test = np.var(ic_test, ddof=1)

        se = np.sqrt(var_train / n_train + var_test / n_test)
        if se < 1e-15:
            # Both series are essentially constant -- no evidence of leakage.
            logger.warning(
                "Standard error is near zero; returning t_stat=0.0"
            )
            return 0.0, 1.0

        t_stat = (mean_train - mean_test) / se

        # One-sided p-value: P(T > t_stat) under the null.
        # Use Welch-Satterthwaite degrees of freedom for the t-distribution.
        df_num = (var_train / n_train + var_test / n_test) ** 2
        df_den = (
            (var_train / n_train) ** 2 / (n_train - 1)
            + (var_test / n_test) ** 2 / (n_test - 1)
        )
        if df_den < 1e-15:
            df = n_train + n_test - 2
        else:
            df = df_num / df_den

        p_value = 1.0 - stats.t.cdf(t_stat, df=df)

        return float(t_stat), float(p_value)

    def test_leakage(
        self,
        ic_train_series: np.ndarray,
        ic_test_series: np.ndarray,
        alpha: float | None = None,
    ) -> LDTResult:
        """Run the full Leakage Detection Test.

        Parameters
        ----------
        ic_train_series : np.ndarray
            Daily IC values from the training partition (within LLM window).
        ic_test_series : np.ndarray
            Daily IC values from the test partition (outside LLM window).
        alpha : float, optional
            Override the instance-level significance level for this call.

        Returns
        -------
        LDTResult
            Contains ``passed=True`` when the factor is **not** flagged for
            memorization leakage (i.e. we fail to reject H_0 at level alpha).
        """
        alpha = alpha if alpha is not None else self.alpha
        z_crit = stats.norm.ppf(1.0 - alpha)

        ic_train_series = np.asarray(ic_train_series, dtype=np.float64)
        ic_test_series = np.asarray(ic_test_series, dtype=np.float64)

        t_stat, p_value = self.compute_ldt_statistic(
            ic_train_series, ic_test_series
        )

        # The factor PASSES if we *fail* to reject H_0.
        # Rejection (leakage detected) happens when t_stat > z_crit.
        passed = t_stat <= z_crit

        result = LDTResult(
            passed=passed,
            t_stat=t_stat,
            p_value=p_value,
            ic_train_mean=float(np.mean(ic_train_series)),
            ic_test_mean=float(np.mean(ic_test_series)),
            alpha=alpha,
        )

        if passed:
            logger.info(
                "LDT PASSED: t_stat=%.4f <= z_crit=%.4f (alpha=%.4f), "
                "IC_train=%.4f, IC_test=%.4f",
                t_stat,
                z_crit,
                alpha,
                result.ic_train_mean,
                result.ic_test_mean,
            )
        else:
            logger.warning(
                "LDT FAILED (leakage detected): t_stat=%.4f > z_crit=%.4f "
                "(alpha=%.4f), IC_train=%.4f, IC_test=%.4f",
                t_stat,
                z_crit,
                alpha,
                result.ic_train_mean,
                result.ic_test_mean,
            )

        return result
