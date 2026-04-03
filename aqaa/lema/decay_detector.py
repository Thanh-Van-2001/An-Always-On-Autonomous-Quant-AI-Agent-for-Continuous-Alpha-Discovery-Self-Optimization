"""
Change-Point Decay Detection (CUSUM + PELT with Consensus).

Provides two complementary change-point detection algorithms for
monitoring IC series in real time (Section 4.3 of the KDD 2026 paper):

1. **CUSUM** (Cumulative Sum) -- a sequential test that accumulates
   evidence of a mean shift relative to a reference level.
2. **PELT** (Pruned Exact Linear Time) -- an optimal segmentation
   algorithm that partitions the series into regimes while penalising
   model complexity.

A **ConsensusDetector** combines both: a decay alarm fires only when
the two methods agree (within a tolerance window of 5 days), reducing
false-positive rate while retaining sensitivity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# CUSUM Detector
# ======================================================================

class CUSUMDetector:
    """Tabular CUSUM test for detecting a downward mean shift.

    The one-sided lower-CUSUM statistic is maintained as:

    .. math::
        S_t^- = \\max(0,\\; S_{t-1}^- - (y_t - \\mu_0) - k)

    where :math:`\\mu_0` is the reference (in-control) mean and *k* is
    the allowance (slack) parameter, typically set to half the expected
    shift magnitude.  A change point is signalled when :math:`S_t^-`
    exceeds the decision threshold *h*.

    We also track the upper CUSUM for completeness:

    .. math::
        S_t^+ = \\max(0,\\; S_{t-1}^+ + (y_t - \\mu_0) - k)

    Parameters
    ----------
    window : int
        Rolling window length for computing the reference mean.
        Default is 20 (trading days).
    k : float
        Allowance (slack) parameter.  Default 0.005.
    h : float
        Decision threshold.  Default 0.05.
    """

    def __init__(
        self,
        window: int = 20,
        k: float = 0.005,
        h: float = 0.05,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self.k = k
        self.h = h

    def detect(
        self,
        ic_series: np.ndarray,
        threshold: Optional[float] = None,
    ) -> List[int]:
        """Run CUSUM on the IC series and return detected change points.

        Parameters
        ----------
        ic_series : np.ndarray
            1-D array of IC observations ordered by time.
        threshold : float, optional
            Override for the decision threshold *h*.

        Returns
        -------
        list of int
            Indices at which the CUSUM statistic exceeded the threshold,
            interpreted as change-point locations.
        """
        ic = np.asarray(ic_series, dtype=np.float64)
        n = len(ic)
        if n < self.window:
            return []

        h = threshold if threshold is not None else self.h
        change_points: List[int] = []

        # Reference mean: rolling mean of the first `window` observations.
        mu0 = float(np.mean(ic[: self.window]))

        s_pos = 0.0  # upper CUSUM
        s_neg = 0.0  # lower CUSUM (tracks downward shifts)

        for t in range(self.window, n):
            deviation = ic[t] - mu0

            # Upper CUSUM (positive shift -- IC improving, less common)
            s_pos = max(0.0, s_pos + deviation - self.k)

            # Lower CUSUM (negative shift -- IC decay)
            s_neg = max(0.0, s_neg - deviation - self.k)

            if s_neg > h:
                change_points.append(t)
                # Reset after alarm
                s_neg = 0.0
                # Update reference mean to post-change level
                start = max(0, t - self.window + 1)
                mu0 = float(np.mean(ic[start: t + 1]))

        return change_points


# ======================================================================
# PELT Detector
# ======================================================================

class PELTDetector:
    """Pruned Exact Linear Time (PELT) change-point detector.

    Implements the PELT algorithm of Killick, Fearnhead & Eckley (2012)
    for detecting changes in the mean of a Gaussian sequence.

    The cost function for a segment :math:`y_{s+1:t}` under a Gaussian
    model with known variance is the negative log-likelihood (up to a
    constant):

    .. math::
        C(y_{s+1:t}) = \\frac{n}{2} \\ln\\hat\\sigma^2

    where :math:`\\hat\\sigma^2` is the sample variance of the segment
    and *n = t - s* is the segment length.  We use the cumulative-sum
    trick for O(1) cost evaluation.

    The PELT pruning rule keeps only those candidate change points whose
    cost is still competitive:

    .. math::
        F(s) + C(y_{s+1:t}) + \\beta \\leq F(t)

    Parameters
    ----------
    min_segment : int
        Minimum segment length (must be >= 2 for variance estimation).
    penalty : float or str
        Penalty per change point.  Use ``"bic"`` for the BIC penalty
        :math:`\\beta = \\ln(n)` (computed automatically) or a positive
        float for a fixed penalty.
    """

    def __init__(
        self,
        min_segment: int = 5,
        penalty: float | str = "bic",
    ) -> None:
        if min_segment < 2:
            raise ValueError("min_segment must be >= 2")
        self.min_segment = min_segment
        self._penalty_spec = penalty

    # ------------------------------------------------------------------
    # Cost function helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cumsums(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-compute cumulative sums and cumulative sums of squares.

        Returns arrays of length ``len(y) + 1`` where index 0 is 0.
        """
        cumsum = np.empty(len(y) + 1, dtype=np.float64)
        cumsum[0] = 0.0
        cumsum[1:] = np.cumsum(y)

        cumsq = np.empty(len(y) + 1, dtype=np.float64)
        cumsq[0] = 0.0
        cumsq[1:] = np.cumsum(y ** 2)

        return cumsum, cumsq

    @staticmethod
    def _segment_cost(
        cumsum: np.ndarray,
        cumsq: np.ndarray,
        s: int,
        t: int,
    ) -> float:
        """Gaussian cost for segment (s, t] (0-indexed, exclusive s).

        .. math::
            C = n \\ln(\\hat\\sigma^2)

        where :math:`\\hat\\sigma^2 = (1/n)\\sum(y_i - \\bar y)^2`.
        We add a small epsilon for numerical safety when the segment
        is perfectly constant.

        Parameters
        ----------
        cumsum, cumsq : np.ndarray
            Pre-computed cumulative sums.
        s, t : int
            Segment defined as observations with indices s+1, ..., t
            (Python slice ``[s+1 : t+1]`` if 0-based, but the cumsums
            are 1-indexed).
        """
        n = t - s
        if n < 2:
            return 0.0

        # sum and sum-of-squares for the segment
        seg_sum = cumsum[t] - cumsum[s]
        seg_sumsq = cumsq[t] - cumsq[s]

        mean = seg_sum / n
        var = seg_sumsq / n - mean ** 2

        # Clamp to avoid log(0)
        eps = 1e-15
        var = max(var, eps)

        return n * np.log(var)

    # ------------------------------------------------------------------
    # PELT algorithm
    # ------------------------------------------------------------------

    def detect(
        self,
        ic_series: np.ndarray,
        penalty: Optional[float] = None,
    ) -> List[int]:
        """Run PELT on the IC series and return optimal change points.

        Parameters
        ----------
        ic_series : np.ndarray
            1-D array of IC observations.
        penalty : float, optional
            Override for the per-change-point penalty.

        Returns
        -------
        list of int
            Sorted indices of detected change points (0-based).
        """
        y = np.asarray(ic_series, dtype=np.float64)
        n = len(y)

        if n < 2 * self.min_segment:
            return []

        # Determine penalty
        if penalty is not None:
            beta = float(penalty)
        elif self._penalty_spec == "bic":
            beta = np.log(n)
        else:
            beta = float(self._penalty_spec)

        cumsum, cumsq = self._build_cumsums(y)

        # F[t] = optimal cost of segmenting y[0:t]
        # Using 1-based indexing internally: F[0] = -beta (convention)
        F = np.full(n + 1, np.inf, dtype=np.float64)
        F[0] = -beta

        # last_cp[t] stores the last change point for the optimal
        # segmentation ending at t.
        last_cp = np.zeros(n + 1, dtype=np.int64)

        # R = set of admissible last change point candidates
        candidates: List[int] = [0]

        for t_star in range(self.min_segment, n + 1):
            # Evaluate cost for each candidate
            best_val = np.inf
            best_s = 0

            admissible: List[int] = []

            for s in candidates:
                seg_len = t_star - s
                if seg_len < self.min_segment:
                    admissible.append(s)
                    continue

                cost = F[s] + self._segment_cost(cumsum, cumsq, s, t_star) + beta

                if cost < best_val:
                    best_val = cost
                    best_s = s

                # PELT pruning: keep s only if it can still be useful
                if F[s] + self._segment_cost(cumsum, cumsq, s, t_star) <= F[t_star]:
                    admissible.append(s)

            F[t_star] = best_val
            last_cp[t_star] = best_s

            # Add t_star as a new candidate and apply pruning
            admissible.append(t_star)

            # Post-hoc pruning against the newly established F[t_star]
            pruned: List[int] = []
            for s in admissible:
                seg_len = t_star - s
                if seg_len < self.min_segment:
                    pruned.append(s)
                    continue
                if F[s] + self._segment_cost(cumsum, cumsq, s, t_star) + beta <= F[t_star]:
                    pruned.append(s)
            # Always keep t_star itself
            if t_star not in pruned:
                pruned.append(t_star)

            candidates = pruned

        # Back-track to recover change points
        change_points: List[int] = []
        idx = n
        while idx > 0:
            cp = int(last_cp[idx])
            if cp > 0:
                change_points.append(cp)
            idx = cp

        change_points.sort()
        return change_points


# ======================================================================
# Consensus Detector
# ======================================================================

class ConsensusDetector:
    """Combines CUSUM and PELT to produce a consensus decay alarm.

    A decay event is flagged only when both detectors report a change
    point within *tolerance* days of each other.  This significantly
    reduces false alarms compared with either method alone.

    Parameters
    ----------
    cusum : CUSUMDetector, optional
        Pre-configured CUSUM detector.
    pelt : PELTDetector, optional
        Pre-configured PELT detector.
    tolerance : int
        Maximum distance (in time indices) between CUSUM and PELT
        change points for them to be considered "agreeing".  Default 5.
    """

    def __init__(
        self,
        cusum: Optional[CUSUMDetector] = None,
        pelt: Optional[PELTDetector] = None,
        tolerance: int = 5,
    ) -> None:
        self.cusum = cusum or CUSUMDetector()
        self.pelt = pelt or PELTDetector()
        self.tolerance = tolerance

    def detect_decay(
        self,
        ic_series: np.ndarray,
    ) -> Tuple[bool, List[int]]:
        """Run both detectors and apply the consensus rule.

        Parameters
        ----------
        ic_series : np.ndarray
            1-D array of IC observations.

        Returns
        -------
        (is_decaying, consensus_points) : tuple
            ``is_decaying`` is ``True`` if at least one consensus change
            point was found.  ``consensus_points`` contains the CUSUM
            indices that matched a PELT change point within *tolerance*.
        """
        ic = np.asarray(ic_series, dtype=np.float64)

        cusum_cps = self.cusum.detect(ic)
        pelt_cps = self.pelt.detect(ic)

        if not cusum_cps or not pelt_cps:
            return False, []

        pelt_arr = np.array(pelt_cps)

        consensus: List[int] = []
        for c in cusum_cps:
            # Check if any PELT change point is within tolerance
            distances = np.abs(pelt_arr - c)
            if np.min(distances) <= self.tolerance:
                consensus.append(c)

        is_decaying = len(consensus) > 0

        if is_decaying:
            logger.info(
                "Consensus decay detected at indices %s "
                "(CUSUM: %s, PELT: %s)",
                consensus,
                cusum_cps,
                pelt_cps,
            )

        return is_decaying, consensus


# ======================================================================
# Convenience wrapper
# ======================================================================

class DecayDetector:
    """High-level decay detector that wraps the consensus approach.

    This is the recommended entry point for callers who want a simple
    boolean answer.

    Parameters
    ----------
    cusum_window : int
        CUSUM rolling window (default 20).
    cusum_k : float
        CUSUM allowance (default 0.005).
    cusum_h : float
        CUSUM decision threshold (default 0.05).
    pelt_min_segment : int
        PELT minimum segment length (default 5).
    pelt_penalty : float or str
        PELT penalty (default ``"bic"``).
    consensus_tolerance : int
        Consensus match tolerance in days (default 5).
    """

    def __init__(
        self,
        cusum_window: int = 20,
        cusum_k: float = 0.005,
        cusum_h: float = 0.05,
        pelt_min_segment: int = 5,
        pelt_penalty: float | str = "bic",
        consensus_tolerance: int = 5,
    ) -> None:
        self._consensus = ConsensusDetector(
            cusum=CUSUMDetector(window=cusum_window, k=cusum_k, h=cusum_h),
            pelt=PELTDetector(
                min_segment=pelt_min_segment, penalty=pelt_penalty
            ),
            tolerance=consensus_tolerance,
        )

    def detect(
        self, ic_series: np.ndarray
    ) -> Tuple[bool, List[int]]:
        """Detect structural IC decay.

        Parameters
        ----------
        ic_series : np.ndarray
            1-D array of IC observations.

        Returns
        -------
        (is_decaying, change_points) : tuple
        """
        return self._consensus.detect_decay(ic_series)
