"""
Temporal Integrity Framework (TIF).

The TIF is the core methodological contribution of the KDD 2026 paper.  It
is a four-step protocol that guarantees factors produced by the CSBA module
are free of data-snooping and LLM memorization bias:

    Step 1  **Data Partition**
            Split the historical dataset into four non-overlapping zones:
            Training, Validation, Buffer, and Test.

    Step 2  **LLM Cutoff Isolation**
            Exclude all data that falls within ``buffer_months`` of the LLM's
            training data cutoff date.  This creates a temporal "firewall"
            between information the LLM could have memorized and the data
            used for out-of-sample evaluation.

    Step 3  **Leakage Detection Test (LDT)**
            Apply the statistical test defined in ``ldt.py``.  If the factor's
            in-sample IC is significantly higher than its out-of-sample IC,
            the factor is flagged for memorization leakage and rejected.

    Step 4  **Live Validation Gate**
            Require 30 calendar days of paper-traded forward performance
            before the factor may enter the live portfolio.

Reference
---------
KDD 2026, Section 3.2 -- Temporal Integrity Framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aqaa.core.types import BacktestResult
from aqaa.csba.ldt import LeakageDetectionTest, LDTResult

logger = logging.getLogger(__name__)

# Default buffer around LLM knowledge cutoff.
DEFAULT_BUFFER_MONTHS = 6

# Default paper-trade validation period (calendar days).
DEFAULT_PAPER_TRADE_DAYS = 30


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DateRange:
    """An inclusive date range [start, end]."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(
                f"DateRange start ({self.start}) > end ({self.end})"
            )

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end

    def __repr__(self) -> str:
        return f"DateRange({self.start} .. {self.end}, {self.days}d)"


@dataclass
class DataPartition:
    """Result of partitioning the dataset via TIF Steps 1-2.

    Attributes:
        train: Date range and data indices for the training zone.
        validation: Date range and data indices for the validation zone.
        buffer: Date range surrounding the LLM cutoff (excluded from both
                train and test).
        test: Date range and data indices for the out-of-sample test zone.
        train_indices: Boolean mask or integer indices into the original data.
        validation_indices: Boolean mask or integer indices.
        buffer_indices: Boolean mask or integer indices (excluded zone).
        test_indices: Boolean mask or integer indices.
        llm_cutoff_date: The LLM's training-data cutoff.
        buffer_months: Width of the exclusion buffer in months.
    """

    train: DateRange
    validation: DateRange
    buffer: DateRange
    test: DateRange

    train_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    validation_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    buffer_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    test_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    llm_cutoff_date: Optional[date] = None
    buffer_months: int = DEFAULT_BUFFER_MONTHS

    def summary(self) -> str:
        """Return a human-readable summary of the partition."""
        lines = [
            "=== Temporal Integrity Framework -- Data Partition ===",
            f"  Train      : {self.train}  ({len(self.train_indices)} rows)",
            f"  Validation : {self.validation}  ({len(self.validation_indices)} rows)",
            f"  Buffer     : {self.buffer}  ({len(self.buffer_indices)} rows)  [EXCLUDED]",
            f"  Test       : {self.test}  ({len(self.test_indices)} rows)",
            f"  LLM cutoff : {self.llm_cutoff_date}",
            f"  Buffer     : {self.buffer_months} months",
        ]
        return "\n".join(lines)


@dataclass
class TIFValidationResult:
    """Aggregated result after running all four TIF steps on a factor.

    Attributes:
        step1_passed: Data partition created successfully.
        step2_passed: LLM cutoff isolation applied (buffer zone excluded).
        step3_ldt_result: The ``LDTResult`` from the leakage detection test.
        step3_passed: Factor passed LDT (no memorization leakage detected).
        step4_paper_trade_required: Whether a 30-day paper trade is required
                                    (always True until completed).
        overall_passed: All automated gates passed (Steps 1-3).  Step 4 is
                        flagged but must be confirmed asynchronously.
        partition: The ``DataPartition`` used.
        backtest_train: Backtest result on the training partition.
        backtest_test: Backtest result on the test partition.
    """

    step1_passed: bool = False
    step2_passed: bool = False
    step3_ldt_result: Optional[LDTResult] = None
    step3_passed: bool = False
    step4_paper_trade_required: bool = True
    overall_passed: bool = False
    partition: Optional[DataPartition] = None
    backtest_train: Optional[BacktestResult] = None
    backtest_test: Optional[BacktestResult] = None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TemporalIntegrityFramework:
    """Implements the four-step TIF protocol.

    Parameters
    ----------
    llm_cutoff_date : date
        The knowledge cutoff of the LLM used for factor code synthesis.
    buffer_months : int
        Number of months to exclude around the LLM cutoff (default 6).
    ldt_alpha : float
        Significance level for the Leakage Detection Test (default 0.01).
    paper_trade_days : int
        Required number of calendar days for the paper-trading gate
        (default 30).
    train_frac : float
        Fraction of the pre-buffer data allocated to training (the rest
        goes to validation).  Default 0.7.
    """

    def __init__(
        self,
        llm_cutoff_date: date,
        buffer_months: int = DEFAULT_BUFFER_MONTHS,
        ldt_alpha: float = 0.01,
        paper_trade_days: int = DEFAULT_PAPER_TRADE_DAYS,
        train_frac: float = 0.7,
    ) -> None:
        self.llm_cutoff_date = llm_cutoff_date
        self.buffer_months = buffer_months
        self.ldt_alpha = ldt_alpha
        self.paper_trade_days = paper_trade_days
        self.train_frac = train_frac

        self._ldt = LeakageDetectionTest(alpha=ldt_alpha)

        logger.info(
            "TIF initialised: cutoff=%s, buffer=%d months, "
            "ldt_alpha=%.4f, paper_trade=%d days",
            llm_cutoff_date,
            buffer_months,
            ldt_alpha,
            paper_trade_days,
        )

    # ------------------------------------------------------------------
    # Step 1 + 2: Data Partition & LLM Cutoff Isolation
    # ------------------------------------------------------------------

    def partition_data(
        self,
        data: Dict[str, Any],
        llm_cutoff_date: Optional[date] = None,
        buffer_months: Optional[int] = None,
    ) -> DataPartition:
        """Partition the dataset into Train / Validation / Buffer / Test zones.

        This method implements TIF Steps 1 and 2 jointly:

        1. The buffer zone spans ``[cutoff - buffer_months, cutoff + buffer_months]``
           and is **entirely excluded** from both training and testing.
        2. Data *before* the buffer start is split into Training (first
           ``train_frac``) and Validation (remainder).
        3. Data *after* the buffer end constitutes the Test zone.

        Parameters
        ----------
        data : dict
            Must contain a ``"dates"`` key with an array-like of
            ``datetime.date`` objects (one per row of the dataset).
        llm_cutoff_date : date, optional
            Override the instance-level cutoff for this call.
        buffer_months : int, optional
            Override the instance-level buffer width.

        Returns
        -------
        DataPartition
            The fully populated partition with date ranges and row indices.

        Raises
        ------
        ValueError
            If the dataset has no dates or the buffer consumes all data.
        """
        cutoff = llm_cutoff_date or self.llm_cutoff_date
        buf_m = buffer_months if buffer_months is not None else self.buffer_months

        dates = np.asarray(data["dates"])
        if len(dates) == 0:
            raise ValueError("Dataset contains no dates.")

        # Sort indices by date to be safe.
        sort_idx = np.argsort(dates)
        sorted_dates = dates[sort_idx]

        data_start = sorted_dates[0]
        data_end = sorted_dates[-1]

        # Buffer boundaries.
        buffer_start = _subtract_months(cutoff, buf_m)
        buffer_end = _add_months(cutoff, buf_m)

        # Clamp to actual data range.
        buffer_start = max(buffer_start, data_start)
        buffer_end = min(buffer_end, data_end)

        # Pre-buffer zone: for training + validation.
        pre_buffer_mask = sorted_dates < buffer_start
        post_buffer_mask = sorted_dates > buffer_end
        buffer_mask = (~pre_buffer_mask) & (~post_buffer_mask)

        pre_idx = sort_idx[pre_buffer_mask]
        post_idx = sort_idx[post_buffer_mask]
        buf_idx = sort_idx[buffer_mask]

        if len(pre_idx) == 0:
            raise ValueError(
                f"No data before buffer start ({buffer_start}). "
                "Cannot create training partition."
            )
        if len(post_idx) == 0:
            raise ValueError(
                f"No data after buffer end ({buffer_end}). "
                "Cannot create test partition."
            )

        # Split pre-buffer into train / validation.
        n_pre = len(pre_idx)
        n_train = max(1, int(n_pre * self.train_frac))
        train_idx = pre_idx[:n_train]
        val_idx = pre_idx[n_train:]

        # If validation is empty, use the last training sample.
        if len(val_idx) == 0:
            logger.warning(
                "Validation set is empty; using last training sample."
            )
            val_idx = train_idx[-1:]

        # Build DateRange objects.
        train_dates = dates[train_idx]
        val_dates = dates[val_idx]
        buf_dates = dates[buf_idx] if len(buf_idx) > 0 else np.array([buffer_start, buffer_end])
        test_dates = dates[post_idx]

        partition = DataPartition(
            train=DateRange(start=_to_date(train_dates.min()), end=_to_date(train_dates.max())),
            validation=DateRange(start=_to_date(val_dates.min()), end=_to_date(val_dates.max())),
            buffer=DateRange(start=_to_date(buffer_start), end=_to_date(buffer_end)),
            test=DateRange(start=_to_date(test_dates.min()), end=_to_date(test_dates.max())),
            train_indices=train_idx,
            validation_indices=val_idx,
            buffer_indices=buf_idx,
            test_indices=post_idx,
            llm_cutoff_date=cutoff,
            buffer_months=buf_m,
        )

        logger.info("TIF partition created:\n%s", partition.summary())
        return partition

    # ------------------------------------------------------------------
    # Steps 1-4: Full validation pipeline
    # ------------------------------------------------------------------

    def validate_factor(
        self,
        factor: Dict[str, Any],
        partitioned_data: DataPartition,
        ic_train: Optional[np.ndarray] = None,
        ic_test: Optional[np.ndarray] = None,
    ) -> TIFValidationResult:
        """Run all four TIF steps on a candidate factor.

        Parameters
        ----------
        factor : dict
            Factor metadata.  Must contain ``"factor_id"`` and ``"code"``
            keys.  Additional keys are passed through for logging.
        partitioned_data : DataPartition
            A partition previously created by ``partition_data()``.
        ic_train : np.ndarray, optional
            Pre-computed IC series on the training partition.  If ``None``
            the caller is responsible for running the backtest first and
            providing the series.
        ic_test : np.ndarray, optional
            Pre-computed IC series on the test partition.

        Returns
        -------
        TIFValidationResult
            Aggregated result of all four validation steps.
        """
        result = TIFValidationResult()
        factor_id = factor.get("factor_id", "<unknown>")

        # ---- Step 1: Data Partition ----
        if partitioned_data is None:
            logger.error("Step 1 FAILED: no partition provided for factor %s", factor_id)
            return result
        result.step1_passed = True
        result.partition = partitioned_data
        logger.info(
            "[%s] Step 1 PASSED: data partitioned into %d train / %d val / "
            "%d buffer / %d test rows.",
            factor_id,
            len(partitioned_data.train_indices),
            len(partitioned_data.validation_indices),
            len(partitioned_data.buffer_indices),
            len(partitioned_data.test_indices),
        )

        # ---- Step 2: LLM Cutoff Isolation ----
        if len(partitioned_data.buffer_indices) > 0:
            logger.info(
                "[%s] Step 2 PASSED: %d rows in buffer zone excluded "
                "(cutoff=%s, buffer=%d months).",
                factor_id,
                len(partitioned_data.buffer_indices),
                partitioned_data.llm_cutoff_date,
                partitioned_data.buffer_months,
            )
        else:
            logger.info(
                "[%s] Step 2 PASSED: buffer zone is empty (no data "
                "overlaps the LLM cutoff window).",
                factor_id,
            )
        result.step2_passed = True

        # ---- Step 3: Leakage Detection Test ----
        if ic_train is None or ic_test is None:
            logger.warning(
                "[%s] Step 3 SKIPPED: IC series not provided.  "
                "Run backtest and supply ic_train/ic_test.",
                factor_id,
            )
            result.step3_passed = False
        else:
            ic_train = np.asarray(ic_train, dtype=np.float64)
            ic_test = np.asarray(ic_test, dtype=np.float64)

            # Drop NaN values.
            ic_train_clean = ic_train[~np.isnan(ic_train)]
            ic_test_clean = ic_test[~np.isnan(ic_test)]

            if len(ic_train_clean) < 2 or len(ic_test_clean) < 2:
                logger.warning(
                    "[%s] Step 3 SKIPPED: insufficient IC observations "
                    "(train=%d, test=%d).",
                    factor_id,
                    len(ic_train_clean),
                    len(ic_test_clean),
                )
                result.step3_passed = False
            else:
                ldt_result = self._ldt.test_leakage(
                    ic_train_clean, ic_test_clean
                )
                result.step3_ldt_result = ldt_result
                result.step3_passed = ldt_result.passed
                logger.info(
                    "[%s] Step 3 %s: LDT t_stat=%.4f, p=%.6f, alpha=%.4f",
                    factor_id,
                    "PASSED" if ldt_result.passed else "FAILED",
                    ldt_result.t_stat,
                    ldt_result.p_value,
                    ldt_result.alpha,
                )

        # ---- Step 4: Live Validation Gate ----
        # This step cannot be completed synchronously; it requires
        # 30 days of forward paper trading managed by FLO/LEMA.
        result.step4_paper_trade_required = True
        logger.info(
            "[%s] Step 4: paper-trade gate flagged -- %d calendar days "
            "of live validation required before deployment.",
            factor_id,
            self.paper_trade_days,
        )

        # ---- Overall verdict (Steps 1-3) ----
        result.overall_passed = (
            result.step1_passed
            and result.step2_passed
            and result.step3_passed
        )
        logger.info(
            "[%s] TIF overall: %s (step4 pending paper-trade).",
            factor_id,
            "PASSED" if result.overall_passed else "FAILED",
        )

        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_months(d: date, months: int) -> date:
    """Add *months* calendar months to date *d*, clamping to month end."""
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, max_day))


def _subtract_months(d: date, months: int) -> date:
    """Subtract *months* calendar months from date *d*."""
    month = d.month - 1 - months
    year = d.year + month // 12
    month = month % 12 + 1
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, max_day))


def _to_date(obj: Any) -> date:
    """Coerce a numpy datetime64 or similar to ``datetime.date``."""
    if isinstance(obj, date):
        return obj
    if hasattr(obj, "astype"):
        # numpy datetime64
        import pandas as pd
        return pd.Timestamp(obj).date()
    return date.fromisoformat(str(obj))
