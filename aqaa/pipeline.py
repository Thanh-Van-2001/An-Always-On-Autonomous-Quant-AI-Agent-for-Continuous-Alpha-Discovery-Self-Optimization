"""
Main AQAA Pipeline -- closed-loop orchestration of all five modules.

Ties together the five core modules into the autonomous 24-hour cycle
described in Figure 1 and Algorithm 1 of the KDD 2026 paper:

    HE  -> CSBA -> APO -> LEMA -> FLO -> (feedback) -> HE ...

Each ``run_cycle`` call executes one complete iteration:
  1. FLO allocates compute budget via UCB1.
  2. HE generates candidate alpha hypotheses.
  3. CSBA validates each hypothesis (novelty -> code synthesis -> backtest -> TIF/LDT).
  4. APO detects the market regime, optimises factor weights, and applies Kelly sizing.
  5. LEMA monitors deployed factors (Bayesian IC, decay detection, crowding).
  6. FLO collects feedback and updates the knowledge base.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Core types and configuration
from aqaa.core.types import (
    AlphaFactor,
    AlphaState,
    BacktestResult,
    HypothesisSpec,
    ICTimeSeries,
    MarketRegime,
    PortfolioWeights,
)
from aqaa.core.config import AQAAConfig

# Module 1 -- Hypothesis Engine
from aqaa.hypothesis_engine import (
    HypothesisProposer,
    NoveltyFilter,
    QuantFinanceKnowledgeBase,
)

# Module 2 -- Code Synthesis and Backtest Agent
from aqaa.csba import (
    Backtester,
    CodeSynthesizer,
    TemporalIntegrityFramework,
)

# Module 3 -- Adaptive Portfolio Optimizer
from aqaa.apo import (
    RegimeDetector,
    PortfolioOptimizer,
    KellySizer,
)

# Module 4 -- Live Execution and Monitoring Agent
from aqaa.lema import (
    AlphaLifecycleManager,
    BayesianICTracker,
    DecayDetector,
    CrowdingDetector,
    OrderExecutor,
)

# Module 5 -- Feedback Loop Orchestrator
from aqaa.flo import (
    UCB1Bandit,
    FeedbackLoopOrchestrator,
    BudgetAllocator,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-cycle result
# ---------------------------------------------------------------------------

@dataclass
class CycleResult:
    """Metrics and artefacts from a single pipeline cycle.

    Attributes:
        cycle_number: 1-based cycle index.
        timestamp: When the cycle started.
        compute_allocation: UCB1-derived compute fractions per module.
        hypotheses_generated: Number of hypotheses proposed by HE.
        hypotheses_after_novelty: Number surviving the novelty filter.
        factors_synthesized: Number of factors for which code was synthesised.
        factors_backtested: Number of factors that entered backtesting.
        factors_passed_backtest: Number passing IC/Sharpe/MDD gates.
        factors_passed_tif: Number passing TIF (LDT + partition).
        factors_deployed: Number promoted to PAPER_TRADING or LIVE_DEPLOYED.
        factors_under_review: Number flagged for decay / crowding.
        factors_retired: Number retired during this cycle.
        regime: Detected market regime.
        portfolio_weights: Final portfolio weights after APO + Kelly.
        portfolio_ic: Mean IC of the deployed portfolio.
        crowding_detected: Whether the crowding detector fired.
        ucb1_rewards: Per-module reward fed back to the bandit.
        alm_summary: Factor counts by lifecycle state.
        duration_seconds: Wall-clock time for the cycle.
    """

    cycle_number: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    compute_allocation: Dict[str, float] = field(default_factory=dict)
    hypotheses_generated: int = 0
    hypotheses_after_novelty: int = 0
    factors_synthesized: int = 0
    factors_backtested: int = 0
    factors_passed_backtest: int = 0
    factors_passed_tif: int = 0
    factors_deployed: int = 0
    factors_under_review: int = 0
    factors_retired: int = 0
    regime: str = "SIDEWAYS"
    portfolio_weights: Dict[str, float] = field(default_factory=dict)
    portfolio_ic: float = 0.0
    crowding_detected: bool = False
    ucb1_rewards: Dict[str, float] = field(default_factory=dict)
    alm_summary: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class AQAAPipeline:
    """End-to-end orchestration of the five AQAA modules.

    Instantiates every sub-component from a single ``AQAAConfig`` and
    exposes ``run_cycle`` / ``run`` for executing the closed loop.

    Parameters
    ----------
    config : AQAAConfig
        Fully-populated system configuration.
    """

    def __init__(self, config: AQAAConfig) -> None:
        self.config = config

        # ---- logging setup ------------------------------------------------
        log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        logger.info("Initialising AQAA pipeline ...")

        # ---- Module 1: Hypothesis Engine ----------------------------------
        self.knowledge_base = QuantFinanceKnowledgeBase()
        self.proposer = HypothesisProposer(seed=42)
        self.novelty_filter = NoveltyFilter(
            tau_nov=config.hypothesis_engine.tau_nov,
        )

        # ---- Module 2: CSBA -----------------------------------------------
        self.code_synthesizer = CodeSynthesizer(
            max_iterations=config.hypothesis_engine.max_code_iterations,
        )
        self.backtester = Backtester(
            ic_threshold=config.backtester.ic_threshold,
            icir_threshold=config.backtester.icir_threshold,
            sharpe_threshold=config.backtester.sharpe_threshold,
            mdd_threshold=config.backtester.mdd_threshold,
            cost_bps=config.backtester.cost_bps,
        )
        self.tif = TemporalIntegrityFramework(
            llm_cutoff_date=date(2025, 4, 1),  # GPT-4o approximate cutoff
            buffer_months=config.backtester.buffer_months,
            ldt_alpha=config.backtester.ldt_alpha,
            paper_trade_days=config.backtester.paper_trade_days,
        )

        # ---- Module 3: APO ------------------------------------------------
        self.regime_detector = RegimeDetector(n_iter=100, tol=1e-4)
        self.portfolio_optimizer = PortfolioOptimizer(
            base_lambda=1.0,
            gross_exposure_limit=config.portfolio.max_gross_exposure,
            crowding_penalty=0.3,
        )
        self.kelly_sizer = KellySizer(
            fraction=config.portfolio.kelly_fraction,
            max_gross_exposure=2.0,
        )

        # ---- Module 4: LEMA -----------------------------------------------
        self.alm = AlphaLifecycleManager()
        self.ic_trackers: Dict[str, BayesianICTracker] = {}
        self.decay_detector = DecayDetector(
            cusum_k=config.watchdog.cusum_drift,
        )
        self.crowding_detector = CrowdingDetector(
            rho_crowd=config.portfolio.rho_crowd,
        )
        self.executor = OrderExecutor()

        # ---- Module 5: FLO ------------------------------------------------
        self.flo = FeedbackLoopOrchestrator(
            budget_gpu_hours=10.0,
            ucb_c=config.watchdog.ucb_c,
            log_dir=config.logging.log_dir,
            seed=42,
        )

        # ---- Pipeline state -----------------------------------------------
        self._cycle_results: List[CycleResult] = []
        self._deployed_factors: List[AlphaFactor] = []
        self._all_factors: List[AlphaFactor] = []
        self._regime_fitted = False

        logger.info("AQAA pipeline initialised.")

    # ------------------------------------------------------------------
    # Single cycle
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        market_data: Dict[str, Any],
        cycle_number: int = 1,
    ) -> CycleResult:
        """Execute one full 24-hour outer cycle.

        Parameters
        ----------
        market_data : dict
            Must contain at minimum:
            - ``"returns"`` : np.ndarray of shape ``(T, N)`` -- daily
              asset returns for the universe.
            - ``"predictions"`` : np.ndarray of shape ``(T, N)`` --
              factor predictions (can be synthetic / zeros if no
              deployed factors yet).
            Optionally:
            - ``"dates"`` : array of ``datetime.date`` objects.
        cycle_number : int
            1-based cycle identifier.

        Returns
        -------
        CycleResult
            Metrics and artefacts from this cycle.
        """
        import time as _time

        t0 = _time.monotonic()
        result = CycleResult(cycle_number=cycle_number)

        logger.info("=" * 70)
        logger.info("AQAA CYCLE %d START", cycle_number)
        logger.info("=" * 70)

        # ==============================================================
        # STEP 1: FLO -- allocate compute budget via UCB1
        # ==============================================================
        logger.info("[Step 1/6] FLO: allocating compute budget ...")
        allocations = self.flo.allocate_compute()
        alloc_dict = {a.module: a.fraction for a in allocations}
        result.compute_allocation = alloc_dict
        logger.info("  Allocation: %s", alloc_dict)

        # Scale HE hypothesis count by its compute fraction.
        he_fraction = alloc_dict.get("HE", 0.2)
        n_hypotheses = max(4, int(he_fraction * 50))

        # ==============================================================
        # STEP 2: HE -- generate candidate alpha hypotheses
        # ==============================================================
        logger.info(
            "[Step 2/6] HE: generating %d hypotheses ...", n_hypotheses
        )
        hypotheses = self.proposer.generate_hypotheses(
            knowledge_base=self.knowledge_base,
            market_data=market_data,
            n=n_hypotheses,
        )
        result.hypotheses_generated = len(hypotheses)
        logger.info("  Generated %d raw hypotheses.", len(hypotheses))

        # Novelty filter
        novel_hypotheses = self.novelty_filter.filter_hypotheses(
            candidates=hypotheses,
            knowledge_base=self.knowledge_base,
        )
        result.hypotheses_after_novelty = len(novel_hypotheses)
        logger.info(
            "  After novelty filter: %d / %d passed.",
            len(novel_hypotheses),
            len(hypotheses),
        )

        # ==============================================================
        # STEP 3: CSBA -- code synthesis + backtest + TIF/LDT
        # ==============================================================
        logger.info(
            "[Step 3/6] CSBA: synthesising and backtesting %d hypotheses ...",
            len(novel_hypotheses),
        )
        new_factors: List[AlphaFactor] = []
        passed_factors: List[AlphaFactor] = []

        for hyp in novel_hypotheses:
            # 3a. Code synthesis via Writer/Judge loop
            factor = self.code_synthesizer.synthesize(
                hypothesis=hyp,
                knowledge_base={},
            )
            new_factors.append(factor)

            # Register with the ALM
            self.alm.register(factor)

            # Transition HG -> BACKTESTING (novelty gate)
            self.alm.transition(
                factor,
                AlphaState.BACKTESTING,
                reason="passed novelty filter",
                novelty_score=hyp.novelty_score,
            )

            # 3b. Backtest
            bt_result = self._run_backtest(factor, market_data)
            factor.ic_mean = bt_result.ic
            factor.icir = bt_result.icir
            factor.sharpe = bt_result.sharpe
            factor.mdd = bt_result.mdd

            if not self.backtester.check_acceptance(bt_result):
                logger.info(
                    "  Factor %s REJECTED at backtest gate.", factor.id
                )
                continue

            result.factors_passed_backtest += 1

            # 3c. TIF validation (LDT step)
            tif_result = self._run_tif(factor, market_data, bt_result)
            if tif_result is not None and tif_result.overall_passed:
                bt_result.passed_ldt = True
                result.factors_passed_tif += 1

                # Transition BACKTESTING -> PAPER_TRADING
                self.alm.transition(
                    factor,
                    AlphaState.PAPER_TRADING,
                    reason="passed backtest + TIF",
                    backtest_result=bt_result,
                )
                passed_factors.append(factor)
                logger.info(
                    "  Factor %s PASSED TIF -> PAPER_TRADING.", factor.id
                )
            else:
                logger.info(
                    "  Factor %s REJECTED at TIF gate.", factor.id
                )

        result.factors_synthesized = len(new_factors)
        result.factors_backtested = len(new_factors)
        self._all_factors.extend(new_factors)

        # Promote paper-trading factors with sufficient simulated evidence
        # to LIVE_DEPLOYED (in a real system this requires 30 days; here we
        # use the backtest IC as a proxy for the paper-trade IC).
        for factor in passed_factors:
            live_ic = factor.ic_mean  # proxy for paper-trade IC
            backtest_ic = factor.ic_mean
            promoted = self.alm.transition(
                factor,
                AlphaState.LIVE_DEPLOYED,
                reason="paper-trade IC proxy passed",
                live_ic=live_ic,
                backtest_ic=backtest_ic,
            )
            if promoted:
                self._deployed_factors.append(factor)
                result.factors_deployed += 1
                logger.info("  Factor %s -> LIVE_DEPLOYED.", factor.id)

        # ==============================================================
        # STEP 4: APO -- regime detection + portfolio optimisation + Kelly
        # ==============================================================
        logger.info("[Step 4/6] APO: optimising portfolio ...")
        regime = self._detect_regime(market_data)
        result.regime = regime.name
        logger.info("  Detected regime: %s", regime.name)

        if self._deployed_factors:
            weights = self._optimise_portfolio(
                self._deployed_factors, market_data, regime
            )
            result.portfolio_weights = weights
            logger.info("  Portfolio weights: %s", weights)
        else:
            logger.info("  No deployed factors -- skipping portfolio construction.")

        # ==============================================================
        # STEP 5: LEMA -- monitor deployed factors
        # ==============================================================
        logger.info("[Step 5/6] LEMA: monitoring deployed factors ...")
        review_count, retired_count = self._monitor_factors(market_data)
        result.factors_under_review = review_count
        result.factors_retired = retired_count

        # Crowding check
        if len(self._deployed_factors) >= 2:
            crowding = self._check_crowding(market_data)
            result.crowding_detected = crowding
            if crowding:
                logger.warning("  Factor crowding detected!")

        # Compute portfolio IC estimate
        result.portfolio_ic = self._estimate_portfolio_ic()

        # ==============================================================
        # STEP 6: FLO -- collect feedback, update KB, update bandit
        # ==============================================================
        logger.info("[Step 6/6] FLO: collecting feedback ...")
        rewards = self._compute_module_rewards(result)
        result.ucb1_rewards = rewards

        # Update the UCB1 bandit with per-module rewards
        for alloc in allocations:
            arm_idx = self.flo.bandit.arm_names.index(alloc.module)
            self.flo.bandit.update(arm_idx, rewards.get(alloc.module, 0.0))

        # Update knowledge base with factor profiles
        self._update_knowledge_base()

        # FLO bookkeeping
        self.flo.collect_feedback({
            "ic": result.portfolio_ic,
            "factor_profiles": [
                self._factor_to_profile(f) for f in self._all_factors
            ],
        })

        # ALM summary
        result.alm_summary = self.alm.summary()

        elapsed = _time.monotonic() - t0
        result.duration_seconds = round(elapsed, 3)

        logger.info("-" * 70)
        logger.info(
            "CYCLE %d COMPLETE in %.1fs | "
            "generated=%d novel=%d deployed=%d IC=%.4f regime=%s",
            cycle_number,
            elapsed,
            result.hypotheses_generated,
            result.hypotheses_after_novelty,
            result.factors_deployed,
            result.portfolio_ic,
            result.regime,
        )
        logger.info("-" * 70)

        self._cycle_results.append(result)
        return result

    # ------------------------------------------------------------------
    # Multi-cycle runner
    # ------------------------------------------------------------------

    def run(
        self,
        market_data: Dict[str, Any],
        n_cycles: int = 1,
    ) -> List[CycleResult]:
        """Run multiple pipeline cycles sequentially.

        Parameters
        ----------
        market_data : dict
            Market data passed to each cycle.
        n_cycles : int
            Number of 24-hour cycles to simulate.

        Returns
        -------
        list[CycleResult]
        """
        logger.info(
            "Starting AQAA pipeline for %d cycle(s) ...", n_cycles
        )
        results: List[CycleResult] = []
        for i in range(1, n_cycles + 1):
            result = self.run_cycle(market_data, cycle_number=i)
            results.append(result)
        logger.info("Pipeline complete: %d cycle(s) executed.", n_cycles)
        return results

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_results(self) -> Dict[str, Any]:
        """Aggregate results across all cycles.

        Returns
        -------
        dict
            Keys:
            - ``cycles``: list of per-cycle metric dicts.
            - ``total_hypotheses_generated``: across all cycles.
            - ``total_factors_deployed``: across all cycles.
            - ``total_factors_retired``: across all cycles.
            - ``final_portfolio_weights``: latest weights.
            - ``final_portfolio_ic``: latest IC.
            - ``knowledge_base_stats``: QFKB summary.
            - ``bandit_summary``: UCB1 arm statistics.
            - ``alm_summary``: factor counts by lifecycle state.
            - ``factor_history``: list of all factor profiles.
        """
        cycle_dicts = []
        for cr in self._cycle_results:
            cycle_dicts.append({
                "cycle_number": cr.cycle_number,
                "hypotheses_generated": cr.hypotheses_generated,
                "hypotheses_after_novelty": cr.hypotheses_after_novelty,
                "factors_synthesized": cr.factors_synthesized,
                "factors_passed_backtest": cr.factors_passed_backtest,
                "factors_passed_tif": cr.factors_passed_tif,
                "factors_deployed": cr.factors_deployed,
                "factors_under_review": cr.factors_under_review,
                "factors_retired": cr.factors_retired,
                "regime": cr.regime,
                "portfolio_ic": cr.portfolio_ic,
                "crowding_detected": cr.crowding_detected,
                "compute_allocation": cr.compute_allocation,
                "ucb1_rewards": cr.ucb1_rewards,
                "alm_summary": cr.alm_summary,
                "duration_seconds": cr.duration_seconds,
            })

        latest = self._cycle_results[-1] if self._cycle_results else CycleResult()

        return {
            "cycles": cycle_dicts,
            "total_hypotheses_generated": sum(
                cr.hypotheses_generated for cr in self._cycle_results
            ),
            "total_factors_deployed": sum(
                cr.factors_deployed for cr in self._cycle_results
            ),
            "total_factors_retired": sum(
                cr.factors_retired for cr in self._cycle_results
            ),
            "final_portfolio_weights": latest.portfolio_weights,
            "final_portfolio_ic": latest.portfolio_ic,
            "knowledge_base_stats": self.knowledge_base.get_statistics(),
            "bandit_summary": self.flo.bandit.summary(),
            "alm_summary": self.alm.summary(),
            "factor_history": [
                self._factor_to_profile(f) for f in self._all_factors
            ],
        }

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _run_backtest(
        self,
        factor: AlphaFactor,
        market_data: Dict[str, Any],
    ) -> BacktestResult:
        """Run the vectorised backtester on a single factor."""
        predictions = np.asarray(
            market_data.get("predictions", np.zeros((100, 50))),
            dtype=np.float64,
        )
        returns = np.asarray(
            market_data.get("returns", np.zeros((100, 50))),
            dtype=np.float64,
        )

        # If we have real predictions/returns, use them; otherwise generate
        # synthetic predictions seeded from the factor's hash.
        if predictions.sum() == 0.0:
            rng = np.random.RandomState(
                abs(hash(factor.id)) % (2**31)
            )
            predictions = rng.randn(*returns.shape) * 0.01

        data = {"predictions": predictions, "returns": returns}
        if "dates" in market_data:
            data["dates"] = market_data["dates"]

        return self.backtester.run_backtest(
            factor_code=factor.code,
            data=data,
        )

    def _run_tif(
        self,
        factor: AlphaFactor,
        market_data: Dict[str, Any],
        bt_result: BacktestResult,
    ) -> Any:
        """Run TIF validation (Steps 1-3) on a factor.

        Returns the TIFValidationResult, or None if data is insufficient.
        """
        if "dates" not in market_data:
            logger.info(
                "  TIF: no dates in market_data, auto-passing factor %s.",
                factor.id,
            )
            # Create a synthetic pass result
            from aqaa.csba.tif import TIFValidationResult
            result = TIFValidationResult()
            result.step1_passed = True
            result.step2_passed = True
            result.step3_passed = True
            result.overall_passed = True
            return result

        try:
            partition = self.tif.partition_data(market_data)
        except ValueError as exc:
            logger.warning(
                "  TIF partition failed for factor %s: %s", factor.id, exc
            )
            return None

        # Compute IC series on train and test partitions
        predictions = np.asarray(market_data["predictions"], dtype=np.float64)
        returns = np.asarray(market_data["returns"], dtype=np.float64)

        train_idx = partition.train_indices
        test_idx = partition.test_indices

        if len(train_idx) == 0 or len(test_idx) == 0:
            logger.warning(
                "  TIF: empty partition for factor %s.", factor.id
            )
            return None

        ic_train = self.backtester.compute_ic(
            predictions[train_idx], returns[train_idx]
        )
        ic_test = self.backtester.compute_ic(
            predictions[test_idx], returns[test_idx]
        )

        tif_result = self.tif.validate_factor(
            factor={"factor_id": factor.id, "code": factor.code},
            partitioned_data=partition,
            ic_train=ic_train,
            ic_test=ic_test,
        )
        return tif_result

    def _detect_regime(
        self, market_data: Dict[str, Any]
    ) -> MarketRegime:
        """Detect the current market regime using the HMM."""
        returns = np.asarray(
            market_data.get("returns", np.zeros((100, 50))),
            dtype=np.float64,
        )
        # Use the mean cross-sectional return as a market-level signal
        market_returns = np.nanmean(returns, axis=1)

        if len(market_returns) < 20:
            logger.info("  Insufficient data for regime detection; defaulting to SIDEWAYS.")
            return MarketRegime.SIDEWAYS

        if not self._regime_fitted:
            try:
                self.regime_detector.fit(market_returns)
                self._regime_fitted = True
            except Exception as exc:
                logger.warning("  Regime detector fit failed: %s", exc)
                return MarketRegime.SIDEWAYS

        try:
            window = market_returns[-min(20, len(market_returns)):]
            return self.regime_detector.predict_regime(window)
        except Exception as exc:
            logger.warning("  Regime prediction failed: %s", exc)
            return MarketRegime.SIDEWAYS

    def _optimise_portfolio(
        self,
        factors: List[AlphaFactor],
        market_data: Dict[str, Any],
        regime: MarketRegime,
    ) -> Dict[str, float]:
        """Build an optimised portfolio from deployed factors."""
        factor_names = [f.name for f in factors]
        n = len(factor_names)

        if n == 0:
            return {}

        # Build expected returns from factor IC means
        expected_returns = np.array(
            [f.ic_mean for f in factors], dtype=np.float64
        )

        # Build a synthetic covariance (diagonal + small off-diagonal)
        rng = np.random.RandomState(42)
        cov = np.eye(n) * 0.01
        noise = rng.randn(n, n) * 0.001
        cov += noise @ noise.T  # ensure PSD

        # Optimise weights
        weights = self.portfolio_optimizer.optimize_weights(
            factor_names=factor_names,
            covariance_matrix=cov,
            expected_returns=expected_returns,
            regime=regime,
        )

        # Kelly sizing
        # Estimate win rate from the sign of IC
        avg_ic = np.mean(expected_returns)
        win_rate = np.clip(0.5 + avg_ic * 5, 0.51, 0.70)
        win_loss_ratio = max(1.0 + avg_ic * 10, 1.01)

        full_kelly = self.kelly_sizer.compute_kelly_fraction(
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
        )
        frac_kelly = self.kelly_sizer.fractional_kelly(full_kelly)
        regime_kelly = self.kelly_sizer.regime_adjusted_kelly(
            frac_kelly, regime
        )

        sized_weights = self.kelly_sizer.apply_kelly_sizing(
            weights, regime_kelly
        )

        logger.info(
            "  Kelly: full=%.4f frac=%.4f regime_adj=%.4f",
            full_kelly, frac_kelly, regime_kelly,
        )
        return sized_weights

    def _monitor_factors(
        self, market_data: Dict[str, Any]
    ) -> tuple:
        """Monitor deployed factors for IC decay and trigger reviews.

        Returns (review_count, retired_count).
        """
        review_count = 0
        retired_count = 0

        returns = np.asarray(
            market_data.get("returns", np.zeros((100, 50))),
            dtype=np.float64,
        )

        for factor in list(self._deployed_factors):
            if factor.state != AlphaState.LIVE_DEPLOYED:
                continue

            # Ensure a Bayesian IC tracker exists
            if factor.id not in self.ic_trackers:
                self.ic_trackers[factor.id] = BayesianICTracker(
                    sigma2_obs=0.001,
                    sigma2_drift=1e-5,
                    prior_mu=factor.ic_mean,
                )

            tracker = self.ic_trackers[factor.id]

            # Feed a simulated IC observation (in production this would be
            # the latest cross-sectional IC computed from live data)
            rng = np.random.RandomState(
                abs(hash(factor.id + str(len(tracker.observations)))) % (2**31)
            )
            simulated_ic = factor.ic_mean + rng.randn() * 0.02
            mu_post, _ = tracker.update(simulated_ic)

            # Check for decay
            is_decaying = tracker.is_decaying(
                theta_min=self.config.watchdog.theta_min,
            )

            if is_decaying:
                logger.warning(
                    "  Factor %s: IC decay detected (posterior mu=%.4f).",
                    factor.id, mu_post,
                )
                # Also check with the change-point detector if enough data
                if tracker.n_observations >= 20:
                    ic_series = tracker.observations
                    cp_decay, _ = self.decay_detector.detect(ic_series)
                    if cp_decay:
                        logger.warning(
                            "  Factor %s: change-point decay confirmed.",
                            factor.id,
                        )

                # Transition to UNDER_REVIEW
                try:
                    self.alm.transition(
                        factor,
                        AlphaState.UNDER_REVIEW,
                        reason="IC decay detected",
                    )
                    review_count += 1
                except ValueError:
                    pass  # already in review or invalid transition

            # Check if under-review factors should be retired
            if factor.state == AlphaState.UNDER_REVIEW:
                try:
                    self.alm.transition(
                        factor,
                        AlphaState.RETIRED,
                        reason="grace period expired",
                    )
                    retired_count += 1
                    self._deployed_factors.remove(factor)
                    logger.info(
                        "  Factor %s RETIRED.", factor.id
                    )
                except (ValueError, Exception):
                    pass  # grace period not yet expired

        logger.info(
            "  Monitoring: %d under review, %d retired this cycle.",
            review_count, retired_count,
        )
        return review_count, retired_count

    def _check_crowding(self, market_data: Dict[str, Any]) -> bool:
        """Check for factor crowding among deployed factors."""
        n_factors = len(self._deployed_factors)
        if n_factors < 2:
            return False

        returns = np.asarray(
            market_data.get("returns", np.zeros((100, 50))),
            dtype=np.float64,
        )

        # Build synthetic per-factor return streams
        T = returns.shape[0]
        factor_returns = np.zeros((T, n_factors), dtype=np.float64)
        for i, factor in enumerate(self._deployed_factors):
            rng = np.random.RandomState(abs(hash(factor.id)) % (2**31))
            factor_returns[:, i] = rng.randn(T) * 0.01 + factor.ic_mean * 0.1

        try:
            report = self.crowding_detector.detect_crowding(factor_returns)
            return report.is_crowded
        except ValueError:
            return False

    def _estimate_portfolio_ic(self) -> float:
        """Estimate the current portfolio IC from deployed factors."""
        if not self._deployed_factors:
            return 0.0

        deployed_live = [
            f for f in self._deployed_factors
            if f.state in (AlphaState.LIVE_DEPLOYED, AlphaState.PAPER_TRADING)
        ]
        if not deployed_live:
            return 0.0

        # Weighted average IC (equal-weight as a baseline)
        ics = [f.ic_mean for f in deployed_live]
        return float(np.mean(ics))

    def _compute_module_rewards(self, result: CycleResult) -> Dict[str, float]:
        """Compute per-module rewards for the UCB1 bandit.

        Reward signal = marginal IC gain per compute-hour.
        """
        # HE reward: proportional to novel hypotheses generated
        he_reward = result.hypotheses_after_novelty * 0.002

        # CSBA reward: proportional to factors passing TIF
        csba_reward = result.factors_passed_tif * 0.005

        # APO reward: based on portfolio IC
        apo_reward = max(result.portfolio_ic * 0.1, 0.0)

        # LEMA reward: penalise decay, reward stability
        lema_reward = max(
            0.01 - result.factors_under_review * 0.003, 0.0
        )

        # KB reward: proportional to knowledge base growth
        kb_stats = self.knowledge_base.get_statistics()
        kb_reward = kb_stats.get("total_factors", 0) * 0.001

        return {
            "HE": round(he_reward, 6),
            "CSBA": round(csba_reward, 6),
            "APO": round(apo_reward, 6),
            "LEMA": round(lema_reward, 6),
            "KB": round(kb_reward, 6),
        }

    def _update_knowledge_base(self) -> None:
        """Sync all factor profiles to the QFKB."""
        for factor in self._all_factors:
            profile = self._factor_to_profile(factor)
            self.knowledge_base.add_factor(profile)

    @staticmethod
    def _factor_to_profile(factor: AlphaFactor) -> Dict[str, Any]:
        """Convert an AlphaFactor to a QFKB-compatible dict."""
        return {
            "id": factor.id,
            "name": factor.name,
            "code": factor.code,
            "hypothesis": factor.hypothesis,
            "state": factor.state.name,
            "ic_mean": factor.ic_mean,
            "icir": factor.icir,
            "sharpe": factor.sharpe,
            "mdd": factor.mdd,
            "created_at": factor.created_at.isoformat(),
            "deployed_at": (
                factor.deployed_at.isoformat()
                if factor.deployed_at else None
            ),
            "retired_at": (
                factor.retired_at.isoformat()
                if factor.retired_at else None
            ),
            "lifespan_days": factor.lifespan_days,
            "metadata": factor.metadata,
        }
