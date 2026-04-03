"""
Code Synthesis inner loop -- Writer/Judge agent pair.

Implements the iterative refinement protocol described in the KDD 2026 paper
(Section 3.1):

    (code*, r*) = argmax_{k <= K}  Judge( Writer(h, kb, code_{k-1}) )

where:
    h    = structured hypothesis from the Hypothesis Engine
    kb   = knowledge base (domain constraints, Qlib API reference, etc.)
    K    = maximum iterations (default 10)
    code = executable Qlib-compatible Python that computes a cross-sectional
           alpha factor
    r    = scalar reward / quality score assigned by the Judge

The Writer agent generates (or refines) factor code given a hypothesis and
the Judge agent evaluates the code for correctness, financial plausibility,
and statistical significance.  In this implementation the LLM calls are
**simulated** with deterministic template-based code generation so that the
module can be tested without an API key.

Template factor families
------------------------
- **Momentum**       : trailing return, rate-of-change, relative strength
- **Mean-reversion** : z-score of price relative to moving average
- **Volatility**     : realised vol, vol-of-vol, Garman-Klass
- **Fundamental**    : earnings yield, book-to-market, ROE
"""

from __future__ import annotations

import copy
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from aqaa.core.types import AlphaFactor, AlphaState, HypothesisSpec

logger = logging.getLogger(__name__)

# Maximum refinement iterations per hypothesis.
DEFAULT_MAX_ITERATIONS = 10

# Quality score threshold -- the Judge must assign at least this score for
# the code to be accepted.
DEFAULT_ACCEPT_SCORE = 0.6


# ---------------------------------------------------------------------------
# Template factor code (Qlib-compatible Python expressions)
# ---------------------------------------------------------------------------

_FACTOR_TEMPLATES: Dict[str, str] = {
    "momentum": textwrap.dedent("""\
        # Momentum factor: {description}
        # Hypothesis ID: {hypothesis_id}
        import numpy as np
        import pandas as pd

        def alpha(df: pd.DataFrame) -> pd.Series:
            \"\"\"Cross-sectional momentum factor.

            Parameters
            ----------
            df : pd.DataFrame
                Must contain 'close' column indexed by (date, asset).

            Returns
            -------
            pd.Series
                Factor values (higher = stronger momentum).
            \"\"\"
            lookback = {lookback}
            returns = df['close'].pct_change(lookback)
            # Winsorise at 1st/99th percentile per date.
            lower = returns.groupby(level=0).transform(lambda x: x.quantile(0.01))
            upper = returns.groupby(level=0).transform(lambda x: x.quantile(0.99))
            factor = returns.clip(lower=lower, upper=upper)
            # Cross-sectional z-score.
            factor = factor.groupby(level=0).transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            return factor
    """),

    "mean_reversion": textwrap.dedent("""\
        # Mean-reversion factor: {description}
        # Hypothesis ID: {hypothesis_id}
        import numpy as np
        import pandas as pd

        def alpha(df: pd.DataFrame) -> pd.Series:
            \"\"\"Cross-sectional mean-reversion factor.

            Parameters
            ----------
            df : pd.DataFrame
                Must contain 'close' column indexed by (date, asset).

            Returns
            -------
            pd.Series
                Factor values (lower = more mean-reverting = higher expected return).
            \"\"\"
            window = {window}
            ma = df['close'].rolling(window=window, min_periods=window // 2).mean()
            zscore = (df['close'] - ma) / (
                df['close'].rolling(window=window, min_periods=window // 2).std() + 1e-8
            )
            # Negate: stocks far below MA are expected to revert up.
            factor = -zscore
            factor = factor.groupby(level=0).transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            return factor
    """),

    "volatility": textwrap.dedent("""\
        # Volatility factor: {description}
        # Hypothesis ID: {hypothesis_id}
        import numpy as np
        import pandas as pd

        def alpha(df: pd.DataFrame) -> pd.Series:
            \"\"\"Cross-sectional low-volatility factor.

            Parameters
            ----------
            df : pd.DataFrame
                Must contain 'close' column indexed by (date, asset).

            Returns
            -------
            pd.Series
                Factor values (lower realised vol = higher expected risk-adjusted return).
            \"\"\"
            window = {window}
            daily_ret = df['close'].pct_change()
            realised_vol = daily_ret.rolling(window=window, min_periods=window // 2).std()
            # Negate: low-vol stocks get a higher factor score.
            factor = -realised_vol
            factor = factor.groupby(level=0).transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            return factor
    """),

    "fundamental": textwrap.dedent("""\
        # Fundamental factor: {description}
        # Hypothesis ID: {hypothesis_id}
        import numpy as np
        import pandas as pd

        def alpha(df: pd.DataFrame) -> pd.Series:
            \"\"\"Cross-sectional fundamental value factor.

            Parameters
            ----------
            df : pd.DataFrame
                Must contain 'earnings' and 'price' columns indexed by
                (date, asset).

            Returns
            -------
            pd.Series
                Factor values (higher earnings yield = more attractive).
            \"\"\"
            earnings_yield = df['earnings'] / (df['price'].abs() + 1e-8)
            factor = earnings_yield.groupby(level=0).transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            return factor
    """),
}

# Default hyper-parameters injected into templates.
_DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "momentum": {"lookback": 20},
    "mean_reversion": {"window": 20},
    "volatility": {"window": 20},
    "fundamental": {},
}


# ---------------------------------------------------------------------------
# Writer Agent
# ---------------------------------------------------------------------------

@dataclass
class WriterOutput:
    """Output of a single Writer iteration."""
    code: str
    iteration: int
    params: Dict[str, Any] = field(default_factory=dict)


class WriterAgent:
    """Generates Qlib-compatible factor code from a hypothesis.

    In the full AQAA system, this would invoke an LLM with a carefully
    structured prompt.  Here, deterministic template expansion is used so
    the module is testable offline.

    Parameters
    ----------
    templates : dict, optional
        Mapping from category name to a Python code template string.
        Falls back to the built-in ``_FACTOR_TEMPLATES``.
    """

    def __init__(
        self,
        templates: Optional[Dict[str, str]] = None,
    ) -> None:
        self.templates = templates or _FACTOR_TEMPLATES

    def generate(
        self,
        hypothesis: HypothesisSpec,
        knowledge_base: Dict[str, Any],
        previous_code: Optional[str] = None,
        iteration: int = 0,
    ) -> WriterOutput:
        """Generate or refine factor code for a hypothesis.

        Parameters
        ----------
        hypothesis : HypothesisSpec
            The structured hypothesis to implement.
        knowledge_base : dict
            Domain constraints and reference material.
        previous_code : str, optional
            Code from the previous iteration (for refinement).
        iteration : int
            Current iteration number (0-indexed).

        Returns
        -------
        WriterOutput
            The generated code and associated metadata.
        """
        category = self._resolve_category(hypothesis)
        template = self.templates.get(category)

        if template is None:
            logger.warning(
                "No template for category '%s'; falling back to momentum.",
                category,
            )
            category = "momentum"
            template = self.templates["momentum"]

        # Merge default params with any knowledge-base overrides.
        params = copy.deepcopy(_DEFAULT_PARAMS.get(category, {}))
        params.update(knowledge_base.get("params", {}))

        # On later iterations, perturb hyper-parameters to explore.
        if iteration > 0 and previous_code is not None:
            params = self._perturb_params(params, iteration)

        format_kwargs = {
            "description": hypothesis.description,
            "hypothesis_id": hypothesis.id,
            **params,
        }

        try:
            code = template.format(**format_kwargs)
        except KeyError as exc:
            logger.error("Template formatting failed: %s", exc)
            code = previous_code or ""

        logger.debug(
            "Writer iteration %d: category=%s, params=%s",
            iteration,
            category,
            params,
        )

        return WriterOutput(code=code, iteration=iteration, params=params)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_category(hypothesis: HypothesisSpec) -> str:
        """Map the hypothesis source_type / description to a template category."""
        cat = hypothesis.source_type.lower().strip()
        # Direct match.
        if cat in _FACTOR_TEMPLATES:
            return cat
        # Keyword heuristic.
        desc = hypothesis.description.lower()
        for key in _FACTOR_TEMPLATES:
            if key.replace("_", " ") in desc or key.replace("_", "-") in desc:
                return key
        # Fall back to momentum.
        return "momentum"

    @staticmethod
    def _perturb_params(
        params: Dict[str, Any],
        iteration: int,
    ) -> Dict[str, Any]:
        """Deterministically perturb numeric hyper-parameters."""
        rng = np.random.RandomState(seed=42 + iteration)
        perturbed = {}
        for k, v in params.items():
            if isinstance(v, int):
                delta = rng.randint(-5, 6)
                perturbed[k] = max(5, v + delta)
            elif isinstance(v, float):
                perturbed[k] = v * (1.0 + rng.uniform(-0.3, 0.3))
            else:
                perturbed[k] = v
        return perturbed


# ---------------------------------------------------------------------------
# Judge Agent
# ---------------------------------------------------------------------------

@dataclass
class JudgeVerdict:
    """Evaluation result from the Judge agent.

    Attributes:
        score: Quality / reward score in [0, 1].
        accepted: Whether the code meets the acceptance threshold.
        feedback: Natural-language feedback for the Writer's next iteration.
        checks: Detailed breakdown of individual checks.
    """
    score: float
    accepted: bool
    feedback: str
    checks: Dict[str, bool] = field(default_factory=dict)


class JudgeAgent:
    """Evaluates generated factor code for correctness and plausibility.

    The Judge performs a series of static checks:
    1. **Syntax** -- code can be parsed by the Python compiler.
    2. **Structure** -- contains a callable ``alpha`` function.
    3. **Safety** -- no disallowed operations (I/O, network, subprocess).
    4. **Financial plausibility** -- uses standard constructs (pct_change,
       rolling, groupby cross-sectional normalisation).

    Parameters
    ----------
    accept_threshold : float
        Minimum score to accept a candidate (default 0.6).
    """

    # Tokens that should NOT appear in factor code.
    _DISALLOWED_TOKENS = frozenset([
        "import os",
        "import sys",
        "subprocess",
        "open(",
        "requests.",
        "urllib",
        "socket",
        "__import__",
        "eval(",
        "exec(",
    ])

    # Tokens we WANT to see (financial domain).
    _DESIRED_TOKENS = [
        "def alpha",
        "pct_change",
        "rolling",
        "groupby",
    ]

    def __init__(self, accept_threshold: float = DEFAULT_ACCEPT_SCORE) -> None:
        self.accept_threshold = accept_threshold

    def evaluate(
        self,
        code: str,
        hypothesis: HypothesisSpec,
    ) -> JudgeVerdict:
        """Score the candidate factor code.

        Parameters
        ----------
        code : str
            The Python source code to evaluate.
        hypothesis : HypothesisSpec
            The originating hypothesis (used for relevance checking).

        Returns
        -------
        JudgeVerdict
        """
        checks: Dict[str, bool] = {}
        feedback_parts: List[str] = []

        # 1. Syntax check.
        syntax_ok = self._check_syntax(code)
        checks["syntax"] = syntax_ok
        if not syntax_ok:
            feedback_parts.append("Code has syntax errors; fix them.")

        # 2. Structure check.
        has_alpha = "def alpha" in code
        checks["structure"] = has_alpha
        if not has_alpha:
            feedback_parts.append(
                "Code must define a function named 'alpha'."
            )

        # 3. Safety check.
        safety_ok = self._check_safety(code)
        checks["safety"] = safety_ok
        if not safety_ok:
            feedback_parts.append(
                "Code contains disallowed operations (I/O, network, eval)."
            )

        # 4. Financial plausibility.
        plausibility_score, plausibility_feedback = self._check_plausibility(
            code
        )
        checks["plausibility"] = plausibility_score >= 0.5
        if plausibility_feedback:
            feedback_parts.append(plausibility_feedback)

        # 5. Non-trivial length check.
        non_trivial = len(code.strip().splitlines()) >= 5
        checks["non_trivial"] = non_trivial
        if not non_trivial:
            feedback_parts.append("Code is too short; add computation logic.")

        # Aggregate score: each check contributes equally.
        n_checks = len(checks)
        raw_score = sum(checks.values()) / n_checks if n_checks > 0 else 0.0
        # Blend in the plausibility detail score.
        score = 0.7 * raw_score + 0.3 * plausibility_score
        score = min(1.0, max(0.0, score))

        accepted = score >= self.accept_threshold
        feedback = " ".join(feedback_parts) if feedback_parts else "Code looks good."

        verdict = JudgeVerdict(
            score=score,
            accepted=accepted,
            feedback=feedback,
            checks=checks,
        )

        logger.debug(
            "Judge verdict: score=%.3f, accepted=%s, checks=%s",
            score,
            accepted,
            checks,
        )
        return verdict

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_syntax(code: str) -> bool:
        """Return True if *code* compiles without syntax errors."""
        try:
            compile(code, "<factor>", "exec")
            return True
        except SyntaxError:
            return False

    def _check_safety(self, code: str) -> bool:
        """Return True if no disallowed tokens are present."""
        for token in self._DISALLOWED_TOKENS:
            if token in code:
                logger.warning("Disallowed token found: '%s'", token)
                return False
        return True

    def _check_plausibility(self, code: str) -> Tuple[float, str]:
        """Heuristic financial-plausibility score in [0, 1]."""
        found = sum(1 for tok in self._DESIRED_TOKENS if tok in code)
        score = found / len(self._DESIRED_TOKENS)
        if score < 0.5:
            feedback = (
                "Consider using standard financial constructs: "
                "pct_change, rolling windows, cross-sectional groupby."
            )
        else:
            feedback = ""
        return score, feedback


# ---------------------------------------------------------------------------
# Code Synthesizer (orchestrator)
# ---------------------------------------------------------------------------

class CodeSynthesizer:
    """Orchestrates the Writer/Judge inner loop for factor code generation.

    Implements the optimization:

        (code*, r*) = argmax_{k <= K}  Judge( Writer(h, kb, code_{k-1}) )

    At each iteration the Writer produces (or refines) code and the Judge
    scores it.  The best-scoring code across all iterations is returned.

    Parameters
    ----------
    writer : WriterAgent, optional
        Custom writer; a default is created if not supplied.
    judge : JudgeAgent, optional
        Custom judge; a default is created if not supplied.
    max_iterations : int
        Maximum refinement iterations per hypothesis (K).
    """

    def __init__(
        self,
        writer: Optional[WriterAgent] = None,
        judge: Optional[JudgeAgent] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self.writer = writer or WriterAgent()
        self.judge = judge or JudgeAgent()
        self.max_iterations = max_iterations

    def synthesize(
        self,
        hypothesis: HypothesisSpec,
        knowledge_base: Optional[Dict[str, Any]] = None,
        max_iterations: Optional[int] = None,
    ) -> AlphaFactor:
        """Run the Writer/Judge loop and return the best factor.

        Parameters
        ----------
        hypothesis : HypothesisSpec
            The structured hypothesis to implement.
        knowledge_base : dict, optional
            Domain constraints passed to the Writer (default empty).
        max_iterations : int, optional
            Override the instance-level maximum.

        Returns
        -------
        AlphaFactor
            The factor whose code received the highest Judge score.
            Its ``state`` is ``HYPOTHESIS_GENERATED`` -- further promotion
            is handled by the Backtester and TIF.
        """
        kb = knowledge_base or {}
        K = max_iterations if max_iterations is not None else self.max_iterations

        best_code: str = ""
        best_score: float = -1.0
        best_verdict: Optional[JudgeVerdict] = None
        previous_code: Optional[str] = None

        logger.info(
            "CodeSynthesizer: starting %d-iteration loop for hypothesis '%s'",
            K,
            hypothesis.id,
        )

        for k in range(K):
            # --- Writer step ---
            writer_out = self.writer.generate(
                hypothesis=hypothesis,
                knowledge_base=kb,
                previous_code=previous_code,
                iteration=k,
            )
            code = writer_out.code

            # --- Judge step ---
            verdict = self.judge.evaluate(code, hypothesis)

            logger.info(
                "  Iteration %d/%d: score=%.3f  accepted=%s",
                k + 1,
                K,
                verdict.score,
                verdict.accepted,
            )

            if verdict.score > best_score:
                best_score = verdict.score
                best_code = code
                best_verdict = verdict

            # Early exit if the Judge fully accepts.
            if verdict.accepted:
                logger.info(
                    "  Early stop at iteration %d (score=%.3f >= threshold).",
                    k + 1,
                    verdict.score,
                )
                break

            # Feed code + feedback into next Writer iteration.
            previous_code = code

        # Build the AlphaFactor output.
        factor = AlphaFactor(
            name=f"factor_{hypothesis.id}",
            code=best_code,
            hypothesis=hypothesis.description,
            state=AlphaState.HYPOTHESIS_GENERATED,
            metadata={
                "synthesis_iterations": k + 1,  # noqa: F821 (k is in scope)
                "judge_score": best_score,
                "judge_checks": best_verdict.checks if best_verdict else {},
                "judge_feedback": best_verdict.feedback if best_verdict else "",
                "hypothesis_id": hypothesis.id,
                "source_type": hypothesis.source_type,
            },
        )

        logger.info(
            "CodeSynthesizer: best code selected after %d iterations "
            "(score=%.3f).",
            k + 1,
            best_score,
        )

        return factor
