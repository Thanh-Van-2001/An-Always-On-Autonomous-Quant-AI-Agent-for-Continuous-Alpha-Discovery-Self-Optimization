"""
Core data types for the AQAA system.

All dataclasses and enums shared across modules are defined here to
provide a single source of truth for the type vocabulary.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AlphaState(Enum):
    """Lifecycle states of an alpha factor within the AQAA pipeline."""

    HYPOTHESIS_GENERATED = auto()
    BACKTESTING = auto()
    PAPER_TRADING = auto()
    LIVE_DEPLOYED = auto()
    UNDER_REVIEW = auto()
    RETIRED = auto()
    ARCHIVED = auto()
    MUTATED = auto()


class MarketRegime(Enum):
    """Broad market-regime classification used by the regime detector."""

    BULL = auto()
    BEAR = auto()
    SIDEWAYS = auto()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HypothesisSpec:
    """A candidate hypothesis produced by the Hypothesis Engine.

    Attributes:
        id: Unique identifier for this hypothesis.
        description: Human-readable description of the investment thesis.
        source_type: Origin of the hypothesis (academic, news, or data).
        ast_hash: SHA-256 hash of the normalised AST for deduplication.
        novelty_score: Cosine-distance novelty vs. existing factor pool [0, 1].
        quality_score: LLM-assigned quality rating [0, 1].
        raw_prompt: The original prompt or snippet that seeded this hypothesis.
        created_at: Timestamp when the hypothesis was generated.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    source_type: str = "academic"  # academic | news | data
    ast_hash: str = ""
    novelty_score: float = 0.0
    quality_score: float = 0.0
    raw_prompt: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def compute_ast_hash(self, code: str) -> str:
        """Compute and store the SHA-256 hash of the given code string."""
        self.ast_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        return self.ast_hash


@dataclass
class BacktestResult:
    """Statistical summary returned by the vectorised backtester.

    Attributes:
        factor_id: Reference to the originating AlphaFactor.
        ic: Mean information coefficient.
        icir: Information coefficient information ratio (IC / std(IC)).
        sharpe: Annualised Sharpe ratio.
        mdd: Maximum drawdown (positive value, e.g. 0.15 = 15%).
        arr: Annualised rate of return.
        calmar: Calmar ratio (ARR / MDD).
        passed_ldt: Whether the factor survives the Lopez de Prado
                     deflated-Sharpe / combinatorial purged CV test.
        paper_trade_passed: Whether the factor passed the paper-trading
                            gate after forward validation.
    """

    factor_id: str = ""
    ic: float = 0.0
    icir: float = 0.0
    sharpe: float = 0.0
    mdd: float = 0.0
    arr: float = 0.0
    calmar: float = 0.0
    passed_ldt: bool = False
    paper_trade_passed: bool = False


@dataclass
class AlphaFactor:
    """Representation of an alpha factor throughout its full lifecycle.

    Attributes:
        id: Unique factor identifier.
        name: Short human-readable name.
        code: Executable Python code that computes the factor values.
        hypothesis: The originating HypothesisSpec description.
        state: Current lifecycle state.
        ic_mean: Running mean information coefficient.
        icir: Running IC information ratio.
        sharpe: Running annualised Sharpe ratio.
        mdd: Running maximum drawdown.
        created_at: When the factor was first created.
        deployed_at: When the factor entered LIVE_DEPLOYED (if ever).
        retired_at: When the factor was retired (if ever).
        lifespan_days: Number of days the factor has been / was live.
        metadata: Arbitrary extra information (provenance, tags, etc.).
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    code: str = ""
    hypothesis: str = ""
    state: AlphaState = AlphaState.HYPOTHESIS_GENERATED
    ic_mean: float = 0.0
    icir: float = 0.0
    sharpe: float = 0.0
    mdd: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    deployed_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    lifespan_days: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def transition(self, new_state: AlphaState) -> None:
        """Transition the factor to *new_state* and record timestamps."""
        self.state = new_state
        if new_state == AlphaState.LIVE_DEPLOYED and self.deployed_at is None:
            self.deployed_at = datetime.now(UTC)
        elif new_state in (AlphaState.RETIRED, AlphaState.ARCHIVED):
            self.retired_at = datetime.now(UTC)
            if self.deployed_at is not None:
                delta = self.retired_at - self.deployed_at
                self.lifespan_days = delta.days


@dataclass
class PortfolioWeights:
    """A snapshot of portfolio weights produced by the portfolio constructor.

    Attributes:
        weights: Mapping from symbol to signed weight.
        regime: The market regime active when weights were computed.
        timestamp: When the snapshot was generated.
        gross_exposure: Sum of absolute weights.
        kelly_fraction: Fractional Kelly sizing scalar applied.
    """

    weights: Dict[str, float] = field(default_factory=dict)
    regime: MarketRegime = MarketRegime.SIDEWAYS
    timestamp: datetime = field(default_factory=datetime.utcnow)
    gross_exposure: float = 1.0
    kelly_fraction: float = 0.5


@dataclass
class TradeOrder:
    """A single order emitted by the execution engine.

    Attributes:
        symbol: Ticker or instrument identifier.
        side: 'BUY' or 'SELL'.
        quantity: Number of shares / contracts / units.
        price: Limit price (0.0 for market orders).
        order_type: 'MARKET', 'LIMIT', 'TWAP', 'VWAP'.
        timestamp: When the order was created.
        execution_algo: Name of the execution algorithm used.
    """

    symbol: str = ""
    side: str = "BUY"
    quantity: float = 0.0
    price: float = 0.0
    order_type: str = "MARKET"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_algo: str = "TWAP"


@dataclass
class ICTimeSeries:
    """Time series of information-coefficient values for a single factor.

    Used by the IC Watchdog for decay detection and regime-change monitoring.

    Attributes:
        values: Ordered list of IC observations.
        timestamps: Corresponding datetime for each observation.
        factor_id: The factor this series belongs to.
    """

    values: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    factor_id: str = ""

    def append(self, value: float, ts: Optional[datetime] = None) -> None:
        """Append a new IC observation."""
        self.values.append(value)
        self.timestamps.append(ts or datetime.now(UTC))

    @property
    def mean(self) -> float:
        """Return the arithmetic mean of the IC series."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def std(self) -> float:
        """Return the sample standard deviation of the IC series."""
        n = len(self.values)
        if n < 2:
            return 0.0
        mu = self.mean
        return (sum((v - mu) ** 2 for v in self.values) / (n - 1)) ** 0.5

    @property
    def icir(self) -> float:
        """Return IC / std(IC), i.e. the information ratio."""
        s = self.std
        if s == 0.0:
            return 0.0
        return self.mean / s
