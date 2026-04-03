"""
CSBA -- Code Synthesis and Backtest Agent.

This module implements the second stage of the AQAA pipeline as described in
the KDD 2026 paper.  It takes a structured ``HypothesisSpec`` from the
Hypothesis Engine, synthesizes executable factor code via a Writer/Judge LLM
loop, back-tests the resulting alpha factor, and subjects it to the
**Temporal Integrity Framework (TIF)** -- a four-step protocol designed to
eliminate data-snooping and memorization bias that arises when LLMs are used
to generate trading strategies.

Submodules
----------
tif
    Temporal Integrity Framework: data partitioning, LLM-cutoff isolation,
    leakage detection, and live-validation gating.
ldt
    Leakage Detection Test -- the statistical core of TIF Step 3.
backtester
    Vectorised backtesting engine with IC, ICIR, Sharpe, MDD, Calmar metrics.
code_synthesizer
    Writer/Judge inner loop for iterative factor code generation.
"""

from aqaa.csba.backtester import Backtester, BacktestMetrics
from aqaa.csba.code_synthesizer import CodeSynthesizer, WriterAgent, JudgeAgent
from aqaa.csba.ldt import LeakageDetectionTest
from aqaa.csba.tif import TemporalIntegrityFramework, DataPartition

__all__ = [
    "Backtester",
    "BacktestMetrics",
    "CodeSynthesizer",
    "WriterAgent",
    "JudgeAgent",
    "LeakageDetectionTest",
    "TemporalIntegrityFramework",
    "DataPartition",
]
