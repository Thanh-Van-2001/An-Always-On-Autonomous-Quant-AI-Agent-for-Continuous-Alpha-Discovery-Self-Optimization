"""
LEMA - Live Execution and Monitoring Agent.

This module implements the real-time monitoring and execution layer of the
AQAA pipeline (Contribution C3 in the KDD 2026 paper).  Key capabilities:

* **Alpha Lifecycle Management (ALM)** -- an 8-state machine that governs
  every alpha factor from hypothesis generation through retirement and
  mutation (Figure 3).
* **Bayesian IC Tracking** -- a local-level / Kalman-filter model for the
  information coefficient, providing a principled posterior predictive
  distribution and a probabilistic decay trigger.
* **Change-point Decay Detection** -- CUSUM and PELT detectors with a
  consensus rule for identifying structural breaks in IC series.
* **Factor Crowding Detection** -- pairwise-correlation monitoring with an
  automatic rebalancing suggestion mechanism.
* **Order Execution** -- simulated TWAP execution and pre-trade risk checks.
"""

from aqaa.lema.alm import AlphaLifecycleManager
from aqaa.lema.bayesian_ic import BayesianICTracker
from aqaa.lema.decay_detector import (
    CUSUMDetector,
    PELTDetector,
    ConsensusDetector,
    DecayDetector,
)
from aqaa.lema.crowding_detector import CrowdingDetector
from aqaa.lema.executor import OrderExecutor, RiskLimits

__all__ = [
    "AlphaLifecycleManager",
    "BayesianICTracker",
    "CUSUMDetector",
    "PELTDetector",
    "ConsensusDetector",
    "DecayDetector",
    "CrowdingDetector",
    "OrderExecutor",
    "RiskLimits",
]
