"""
FLO - Feedback Loop Orchestrator.

This module implements the outer control loop of the AQAA system as described
in the KDD 2026 paper (Section 4, Figure 4).  It coordinates all five
subsystem modules through a 24-hour autonomous cycle and uses a UCB1
multi-armed bandit to dynamically allocate compute resources across:

    HE   -- Hypothesis Engine
    CSBA -- Code Synthesis and Backtest Agent
    APO  -- Adaptive Portfolio Optimizer
    LEMA -- Live Execution and Monitoring Agent
    KB   -- Knowledge Base maintenance

The UCB1 reward signal is the *marginal IC gain per compute-hour*, ensuring
that the system automatically shifts resources toward whichever module is
currently yielding the highest information-coefficient improvement per unit
of compute budget.

Submodules
----------
ucb1
    UCB1 Multi-Armed Bandit for compute allocation.
orchestrator
    FeedbackLoopOrchestrator and BudgetAllocator.
"""

from aqaa.flo.ucb1 import UCB1Bandit
from aqaa.flo.orchestrator import FeedbackLoopOrchestrator, BudgetAllocator

__all__ = [
    "UCB1Bandit",
    "FeedbackLoopOrchestrator",
    "BudgetAllocator",
]
