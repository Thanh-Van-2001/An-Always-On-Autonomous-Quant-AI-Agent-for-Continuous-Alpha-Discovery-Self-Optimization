"""
UCB1 Multi-Armed Bandit for AQAA compute allocation.

Implements the Upper Confidence Bound 1 algorithm (Auer et al., 2002) to
dynamically allocate GPU-hours across the five AQAA modules.  The reward
signal is the marginal IC gain per compute-hour, so arms that are currently
producing the highest information-coefficient improvement per unit of
compute will be preferentially selected.

The exploration constant *c* defaults to 1.5, matching the value reported
in the KDD 2026 paper (Section 4.2, Eq. 7).

Usage
-----
>>> bandit = UCB1Bandit()
>>> arm = bandit.select_arm()
>>> bandit.update(arm, reward=0.012)
>>> bandit.get_allocation()
{'HE': 0.22, 'CSBA': 0.28, ...}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# The five AQAA modules that compete for compute budget.
ARM_NAMES: List[str] = ["HE", "CSBA", "APO", "LEMA", "KB"]


@dataclass
class ArmStats:
    """Running statistics for a single bandit arm.

    Attributes:
        name: Human-readable module identifier.
        total_reward: Cumulative reward collected from this arm.
        n_pulls: Number of times this arm has been selected.
        reward_history: Full history of per-pull rewards for analysis.
    """

    name: str = ""
    total_reward: float = 0.0
    n_pulls: int = 0
    reward_history: List[float] = field(default_factory=list)

    @property
    def empirical_mean(self) -> float:
        """Q(a) -- average reward per pull."""
        if self.n_pulls == 0:
            return 0.0
        return self.total_reward / self.n_pulls


class UCB1Bandit:
    """Upper Confidence Bound 1 bandit for AQAA compute allocation.

    Each arm corresponds to one of the five AQAA modules.  The policy
    selects the arm that maximises:

        a_t = argmax_a [ Q(a) + c * sqrt( ln(t) / n(a) ) ]

    where *Q(a)* is the empirical mean reward, *t* is the total number of
    pulls so far, *n(a)* is the number of times arm *a* has been pulled,
    and *c* is the exploration constant.

    Parameters
    ----------
    c : float
        Exploration constant.  Default 1.5 per the KDD 2026 paper.
    arm_names : list[str] | None
        Override the default module names if needed.
    seed : int | None
        Random seed for tie-breaking reproducibility.
    """

    def __init__(
        self,
        c: float = 1.5,
        arm_names: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.c = c
        self.arm_names: List[str] = list(arm_names or ARM_NAMES)
        self.n_arms: int = len(self.arm_names)
        self.arms: List[ArmStats] = [
            ArmStats(name=name) for name in self.arm_names
        ]
        self.t: int = 0  # total pulls across all arms
        self.rng = np.random.default_rng(seed)

        # Global history: list of (arm_index, reward) per step.
        self.selection_history: List[int] = []
        self.reward_history: List[float] = []

    # ------------------------------------------------------------------
    # Core UCB1 interface
    # ------------------------------------------------------------------

    def select_arm(self) -> int:
        """Select the arm with the highest UCB1 index.

        During the initial round-robin phase (t < n_arms), each arm is
        pulled once to ensure all arms have at least one observation.

        Returns
        -------
        int
            Index of the selected arm in ``self.arm_names``.
        """
        # Round-robin initialisation: pull each arm once.
        for i, arm in enumerate(self.arms):
            if arm.n_pulls == 0:
                logger.debug("UCB1 init phase: pulling arm %d (%s)", i, arm.name)
                return i

        # Compute UCB1 index for every arm.
        ln_t = np.log(self.t)
        ucb_values = np.array([
            arm.empirical_mean + self.c * np.sqrt(ln_t / arm.n_pulls)
            for arm in self.arms
        ])

        # Break ties randomly for fairness.
        max_val = ucb_values.max()
        candidates = np.where(np.isclose(ucb_values, max_val))[0]
        selected = int(self.rng.choice(candidates))

        logger.debug(
            "UCB1 values: %s -> selected arm %d (%s)",
            dict(zip(self.arm_names, ucb_values.round(6))),
            selected,
            self.arm_names[selected],
        )
        return selected

    def update(self, arm: int, reward: float) -> None:
        """Update the statistics for *arm* after observing *reward*.

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Observed reward (marginal IC gain per compute-hour).
        """
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(
                f"arm index {arm} out of range [0, {self.n_arms})"
            )

        self.arms[arm].total_reward += reward
        self.arms[arm].n_pulls += 1
        self.arms[arm].reward_history.append(reward)
        self.t += 1

        self.selection_history.append(arm)
        self.reward_history.append(reward)

        logger.debug(
            "Updated arm %d (%s): n=%d, Q=%.6f, reward=%.6f",
            arm,
            self.arms[arm].name,
            self.arms[arm].n_pulls,
            self.arms[arm].empirical_mean,
            reward,
        )

    # ------------------------------------------------------------------
    # Allocation helpers
    # ------------------------------------------------------------------

    def get_allocation(self) -> Dict[str, float]:
        """Convert empirical pull counts into compute-fraction allocation.

        Returns a dict mapping arm names to their fraction of total compute,
        computed as ``n(a) / t``.  During the early phase when no arms have
        been pulled, returns a uniform allocation.

        Returns
        -------
        dict[str, float]
            Mapping from module name to fraction in [0, 1].  Sums to 1.0.
        """
        if self.t == 0:
            uniform = 1.0 / self.n_arms
            return {name: uniform for name in self.arm_names}

        return {
            arm.name: arm.n_pulls / self.t
            for arm in self.arms
        }

    def get_ucb_values(self) -> Dict[str, float]:
        """Return the current UCB1 index for each arm.

        Useful for logging and diagnostics.

        Returns
        -------
        dict[str, float]
            Mapping from module name to UCB1 index value.
        """
        if self.t == 0:
            return {name: float("inf") for name in self.arm_names}

        ln_t = np.log(self.t)
        return {
            arm.name: (
                arm.empirical_mean + self.c * np.sqrt(ln_t / arm.n_pulls)
                if arm.n_pulls > 0
                else float("inf")
            )
            for arm in self.arms
        }

    # ------------------------------------------------------------------
    # Simulation for analysis / paper figures
    # ------------------------------------------------------------------

    def simulate(
        self,
        n_steps: int,
        reward_fn: Optional[Any] = None,
        reward_means: Optional[List[float]] = None,
        reward_std: float = 0.005,
    ) -> Dict[str, Any]:
        """Run a simulation of the UCB1 policy for *n_steps* rounds.

        This is intended for generating arm-selection distributions like
        Figure 4b in the KDD 2026 paper.

        Parameters
        ----------
        n_steps : int
            Number of simulation rounds.
        reward_fn : callable | None
            Optional ``reward_fn(arm_index, step) -> float``.  If provided,
            *reward_means* and *reward_std* are ignored.
        reward_means : list[float] | None
            True mean reward for each arm.  If None, defaults to
            ``[0.010, 0.015, 0.008, 0.012, 0.006]`` (representative
            marginal IC gains per GPU-hour for HE, CSBA, APO, LEMA, KB).
        reward_std : float
            Standard deviation of Gaussian noise added to rewards.

        Returns
        -------
        dict
            Keys: ``selections`` (array of arm indices), ``rewards``,
            ``cumulative_rewards``, ``allocation_history`` (list of dicts),
            ``final_allocation``.
        """
        if reward_means is None:
            reward_means = [0.010, 0.015, 0.008, 0.012, 0.006]
        if len(reward_means) != self.n_arms:
            raise ValueError(
                f"reward_means length {len(reward_means)} != n_arms {self.n_arms}"
            )

        means = np.asarray(reward_means, dtype=np.float64)
        selections: List[int] = []
        rewards: List[float] = []
        allocation_history: List[Dict[str, float]] = []

        for step in range(n_steps):
            arm = self.select_arm()

            if reward_fn is not None:
                r = float(reward_fn(arm, step))
            else:
                r = float(self.rng.normal(means[arm], reward_std))

            self.update(arm, r)
            selections.append(arm)
            rewards.append(r)

            # Record allocation snapshot every 10 steps to keep memory bounded.
            if (step + 1) % 10 == 0 or step == n_steps - 1:
                allocation_history.append(self.get_allocation())

        return {
            "selections": np.array(selections),
            "rewards": np.array(rewards),
            "cumulative_rewards": np.cumsum(rewards),
            "allocation_history": allocation_history,
            "final_allocation": self.get_allocation(),
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict suitable for JSON serialisation."""
        return {
            "total_steps": self.t,
            "c": self.c,
            "arms": {
                arm.name: {
                    "n_pulls": arm.n_pulls,
                    "empirical_mean": round(arm.empirical_mean, 8),
                    "total_reward": round(arm.total_reward, 8),
                }
                for arm in self.arms
            },
            "allocation": self.get_allocation(),
        }

    def __repr__(self) -> str:
        alloc = self.get_allocation()
        parts = [f"{k}={v:.2%}" for k, v in alloc.items()]
        return f"UCB1Bandit(t={self.t}, c={self.c}, alloc=[{', '.join(parts)}])"
