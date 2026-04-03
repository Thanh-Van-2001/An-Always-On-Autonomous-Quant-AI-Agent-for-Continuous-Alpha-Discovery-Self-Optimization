"""
Tests for the UCB1 multi-armed bandit module.

Validates that the best arm is selected most frequently, exploration occurs
in early rounds, the UCB formula is correct, and simulate() produces
valid distributions.
"""

import numpy as np
import pytest

from aqaa.flo.ucb1 import ARM_NAMES, ArmStats, UCB1Bandit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bandit():
    return UCB1Bandit(c=1.5, seed=42)


# ---------------------------------------------------------------------------
# Test: Best arm selected most frequently
# ---------------------------------------------------------------------------

class TestBestArmConvergence:
    """In the long run, the arm with the highest mean should dominate."""

    def test_best_arm_dominates(self):
        """Arm 1 (CSBA) has the highest mean reward -> most pulls."""
        bandit = UCB1Bandit(c=1.5, seed=42)
        reward_means = [0.010, 0.025, 0.008, 0.012, 0.006]

        result = bandit.simulate(
            n_steps=2000,
            reward_means=reward_means,
            reward_std=0.003,
        )

        alloc = result["final_allocation"]
        # Arm 1 (CSBA) should have the largest allocation
        best_arm_name = "CSBA"
        assert alloc[best_arm_name] == max(alloc.values())

    def test_best_arm_majority_pulls(self):
        """The best arm should receive more than 25% of total pulls
        after sufficient rounds (well above the 20% uniform baseline)."""
        bandit = UCB1Bandit(c=0.5, seed=99)
        reward_means = [0.005, 0.030, 0.005, 0.005, 0.005]

        result = bandit.simulate(
            n_steps=5000,
            reward_means=reward_means,
            reward_std=0.003,
        )

        # CSBA (arm 1) has the highest mean; with lower c for less
        # exploration it should clearly dominate over 5000 rounds.
        assert result["final_allocation"]["CSBA"] > 0.25


# ---------------------------------------------------------------------------
# Test: Exploration in early rounds (round-robin)
# ---------------------------------------------------------------------------

class TestExploration:
    """Early rounds should explore all arms equally."""

    def test_round_robin_initialisation(self, bandit):
        """Each arm should be pulled once in the first n_arms rounds."""
        # Manually step through the init phase
        selected = []
        for _ in range(bandit.n_arms):
            arm = bandit.select_arm()
            selected.append(arm)
            bandit.update(arm, 0.01)

        # All arms should have been pulled exactly once
        assert set(selected) == set(range(bandit.n_arms))
        for arm in bandit.arms:
            assert arm.n_pulls == 1

    def test_all_arms_explored_early(self):
        """In the first 2*n_arms rounds, every arm should be pulled
        at least once."""
        bandit = UCB1Bandit(c=1.5, seed=42)
        for _ in range(2 * bandit.n_arms):
            arm = bandit.select_arm()
            bandit.update(arm, 0.01)

        for arm_stat in bandit.arms:
            assert arm_stat.n_pulls >= 1


# ---------------------------------------------------------------------------
# Test: UCB formula correctness
# ---------------------------------------------------------------------------

class TestUCBFormula:
    """Verify the UCB1 index computation."""

    def test_ucb_values_after_updates(self):
        """Manually verify UCB values match the formula."""
        bandit = UCB1Bandit(c=2.0, seed=42)
        # Initialise all arms
        rewards = [0.01, 0.02, 0.015, 0.012, 0.008]
        for i in range(bandit.n_arms):
            bandit.update(i, rewards[i])

        # Now t = 5, each arm has n=1
        ucb_vals = bandit.get_ucb_values()

        for i, arm in enumerate(bandit.arms):
            expected_exploit = arm.empirical_mean
            expected_explore = bandit.c * np.sqrt(np.log(bandit.t) / arm.n_pulls)
            expected_ucb = expected_exploit + expected_explore

            actual_ucb = ucb_vals[arm.name]
            assert abs(actual_ucb - expected_ucb) < 1e-10, (
                f"Arm {arm.name}: expected {expected_ucb}, got {actual_ucb}"
            )

    def test_ucb_values_before_any_pulls(self):
        """Before any pulls, all UCB values should be inf."""
        bandit = UCB1Bandit(c=1.5, seed=42)
        ucb_vals = bandit.get_ucb_values()
        for name, val in ucb_vals.items():
            assert val == float("inf")

    def test_exploration_term_decreases(self):
        """As an arm is pulled more, its exploration bonus decreases."""
        bandit = UCB1Bandit(c=1.5, seed=42)
        # Pull arm 0 many times
        for _ in range(bandit.n_arms):
            arm = bandit.select_arm()
            bandit.update(arm, 0.01)

        # Pull arm 0 repeatedly
        for _ in range(50):
            bandit.update(0, 0.01)

        ucb_vals = bandit.get_ucb_values()
        # Arm 0 has many pulls -> small exploration bonus
        # Other arms have 1 pull -> large exploration bonus
        arm0_ucb = ucb_vals[bandit.arm_names[0]]
        # The UCB for arm 0 should be dominated by exploitation
        # while other arms have high exploration bonuses
        for i in range(1, bandit.n_arms):
            other_ucb = ucb_vals[bandit.arm_names[i]]
            # Other arms should have higher UCB due to exploration
            assert other_ucb > arm0_ucb


# ---------------------------------------------------------------------------
# Test: ArmStats
# ---------------------------------------------------------------------------

class TestArmStats:

    def test_empirical_mean_no_pulls(self):
        arm = ArmStats(name="test")
        assert arm.empirical_mean == 0.0

    def test_empirical_mean_after_updates(self):
        arm = ArmStats(name="test")
        arm.total_reward = 0.15
        arm.n_pulls = 3
        assert abs(arm.empirical_mean - 0.05) < 1e-10


# ---------------------------------------------------------------------------
# Test: simulate() produces valid distributions
# ---------------------------------------------------------------------------

class TestSimulate:
    """Verify simulate() output structure and consistency."""

    def test_output_structure(self):
        bandit = UCB1Bandit(c=1.5, seed=42)
        result = bandit.simulate(n_steps=100, reward_std=0.005)

        assert "selections" in result
        assert "rewards" in result
        assert "cumulative_rewards" in result
        assert "allocation_history" in result
        assert "final_allocation" in result

        assert len(result["selections"]) == 100
        assert len(result["rewards"]) == 100
        assert len(result["cumulative_rewards"]) == 100

    def test_selections_are_valid_indices(self):
        bandit = UCB1Bandit(c=1.5, seed=42)
        result = bandit.simulate(n_steps=200, reward_std=0.005)

        selections = result["selections"]
        assert all(0 <= s < bandit.n_arms for s in selections)

    def test_allocation_sums_to_one(self):
        bandit = UCB1Bandit(c=1.5, seed=42)
        result = bandit.simulate(n_steps=100, reward_std=0.005)

        alloc = result["final_allocation"]
        assert abs(sum(alloc.values()) - 1.0) < 1e-10

    def test_cumulative_rewards_monotone(self):
        """Cumulative rewards should be monotonically non-decreasing
        when all reward_means are positive."""
        bandit = UCB1Bandit(c=1.5, seed=42)
        result = bandit.simulate(
            n_steps=200,
            reward_means=[0.01, 0.02, 0.015, 0.012, 0.008],
            reward_std=0.001,  # very low noise -> all positive rewards
        )
        cum = result["cumulative_rewards"]
        # With very small std and positive means, rewards should be positive
        # so cumulative should be increasing
        diffs = np.diff(cum)
        assert np.all(diffs > 0)

    def test_custom_reward_fn(self):
        """Using a custom reward function."""
        bandit = UCB1Bandit(c=1.5, seed=42)

        def my_reward(arm_idx, step):
            return 0.01 * (arm_idx + 1)

        result = bandit.simulate(n_steps=50, reward_fn=my_reward)
        assert len(result["rewards"]) == 50
        # All rewards should be deterministic multiples of 0.01
        for r in result["rewards"]:
            assert r in [0.01, 0.02, 0.03, 0.04, 0.05]

    def test_wrong_reward_means_length_raises(self):
        bandit = UCB1Bandit(c=1.5, seed=42)
        with pytest.raises(ValueError, match="reward_means length"):
            bandit.simulate(n_steps=10, reward_means=[0.01, 0.02])


# ---------------------------------------------------------------------------
# Test: get_allocation
# ---------------------------------------------------------------------------

class TestGetAllocation:

    def test_uniform_before_pulls(self):
        bandit = UCB1Bandit(c=1.5, seed=42)
        alloc = bandit.get_allocation()
        expected = 1.0 / bandit.n_arms
        for v in alloc.values():
            assert abs(v - expected) < 1e-10

    def test_allocation_after_pulls(self):
        bandit = UCB1Bandit(c=1.5, seed=42)
        bandit.update(0, 0.01)
        bandit.update(0, 0.01)
        bandit.update(1, 0.01)

        alloc = bandit.get_allocation()
        assert abs(alloc["HE"] - 2 / 3) < 1e-10
        assert abs(alloc["CSBA"] - 1 / 3) < 1e-10


# ---------------------------------------------------------------------------
# Test: update validation
# ---------------------------------------------------------------------------

class TestUpdateValidation:

    def test_invalid_arm_index_raises(self):
        bandit = UCB1Bandit(c=1.5, seed=42)
        with pytest.raises(ValueError, match="arm index"):
            bandit.update(-1, 0.01)
        with pytest.raises(ValueError, match="arm index"):
            bandit.update(5, 0.01)

    def test_total_steps_tracked(self):
        bandit = UCB1Bandit(c=1.5, seed=42)
        for i in range(10):
            bandit.update(i % bandit.n_arms, 0.01)
        assert bandit.t == 10


# ---------------------------------------------------------------------------
# Test: summary and repr
# ---------------------------------------------------------------------------

class TestSummaryRepr:

    def test_summary_keys(self):
        bandit = UCB1Bandit(c=1.5, seed=42)
        bandit.update(0, 0.01)
        s = bandit.summary()
        assert "total_steps" in s
        assert "c" in s
        assert "arms" in s
        assert "allocation" in s
        assert s["total_steps"] == 1

    def test_repr(self):
        bandit = UCB1Bandit(c=1.5, seed=42)
        r = repr(bandit)
        assert "UCB1Bandit" in r
        assert "t=0" in r
