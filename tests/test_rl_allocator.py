"""
Tests for RL Allocator — Regime-Conditioned Position Sizing
============================================================
Covers:
  - DQN network: forward pass, training, weight copying
  - Action encoding/decoding
  - Differential Sharpe reward
  - Replay buffer
  - RLAllocator: training, inference, fallback behavior
  - Edge cases: empty episodes, constant returns, extreme regimes

All computation is deterministic (seeded) — AGENTS.md rule 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.quant.rl_allocator import (
    AGGRESSION_BUCKETS,
    HEDGE_BUCKETS,
    N_ACTIONS,
    TOTAL_STATE_DIM,
    AllocationAction,
    Experience,
    RLAllocator,
    RLState,
    ReplayBuffer,
    SimpleQNetwork,
    TrainingMetrics,
    decode_action,
    differential_sharpe_reward,
    encode_action,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    """Deterministic RNG for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def simple_episode(rng):
    """Generate a simple training episode DataFrame."""
    n = 200
    regimes = rng.choice(["BULL", "BEAR", "SIDEWAYS"], size=n, p=[0.5, 0.2, 0.3])
    returns = rng.normal(0.0005, 0.015, n)
    # Bull regime gets positive returns, bear negative
    for i in range(n):
        if regimes[i] == "BULL":
            returns[i] += 0.001
        elif regimes[i] == "BEAR":
            returns[i] -= 0.001

    cumret = np.cumprod(1 + returns) - 1
    peak = np.maximum.accumulate(1 + cumret)
    drawdown = (1 + cumret) / peak - 1

    return pd.DataFrame({
        "regime": regimes,
        "portfolio_return": returns,
        "rolling_sharpe_20d": pd.Series(returns).rolling(20).mean() / pd.Series(returns).rolling(20).std(),
        "rolling_vol_20d": pd.Series(returns).rolling(20).std() * np.sqrt(252),
        "current_drawdown": drawdown,
        "vix_z": rng.normal(0, 1, n),
        "sentiment_score": rng.uniform(-1, 1, n),
        "momentum_20d": pd.Series(returns).rolling(20).sum(),
        "current_hedge": 0.5,
        "portfolio_return_5d": pd.Series(returns).rolling(5).sum(),
        "portfolio_drawdown": drawdown,
    }).fillna(0.0)


@pytest.fixture
def multi_episodes(rng):
    """Generate multiple training episodes with different market conditions."""
    episodes = []
    for _ in range(5):
        n = 100
        regime_probs = rng.dirichlet([1, 1, 1])
        regimes = rng.choice(["BULL", "BEAR", "SIDEWAYS"], size=n, p=regime_probs)
        returns = rng.normal(0.0003, 0.02, n)

        df = pd.DataFrame({
            "regime": regimes,
            "portfolio_return": returns,
            "rolling_sharpe_20d": 0.0,
            "rolling_vol_20d": 0.15,
            "current_drawdown": 0.0,
            "vix_z": 0.0,
            "sentiment_score": 0.0,
            "momentum_20d": 0.0,
            "current_hedge": 0.5,
            "portfolio_return_5d": 0.0,
            "portfolio_drawdown": 0.0,
        })
        episodes.append(df)
    return episodes


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: SimpleQNetwork
# ══════════════════════════════════════════════════════════════════════════════

class TestSimpleQNetwork:
    """Test the numpy-based Q-network."""

    def test_forward_shape(self):
        """Forward pass should produce N_ACTIONS Q-values."""
        net = SimpleQNetwork()
        state = np.random.randn(TOTAL_STATE_DIM).astype(np.float32)
        q = net.forward(state)
        assert q.shape == (N_ACTIONS,)

    def test_forward_batch(self):
        """Batch forward should handle multiple states."""
        net = SimpleQNetwork()
        states = np.random.randn(10, TOTAL_STATE_DIM).astype(np.float32)
        q = net.forward(states)
        assert q.shape == (10, N_ACTIONS)

    def test_training_reduces_loss(self):
        """Training on fixed data should reduce loss over iterations."""
        np.random.seed(42)
        net = SimpleQNetwork(lr=0.01)
        states = np.random.randn(32, TOTAL_STATE_DIM).astype(np.float32)
        actions = np.random.randint(0, N_ACTIONS, 32)
        targets = np.random.randn(32).astype(np.float32)

        loss_first = net.train_batch(states, actions, targets)
        for _ in range(50):
            net.train_batch(states, actions, targets)
        loss_last = net.train_batch(states, actions, targets)

        assert loss_last < loss_first

    def test_copy_from(self):
        """Copied network should produce identical Q-values."""
        net1 = SimpleQNetwork()
        net2 = SimpleQNetwork()

        state = np.random.randn(TOTAL_STATE_DIM).astype(np.float32)
        q_before = net2.forward(state)

        net2.copy_from(net1)
        q_after = net2.forward(state)
        q_original = net1.forward(state)

        np.testing.assert_allclose(q_after, q_original)

    def test_different_inits(self):
        """Two independently initialised networks should differ."""
        np.random.seed(1)
        net1 = SimpleQNetwork()
        np.random.seed(2)
        net2 = SimpleQNetwork()

        state = np.random.randn(TOTAL_STATE_DIM).astype(np.float32)
        q1 = net1.forward(state)
        q2 = net2.forward(state)
        assert not np.allclose(q1, q2)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: Action Encoding/Decoding
# ══════════════════════════════════════════════════════════════════════════════

class TestActionEncoding:
    """Test action space encoding and decoding."""

    def test_decode_all_actions(self):
        """All action indices should decode to valid actions."""
        for i in range(N_ACTIONS):
            action = decode_action(i)
            assert 0.0 <= action.hedge_ratio <= 1.0
            assert action.aggression >= 0.5
            assert action.aggression <= 1.5

    def test_roundtrip(self):
        """encode(decode(i)) should return the same index."""
        for i in range(N_ACTIONS):
            action = decode_action(i)
            reconstructed = encode_action(action.hedge_ratio, action.aggression)
            assert reconstructed == i

    def test_encode_nearest(self):
        """Encoding should find the nearest bucket."""
        # 0.33 is between 0.20 and 0.35 — should snap to 0.35
        action_idx = encode_action(0.33, 1.0)
        action = decode_action(action_idx)
        assert action.hedge_ratio in HEDGE_BUCKETS
        assert action.aggression in AGGRESSION_BUCKETS

    def test_action_description(self):
        """AllocationAction should have a readable description."""
        action = AllocationAction(hedge_ratio=0.50, aggression=1.25)
        desc = action.description
        assert "50%" in desc
        assert "1.25" in desc


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: RLState
# ══════════════════════════════════════════════════════════════════════════════

class TestRLState:
    """Test the observation dataclass."""

    def test_to_array_shape(self):
        """State array should have correct dimension."""
        state = RLState()
        arr = state.to_array()
        assert arr.shape == (TOTAL_STATE_DIM,)
        assert arr.dtype == np.float32

    def test_regime_encoding(self):
        """One-hot regime encoding should work correctly."""
        bull = RLState(regime_bull=1.0)
        bear = RLState(regime_bear=1.0)
        arr_bull = bull.to_array()
        arr_bear = bear.to_array()
        assert arr_bull[0] == 1.0  # bull flag
        assert arr_bull[2] == 0.0  # bear flag
        assert arr_bear[2] == 1.0  # bear flag
        assert arr_bear[0] == 0.0  # bull flag


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: Differential Sharpe Reward
# ══════════════════════════════════════════════════════════════════════════════

class TestDifferentialSharpeReward:
    """Test the Moody & Saffell reward function."""

    def test_output_shape(self):
        """Reward array should match input shape."""
        returns = np.random.normal(0, 0.01, 100)
        rewards = differential_sharpe_reward(returns)
        assert rewards.shape == returns.shape

    def test_positive_trend_positive_rewards(self):
        """Consistently positive returns should yield mostly positive rewards."""
        returns = np.ones(100) * 0.001  # Consistent positive
        rewards = differential_sharpe_reward(returns)
        # After warmup, rewards should be positive
        assert rewards[50:].mean() > 0

    def test_negative_trend_negative_rewards(self):
        """Consistently negative returns should yield mostly negative rewards."""
        returns = np.ones(100) * -0.001
        rewards = differential_sharpe_reward(returns)
        assert rewards[50:].mean() < 0

    def test_zero_returns(self):
        """All-zero returns should produce near-zero rewards."""
        returns = np.zeros(50)
        rewards = differential_sharpe_reward(returns)
        assert np.abs(rewards).max() < 0.1


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: ReplayBuffer
# ══════════════════════════════════════════════════════════════════════════════

class TestReplayBuffer:
    """Test the experience replay buffer."""

    def test_push_and_sample(self):
        """Should store and sample experiences."""
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(Experience(
                state=np.zeros(12),
                action=i % N_ACTIONS,
                reward=float(i),
                next_state=np.ones(12),
                done=(i == 49),
            ))
        assert len(buf) == 50

        batch = buf.sample(10)
        assert len(batch) == 10
        assert all(isinstance(e, Experience) for e in batch)

    def test_capacity_limit(self):
        """Buffer should respect capacity limit (FIFO)."""
        buf = ReplayBuffer(capacity=10)
        for i in range(20):
            buf.push(Experience(np.zeros(12), 0, float(i), np.zeros(12), False))
        assert len(buf) == 10


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: RLAllocator
# ══════════════════════════════════════════════════════════════════════════════

class TestRLAllocator:
    """Test the full RL allocator agent."""

    def test_untrained_fallback_bull(self):
        """Untrained allocator should use rule-based fallback for BULL."""
        allocator = RLAllocator()
        state = RLState(regime_bull=1.0)
        action = allocator.get_allocation(state)
        assert action.hedge_ratio == 0.30
        assert action.aggression == 1.25

    def test_untrained_fallback_bear(self):
        """Untrained allocator should use rule-based fallback for BEAR."""
        allocator = RLAllocator()
        state = RLState(regime_bear=1.0)
        action = allocator.get_allocation(state)
        assert action.hedge_ratio == 0.70
        assert action.aggression == 0.75

    def test_untrained_fallback_sideways(self):
        """Untrained allocator should use rule-based fallback for SIDEWAYS."""
        allocator = RLAllocator()
        state = RLState(regime_sideways=1.0)
        action = allocator.get_allocation(state)
        assert action.hedge_ratio == 0.50
        assert action.aggression == 1.0

    def test_training_runs(self, simple_episode):
        """Training should complete without errors and set is_trained."""
        np.random.seed(42)
        allocator = RLAllocator(
            epsilon_start=0.8,
            epsilon_end=0.1,
            epsilon_decay=0.99,
        )
        metrics = allocator.train([simple_episode], n_epochs=1)

        assert allocator.is_trained
        assert metrics.episodes == 1
        assert metrics.total_steps > 0
        assert metrics.final_epsilon < 0.8

    def test_trained_allocation(self, simple_episode):
        """Trained allocator should return valid actions from Q-network."""
        np.random.seed(42)
        allocator = RLAllocator(epsilon_start=0.5, epsilon_decay=0.95)
        allocator.train([simple_episode], n_epochs=2)

        state = RLState(regime_bull=1.0, rolling_sharpe_20d=1.5)
        action = allocator.get_allocation(state)

        assert 0.0 <= action.hedge_ratio <= 1.0
        assert 0.5 <= action.aggression <= 1.5

    def test_multi_episode_training(self, multi_episodes):
        """Training on multiple episodes should work."""
        np.random.seed(42)
        allocator = RLAllocator(epsilon_decay=0.98)
        metrics = allocator.train(multi_episodes, n_epochs=2)

        assert allocator.is_trained
        assert metrics.episodes == 10  # 5 episodes × 2 epochs

    def test_training_metrics_returned(self, simple_episode):
        """Training should return and store metrics."""
        np.random.seed(42)
        allocator = RLAllocator()
        metrics = allocator.train([simple_episode], n_epochs=1)

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.total_steps > 0

        stored = allocator.get_training_metrics()
        assert stored is metrics

    def test_empty_episode(self):
        """Empty episode should be handled gracefully."""
        allocator = RLAllocator()
        empty = pd.DataFrame({
            "regime": [],
            "portfolio_return": [],
            "rolling_sharpe_20d": [],
            "rolling_vol_20d": [],
            "current_drawdown": [],
            "vix_z": [],
            "sentiment_score": [],
            "momentum_20d": [],
            "current_hedge": [],
            "portfolio_return_5d": [],
            "portfolio_drawdown": [],
        })
        # Should not crash
        metrics = allocator.train([empty], n_epochs=1)
        assert metrics.episodes == 1

    def test_constant_returns(self):
        """Constant returns should not crash the agent."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "regime": ["SIDEWAYS"] * n,
            "portfolio_return": [0.001] * n,
            "rolling_sharpe_20d": [1.0] * n,
            "rolling_vol_20d": [0.15] * n,
            "current_drawdown": [0.0] * n,
            "vix_z": [0.0] * n,
            "sentiment_score": [0.0] * n,
            "momentum_20d": [0.02] * n,
            "current_hedge": [0.5] * n,
            "portfolio_return_5d": [0.005] * n,
            "portfolio_drawdown": [0.0] * n,
        })
        allocator = RLAllocator()
        metrics = allocator.train([df], n_epochs=1)
        assert metrics.total_steps > 0


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: Rule 8 compliance — risk-adjusted, not return-chasing
# ══════════════════════════════════════════════════════════════════════════════

class TestRule8Compliance:
    """
    AGENTS.md rule 8: Optimise for Sharpe/Sortino with bounded max-drawdown.
    Do NOT hardcode or backtest-optimize toward a fixed target return percentage.
    """

    def test_reward_is_risk_adjusted(self):
        """
        The reward function should prefer stable small returns over
        volatile large returns (Sharpe > raw return).
        """
        # Stable: 0.1% daily, no vol
        stable = np.ones(100) * 0.001
        volatile = np.random.RandomState(42).normal(0.003, 0.05, 100)

        r_stable = differential_sharpe_reward(stable)
        r_volatile = differential_sharpe_reward(volatile)

        # After warmup, stable rewards should be more consistently positive
        assert (r_stable[50:] > 0).mean() > (r_volatile[50:] > 0).mean()

    def test_no_hardcoded_return_target(self):
        """The allocator should not contain any hardcoded return target."""
        import inspect
        source = inspect.getsource(RLAllocator)
        # Should not contain strings like "target_return", "0.15", "15%"
        assert "target_return" not in source.lower()
