"""
RL Allocator — Regime-Conditioned Position Sizing
===================================================
Learns allocation weights and hedge ratios as a function of
(regime_state, signal_features, portfolio_state) instead of
using hard-coded rules.

Architecture (AGENTS.md rule 1 — LLMs never compute numbers):
  All computation here is deterministic RL — no LLM involvement.

Design choices:
  - **DQN** (not PPO) for the first version because:
    (a) The action space is naturally discrete (regime-dependent
        allocation bucket choices), keeping it simple.
    (b) DQN is more sample-efficient for small state spaces.
    (c) The finance RL literature (Jiang et al. 2017, Moody & Saffell 2001)
        shows DQN works well for portfolio allocation.
  - **Reward = daily risk-adjusted PnL** (differential Sharpe ratio from
    Moody & Saffell 2001), not raw returns — prevents the optimizer from
    chasing volatility.
  - **Walk-forward training** — the agent is NEVER evaluated on data it
    trained on.  The backtest uses the same expanding-window protocol as
    the rest of the quant engine (backtest.py).
  - **Action space** = discrete allocation buckets for hedge ratio and
    strategy weights, not continuous values — reduces instability.

Reference: docs/ARTHA_ARCHITECTURE.md §6.4
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────────

# State features
N_REGIME_FEATURES = 3     # regime_id (one-hot: bull/bear/sideways)
N_SIGNAL_FEATURES = 6     # rolling Sharpe, vol, drawdown, VIX_z, sentiment, momentum
N_PORTFOLIO_FEATURES = 3  # current_hedge, current_return_5d, current_dd
TOTAL_STATE_DIM = N_REGIME_FEATURES + N_SIGNAL_FEATURES + N_PORTFOLIO_FEATURES

# Discrete action space
# Each action maps to a (hedge_ratio, aggression_level) pair
HEDGE_BUCKETS = [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0]
AGGRESSION_BUCKETS = [0.5, 0.75, 1.0, 1.25, 1.5]  # Multiplier on base allocation
N_ACTIONS = len(HEDGE_BUCKETS) * len(AGGRESSION_BUCKETS)  # 35

# Training hyperparameters
DEFAULT_GAMMA = 0.99          # Discount factor
DEFAULT_LR = 1e-3             # Learning rate
DEFAULT_EPSILON_START = 1.0   # Initial exploration
DEFAULT_EPSILON_END = 0.05    # Final exploration
DEFAULT_EPSILON_DECAY = 0.995 # Per-episode decay
REPLAY_BUFFER_SIZE = 10_000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 50       # Update target network every N steps
MIN_REPLAY_SIZE = 500         # Minimum buffer size before training

# Reward scaling
REWARD_WINDOW = 20  # Days for rolling differential Sharpe


# ── Data Types ──────────────────────────────────────────────────────────────────

class RegimeId(Enum):
    BULL = 0
    SIDEWAYS = 1
    BEAR = 2


@dataclass
class AllocationAction:
    """Decoded action from the RL agent."""
    hedge_ratio: float
    aggression: float  # Multiplier on base strategy allocation

    @property
    def description(self) -> str:
        return f"hedge={self.hedge_ratio:.0%}, aggression={self.aggression:.2f}×"


@dataclass
class RLState:
    """Observation for the RL agent at a single timestep."""
    # Regime (one-hot)
    regime_bull: float = 0.0
    regime_sideways: float = 0.0
    regime_bear: float = 0.0

    # Signal features (normalised)
    rolling_sharpe_20d: float = 0.0
    rolling_vol_20d: float = 0.0
    current_drawdown: float = 0.0
    vix_z: float = 0.0
    sentiment_score: float = 0.0
    momentum_20d: float = 0.0

    # Portfolio state
    current_hedge: float = 0.5
    portfolio_return_5d: float = 0.0
    portfolio_drawdown: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.regime_bull, self.regime_sideways, self.regime_bear,
            self.rolling_sharpe_20d, self.rolling_vol_20d,
            self.current_drawdown, self.vix_z, self.sentiment_score,
            self.momentum_20d,
            self.current_hedge, self.portfolio_return_5d,
            self.portfolio_drawdown,
        ], dtype=np.float32)


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""
    episodes: int = 0
    total_steps: int = 0
    mean_episode_reward: float = 0.0
    mean_episode_sharpe: float = 0.0
    best_episode_sharpe: float = float("-inf")
    final_epsilon: float = 1.0
    loss_history: list[float] = field(default_factory=list)


# ── Experience Replay Buffer ────────────────────────────────────────────────────

@dataclass
class Experience:
    """A single (s, a, r, s', done) transition."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size circular replay buffer for DQN."""

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        self._buffer: deque[Experience] = deque(maxlen=capacity)

    def push(self, exp: Experience):
        self._buffer.append(exp)

    def sample(self, batch_size: int) -> list[Experience]:
        indices = np.random.choice(len(self._buffer), batch_size, replace=False)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


# ── DQN Network (numpy-only — no torch dependency) ─────────────────────────────

class SimpleQNetwork:
    """
    Minimal 2-hidden-layer Q-network implemented in pure numpy.

    Why not PyTorch?  The state space is 12-dim, action space is 35 —
    this is small enough that a numpy MLP trains in seconds, and we
    avoid adding torch as a dependency for the quant engine.

    Architecture: 12 → 64 → 32 → 35 (ReLU activations, He init).
    """

    def __init__(
        self,
        state_dim: int = TOTAL_STATE_DIM,
        n_actions: int = N_ACTIONS,
        hidden1: int = 64,
        hidden2: int = 32,
        lr: float = DEFAULT_LR,
    ):
        self.lr = lr
        # He initialisation
        self.W1 = np.random.randn(state_dim, hidden1).astype(np.float32) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.W2 = np.random.randn(hidden1, hidden2).astype(np.float32) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2, dtype=np.float32)
        self.W3 = np.random.randn(hidden2, n_actions).astype(np.float32) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(n_actions, dtype=np.float32)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass — returns Q-values for all actions."""
        # state shape: (batch, state_dim) or (state_dim,)
        single = state.ndim == 1
        if single:
            state = state.reshape(1, -1)

        h1 = np.maximum(0, state @ self.W1 + self.b1)   # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)       # ReLU
        q = h2 @ self.W3 + self.b3                         # Linear output

        return q[0] if single else q

    def train_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """
        One gradient descent step on a batch of (state, action, target_q).

        Returns the mean squared loss.
        """
        batch_size = len(states)

        # ── Forward pass (keeping intermediates for backprop) ───────────
        h1_pre = states @ self.W1 + self.b1
        h1 = np.maximum(0, h1_pre)
        h2_pre = h1 @ self.W2 + self.b2
        h2 = np.maximum(0, h2_pre)
        q_all = h2 @ self.W3 + self.b3

        # Extract Q-values for the taken actions
        q_pred = q_all[np.arange(batch_size), actions.astype(int)]
        error = q_pred - targets
        loss = float(np.mean(error ** 2))

        # ── Backward pass ──────────────────────────────────────────────
        # dL/dq for the selected actions only
        dq = np.zeros_like(q_all)
        dq[np.arange(batch_size), actions.astype(int)] = 2.0 * error / batch_size

        # Layer 3
        dW3 = h2.T @ dq
        db3 = dq.sum(axis=0)
        dh2 = dq @ self.W3.T

        # Layer 2 (ReLU)
        dh2_pre = dh2 * (h2_pre > 0).astype(np.float32)
        dW2 = h1.T @ dh2_pre
        db2 = dh2_pre.sum(axis=0)
        dh1 = dh2_pre @ self.W2.T

        # Layer 1 (ReLU)
        dh1_pre = dh1 * (h1_pre > 0).astype(np.float32)
        dW1 = states.T @ dh1_pre
        db1 = dh1_pre.sum(axis=0)

        # ── Gradient descent with clipping ─────────────────────────────
        max_grad = 1.0
        for dW in [dW1, dW2, dW3]:
            np.clip(dW, -max_grad, max_grad, out=dW)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        return loss

    def copy_from(self, other: "SimpleQNetwork"):
        """Copy weights from another network (for target network update)."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()


# ── Action Encoding/Decoding ───────────────────────────────────────────────────

def decode_action(action_idx: int) -> AllocationAction:
    """Convert action index to (hedge_ratio, aggression) pair."""
    hedge_idx = action_idx // len(AGGRESSION_BUCKETS)
    aggr_idx = action_idx % len(AGGRESSION_BUCKETS)
    return AllocationAction(
        hedge_ratio=HEDGE_BUCKETS[hedge_idx],
        aggression=AGGRESSION_BUCKETS[aggr_idx],
    )


def encode_action(hedge_ratio: float, aggression: float) -> int:
    """Find the closest action index for a given (hedge, aggression) pair."""
    best_h = min(range(len(HEDGE_BUCKETS)), key=lambda i: abs(HEDGE_BUCKETS[i] - hedge_ratio))
    best_a = min(range(len(AGGRESSION_BUCKETS)), key=lambda i: abs(AGGRESSION_BUCKETS[i] - aggression))
    return best_h * len(AGGRESSION_BUCKETS) + best_a


# ── Reward Function ────────────────────────────────────────────────────────────

def differential_sharpe_reward(
    returns: np.ndarray,
    window: int = REWARD_WINDOW,
) -> np.ndarray:
    """
    Differential Sharpe Ratio (Moody & Saffell, 2001).

    Instead of using raw return as reward (which encourages risk-seeking),
    this reward tracks the marginal change in Sharpe ratio:

      dS/dt ≈ (B_{t-1} * R_t - A_{t-1}) / (B_{t-1} - A_{t-1}²)^{3/2}

    where A_t, B_t are EMA of returns and squared returns.

    This directly optimises for risk-adjusted performance (AGENTS.md rule 8).
    """
    rewards = np.zeros_like(returns)
    eta = 2.0 / (window + 1)  # EMA decay

    A = 0.0  # EMA of returns
    B = 1e-6  # EMA of squared returns (small init to avoid div/0)

    for t in range(len(returns)):
        r = returns[t]

        # Differential Sharpe
        denom = (B - A * A) ** 1.5
        if abs(denom) > 1e-10:
            dS = (B * r - 0.5 * A * (r * r)) / denom
        else:
            dS = r  # Fallback to raw return

        rewards[t] = dS

        # Update EMAs
        A = A + eta * (r - A)
        B = B + eta * (r * r - B)

    return rewards


# ── RL Allocator Agent ──────────────────────────────────────────────────────────

class RLAllocator:
    """
    DQN-based regime-conditioned position sizing agent.

    The agent observes (regime, signals, portfolio_state) and outputs
    (hedge_ratio, aggression_multiplier).

    Usage::

        allocator = RLAllocator()
        metrics = allocator.train(historical_data)
        action = allocator.get_allocation(current_state)
    """

    def __init__(
        self,
        gamma: float = DEFAULT_GAMMA,
        lr: float = DEFAULT_LR,
        epsilon_start: float = DEFAULT_EPSILON_START,
        epsilon_end: float = DEFAULT_EPSILON_END,
        epsilon_decay: float = DEFAULT_EPSILON_DECAY,
    ):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Networks
        self.q_network = SimpleQNetwork(lr=lr)
        self.target_network = SimpleQNetwork(lr=lr)
        self.target_network.copy_from(self.q_network)

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

        # Training state
        self._step_count = 0
        self._is_trained = False
        self._training_metrics: Optional[TrainingMetrics] = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def get_allocation(self, state: RLState) -> AllocationAction:
        """
        Get the allocation action for the current state.

        Uses the trained Q-network (greedy policy, no exploration).
        Falls back to rule-based defaults if not trained.
        """
        if not self._is_trained:
            # Rule-based fallback (same as live_strategy.py)
            if state.regime_bear > 0.5:
                return AllocationAction(hedge_ratio=0.70, aggression=0.75)
            elif state.regime_bull > 0.5:
                return AllocationAction(hedge_ratio=0.30, aggression=1.25)
            else:
                return AllocationAction(hedge_ratio=0.50, aggression=1.0)

        q_values = self.q_network.forward(state.to_array())
        action_idx = int(np.argmax(q_values))
        return decode_action(action_idx)

    def train(
        self,
        episodes: list[pd.DataFrame],
        n_epochs: int = 3,
    ) -> TrainingMetrics:
        """
        Train the RL allocator on historical episodes.

        Each episode is a DataFrame with columns:
          - regime: str ("BULL", "BEAR", "SIDEWAYS")
          - rolling_sharpe_20d, rolling_vol_20d, current_drawdown,
            vix_z, sentiment_score, momentum_20d: float
          - portfolio_return: float (daily return of the portfolio)

        The agent learns which (hedge, aggression) choices maximise
        differential Sharpe across different regime conditions.

        Parameters
        ----------
        episodes : list[pd.DataFrame]
            Walk-forward episodes (each is a contiguous period).
        n_epochs : int
            Number of passes over all episodes.

        Returns
        -------
        TrainingMetrics
        """
        metrics = TrainingMetrics()
        all_rewards = []

        for epoch in range(n_epochs):
            for episode_df in episodes:
                episode_reward = self._train_episode(episode_df)
                all_rewards.append(episode_reward)
                metrics.episodes += 1

                # Decay epsilon
                self.epsilon = max(
                    self.epsilon_end,
                    self.epsilon * self.epsilon_decay,
                )

        metrics.total_steps = self._step_count
        metrics.final_epsilon = self.epsilon
        if all_rewards:
            metrics.mean_episode_reward = float(np.mean(all_rewards))
            metrics.best_episode_sharpe = float(np.max(all_rewards))

        self._is_trained = True
        self._training_metrics = metrics

        logger.info(
            "RL allocator trained: %d episodes, %d steps, "
            "mean_reward=%.4f, epsilon=%.3f",
            metrics.episodes, metrics.total_steps,
            metrics.mean_episode_reward, metrics.final_epsilon,
        )

        return metrics

    def _train_episode(self, episode_df: pd.DataFrame) -> float:
        """Run one training episode, returns total reward."""
        if len(episode_df) < 5:
            return 0.0

        # Build states from the episode
        states = self._build_states(episode_df)
        returns = episode_df["portfolio_return"].values

        # Compute differential Sharpe rewards
        rewards = differential_sharpe_reward(returns)

        total_reward = 0.0
        current_hedge = 0.5  # Start neutral
        current_aggression = 1.0

        for t in range(len(states) - 1):
            state = states[t]
            state_arr = state.to_array()

            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                action_idx = np.random.randint(N_ACTIONS)
            else:
                q_vals = self.q_network.forward(state_arr)
                action_idx = int(np.argmax(q_vals))

            action = decode_action(action_idx)
            current_hedge = action.hedge_ratio
            current_aggression = action.aggression

            # Update next state's portfolio features
            next_state = states[t + 1]
            next_state.current_hedge = current_hedge

            next_arr = next_state.to_array()
            done = (t == len(states) - 2)

            # Scale reward by aggression (higher aggression amplifies both gain and loss)
            scaled_reward = rewards[t] * current_aggression

            # Store experience
            self.replay_buffer.push(Experience(
                state=state_arr,
                action=action_idx,
                reward=scaled_reward,
                next_state=next_arr,
                done=done,
            ))

            total_reward += scaled_reward
            self._step_count += 1

            # Train on replay batch
            if len(self.replay_buffer) >= MIN_REPLAY_SIZE:
                loss = self._train_step()
                if self._step_count % 100 == 0:
                    logger.debug("Step %d: loss=%.6f", self._step_count, loss)

            # Update target network periodically
            if self._step_count % TARGET_UPDATE_FREQ == 0:
                self.target_network.copy_from(self.q_network)

        return total_reward

    def _train_step(self) -> float:
        """One DQN training step on a mini-batch from replay buffer."""
        batch = self.replay_buffer.sample(min(BATCH_SIZE, len(self.replay_buffer)))

        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch], dtype=np.float32)

        # Compute targets: r + γ * max_a' Q_target(s', a') * (1 - done)
        next_q = self.target_network.forward(next_states)
        max_next_q = np.max(next_q, axis=1)
        targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.q_network.train_batch(states, actions, targets)
        return loss

    def _build_states(self, df: pd.DataFrame) -> list[RLState]:
        """Convert episode DataFrame to list of RLState observations."""
        states = []
        for _, row in df.iterrows():
            regime = row.get("regime", "SIDEWAYS")
            state = RLState(
                regime_bull=1.0 if regime == "BULL" else 0.0,
                regime_sideways=1.0 if regime == "SIDEWAYS" else 0.0,
                regime_bear=1.0 if regime == "BEAR" else 0.0,
                rolling_sharpe_20d=float(row.get("rolling_sharpe_20d", 0.0)),
                rolling_vol_20d=float(row.get("rolling_vol_20d", 0.0)),
                current_drawdown=float(row.get("current_drawdown", 0.0)),
                vix_z=float(row.get("vix_z", 0.0)),
                sentiment_score=float(row.get("sentiment_score", 0.0)),
                momentum_20d=float(row.get("momentum_20d", 0.0)),
                current_hedge=float(row.get("current_hedge", 0.5)),
                portfolio_return_5d=float(row.get("portfolio_return_5d", 0.0)),
                portfolio_drawdown=float(row.get("portfolio_drawdown", 0.0)),
            )
            states.append(state)
        return states

    def get_training_metrics(self) -> Optional[TrainingMetrics]:
        """Return metrics from the last training run."""
        return self._training_metrics
