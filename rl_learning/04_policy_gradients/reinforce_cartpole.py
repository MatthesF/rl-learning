"""Vanilla REINFORCE on CartPole-v1.

First policy-gradient agent: the network is the policy, not a Q-function.
Monte Carlo, so there is no replay buffer, no target network, and no
bootstrapping — one update after each full episode.

    python rl_learning/04_policy_gradients/reinforce_cartpole.py
"""

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ── hyperparameters ──────────────────────────────────────
SEED = 0
EPISODES = 1000
GAMMA = 0.99
LR = 1e-3
RECORD_VIDEO = False
SHOW_PLOT = True


def create_policy_network(n_obs, n_actions):
    """Outputs logits over actions — a policy, not Q-values."""
    return nn.Sequential(
        nn.Linear(n_obs, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, n_actions),
    )


def sample_action(state, policy):
    """Sample from pi(a|s); the log-prob must stay attached to the graph."""
    logits = policy(torch.as_tensor(state, dtype=torch.float32))
    dist = D.Categorical(probs=torch.softmax(logits, dim=-1))
    action = dist.sample()
    return int(action.item()), dist.log_prob(action)


def greedy_action(state, policy):
    """Most likely action — used for evaluation and video."""
    with torch.no_grad():
        logits = policy(torch.as_tensor(state, dtype=torch.float32))
    return int(torch.argmax(logits).item())


def calculate_returns(rewards, gamma):
    """G_t = r_{t+1} + gamma * G_{t+1}, computed backwards."""
    returns = []
    G = 0.0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.append(G)
    return list(reversed(returns))


def run_episode(env, policy, gamma):
    """Play one full episode; no update until it ends (Monte Carlo).

    States are collected before each step, so states[t] is the state the
    action was chosen in — the baseline version needs that alignment.
    """
    state, _ = env.reset()
    states, log_probs, rewards = [], [], []
    done = False

    while not done:
        states.append(np.asarray(state).copy())
        action, log_prob = sample_action(state, policy)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        log_probs.append(log_prob)
        rewards.append(float(reward))

    return log_probs, calculate_returns(rewards, gamma), sum(rewards), states


def compute_loss(log_probs, returns, gamma):
    """L = -sum_t gamma^t * G_t * log pi(a_t|s_t)  (Sutton & Barto 13.3)."""
    terms = [
        -(gamma ** t) * G * log_prob
        for t, (log_prob, G) in enumerate(zip(log_probs, returns))
    ]
    return torch.stack(terms).sum()


def train_reinforce(env, policy, optimizer, num_episodes=EPISODES, gamma=GAMMA,
                    desc="REINFORCE CartPole"):
    returns = []

    for _ in tqdm(range(num_episodes), desc=desc):
        log_probs, discounted_returns, episode_return, _ = run_episode(env, policy, gamma)
        loss = compute_loss(log_probs, discounted_returns, gamma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        returns.append(episode_return)

    return returns


def evaluate(env, policy, n_episodes=20):
    totals = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done, total = False, 0.0
        while not done:
            state, reward, terminated, truncated, _ = env.step(greedy_action(state, policy))
            done = terminated or truncated
            total += float(reward)
        totals.append(total)
    return float(np.mean(totals))


def moving_average(values, window=100):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def record_video(policy, folder="videos", name_prefix="cartpole-reinforce"):
    from gymnasium.wrappers import RecordVideo

    Path(folder).mkdir(parents=True, exist_ok=True)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=folder, name_prefix=name_prefix,
                      episode_trigger=lambda ep: True)

    state, _ = env.reset()
    done, total = False, 0.0
    while not done:
        state, reward, terminated, truncated, _ = env.step(greedy_action(state, policy))
        done = terminated or truncated
        total += float(reward)
    env.close()
    return total


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make("CartPole-v1")   # no rendering during training
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    policy = create_policy_network(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    returns = train_reinforce(env, policy, optimizer)
    print(f"Greedy eval: {evaluate(env, policy):.1f} / 500")

    if RECORD_VIDEO:
        print(f"Video return: {record_video(policy):.0f} (saved under ./videos/)")

    if SHOW_PLOT:
        plt.figure(figsize=(8, 4))
        plt.plot(moving_average(np.asarray(returns, dtype=float)))
        plt.xlabel("Episode")
        plt.ylabel("Moving average return (100)")
        plt.title("REINFORCE on CartPole")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
