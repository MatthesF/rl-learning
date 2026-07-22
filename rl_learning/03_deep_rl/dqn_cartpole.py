"""DQN on CartPole-v1.

A small, readable single-file DQN: replay buffer, target network,
step-based epsilon decay, warmup, and Huber loss. Grown from notebook
experiments and cleaned up here.

    python rl_learning/03_deep_rl/dqn_cartpole.py
"""

import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# ── hyperparameters ──────────────────────────────────────
EPISODES = 600
BATCH_SIZE = 64
WARMUP_STEPS = 1_000       # collect this many transitions before learning
BUFFER_CAPACITY = 50_000
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 10_000   # measured in env steps, not episodes
TARGET_SYNC_EVERY = 200    # hard-copy online -> target this often
SEED = 0
RECORD_VIDEO = False
SHOW_PLOT = True


def create_q_network(n_obs, n_actions):
    return nn.Sequential(
        nn.Linear(n_obs, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, n_actions),
    )


def choose_action(state, env, epsilon, network):
    if np.random.random() < epsilon:
        return int(env.action_space.sample())
    with torch.no_grad():
        q = network(torch.as_tensor(state, dtype=torch.float32))
    return int(torch.argmax(q).item())


def epsilon_by_step(start, end, decay_steps, t):
    """Linear epsilon decay, measured in environment steps."""
    if decay_steps <= 0:
        return end
    return start + (end - start) * min(1.0, t / decay_steps)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)   # O(1) eviction of oldest

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def compute_td_target(network, target_network, next_states, rewards, dones, gamma):
    """Vanilla DQN: the target net both picks and evaluates the max action."""
    next_q = target_network(next_states).max(dim=1).values
    return rewards + gamma * next_q * (1.0 - dones)


def train_step(network, target_network, buffer, optimizer, batch_size, gamma,
               td_target_fn=compute_td_target, max_grad_norm=10.0):
    states, actions, rewards, next_states, dones = zip(*buffer.sample(batch_size))
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Pick Q(s, a) for the action actually taken in each transition.
    q_taken = network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        td_target = td_target_fn(network, target_network, next_states, rewards, dones, gamma)

    loss = F.smooth_l1_loss(q_taken, td_target)   # Huber: robust to large TD errors
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
    optimizer.step()


def train_dqn(env, network, target_network, buffer, optimizer,
              num_episodes=EPISODES, batch_size=BATCH_SIZE, warmup_steps=WARMUP_STEPS,
              gamma=GAMMA, eps_start=EPS_START, eps_end=EPS_END,
              eps_decay_steps=EPS_DECAY_STEPS, target_sync_every=TARGET_SYNC_EVERY,
              td_target_fn=compute_td_target, desc="DQN CartPole"):
    total_steps = 0
    returns = []

    for _ in tqdm(range(num_episodes), desc=desc):
        state, _ = env.reset()
        done = False
        episode_return = 0.0

        while not done:
            epsilon = epsilon_by_step(eps_start, eps_end, eps_decay_steps, total_steps)
            action = choose_action(state, env, epsilon, network)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Only "terminated" masks the bootstrap; a time-limit still bootstraps.
            buffer.push(state, action, reward, next_state, float(terminated))
            state = next_state
            done = terminated or truncated
            episode_return += float(reward)
            total_steps += 1

            if len(buffer) >= max(warmup_steps, batch_size):
                train_step(network, target_network, buffer, optimizer, batch_size, gamma, td_target_fn)

            if total_steps % target_sync_every == 0:
                target_network.load_state_dict(network.state_dict())

        returns.append(episode_return)

    return returns


def evaluate(env, network, n_episodes=20):
    totals = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done, total = False, 0.0
        while not done:
            action = choose_action(state, env, 0.0, network)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += float(reward)
        totals.append(total)
    return float(np.mean(totals))


def moving_average(values, window=100):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def build_networks(env):
    """Online + target nets, target starting as an exact copy."""
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    online = create_q_network(obs_dim, n_actions)
    target = create_q_network(obs_dim, n_actions)
    target.load_state_dict(online.state_dict())
    target.eval()
    return online, target


def record_video(network, folder="videos"):
    from gymnasium.wrappers import RecordVideo

    Path(folder).mkdir(parents=True, exist_ok=True)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=folder, name_prefix="cartpole-dqn",
                      episode_trigger=lambda ep: True)

    state, _ = env.reset()
    done, total = False, 0.0
    while not done:
        action = choose_action(state, env, 0.0, network)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += float(reward)
    env.close()
    return total


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make("CartPole-v1")   # no rendering during training
    online, target = build_networks(env)
    buffer = ReplayBuffer(BUFFER_CAPACITY)
    optimizer = optim.Adam(online.parameters(), lr=LR)

    returns = train_dqn(env, online, target, buffer, optimizer)
    print(f"Greedy eval: {evaluate(env, online):.1f} / 500")

    if RECORD_VIDEO:
        print(f"Video return: {record_video(online):.0f} (saved under ./videos/)")

    if SHOW_PLOT:
        plt.figure(figsize=(8, 4))
        plt.plot(moving_average(np.asarray(returns, dtype=float)))
        plt.xlabel("Episode")
        plt.ylabel("Moving average return (100)")
        plt.title("DQN on CartPole")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
