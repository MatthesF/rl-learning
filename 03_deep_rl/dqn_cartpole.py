"""DQN on CartPole-v1.

Same Q-learning target as the tabular version, but Q is a network:

    loss = Huber( Q(s,a),  r + gamma * max_a' Q_target(s',a') )

Replacing the table with a network breaks two assumptions, and the two famous tricks here
are exactly the repairs:

    replay buffer    consecutive steps are almost identical, and gradient descent wants
                     independent samples, so store transitions and sample randomly
    target network   the target is computed from the same weights being trained, so it
                     moves every step; freeze a copy and refresh it now and then

    python 03_deep_rl/dqn_cartpole.py
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

SEED = 0
EPISODES = 600
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_CAPACITY = 50_000
WARMUP_STEPS = 1_000        # random experience to collect before the first update
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 10_000    # env steps, not episodes: early episodes last ~20 steps
TARGET_SYNC_EVERY = 200     # env steps between hard copies online -> target
MAX_GRAD_NORM = 10.0
RECORD_VIDEO = False
SHOW_PLOT = True


def create_q_network(n_obs, n_actions):
    """One output per action, so a single forward pass gives Q(s, ·)."""
    return nn.Sequential(
        nn.Linear(n_obs, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, n_actions),
    )


def build_networks(env):
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    online = create_q_network(obs_dim, n_actions)
    target = create_q_network(obs_dim, n_actions)
    target.load_state_dict(online.state_dict())
    target.eval()
    return online, target


def choose_action(state, env, epsilon, network):
    if np.random.random() < epsilon:
        return int(env.action_space.sample())
    with torch.no_grad():
        q = network(torch.as_tensor(state, dtype=torch.float32))
    return int(torch.argmax(q).item())


def epsilon_by_step(step, start=EPS_START, end=EPS_END, decay_steps=EPS_DECAY_STEPS):
    return start + (end - start) * min(1.0, step / decay_steps)


class ReplayBuffer:
    def __init__(self, capacity=BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)   # deque evicts the oldest in O(1)

    def push(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def compute_td_target(online, target, next_states, rewards, terminateds, gamma):
    """Vanilla DQN: the target net both picks and evaluates the best next action."""
    next_q = target(next_states).max(dim=1).values
    return rewards + gamma * next_q * (1.0 - terminateds)


def train_step(online, target, buffer, optimizer, td_target_fn=compute_td_target):
    states, actions, rewards, next_states, terminateds = zip(*buffer.sample(BATCH_SIZE))
    states = torch.as_tensor(np.array(states), dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.int64)
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32)
    terminateds = torch.as_tensor(terminateds, dtype=torch.float32)

    # The net outputs Q for every action; keep the one that was actually taken.
    q_taken = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        td_target = td_target_fn(online, target, next_states, rewards, terminateds, GAMMA)

    loss = F.smooth_l1_loss(q_taken, td_target)   # Huber survives the odd huge TD error
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), MAX_GRAD_NORM)
    optimizer.step()


def train_dqn(env, online, target, buffer, optimizer, td_target_fn=compute_td_target,
              episodes=EPISODES, desc="DQN CartPole"):
    returns = []
    step = 0

    for _ in tqdm(range(episodes), desc=desc):
        state, _ = env.reset()
        episode_return, done = 0.0, False

        while not done:
            action = choose_action(state, env, epsilon_by_step(step), online)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Store `terminated`, not `done`: hitting the 500-step limit is not a real
            # ending, so those transitions should still bootstrap from s'.
            buffer.push(state, action, reward, next_state, float(terminated))

            state = next_state
            done = terminated or truncated
            episode_return += float(reward)
            step += 1

            if len(buffer) >= max(WARMUP_STEPS, BATCH_SIZE):
                train_step(online, target, buffer, optimizer, td_target_fn)

            if step % TARGET_SYNC_EVERY == 0:
                target.load_state_dict(online.state_dict())

        returns.append(episode_return)

    return returns


def evaluate(env, network, n_episodes=20):
    totals = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done, total = False, 0.0
        while not done:
            state, reward, terminated, truncated, _ = env.step(
                choose_action(state, env, 0.0, network)
            )
            done = terminated or truncated
            total += float(reward)
        totals.append(total)
    return float(np.mean(totals))


def moving_average(values, window=100):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def record_video(network, folder="videos", name_prefix="cartpole-dqn"):
    from gymnasium.wrappers import RecordVideo

    Path(folder).mkdir(parents=True, exist_ok=True)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=folder, name_prefix=name_prefix,
                      episode_trigger=lambda ep: True)

    state, _ = env.reset()
    done, total = False, 0.0
    while not done:
        state, reward, terminated, truncated, _ = env.step(
            choose_action(state, env, 0.0, network)
        )
        done = terminated or truncated
        total += float(reward)
    env.close()
    return total


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make("CartPole-v1")   # no rendering during training
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    online, target = build_networks(env)
    buffer = ReplayBuffer()
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
