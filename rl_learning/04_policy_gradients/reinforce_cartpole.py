"""Vanilla REINFORCE on CartPole-v1.

The first method here with no Q at all. The network *is* the policy: it outputs a probability
per action, we sample from it, and afterwards we push up the log-probability of whatever the
good episodes did:

    loss = -sum_t  gamma^t * G_t * log pi(a_t | s_t)

No replay buffer, no target network, no bootstrapping. Exploration is free, because the policy
is random by construction, so there is no epsilon to decay.

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
    """The returned log-prob must stay attached: it is the only path for gradients."""
    logits = policy(torch.as_tensor(state, dtype=torch.float32))
    dist = D.Categorical(logits=logits)   # logits, not probs: softmax happens internally
    action = dist.sample()
    return int(action.item()), dist.log_prob(action)


def greedy_action(state, policy):
    with torch.no_grad():
        logits = policy(torch.as_tensor(state, dtype=torch.float32))
    return int(torch.argmax(logits).item())


def calculate_returns(rewards, gamma):
    """G_t = r_t + gamma * G_{t+1}, accumulated backwards in one pass."""
    returns = []
    G = 0.0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.append(G)
    return list(reversed(returns))


def run_episode(env, policy, gamma=GAMMA):
    """Play a full episode; Monte Carlo means no update until it ends.

    States are appended before stepping, so states[t] is the state the action was chosen
    in. The baseline script needs that alignment to compute V(s_t).
    """
    state, _ = env.reset()
    states, log_probs, rewards = [], [], []
    done = False

    while not done:
        states.append(np.asarray(state, dtype=np.float32))
        action, log_prob = sample_action(state, policy)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        log_probs.append(log_prob)
        rewards.append(float(reward))

    return log_probs, calculate_returns(rewards, gamma), sum(rewards), states


def compute_loss(log_probs, returns, gamma=GAMMA):
    terms = [
        -(gamma ** t) * G * log_prob
        for t, (log_prob, G) in enumerate(zip(log_probs, returns))
    ]
    return torch.stack(terms).sum()   # a list of tensors needs stacking, not sum()


def train_reinforce(env, policy, optimizer, episodes=EPISODES, gamma=GAMMA,
                    desc="REINFORCE CartPole"):
    returns = []

    for _ in tqdm(range(episodes), desc=desc):
        log_probs, discounted_returns, episode_return, _ = run_episode(env, policy, gamma)
        loss = compute_loss(log_probs, discounted_returns, gamma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        returns.append(episode_return)

    return returns


def evaluate(env, policy, n_episodes=20):
    """Greedy: sample while training, take the argmax when measuring."""
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
