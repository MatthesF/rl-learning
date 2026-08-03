"""Generalized Advantage Estimation on CartPole-v1.

Between Monte Carlo and one-step TD:

    A_t = delta_t + (gamma*lam) A_{t+1}
    delta_t = r + gamma V(s') - V(s)

lam=1 → full return, lam=0 → one-step, 0.95 usual. Same networks/loss as baseline;
PPO reuses this advantage unchanged.

    python gae_cartpole.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from reinforce_baseline_cartpole import create_value_network
from reinforce_cartpole import (
    EPISODES,
    GAMMA,
    LR,
    RECORD_VIDEO,
    SEED,
    SHOW_PLOT,
    create_policy_network,
    evaluate,
    moving_average,
    record_video,
    sample_action,
)

LAMBDA = 0.95


def collect_episode(env, policy, value_network):
    state, _ = env.reset()
    states, log_probs, rewards, values, terminateds = [], [], [], [], []
    done = False

    while not done:
        states.append(np.asarray(state, dtype=np.float32))
        action, log_prob = sample_action(state, policy)

        with torch.no_grad():
            value = value_network(torch.as_tensor(state, dtype=torch.float32))

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        log_probs.append(log_prob)
        rewards.append(float(reward))
        values.append(float(value.squeeze(-1)))
        terminateds.append(terminated)

    # Terminal: no bootstrap. Truncated (time limit): still bootstrap from V(s).
    with torch.no_grad():
        next_value = 0.0 if terminated else float(
            value_network(torch.as_tensor(state, dtype=torch.float32)).squeeze(-1)
        )

    return log_probs, rewards, values, terminateds, next_value, states


def compute_gae(rewards, values, terminateds, next_value, gamma=GAMMA, lam=LAMBDA):
    bootstrapped = np.append(np.asarray(values, dtype=np.float32), next_value)
    advantages = np.zeros(len(rewards), dtype=np.float32)
    advantage = 0.0

    for t in reversed(range(len(rewards))):
        mask = 0.0 if terminateds[t] else 1.0  # cut the chain at true terminals
        delta = rewards[t] + gamma * bootstrapped[t + 1] * mask - bootstrapped[t]
        advantage = delta + gamma * lam * mask * advantage
        advantages[t] = advantage

    value_targets = advantages + np.asarray(values, dtype=np.float32)  # A + V = return target
    return advantages, value_targets


def compute_losses(log_probs, states, advantages, value_targets, value_network,
                   gamma=GAMMA):
    states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    value_targets_t = torch.as_tensor(value_targets, dtype=torch.float32)
    log_probs_t = torch.stack(log_probs)

    advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    discounts = gamma ** torch.arange(len(advantages_t), dtype=torch.float32)
    policy_loss = -(discounts * advantages_t * log_probs_t).sum()

    values = value_network(states_t).squeeze(-1)
    value_loss = F.smooth_l1_loss(values, value_targets_t)

    return policy_loss, value_loss


def train_gae(env, policy, value_network, policy_optimizer, value_optimizer,
              episodes=EPISODES, gamma=GAMMA, lam=LAMBDA, desc="GAE CartPole"):
    returns = []

    for _ in tqdm(range(episodes), desc=desc):
        log_probs, rewards, values, terminateds, next_value, states = collect_episode(
            env, policy, value_network
        )
        advantages, value_targets = compute_gae(
            rewards, values, terminateds, next_value, gamma, lam
        )
        policy_loss, value_loss = compute_losses(
            log_probs, states, advantages, value_targets, value_network, gamma
        )

        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        policy_optimizer.step()
        value_optimizer.step()

        returns.append(sum(rewards))

    return returns


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    obs_dim = env.observation_space.shape[0]
    policy = create_policy_network(obs_dim, env.action_space.n)
    value_network = create_value_network(obs_dim)
    policy_optimizer = optim.Adam(policy.parameters(), lr=LR)
    value_optimizer = optim.Adam(value_network.parameters(), lr=LR)

    returns = train_gae(env, policy, value_network, policy_optimizer, value_optimizer)
    print(f"Greedy eval (GAE, lambda={LAMBDA}): {evaluate(env, policy):.1f} / 500")

    if RECORD_VIDEO:
        print(f"Video return: {record_video(policy, name_prefix='cartpole-gae'):.0f} "
              f"(saved under ./videos/)")

    if SHOW_PLOT:
        plt.figure(figsize=(8, 4))
        plt.plot(moving_average(np.asarray(returns, dtype=float)))
        plt.xlabel("Episode")
        plt.ylabel("Moving average return (100)")
        plt.title(f"GAE on CartPole (lambda = {LAMBDA})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
