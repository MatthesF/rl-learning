"""REINFORCE with a learned baseline on CartPole-v1.

Raw returns credit every action in a good state. Subtract V(s):

    A_t = G_t - V(s_t)

Same expected gradient, much lower variance.

    python reinforce_baseline_cartpole.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

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
    run_episode,
)


def create_value_network(n_obs):
    return nn.Sequential(
        nn.Linear(n_obs, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 1),
    )


def compute_losses(log_probs, states, returns, value_network, gamma=GAMMA):
    states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    returns_t = torch.as_tensor(returns, dtype=torch.float32)
    log_probs_t = torch.stack(log_probs)

    values = value_network(states_t).squeeze(-1)

    # detach: the policy must not lower its loss by corrupting the critic instead.
    advantages = returns_t - values.detach()

    discounts = gamma ** torch.arange(len(returns_t), dtype=torch.float32)
    policy_loss = -(discounts * advantages * log_probs_t).sum()
    value_loss = F.smooth_l1_loss(values, returns_t)

    return policy_loss, value_loss


def train_reinforce_baseline(env, policy, value_network, policy_optimizer, value_optimizer,
                             episodes=EPISODES, gamma=GAMMA, desc="REINFORCE + baseline"):
    returns = []

    for _ in tqdm(range(episodes), desc=desc):
        log_probs, discounted_returns, episode_return, states = run_episode(env, policy, gamma)

        # Both losses come from the same V estimate; stepping the critic first would give
        # the policy a baseline that has already seen this episode's answer.
        policy_loss, value_loss = compute_losses(
            log_probs, states, discounted_returns, value_network, gamma
        )

        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        policy_optimizer.step()
        value_optimizer.step()

        returns.append(episode_return)

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

    returns = train_reinforce_baseline(
        env, policy, value_network, policy_optimizer, value_optimizer
    )
    print(f"Greedy eval (baseline): {evaluate(env, policy):.1f} / 500")

    if RECORD_VIDEO:
        video_return = record_video(policy, name_prefix="cartpole-reinforce-baseline")
        print(f"Video return: {video_return:.0f} (saved under ./videos/)")

    if SHOW_PLOT:
        plt.figure(figsize=(8, 4))
        plt.plot(moving_average(np.asarray(returns, dtype=float)))
        plt.xlabel("Episode")
        plt.ylabel("Moving average return (100)")
        plt.title("REINFORCE with baseline on CartPole")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
