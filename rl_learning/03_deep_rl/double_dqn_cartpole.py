"""Double DQN on CartPole-v1.

Same loop as dqn_cartpole.py, but the target splits two jobs:
  - the online net SELECTS the next action (argmax)
  - the target net EVALUATES it

This curbs the overestimation you get from taking max over noisy Q-values.

    python rl_learning/03_deep_rl/double_dqn_cartpole.py
"""

import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from dqn_cartpole import (
    BUFFER_CAPACITY,
    LR,
    RECORD_VIDEO,
    SEED,
    SHOW_PLOT,
    ReplayBuffer,
    build_networks,
    evaluate,
    moving_average,
    record_video,
    train_dqn,
)


def compute_td_target_double(network, target_network, next_states, rewards, dones, gamma):
    next_actions = network(next_states).argmax(dim=1)                      # online selects
    next_q = target_network(next_states).gather(
        1, next_actions.unsqueeze(1)
    ).squeeze(1)                                                            # target evaluates
    return rewards + gamma * next_q * (1.0 - dones)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make("CartPole-v1")
    online, target = build_networks(env)
    buffer = ReplayBuffer(BUFFER_CAPACITY)
    optimizer = optim.Adam(online.parameters(), lr=LR)

    returns = train_dqn(
        env, online, target, buffer, optimizer,
        td_target_fn=compute_td_target_double,
        desc="Double DQN CartPole",
    )
    print(f"Greedy eval (Double DQN): {evaluate(env, online):.1f} / 500")

    if RECORD_VIDEO:
        print(f"Video return: {record_video(online):.0f} (saved under ./videos/)")

    if SHOW_PLOT:
        plt.figure(figsize=(8, 4))
        plt.plot(moving_average(np.asarray(returns, dtype=float)))
        plt.xlabel("Episode")
        plt.ylabel("Moving average return (100)")
        plt.title("Double DQN on CartPole")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
