"""Double DQN on CartPole-v1.

`max_a' Q(s',a')` is biased upwards: the max of noisy estimates lands on whichever action
got lucky, and that error gets baked into the target. Double DQN splits the two jobs the max
was doing at once:

    online net   SELECTS the next action    argmax_a' Q_online(s',a')
    target net   EVALUATES it               Q_target(s', that action)

Two independent sets of weights have to agree before an action looks good, which removes most
of the built-in optimism. Everything else — buffer, target sync, loss — is imported unchanged
from dqn_cartpole.py, so the only difference between the two runs is this target.

    python 03_deep_rl/double_dqn_cartpole.py
"""

import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from dqn_cartpole import (
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


def compute_double_td_target(online, target, next_states, rewards, terminateds, gamma):
    next_actions = online(next_states).argmax(dim=1)
    next_q = target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
    return rewards + gamma * next_q * (1.0 - terminateds)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    online, target = build_networks(env)
    buffer = ReplayBuffer()
    optimizer = optim.Adam(online.parameters(), lr=LR)

    returns = train_dqn(
        env, online, target, buffer, optimizer,
        td_target_fn=compute_double_td_target,
        desc="Double DQN CartPole",
    )
    print(f"Greedy eval (Double DQN): {evaluate(env, online):.1f} / 500")

    if RECORD_VIDEO:
        video_return = record_video(online, name_prefix="cartpole-double-dqn")
        print(f"Video return: {video_return:.0f} (saved under ./videos/)")

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
