"""One-step actor-critic on CartPole-v1 (TD instead of Monte Carlo).

Where REINFORCE waits for the full return G_t, the critic here bootstraps
after a single step:

    delta = r + gamma * V(s') - V(s)

so the networks are updated on *every* timestep instead of once per episode.
Follows the episodic one-step actor-critic pseudocode in Sutton & Barto 13.5,
including the discount accumulator I = gamma^t on the policy update.

    python rl_learning/04_policy_gradients/actor_critic_cartpole.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from reinforce_baseline_cartpole import create_value_network
from reinforce_cartpole import (
    EPISODES,
    GAMMA,
    RECORD_VIDEO,
    SEED,
    SHOW_PLOT,
    create_policy_network,
    evaluate,
    moving_average,
    record_video,
    sample_action,
)

POLICY_LR = 3e-4
VALUE_LR = 3e-4


def run_actor_critic_episode(env, policy, value_network, policy_optimizer,
                             value_optimizer, gamma=GAMMA):
    """Play one episode, updating both networks after every step."""
    state, _ = env.reset()
    episode_return = 0.0
    done = False

    # I = gamma^t, the discount accumulator from the book's pseudocode.
    discount = 1.0

    while not done:
        action, log_prob = sample_action(state, policy)
        value = value_network(torch.as_tensor(state, dtype=torch.float32)).squeeze(-1)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_return += float(reward)

        # A terminal state has V = 0; a time-limit cutoff should still bootstrap.
        with torch.no_grad():
            if terminated:
                next_value = torch.zeros_like(value)
            else:
                next_value = value_network(
                    torch.as_tensor(next_state, dtype=torch.float32)
                ).squeeze(-1)

        td_target = float(reward) + gamma * next_value   # already detached
        delta = td_target - value

        # theta <- theta + alpha * I * delta * grad log pi   (I only on the actor)
        policy_loss = -(discount * delta.detach() * log_prob)
        # The 1/2 gives exactly w <- w + alpha * delta * grad V.
        value_loss = 0.5 * delta.pow(2)

        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        policy_optimizer.step()
        value_optimizer.step()

        discount *= gamma
        state = next_state

    return episode_return


def train_actor_critic(env, policy, value_network, policy_optimizer, value_optimizer,
                       num_episodes=EPISODES, gamma=GAMMA, desc="Actor-critic CartPole"):
    returns = []

    for _ in tqdm(range(num_episodes), desc=desc):
        returns.append(
            run_actor_critic_episode(
                env, policy, value_network, policy_optimizer, value_optimizer, gamma
            )
        )

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
    policy_optimizer = optim.Adam(policy.parameters(), lr=POLICY_LR)
    value_optimizer = optim.Adam(value_network.parameters(), lr=VALUE_LR)

    returns = train_actor_critic(
        env, policy, value_network, policy_optimizer, value_optimizer
    )
    print(f"Greedy eval (actor-critic): {evaluate(env, policy):.1f} / 500")

    if RECORD_VIDEO:
        video_return = record_video(policy, name_prefix="cartpole-actor-critic")
        print(f"Video return: {video_return:.0f} (saved under ./videos/)")

    if SHOW_PLOT:
        plt.figure(figsize=(8, 4))
        plt.plot(moving_average(np.asarray(returns, dtype=float)))
        plt.xlabel("Episode")
        plt.ylabel("Moving average return (100)")
        plt.title("One-step actor-critic on CartPole")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
