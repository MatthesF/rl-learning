"""PPO on Pendulum-v1 — continuous actions.

Same clip / GAE / minibatch reuse as ppo_cartpole.py. Only the policy changes:

    mean = network(s)
    std  = exp(log_std)          free parameter, shared across states
    a ~ Normal(mean, std)

Two traps:

    clipping   store the raw sample, clip only what the env sees
    gamma      0.9, not 0.99 — 200 steps of −16..0 makes γ=0.99 targets unfittable

    python ppo_pendulum.py
"""

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from gae_cartpole import LAMBDA, compute_gae
from ppo_cartpole import CLIP, EPOCHS, LR, MAX_GRAD_NORM, MINIBATCH_SIZE, ROLLOUT_STEPS
from reinforce_baseline_cartpole import create_value_network
from reinforce_cartpole import (
    RECORD_VIDEO,
    SEED,
    SHOW_PLOT,
    create_policy_network,
    moving_average,
)

ENV_ID = "Pendulum-v1"
TOTAL_STEPS = 300_000
GAMMA = 0.9
ENTROPY_COEF = 0.0   # log_std is the exploration; an entropy bonus fights it


class GaussianPolicy(nn.Module):
    """mean(s) from a net; state-independent log_std (init 0 → std=1)."""

    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.mean = create_policy_network(n_obs, n_actions)
        self.log_std = nn.Parameter(torch.zeros(n_actions))

    def distribution(self, states):
        return D.Normal(self.mean(states), self.log_std.exp())


def action_log_prob(dist, actions):
    return dist.log_prob(actions).sum(-1)  # product over dims → sum of logs


def collect_rollout(env, state, policy, value_network, n_steps=ROLLOUT_STEPS):
    low, high = env.action_space.low, env.action_space.high
    states, actions, old_log_probs = [], [], []
    rewards, values, dones = [], [], []
    episode_returns, episode_return = [], 0.0

    for _ in range(n_steps):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            dist = policy.distribution(state_t)
            action = dist.sample()
            log_prob = action_log_prob(dist, action)
            value = value_network(state_t).squeeze(-1)

        next_state, reward, terminated, truncated, _ = env.step(
            np.clip(action.numpy(), low, high)
        )
        done = terminated or truncated
        episode_return += float(reward)

        if truncated:  # real future, just cut short
            with torch.no_grad():
                last_v = float(value_network(
                    torch.as_tensor(next_state, dtype=torch.float32)).squeeze(-1))
            reward = reward + GAMMA * last_v

        states.append(np.asarray(state, dtype=np.float32))
        actions.append(action.numpy())  # unclipped — ratio must match what was sampled
        old_log_probs.append(float(log_prob))
        rewards.append(float(reward))
        values.append(float(value))
        dones.append(done)

        if done:
            episode_returns.append(episode_return)
            episode_return = 0.0
            next_state, _ = env.reset()

        state = next_state

    with torch.no_grad():
        next_value = 0.0 if dones[-1] else float(
            value_network(torch.as_tensor(state, dtype=torch.float32)).squeeze(-1)
        )

    return state, {
        "states": states,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "rewards": rewards,
        "values": values,
        "dones": dones,
        "next_value": next_value,
        "episode_returns": episode_returns,
    }


def ppo_losses(policy, value_network, states, actions, old_log_probs, advantages,
               value_targets):
    dist = policy.distribution(states)
    new_log_probs = action_log_prob(dist, actions)
    ratio = torch.exp(new_log_probs - old_log_probs)

    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - CLIP, 1.0 + CLIP) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    policy_loss = policy_loss - ENTROPY_COEF * dist.entropy().sum(-1).mean()

    values = value_network(states).squeeze(-1)
    value_loss = F.smooth_l1_loss(values, value_targets)
    return policy_loss, value_loss


def update(policy, value_network, policy_optimizer, value_optimizer, rollout,
           advantages, value_targets):
    states = torch.as_tensor(np.asarray(rollout["states"]), dtype=torch.float32)
    actions = torch.as_tensor(np.asarray(rollout["actions"]), dtype=torch.float32)
    old_log_probs = torch.as_tensor(rollout["old_log_probs"], dtype=torch.float32)
    value_targets = torch.as_tensor(value_targets, dtype=torch.float32)

    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(states)
    for _ in range(EPOCHS):
        order = torch.randperm(n)
        for start in range(0, n, MINIBATCH_SIZE):
            batch = order[start:start + MINIBATCH_SIZE]
            policy_loss, value_loss = ppo_losses(
                policy, value_network,
                states[batch], actions[batch], old_log_probs[batch],
                advantages[batch], value_targets[batch],
            )
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            nn.utils.clip_grad_norm_(value_network.parameters(), MAX_GRAD_NORM)
            policy_optimizer.step()
            value_optimizer.step()


def train_ppo(env, policy, value_network, policy_optimizer, value_optimizer,
              total_steps=TOTAL_STEPS, gamma=GAMMA, lam=LAMBDA, desc="PPO Pendulum"):
    state, _ = env.reset()
    returns = []

    with tqdm(total=total_steps, desc=desc, unit="step") as pbar:
        for _ in range(total_steps // ROLLOUT_STEPS):
            state, rollout = collect_rollout(env, state, policy, value_network)
            advantages, value_targets = compute_gae(
                rollout["rewards"], rollout["values"], rollout["dones"],
                rollout["next_value"], gamma, lam,
            )
            update(policy, value_network, policy_optimizer, value_optimizer,
                   rollout, advantages, value_targets)

            returns.extend(rollout["episode_returns"])
            pbar.update(ROLLOUT_STEPS)
            if returns:
                pbar.set_postfix(ret=f"{np.mean(returns[-20:]):.0f}",
                                 std=f"{policy.log_std.exp().item():.2f}")

    return returns


def greedy_action(state, policy, low, high):
    with torch.no_grad():
        mean = policy.mean(torch.as_tensor(state, dtype=torch.float32))
    return np.clip(mean.numpy(), low, high)


def evaluate(env, policy, n_episodes=20):
    low, high = env.action_space.low, env.action_space.high
    totals = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done, total = False, 0.0
        while not done:
            state, reward, terminated, truncated, _ = env.step(
                greedy_action(state, policy, low, high)
            )
            done = terminated or truncated
            total += float(reward)
        totals.append(total)
    return float(np.mean(totals))


def record_video(policy, folder="videos", name_prefix="pendulum-ppo"):
    from gymnasium.wrappers import RecordVideo

    Path(folder).mkdir(parents=True, exist_ok=True)
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=folder, name_prefix=name_prefix,
                      episode_trigger=lambda ep: True)

    low, high = env.action_space.low, env.action_space.high
    state, _ = env.reset()
    done, total = False, 0.0
    while not done:
        state, reward, terminated, truncated, _ = env.step(
            greedy_action(state, policy, low, high)
        )
        done = terminated or truncated
        total += float(reward)
    env.close()
    return total


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make(ENV_ID)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = GaussianPolicy(obs_dim, action_dim)
    value_network = create_value_network(obs_dim)
    policy_optimizer = optim.Adam(policy.parameters(), lr=LR)
    value_optimizer = optim.Adam(value_network.parameters(), lr=LR)

    returns = train_ppo(env, policy, value_network, policy_optimizer, value_optimizer)
    print(f"Episodes played: {len(returns)}")
    print(f"Greedy eval (PPO): {evaluate(env, policy):.1f}  (random ~ -1200)")

    if RECORD_VIDEO:
        print(f"Video return: {record_video(policy):.0f} (saved under ./videos/)")

    if SHOW_PLOT:
        plt.figure(figsize=(8, 4))
        plt.plot(moving_average(np.asarray(returns, dtype=float), window=20))
        plt.xlabel("Episode")
        plt.ylabel("Moving average return (20)")
        plt.title("PPO on Pendulum (continuous)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
