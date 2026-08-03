"""PPO on Pendulum-v1 — the same algorithm with a continuous action.

Every script so far picked from a short list of actions, so a policy was a probability per
entry and log pi(a|s) was a lookup. Pendulum wants a torque anywhere in [-2, 2], so there is
nothing to look up in. The network describes a distribution over the real line instead:

    mean = network(s)          one number per action dimension
    std  = exp(log_std)        a free parameter, the same in every state
    a ~ Normal(mean, std)

Exploration is now that std, and it shrinks on its own as the policy gets confident (the
progress bar prints it). Everything else — clip, GAE, minibatch reuse, the constants — is
imported from ppo_cartpole.py unchanged, which is the point of this file.

Two traps that do not exist in the discrete case:

    clipping   Normal can sample outside [-2, 2]. Clip what the env sees, store the raw
               sample, because that is the action log pi has to be evaluated at.
    gamma      0.9, not 0.99. Pendulum never terminates, it is cut off after 200 steps, and
               every step pays -16..0. With 0.99 the critic has to nail returns near -900
               before the advantages mean anything; with 0.9 the targets are small enough to
               fit, and learning takes off after ~60k steps.

    python 04_policy_gradients/ppo_pendulum.py
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
GAMMA = 0.9          # see the docstring: 200 steps of -16..0, so 0.99 is unfittable
ENTROPY_COEF = 0.0   # log_std handles exploration; a bonus would only fight it


class GaussianPolicy(nn.Module):
    """The same network as before, read differently.

    create_policy_network maps a state to n_actions numbers. On CartPole those were logits,
    one per available action; here the single number is the *mean* torque.

    The spread is a bare parameter rather than a network output: the policy is equally unsure
    in every state, it just gets less unsure over training. log_std = 0 means std = 1, wide
    enough to cover a good part of [-2, 2] on the first rollout.
    """

    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.mean = create_policy_network(n_obs, n_actions)
        self.log_std = nn.Parameter(torch.zeros(n_actions))

    def distribution(self, states):
        return D.Normal(self.mean(states), self.log_std.exp())


def action_log_prob(dist, actions):
    """Sum over action dimensions: independent dimensions multiply, so their logs add."""
    return dist.log_prob(actions).sum(-1)


def collect_rollout(env, state, policy, value_network, n_steps=ROLLOUT_STEPS):
    """Identical to the CartPole version except for how the action is handled."""
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

        states.append(np.asarray(state, dtype=np.float32))
        # The *unclipped* sample. Clipping is the env's business; the ratio has to be
        # computed at the action the old policy actually drew, or log pi(a|s) is a lie.
        actions.append(action.numpy())
        old_log_probs.append(float(log_prob))
        rewards.append(float(reward))
        values.append(float(value))
        dones.append(done)

        episode_return += float(reward)
        if done:
            episode_returns.append(episode_return)
            episode_return = 0.0
            next_state, _ = env.reset()

        state = next_state

    # Pendulum has no failure state, so this branch is always the bootstrap one — the
    # episode was cut off mid-swing and V(s) is the best guess at what was left.
    with torch.no_grad():
        next_value = 0.0 if dones[-1] else float(
            value_network(torch.as_tensor(state, dtype=torch.float32)).squeeze(-1)
        )

    rollout = {
        "states": states,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "rewards": rewards,
        "values": values,
        "dones": dones,
        "next_value": next_value,
        "episode_returns": episode_returns,
    }
    return state, rollout


def ppo_losses(policy, value_network, states, actions, old_log_probs, advantages,
               value_targets):
    """Word for word the CartPole objective; only the distribution underneath differs."""
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
    """EPOCHS passes over the rollout in shuffled minibatches.

    One line differs from ppo_cartpole.update: actions are float32 coordinates now, not
    int64 indices into a list.
    """
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
                # std is worth watching: it is this policy's entire exploration schedule.
                pbar.set_postfix(ret=f"{np.mean(returns[-20:]):.0f}",
                                 std=f"{policy.log_std.exp().item():.2f}")

    return returns


def greedy_action(state, policy, low, high):
    """Greedy for a Gaussian means the mean, not a sample."""
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

    obs_dim = env.observation_space.shape[0]        # 3: cos, sin, angular velocity
    action_dim = env.action_space.shape[0]          # 1: torque, continuous
    policy = GaussianPolicy(obs_dim, action_dim)
    value_network = create_value_network(obs_dim)
    policy_optimizer = optim.Adam(policy.parameters(), lr=LR)
    value_optimizer = optim.Adam(value_network.parameters(), lr=LR)

    returns = train_ppo(env, policy, value_network, policy_optimizer, value_optimizer)
    print(f"Episodes played: {len(returns)}")
    # No 500-step cap to compare against here: a return is a sum of -16..0 penalties, so
    # -1200 is a pendulum hanging down and anything above -300 is a real swing-up.
    print(f"Greedy eval (PPO): {evaluate(env, policy):.1f}  (random ~ -1200)")

    if RECORD_VIDEO:
        print(f"Video return: {record_video(policy):.0f} (saved under ./videos/)")

    if SHOW_PLOT:
        plt.figure(figsize=(8, 4))
        plt.plot(moving_average(np.asarray(returns, dtype=float), window=20))
        plt.xlabel("Episode")
        plt.ylabel("Moving average return (20)")
        plt.title("PPO on Pendulum (continuous actions)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
