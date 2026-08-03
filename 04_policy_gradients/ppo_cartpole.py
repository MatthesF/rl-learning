"""PPO on CartPole-v1.

Every script before this one threw the episode away after a single gradient step. PPO reuses
a whole rollout ten times over, which is only safe if the policy cannot run away from the data
it was collected with. The clip is that safety belt:

    ratio = pi_new(a|s) / pi_old(a|s)
    loss  = -min( ratio * A,  clip(ratio, 1-eps, 1+eps) * A )

Read the min as: once an action has become much more likely than it was when the data was
collected, the clipped branch is the smaller one, its gradient is flat, and that sample stops
pushing. Good actions get promoted, but only so far per rollout — the answer to the saturation
that kills actor_critic_cartpole.py.

Three things change compared to gae_cartpole.py:

    rollout      ROLLOUT_STEPS of experience crossing episode boundaries, not one episode
    old data     store actions and *detached* log-probs, so log pi can be recomputed later
    reuse        EPOCHS passes over the same rollout in shuffled minibatches

The advantages come from compute_gae, imported unchanged. The gamma^t term the earlier scripts
carried is gone: PPO averages over the batch instead, as everyone actually runs it.

    python 04_policy_gradients/ppo_cartpole.py
"""

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
from reinforce_baseline_cartpole import create_value_network
from reinforce_cartpole import (
    GAMMA,
    RECORD_VIDEO,
    SEED,
    SHOW_PLOT,
    create_policy_network,
    evaluate,
    moving_average,
    record_video,
)

TOTAL_STEPS = 150_000
ROLLOUT_STEPS = 2048     # experience collected before each update
EPOCHS = 10              # passes over that same experience
MINIBATCH_SIZE = 64
CLIP = 0.2               # how far pi_new may drift from pi_old
LR = 3e-4
ENTROPY_COEF = 0.01      # small bonus for staying unsure
MAX_GRAD_NORM = 0.5


def collect_rollout(env, state, policy, value_network, n_steps=ROLLOUT_STEPS):
    """Run n_steps, resetting and carrying on whenever an episode ends.

    Unlike collect_episode in gae_cartpole.py, the log-probs are stored as plain floats.
    They are the *old* policy's opinion, a fixed reference point for the ratio, so they
    must not be attached to a graph that later gets differentiated ten more times.
    """
    states, actions, old_log_probs = [], [], []
    rewards, values, dones = [], [], []
    episode_returns, episode_return = [], 0.0

    for _ in range(n_steps):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            dist = D.Categorical(logits=policy(state_t))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = value_network(state_t).squeeze(-1)

        next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
        done = terminated or truncated

        # Everything is recorded for s_t, the state the action was chosen in.
        states.append(np.asarray(state, dtype=np.float32))
        actions.append(int(action.item()))
        old_log_probs.append(float(log_prob))
        rewards.append(float(reward))
        values.append(float(value))
        # `done`, not `terminated`: whatever the reason, the next stored state comes from a
        # fresh reset, so the advantage chain has to be cut here. That treats CartPole's
        # 500-step limit as a real ending, which is the one approximation in this file.
        dones.append(done)

        episode_return += float(reward)
        if done:
            episode_returns.append(episode_return)
            episode_return = 0.0
            next_state, _ = env.reset()

        state = next_state

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
    """The clipped surrogate objective, plus the usual critic regression."""
    dist = D.Categorical(logits=policy(states))
    new_log_probs = dist.log_prob(actions)

    # A ratio, computed in log space: exp(log a - log b) == a / b.
    ratio = torch.exp(new_log_probs - old_log_probs)

    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - CLIP, 1.0 + CLIP) * advantages
    # min() over the two branches is what makes the clip one-sided: it never rewards a
    # move past the boundary, but a move that made things worse is not let off the hook.
    policy_loss = -torch.min(unclipped, clipped).mean()
    policy_loss = policy_loss - ENTROPY_COEF * dist.entropy().mean()

    values = value_network(states).squeeze(-1)
    value_loss = F.smooth_l1_loss(values, value_targets)

    return policy_loss, value_loss


def update(policy, value_network, policy_optimizer, value_optimizer, rollout,
           advantages, value_targets):
    """EPOCHS passes over the same rollout, in freshly shuffled minibatches."""
    states = torch.as_tensor(np.asarray(rollout["states"]), dtype=torch.float32)
    actions = torch.as_tensor(rollout["actions"], dtype=torch.int64)
    old_log_probs = torch.as_tensor(rollout["old_log_probs"], dtype=torch.float32)
    value_targets = torch.as_tensor(value_targets, dtype=torch.float32)

    # Whiten once over the whole rollout; per-minibatch statistics would be far too noisy.
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(states)
    for _ in range(EPOCHS):
        # Shuffling matters: fixed minibatches would correlate each update with a slice
        # of the trajectory rather than the rollout as a whole.
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
              total_steps=TOTAL_STEPS, gamma=GAMMA, lam=LAMBDA, desc="PPO CartPole"):
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
                pbar.set_postfix(ret=f"{np.mean(returns[-20:]):.0f}")

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

    returns = train_ppo(env, policy, value_network, policy_optimizer, value_optimizer)
    print(f"Episodes played: {len(returns)}")
    print(f"Greedy eval (PPO): {evaluate(env, policy):.1f} / 500")

    if RECORD_VIDEO:
        print(f"Video return: {record_video(policy, name_prefix='cartpole-ppo'):.0f} "
              f"(saved under ./videos/)")

    if SHOW_PLOT:
        plt.figure(figsize=(8, 4))
        plt.plot(moving_average(np.asarray(returns, dtype=float), window=20))
        plt.xlabel("Episode")
        plt.ylabel("Moving average return (20)")
        plt.title("PPO on CartPole")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
