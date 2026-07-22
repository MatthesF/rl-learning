"""Tabular Q-learning on FrozenLake-v1.

First real RL agent: a plain Q-table updated with the Bellman rule.
Written in a notebook while learning, then cleaned up into this script.

Tweak the hyperparameters below and run:

    python rl_learning/01_tabular/q_learning_frozenlake.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# ── hyperparameters ──────────────────────────────────────
SEED = 0
SLIPPERY = False     # True = stochastic ice (much harder)
EPISODES = 5000
EPSILON = 0.5        # exploration rate (kept fixed for simplicity)
ALPHA = 0.1          # learning rate
GAMMA = 0.99         # discount factor
EVAL_EVERY = 100     # measure the greedy policy this often
SHOW_PLOT = True


def epsilon_greedy(env, state, q, epsilon):
    if np.random.rand() < epsilon:
        return int(env.action_space.sample())
    return int(np.argmax(q[state]))


def q_learning_episode(env, q, epsilon, alpha, gamma):
    """Play one episode and update Q in place."""
    state, _ = env.reset()
    while True:
        action = epsilon_greedy(env, state, q, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # A terminal step has no future, so the target is just the reward.
        target = reward if done else reward + gamma * np.max(q[next_state])
        q[state, action] += alpha * (target - q[state, action])

        state = next_state
        if done:
            break
    return q


def greedy_return(env, q):
    """One greedy rollout (epsilon = 0)."""
    state, _ = env.reset()
    total = 0.0
    while True:
        state, reward, terminated, truncated, _ = env.step(int(np.argmax(q[state])))
        total += float(reward)
        if terminated or truncated:
            break
    return total


def evaluate(env, q, n_episodes=100):
    return float(np.mean([greedy_return(env, q) for _ in range(n_episodes)]))


def train(env, num_episodes, epsilon, alpha, gamma, eval_every=100, eval_episodes=50):
    q = np.zeros((env.observation_space.n, env.action_space.n))
    xs, ys = [], []
    for episode in range(num_episodes):
        q = q_learning_episode(env, q, epsilon, alpha, gamma)
        if episode % eval_every == 0 or episode == num_episodes - 1:
            xs.append(episode)
            ys.append(evaluate(env, q, eval_episodes))
    return q, np.asarray(xs), np.asarray(ys)


def main():
    np.random.seed(SEED)
    env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    q, xs, ys = train(env, EPISODES, EPSILON, ALPHA, GAMMA, EVAL_EVERY)

    mode = "slippery" if SLIPPERY else "deterministic"
    print(f"FrozenLake ({mode}) — greedy success rate: {evaluate(env, q, 200):.3f}")

    if SHOW_PLOT:
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys)
        plt.xlabel("Training episode")
        plt.ylabel("Greedy success rate")
        plt.title(f"Q-learning on FrozenLake ({mode})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
