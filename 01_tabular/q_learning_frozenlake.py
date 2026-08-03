"""Tabular Q-learning on FrozenLake-v1.

One number per (state, action) in a 16x4 array, nudged towards the best thing we
currently believe about the next state:

    Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

Set SLIPPERY = True for the stochastic ice, which is a much harder problem.

    python 01_tabular/q_learning_frozenlake.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

SEED = 0
SLIPPERY = False
EPISODES = 5000
EPSILON = 0.5      # fixed, not decayed: FrozenLake needs a lot of exploration
ALPHA = 0.1
GAMMA = 0.99
EVAL_EVERY = 100
SHOW_PLOT = True


def epsilon_greedy(q, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return int(np.random.randint(n_actions))
    return int(np.argmax(q[state]))


def q_learning_episode(env, q, epsilon, alpha, gamma):
    """Play one episode, updating q in place."""
    state, _ = env.reset()
    done = False

    while not done:
        action = epsilon_greedy(q, state, epsilon, env.action_space.n)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Nothing follows a terminal state, so its target is just the reward.
        target = reward if terminated else reward + gamma * np.max(q[next_state])
        q[state, action] += alpha * (target - q[state, action])

        state = next_state


def evaluate(env, q, n_episodes=100):
    """Return with epsilon = 0. FrozenLake pays 1 for the goal, so this is a success rate."""
    total = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            state, reward, terminated, truncated, _ = env.step(int(np.argmax(q[state])))
            done = terminated or truncated
            total += float(reward)
    return total / n_episodes


def train(env, episodes=EPISODES, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA,
          eval_every=EVAL_EVERY):
    q = np.zeros((env.observation_space.n, env.action_space.n))
    curve = []

    for episode in range(episodes):
        q_learning_episode(env, q, epsilon, alpha, gamma)
        if episode % eval_every == 0:
            curve.append((episode, evaluate(env, q, 50)))

    return q, np.asarray(curve)


def main():
    np.random.seed(SEED)
    env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY)
    env.reset(seed=SEED)

    q, curve = train(env)

    mode = "slippery" if SLIPPERY else "deterministic"
    print(f"FrozenLake ({mode}) greedy success rate: {evaluate(env, q, 200):.3f}")

    if SHOW_PLOT:
        plt.figure(figsize=(8, 4))
        plt.plot(curve[:, 0], curve[:, 1])
        plt.xlabel("Episode")
        plt.ylabel("Greedy success rate")
        plt.title(f"Q-learning on FrozenLake ({mode})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
