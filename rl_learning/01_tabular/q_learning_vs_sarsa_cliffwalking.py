"""Q-learning vs SARSA on CliffWalking-v0.

The classic contrast between off-policy and on-policy control:
- Q-learning tends to hug the cliff (short, risky).
- SARSA tends to take a safer, longer path, because it learns from the
  exploratory actions it actually takes.

    python rl_learning/01_tabular/q_learning_vs_sarsa_cliffwalking.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# ── hyperparameters ──────────────────────────────────────
SEED = 0
EPISODES = 500
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 1.0
EVAL_EVERY = 10
EVAL_EPISODES = 50
MAX_EPISODE_STEPS = 100   # stop bad early policies from wandering forever
SHOW_PLOT = True


def epsilon_greedy(env, state, q, epsilon):
    if np.random.rand() < epsilon:
        return int(env.action_space.sample())
    return int(np.argmax(q[state]))


def q_learning_episode(env, q, epsilon, alpha, gamma):
    """Off-policy: bootstrap with the best next action."""
    state, _ = env.reset()
    while True:
        action = epsilon_greedy(env, state, q, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        target = reward if done else reward + gamma * np.max(q[next_state])
        q[state, action] += alpha * (target - q[state, action])

        state = next_state
        if done:
            break
    return q


def sarsa_episode(env, q, epsilon, alpha, gamma):
    """On-policy: bootstrap with the action we actually take next."""
    state, _ = env.reset()
    action = epsilon_greedy(env, state, q, epsilon)
    while True:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done:
            q[state, action] += alpha * (reward - q[state, action])
            break

        next_action = epsilon_greedy(env, next_state, q, epsilon)
        target = reward + gamma * q[next_state, next_action]
        q[state, action] += alpha * (target - q[state, action])
        state, action = next_state, next_action
    return q


def greedy_return(env, q):
    state, _ = env.reset()
    total = 0.0
    while True:
        state, reward, terminated, truncated, _ = env.step(int(np.argmax(q[state])))
        total += float(reward)
        if terminated or truncated:
            break
    return total


def evaluate(env, q, n_episodes=50):
    return float(np.mean([greedy_return(env, q) for _ in range(n_episodes)]))


def train(env, update_fn, num_episodes, epsilon, alpha, gamma, eval_every, eval_episodes):
    q = np.zeros((env.observation_space.n, env.action_space.n))
    xs, ys = [], []
    for episode in range(num_episodes):
        q = update_fn(env, q, epsilon, alpha, gamma)
        if episode % eval_every == 0 or episode == num_episodes - 1:
            xs.append(episode)
            ys.append(evaluate(env, q, eval_episodes))
    return q, np.asarray(xs), np.asarray(ys)


def seed_everything(env, seed):
    """Reset numpy + env RNGs so two runs with the same SEED match."""
    np.random.seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)


def main():
    env = gym.make("CliffWalking-v0", max_episode_steps=MAX_EPISODE_STEPS)

    # Re-seed before each algorithm so the comparison is fair.
    seed_everything(env, SEED)
    q_ql, x_ql, y_ql = train(env, q_learning_episode, EPISODES, EPSILON, ALPHA, GAMMA,
                             EVAL_EVERY, EVAL_EPISODES)
    seed_everything(env, SEED)
    q_sa, x_sa, y_sa = train(env, sarsa_episode, EPISODES, EPSILON, ALPHA, GAMMA,
                             EVAL_EVERY, EVAL_EPISODES)

    print(f"Final greedy return — Q-learning: {evaluate(env, q_ql, 200):.1f}")
    print(f"Final greedy return — SARSA:      {evaluate(env, q_sa, 200):.1f}")

    if SHOW_PLOT:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_ql, y_ql, label="Q-learning")
        ax.plot(x_sa, y_sa, label="SARSA")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Avg greedy return")
        ax.set_title("CliffWalking: Q-learning vs SARSA")
        ax.set_ylim(-100, 0)   # ignore the huge negative spike from early runs
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
