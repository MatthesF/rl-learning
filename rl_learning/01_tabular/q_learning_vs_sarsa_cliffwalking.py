"""Q-learning vs SARSA on CliffWalking-v0.

Same table, same exploration, one difference in the target:

    Q-learning  r + gamma * max_a' Q(s',a')    the best next action
    SARSA       r + gamma * Q(s',a')           the next action it will really take

So Q-learning learns the optimal route, which runs along the cliff edge, and keeps falling
off it while epsilon is still non-zero. SARSA's target includes its own random slips, so it
learns that the cliff edge is expensive and walks around. Off-policy learns the best policy;
on-policy learns the best policy *given that you explore*.

    python rl_learning/01_tabular/q_learning_vs_sarsa_cliffwalking.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from q_learning_frozenlake import epsilon_greedy

SEED = 0
EPISODES = 500
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 1.0        # undiscounted: every step costs -1 until the goal
EVAL_EVERY = 10
MAX_EPISODE_STEPS = 100
SHOW_PLOT = True


def q_learning_episode(env, q, epsilon, alpha, gamma):
    state, _ = env.reset()
    done = False

    while not done:
        action = epsilon_greedy(q, state, epsilon, env.action_space.n)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        target = reward if terminated else reward + gamma * np.max(q[next_state])
        q[state, action] += alpha * (target - q[state, action])

        state = next_state


def sarsa_episode(env, q, epsilon, alpha, gamma):
    state, _ = env.reset()
    action = epsilon_greedy(q, state, epsilon, env.action_space.n)
    done = False

    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Choose a' before updating: it is part of the target, not just the next move.
        next_action = epsilon_greedy(q, next_state, epsilon, env.action_space.n)
        target = reward if terminated else reward + gamma * q[next_state, next_action]
        q[state, action] += alpha * (target - q[state, action])

        state, action = next_state, next_action


def greedy_return(env, q, n_episodes=50):
    total = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            state, reward, terminated, truncated, _ = env.step(int(np.argmax(q[state])))
            done = terminated or truncated
            total += float(reward)
    return total / n_episodes


def train(env, episode_fn, episodes=EPISODES, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA,
          eval_every=EVAL_EVERY):
    """Re-seeds first, so both algorithms see the same random stream."""
    np.random.seed(SEED)
    env.reset(seed=SEED)

    q = np.zeros((env.observation_space.n, env.action_space.n))
    curve = []

    for episode in range(episodes):
        episode_fn(env, q, epsilon, alpha, gamma)
        if episode % eval_every == 0:
            curve.append((episode, greedy_return(env, q)))

    return q, np.asarray(curve)


def main():
    env = gym.make("CliffWalking-v0", max_episode_steps=MAX_EPISODE_STEPS)

    q_ql, curve_ql = train(env, q_learning_episode)
    q_sarsa, curve_sarsa = train(env, sarsa_episode)

    print(f"Greedy return — Q-learning: {greedy_return(env, q_ql, 200):.1f}")
    print(f"Greedy return — SARSA:      {greedy_return(env, q_sarsa, 200):.1f}")

    if SHOW_PLOT:
        plt.figure(figsize=(8, 4))
        plt.plot(curve_ql[:, 0], curve_ql[:, 1], label="Q-learning")
        plt.plot(curve_sarsa[:, 0], curve_sarsa[:, 1], label="SARSA")
        plt.xlabel("Episode")
        plt.ylabel("Greedy return")
        plt.title("CliffWalking: off-policy vs on-policy")
        plt.ylim(-100, 0)   # early policies wander and would flatten everything else
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
