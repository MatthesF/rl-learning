"""n-step SARSA with tile coding on MountainCar-v0.

Same tiles, but the target waits for n rewards before bootstrapping:

    G = r_1 + … + gamma^(n-1)*r_n + gamma^n * Q(s_n, a_n)

Update for step τ happens at τ+n (Sutton & Barto). n=1 is ordinary SARSA.

    python n_step_sarsa_mountaincar.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from one_step_sarsa_mountaincar import (
    ALPHA,
    EPSILON,
    GAMMA,
    SEED,
    SHOW_PLOT,
    TileCoder,
    epsilon_greedy,
    evaluate,
    moving_average,
    q_values,
)

EPISODES = 3000
N_STEPS = 4


def n_step_sarsa_episode(env, weights, coder, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA,
                         n_steps=N_STEPS):
    """One episode, updating weights in place. Returns the episode return."""
    step_size = alpha / coder.n_tilings
    n_actions = env.action_space.n

    state, _ = env.reset()
    action = epsilon_greedy(q_values(weights, coder(state)), epsilon, n_actions)

    # rewards[0] is a dummy so that rewards[i] is R_i, matching the book's indexing.
    states, actions, rewards = [state], [action], [0.0]
    total = 0.0
    T = np.inf   # the step the episode ends on; unknown until it happens
    t = 0

    while True:
        if t < T:
            next_state, reward, terminated, truncated, _ = env.step(actions[t])
            states.append(next_state)
            rewards.append(float(reward))
            total += float(reward)

            if terminated or truncated:
                T = t + 1
            else:
                actions.append(
                    epsilon_greedy(q_values(weights, coder(next_state)), epsilon, n_actions)
                )

        tau = t - n_steps + 1   # the step whose n rewards are now all known
        if tau >= 0:
            G = sum(
                gamma ** (i - tau - 1) * rewards[i]
                for i in range(tau + 1, int(min(tau + n_steps, T)) + 1)
            )
            # Only bootstrap if the n-step horizon has not run past the end.
            if tau + n_steps < T:
                bootstrap_q = q_values(weights, coder(states[tau + n_steps]))
                G += gamma ** n_steps * bootstrap_q[actions[tau + n_steps]]

            tiles = coder(states[tau])
            weights[actions[tau], tiles] += step_size * (
                G - weights[actions[tau], tiles].sum()
            )

        if tau == T - 1:   # the last step has been updated
            break
        t += 1

    return total


def main():
    np.random.seed(SEED)
    env = gym.make("MountainCar-v0")
    env.reset(seed=SEED)

    coder = TileCoder(env)
    weights = np.zeros((env.action_space.n, coder.n_features))

    returns = []
    for _ in tqdm(range(EPISODES), desc=f"{N_STEPS}-step SARSA MountainCar"):
        returns.append(n_step_sarsa_episode(env, weights, coder))

    print(f"Greedy eval (n={N_STEPS}): {evaluate(env, weights, coder):.1f}")

    if SHOW_PLOT:
        plt.figure(figsize=(8, 4))
        plt.plot(moving_average(np.asarray(returns, dtype=float)))
        plt.xlabel("Episode")
        plt.ylabel("Moving average return (100)")
        plt.title(f"{N_STEPS}-step SARSA on MountainCar")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
