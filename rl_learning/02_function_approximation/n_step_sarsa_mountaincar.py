"""n-step SARSA with tile coding on MountainCar-v0.

Same tiles and features as one_step_sarsa_mountaincar.py, but the target
looks n steps ahead (Sutton & Barto, tau = t - n + 1). Read the one-step
version first; this is the harder follow-up.

    python rl_learning/02_function_approximation/n_step_sarsa_mountaincar.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Reuse the tile-coding helpers from the one-step version.
from one_step_sarsa_mountaincar import (
    active_tiles,
    choose_action,
    generate_tilings,
    initialize_weights,
    moving_average,
    q_value,
)

# ── hyperparameters ──────────────────────────────────────
SEED = 0
EPISODES = 3000
N_STEPS = 4
N_BINS = 10
N_TILINGS = 10
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 1.0
SHOW_PLOT = True


def run_n_step_sarsa_episode(env, weights, tilings, n_bins, epsilon, alpha, gamma,
                             n_tilings, n_steps):
    state, _ = env.reset()
    action = choose_action(state, weights, tilings, n_bins, epsilon, env)

    # rewards[0] is a dummy so that rewards[t] means R_t (textbook indexing).
    states, actions, rewards = [state], [action], [0.0]
    t, T = 0, np.inf
    total_reward = 0.0
    step_size = alpha / n_tilings

    while True:
        if t < T:
            next_state, reward, terminated, truncated, _ = env.step(actions[t])
            done = terminated or truncated
            states.append(next_state)
            rewards.append(float(reward))
            total_reward += float(reward)
            if done:
                T = t + 1
            else:
                actions.append(choose_action(next_state, weights, tilings, n_bins, epsilon, env))

        tau = t - n_steps + 1   # the time step we can finally update
        if tau >= 0:
            # Discounted sum of the next n rewards ...
            G = 0.0
            for i in range(tau + 1, int(min(tau + n_steps, T)) + 1):
                G += (gamma ** (i - tau - 1)) * rewards[i]
            # ... plus a bootstrap, unless the episode already ended.
            if tau + n_steps < T:
                s, a = states[tau + n_steps], actions[tau + n_steps]
                G += (gamma ** n_steps) * q_value(s, a, weights, tilings, n_bins)

            s_tau, a_tau = states[tau], actions[tau]
            tiles = active_tiles(s_tau, tilings, n_bins)
            weights[a_tau, tiles] += step_size * (G - q_value(s_tau, a_tau, weights, tilings, n_bins))

        if tau == T - 1:
            break
        t += 1

    return weights, total_reward


def main():
    np.random.seed(SEED)
    env = gym.make("MountainCar-v0")
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    tilings = generate_tilings(env, N_BINS, N_TILINGS)
    weights = initialize_weights(env, N_TILINGS, N_BINS)

    returns = []
    for episode in range(EPISODES):
        weights, total_reward = run_n_step_sarsa_episode(
            env, weights, tilings, N_BINS, EPSILON, ALPHA, GAMMA, N_TILINGS, N_STEPS
        )
        returns.append(total_reward)
        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1:5d} | avg100 = {np.mean(returns[-100:]):7.1f} | n={N_STEPS}")

    greedy = [run_n_step_sarsa_episode(env, weights, tilings, N_BINS, 0.0, 0.0, GAMMA,
                                       N_TILINGS, N_STEPS)[1] for _ in range(50)]
    print(f"Greedy eval mean (n={N_STEPS}): {np.mean(greedy):.1f}")

    if SHOW_PLOT:
        plt.figure(figsize=(8, 4))
        plt.plot(moving_average(np.asarray(returns, dtype=float)))
        plt.xlabel("Episode")
        plt.ylabel("Moving average return (100)")
        plt.title(f"n-step SARSA (n={N_STEPS}) on MountainCar")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
