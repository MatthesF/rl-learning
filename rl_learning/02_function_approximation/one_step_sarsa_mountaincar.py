"""One-step SARSA with tile coding on MountainCar-v0.

Start here for function approximation. MountainCar's state is continuous,
so we turn (position, velocity) into a handful of active "tiles" and keep
a linear Q on top of them.

Built in a notebook first, then refined into this script for readability.

    python rl_learning/02_function_approximation/one_step_sarsa_mountaincar.py
"""

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# ── hyperparameters ──────────────────────────────────────
SEED = 0
EPISODES = 5000
N_BINS = 10          # tiles per axis
N_TILINGS = 10       # number of overlapping grids
EPSILON = 0.1
ALPHA = 0.5          # divided by N_TILINGS inside the update
GAMMA = 1.0
RECORD_VIDEO = False
SHOW_PLOT = True


# ── tile coding ──────────────────────────────────────────
def create_tiling(env, n_bins, position_offset, velocity_offset):
    """One grid of bin edges, shifted by an offset."""
    low, high = env.observation_space.low, env.observation_space.high
    position_edges = np.linspace(low[0], high[0], n_bins + 1)[1:-1] + position_offset
    velocity_edges = np.linspace(low[1], high[1], n_bins + 1)[1:-1] + velocity_offset
    return position_edges, velocity_edges


def generate_tilings(env, n_bins, n_tilings):
    """Several grids, each nudged by a small asymmetric (1, 3) offset."""
    low, high = env.observation_space.low, env.observation_space.high
    position_width = (high[0] - low[0]) / n_bins
    velocity_width = (high[1] - low[1]) / n_bins

    tilings = []
    for i in range(n_tilings):
        pos_offset = (i / n_tilings) * position_width
        vel_offset = (((3 * i) % n_tilings) / n_tilings) * velocity_width
        tilings.append(create_tiling(env, n_bins, pos_offset, vel_offset))
    return tilings


def active_tiles(state, tilings, n_bins):
    """One feature index per tiling."""
    indices = []
    per_tiling = n_bins * n_bins
    for t, (pos_edges, vel_edges) in enumerate(tilings):
        pos_bin = int(np.digitize(state[0], pos_edges))
        vel_bin = int(np.digitize(state[1], vel_edges))
        indices.append(t * per_tiling + pos_bin + vel_bin * n_bins)
    return np.asarray(indices, dtype=np.int64)


def initialize_weights(env, n_tilings, n_bins):
    return np.zeros((env.action_space.n, n_tilings * n_bins * n_bins))


def q_value(state, action, weights, tilings, n_bins):
    """Q(s, a) = sum of weights on the active tiles."""
    return float(np.sum(weights[action, active_tiles(state, tilings, n_bins)]))


def choose_action(state, weights, tilings, n_bins, epsilon, env):
    if np.random.random() < epsilon:
        return int(env.action_space.sample())
    q = np.array([q_value(state, a, weights, tilings, n_bins)
                  for a in range(env.action_space.n)])
    best = np.flatnonzero(q == q.max())   # random tie-break when Q is flat
    return int(np.random.choice(best))


# ── SARSA ────────────────────────────────────────────────
def run_sarsa_episode(env, weights, tilings, n_bins, epsilon, alpha, gamma, n_tilings):
    state, _ = env.reset()
    action = choose_action(state, weights, tilings, n_bins, epsilon, env)
    total_reward = 0.0
    step_size = alpha / n_tilings   # several tiles move per update

    while True:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

        tiles = active_tiles(state, tilings, n_bins)
        old_q = q_value(state, action, weights, tilings, n_bins)

        if done:
            weights[action, tiles] += step_size * (reward - old_q)
            break

        next_action = choose_action(next_state, weights, tilings, n_bins, epsilon, env)
        target = reward + gamma * q_value(next_state, next_action, weights, tilings, n_bins)
        weights[action, tiles] += step_size * (target - old_q)
        state, action = next_state, next_action

    return weights, total_reward


# ── helpers ──────────────────────────────────────────────
def moving_average(values, window=100):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def record_video(weights, tilings, n_bins, folder="videos"):
    from gymnasium.wrappers import RecordVideo

    Path(folder).mkdir(parents=True, exist_ok=True)
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=folder, name_prefix="mountaincar-sarsa",
                      episode_trigger=lambda ep: True)

    state, _ = env.reset()
    done, total = False, 0.0
    while not done:
        action = choose_action(state, weights, tilings, n_bins, 0.0, env)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += float(reward)
    env.close()
    return total


def main():
    np.random.seed(SEED)
    env = gym.make("MountainCar-v0")
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    tilings = generate_tilings(env, N_BINS, N_TILINGS)
    weights = initialize_weights(env, N_TILINGS, N_BINS)

    returns = []
    for episode in range(EPISODES):
        weights, total_reward = run_sarsa_episode(
            env, weights, tilings, N_BINS, EPSILON, ALPHA, GAMMA, N_TILINGS
        )
        returns.append(total_reward)
        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1:5d} | avg100 = {np.mean(returns[-100:]):7.1f}")

    greedy = [run_sarsa_episode(env, weights, tilings, N_BINS, 0.0, 0.0, GAMMA, N_TILINGS)[1]
              for _ in range(100)]
    print(f"Greedy eval mean: {np.mean(greedy):.1f}")

    if RECORD_VIDEO:
        print(f"Video return: {record_video(weights, tilings, N_BINS):.0f} (saved under ./videos/)")

    if SHOW_PLOT:
        plt.figure(figsize=(8, 4))
        plt.plot(moving_average(np.asarray(returns, dtype=float)))
        plt.xlabel("Episode")
        plt.ylabel("Moving average return (100)")
        plt.title("One-step tile-coding SARSA on MountainCar")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
