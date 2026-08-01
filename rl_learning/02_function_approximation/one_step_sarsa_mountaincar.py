"""One-step SARSA with tile coding on MountainCar-v0.

MountainCar's state is two continuous numbers, so a table has nowhere to put it. Tile coding
lays several coarse grids over the state space, each shifted a little, and a state lights up
one tile per grid. Q is then just a sum of the weights on the active tiles:

    Q(s,a) = sum of weights[a, active_tiles(s)]

which makes the SARSA update the same one as the tabular version, spread over a handful of
weights instead of a single cell.

MountainCar's 200-step limit counts as the end of the episode here, which is why an untrained
agent sits at exactly -200: one point of cost per step, goal never reached.

    python rl_learning/02_function_approximation/one_step_sarsa_mountaincar.py
"""

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

SEED = 0
EPISODES = 5000
N_BINS = 10        # cells per axis in one grid
N_TILINGS = 10     # number of shifted grids
EPSILON = 0.1
ALPHA = 0.5        # divided by N_TILINGS in the update
GAMMA = 1.0
RECORD_VIDEO = False
SHOW_PLOT = True


class TileCoder:
    """Maps a continuous state to one active feature index per tiling.

    Each tiling is offset by a fraction of a cell width. Shifting by whole cells would
    just reproduce the same grid, which is the easy way to get this silently wrong.
    """

    def __init__(self, env, n_bins=N_BINS, n_tilings=N_TILINGS):
        low, high = env.observation_space.low, env.observation_space.high
        cell_widths = (high - low) / n_bins

        self.n_bins = n_bins
        self.n_tilings = n_tilings
        self.n_features = n_tilings * n_bins * n_bins
        self.grids = []

        for i in range(n_tilings):
            # (1, 3) offsets: moving both axes by the same step lines the grids up.
            offsets = np.array([i, (3 * i) % n_tilings]) / n_tilings * cell_widths
            # n_bins cells need n_bins + 1 edges, and digitize wants the inner ones.
            self.grids.append([
                np.linspace(low[d], high[d], n_bins + 1)[1:-1] + offsets[d]
                for d in range(len(low))
            ])

    def __call__(self, state):
        cells = self.n_bins * self.n_bins
        return np.array([
            tiling * cells
            + int(np.digitize(state[0], position_edges))
            + int(np.digitize(state[1], velocity_edges)) * self.n_bins
            for tiling, (position_edges, velocity_edges) in enumerate(self.grids)
        ])


def q_values(weights, tiles):
    """Q(s, ·) for every action at once: sum the active weights per row."""
    return weights[:, tiles].sum(axis=1)


def epsilon_greedy(q, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return int(np.random.randint(n_actions))
    best = np.flatnonzero(q == q.max())   # break ties randomly while Q is still flat
    return int(np.random.choice(best))


def sarsa_episode(env, weights, coder, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA):
    """One episode, updating weights in place. Returns the episode return."""
    step_size = alpha / coder.n_tilings   # n_tilings weights move on every update
    n_actions = env.action_space.n

    state, _ = env.reset()
    tiles = coder(state)
    action = epsilon_greedy(q_values(weights, tiles), epsilon, n_actions)
    total, done = 0.0, False

    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += float(reward)

        next_tiles = coder(next_state)
        next_q = q_values(weights, next_tiles)
        next_action = epsilon_greedy(next_q, epsilon, n_actions)

        target = reward if done else reward + gamma * next_q[next_action]
        weights[action, tiles] += step_size * (target - weights[action, tiles].sum())

        tiles, action = next_tiles, next_action

    return total


def greedy_action(weights, coder, state):
    return int(np.argmax(q_values(weights, coder(state))))


def greedy_return(env, weights, coder, n_episodes=20):
    total = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            state, reward, terminated, truncated, _ = env.step(
                greedy_action(weights, coder, state)
            )
            done = terminated or truncated
            total += float(reward)
    return total / n_episodes


def moving_average(values, window=100):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def record_video(weights, coder, folder="videos", name_prefix="mountaincar-sarsa"):
    from gymnasium.wrappers import RecordVideo

    Path(folder).mkdir(parents=True, exist_ok=True)
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=folder, name_prefix=name_prefix,
                      episode_trigger=lambda ep: True)

    state, _ = env.reset()
    done, total = False, 0.0
    while not done:
        state, reward, terminated, truncated, _ = env.step(
            greedy_action(weights, coder, state)
        )
        done = terminated or truncated
        total += float(reward)
    env.close()
    return total


def main():
    np.random.seed(SEED)
    env = gym.make("MountainCar-v0")
    env.reset(seed=SEED)

    coder = TileCoder(env)
    weights = np.zeros((env.action_space.n, coder.n_features))

    returns = []
    for _ in tqdm(range(EPISODES), desc="1-step SARSA MountainCar"):
        returns.append(sarsa_episode(env, weights, coder))

    print(f"Greedy eval: {greedy_return(env, weights, coder):.1f} (untrained is -200)")

    if RECORD_VIDEO:
        print(f"Video return: {record_video(weights, coder):.0f} (saved under ./videos/)")

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
