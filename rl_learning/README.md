# Reinforcement Learning, from scratch

A hands-on walk through classic reinforcement learning, one small algorithm at a time:
tabular methods, then linear function approximation, then deep Q-learning, then policy
gradients.

Each algorithm lives in its own self-contained script you can read top to bottom in a
few minutes, run in one command, and tweak from a small block of hyperparameters at the top.

## How this was built

I wrote the first versions of everything here in a Jupyter notebook while learning,
making the usual beginner mistakes and fixing them one by one. Once each algorithm
actually worked, I sat down with an LLM to refactor the messy notebook cells into these
clean, commented `.py` files, keeping the logic identical but making it readable.

So this folder is the *polished* version of a genuine learning process. The comments
call out the specific bugs I hit, because those were the moments that taught me the most.

## The path

1. **Tabular** — a plain table of `Q(state, action)` values. No neural nets, no
   approximation. The place to understand TD updates, exploration, and on-policy vs
   off-policy control.
2. **Function approximation** — the state becomes continuous, so a table no longer fits.
   Tile coding turns it into sparse features and Q becomes a linear function of weights.
3. **Deep RL** — replace the linear model with a neural network (DQN), add a replay
   buffer and a target network, then fix the instabilities with Double DQN.
4. **Policy gradients** — stop learning values and optimise the policy directly. REINFORCE
   samples actions from its own distribution and reinforces whatever worked, then a learned
   baseline `V(s)` cuts the variance. Actor-critic replaces the Monte Carlo return with a
   one-step TD error so the update can happen every timestep.

```text
rl_learning/
  01_tabular/
    q_learning_frozenlake.py              first Q-table
    q_learning_vs_sarsa_cliffwalking.py   off-policy vs on-policy
  02_function_approximation/
    one_step_sarsa_mountaincar.py         start here
    n_step_sarsa_mountaincar.py           n-step returns (harder)
  03_deep_rl/
    dqn_cartpole.py                       DQN
    double_dqn_cartpole.py                Double DQN
  04_policy_gradients/
    reinforce_cartpole.py                 start here
    reinforce_baseline_cartpole.py        + learned baseline V(s)
    actor_critic_cartpole.py              TD actor-critic (I = gamma^t)
```


## What "good" looks like

| Algorithm | Environment | Rough target |
|-----------|-------------|--------------|
| Q-learning | FrozenLake (deterministic) | success rate ~1.0 |
| Q-learning | FrozenLake (slippery) | success rate ~0.7-0.85 |
| Q-learning vs SARSA | CliffWalking | ~-13 (risky) vs ~-17 (safer) |
| Tile-coding SARSA | MountainCar | returns climb from -200 toward ~-110 to -150 |
| DQN / Double DQN | CartPole | greedy eval well above 200, often near 500 |
| REINFORCE | CartPole | moving average ~300-400, but visibly noisier than DQN |
| REINFORCE + baseline | CartPole | similar level, noticeably smoother curve |
| One-step actor-critic | CartPole | can climb toward 200+, but online TD updates are slower and more fragile than REINFORCE |

Numbers vary with seed and hyperparameters. Always judge a policy with **greedy
evaluation** (epsilon = 0), not the noisy training curve while exploration is still high.

## Setup

These scripts need `gymnasium`, `numpy`, `matplotlib`, `torch`, and `tqdm`.

Using the project conda environment:

```bash
conda activate rl-sim2real
```

Or a minimal standalone install:

```bash
pip install gymnasium numpy matplotlib torch tqdm
```

## Running

Every script runs with no arguments. Change behaviour by editing the small
`hyperparameters` block at the top of the file.

```bash
python rl_learning/01_tabular/q_learning_frozenlake.py
python rl_learning/01_tabular/q_learning_vs_sarsa_cliffwalking.py
python rl_learning/02_function_approximation/one_step_sarsa_mountaincar.py
python rl_learning/02_function_approximation/n_step_sarsa_mountaincar.py
python rl_learning/03_deep_rl/dqn_cartpole.py
python rl_learning/03_deep_rl/double_dqn_cartpole.py
python rl_learning/04_policy_gradients/reinforce_cartpole.py
python rl_learning/04_policy_gradients/reinforce_baseline_cartpole.py
python rl_learning/04_policy_gradients/actor_critic_cartpole.py
```

Set `RECORD_VIDEO = True` in the MountainCar / CartPole scripts to save a greedy
rollout as an `.mp4` under `videos/`.

## Design choices

- One algorithm per file, readable end to end (CleanRL-style).
- Hyperparameters as named constants at the top, not buried in the code.
- Learning curves use a moving average so the trend is visible through the noise.
- No rendering during training; video is opt-in for demos.
