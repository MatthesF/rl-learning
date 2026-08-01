# rl-learning

Classic reinforcement learning from scratch, one algorithm per file. No framework, no config
system, no abstractions to trace through — every script is self-contained, runs with
`python <file>`, and is short enough to read in one sitting.

Written while learning: first in a notebook, then cleaned up into these files.

## The path

```text
01_tabular/
  q_learning_frozenlake.py              a Q-table you can print
  q_learning_vs_sarsa_cliffwalking.py   off-policy vs on-policy, same plot

02_function_approximation/
  one_step_sarsa_mountaincar.py         continuous state -> tile features -> linear Q
  n_step_sarsa_mountaincar.py           look n steps ahead before bootstrapping

03_deep_rl/
  dqn_cartpole.py                       Q is a network: replay buffer + target network
  double_dqn_cartpole.py                one line less overestimation

04_policy_gradients/
  reinforce_cartpole.py                 learn the policy directly, no Q at all
  reinforce_baseline_cartpole.py         subtract V(s) to cut the variance
  actor_critic_cartpole.py              bootstrap instead of waiting for the return
  gae_cartpole.py                       one lambda between those two extremes
```

Later files import from earlier ones in the same folder, so the diff between two algorithms
is the algorithm.

## Everything here is one of two updates

**Value methods** nudge an estimate towards a target:

```text
Q(s,a) += alpha * (target - Q(s,a))
```

Only `target` changes:

| | `target` |
|---|---|
| Q-learning | `r + gamma * max_a' Q(s',a')` |
| SARSA | `r + gamma * Q(s',a')`, `a'` = the action actually taken next |
| n-step SARSA | `r_1 + ... + gamma^(n-1) r_n + gamma^n Q(s_n,a_n)` |
| DQN | Q-learning's target, but `Q` is a network and `s'` goes through a frozen copy |
| Double DQN | `r + gamma * Q_target(s', argmax_a' Q_online(s',a'))` |

**Policy methods** push up the log-probability of actions that went well:

```text
loss = -sum_t  gamma^t * weight_t * log pi(a_t | s_t)
```

Only `weight` changes:

| | `weight_t` |
|---|---|
| REINFORCE | `G_t`, the return that actually followed |
| + baseline | `G_t - V(s_t)` |
| Actor-critic | `r + gamma * V(s') - V(s)`, one step only |
| GAE | those TD errors, exponentially weighted by `lambda` |

Two details show up in every single file:

- **Terminal states have no future.** On `terminated` the target is just `r`. A `truncated`
  time limit is not a terminal state, so it still bootstraps from `s'`.
- **Evaluate greedily.** The training curve is noisy because exploration is on. Judge a
  policy with `epsilon = 0` (or `argmax` over the policy).

## What "good" looks like

| Script | Rough target |
|---|---|
| Q-learning, FrozenLake (deterministic) | success rate ~1.0 |
| Q-learning, FrozenLake (slippery) | ~0.7-0.85 |
| Q-learning vs SARSA, CliffWalking | ~-13 along the cliff vs ~-17 the safe way |
| Tile-coding SARSA, MountainCar | -200 climbing to ~-150 or better |
| DQN / Double DQN, CartPole | greedy eval near the 500 cap |
| REINFORCE, CartPole | ~300-400, noticeably noisier than DQN |
| REINFORCE + baseline | same level, smoother |
| Actor-critic | reaches the cap on a good run, collapses to ~9 on a bad one |
| GAE, `lambda=0.95` | into the 300s; greedy eval anywhere from ~200 to the cap |

Numbers move a lot with the seed, and the policy-gradient scripts move most.

## Run

```bash
pip install gymnasium numpy matplotlib torch tqdm
python rl_learning/01_tabular/q_learning_frozenlake.py
```

Every script takes no arguments. Change behaviour by editing the constants at the top —
`SHOW_PLOT`, `RECORD_VIDEO`, `EPISODES`, and the algorithm's own hyperparameters.
