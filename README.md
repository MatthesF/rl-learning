# rl-learning

Classic RL from scratch, one algorithm per file. No framework, no base classes.
Every script runs with `python <file>` and is short enough to read in one sitting.

```text
01_tabular/
  q_learning_frozenlake.py              a Q-table you can print
  q_learning_vs_sarsa_cliffwalking.py   off-policy vs on-policy, same plot

02_function_approximation/
  one_step_sarsa_mountaincar.py         continuous state → tile features → linear Q
  n_step_sarsa_mountaincar.py           look n steps ahead before bootstrapping

03_deep_rl/
  dqn_cartpole.py                       Q is a network: replay buffer + target network
  double_dqn_cartpole.py                one line less overestimation

04_policy_gradients/
  reinforce_cartpole.py                 learn the policy directly, no Q
  reinforce_baseline_cartpole.py        subtract V(s) to cut the variance
  actor_critic_cartpole.py              bootstrap instead of waiting for the return
  gae_cartpole.py                       one lambda between those two extremes
  ppo_cartpole.py                       reuse each rollout, clipped so it stays safe
  ppo_pendulum.py                       same PPO, continuous action
```

Later files import from earlier ones in the same folder, so the diff between two
algorithms is the algorithm. Run from the folder that holds the script:

```bash
pip install -r requirements.txt
cd 01_tabular && python q_learning_frozenlake.py
cd ../04_policy_gradients && python ppo_cartpole.py
```

## Two updates

**Value methods** nudge an estimate towards a target:

```text
Q(s,a) += alpha * (target - Q(s,a))
```

| | `target` |
|---|---|
| Q-learning | `r + gamma * max_a' Q(s',a')` |
| SARSA | `r + gamma * Q(s',a')` — `a'` actually taken next |
| n-step SARSA | `r_1 + … + gamma^n Q(s_n,a_n)` |
| DQN | Q-learning, with `Q` a network and `s'` through a frozen copy |
| Double DQN | `r + gamma * Q_target(s', argmax Q_online(s',·))` |

**Policy methods** push up log-probability of actions that went well:

```text
loss = -sum_t  gamma^t * weight_t * log pi(a_t | s_t)
```

| | `weight_t` |
|---|---|
| REINFORCE | `G_t` |
| + baseline | `G_t - V(s_t)` |
| Actor-critic | `r + gamma V(s') - V(s)` |
| GAE | those TD errors, exponentially weighted by `lambda` |

PPO keeps GAE's `A_t` but swaps `log pi` for the clipped ratio
`pi_new / pi_old`. Once an action has become much likelier than when the data
was collected, the clipped branch is flat and that sample stops pushing — so
the same rollout can be trained on ten times.

`ppo_pendulum.py` is the same clip with a `Normal` instead of a `Categorical`:
the network outputs a mean, one free parameter holds the spread, and
`log pi(a|s)` is a density.

Two rules everywhere:

- **Terminal states have no future.** On `terminated` the target is just `r`.
  A `truncated` time limit still bootstraps from `s'`.
- **Evaluate greedily.** Judge a policy with `epsilon = 0`, `argmax`, or the
  Gaussian mean — not the noisy training curve.

## What "good" looks like

| Script | Rough target |
|---|---|
| Q-learning, FrozenLake (deterministic) | success ~1.0 |
| Q-learning, FrozenLake (slippery) | ~0.7–0.85 |
| Q-learning vs SARSA, CliffWalking | ~−13 along the cliff vs ~−17 the safe way |
| Tile-coding SARSA, MountainCar | −200 → ~−150 or better |
| DQN / Double DQN, CartPole | greedy near 500 |
| REINFORCE, CartPole | ~300–400, noisier than DQN |
| REINFORCE + baseline | same level, smoother |
| Actor-critic | hits 500 on a good run, collapses to ~9 on a bad one |
| GAE, `lambda=0.95` | into the 300s |
| PPO, CartPole | ~400 within 150k steps, greedy at the cap |
| PPO, Pendulum | −1200 → about −180 in 300k steps; below −300 is a swing-up |

No script takes arguments. Edit the constants at the top (`SHOW_PLOT`,
`RECORD_VIDEO`, `EPISODES`, learning rates).
