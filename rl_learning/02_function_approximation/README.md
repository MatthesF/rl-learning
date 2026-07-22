# 02 · Function approximation (tile coding)

MountainCar's state is two continuous numbers, `(position, velocity)`. A lookup table
would need infinitely many cells, so we need features instead. Tile coding is the classic,
intuitive answer: lay several coarse grids over the state space and let a state light up
one tile in each grid.

## The idea

- One **tiling** is a grid over position x velocity. A state falls into exactly one cell.
- Several tilings, each shifted by a small offset, mean a state activates several tiles at
  once. Together they give a surprisingly smooth approximation from coarse pieces.
- Q becomes linear: `Q(s, a) = sum of the weights on the tiles that are active`.

Learning is then the same SARSA idea as before, applied to the weights on the active tiles.

## Files

| File | Start here? | What it shows |
|------|-------------|---------------|
| `one_step_sarsa_mountaincar.py` | **Yes** | Tile coding + classic 1-step SARSA |
| `n_step_sarsa_mountaincar.py` | later | Same features, target that looks n steps ahead |

Read the one-step file first. The n-step version reuses its tile-coding helpers and only
changes how the target is computed.

## Key concepts

- **Active tiles**: a state maps to `n_tilings` feature indices (one per grid).
- **Linear Q**: sum the weights on those indices for the chosen action.
- **Step size**: use `alpha / n_tilings`, because every update moves several weights at once.
- **n-step return**: instead of bootstrapping after one step, sum `n` discounted rewards
  and then bootstrap with `gamma^n * Q(s_{tau+n}, a_{tau+n})`. Updates lag by `n` steps
  (`tau = t - n + 1`). With `n = 1` it collapses back to ordinary SARSA.

## Bugs I actually hit

1. **Wrong number of bin edges.** `linspace(low, high, n_bins)` gives `n_bins` edges, not
   `n_bins` cells — off by one until I used `n_bins + 1` edges.
2. **Offsets too large.** Shifting tilings by whole bins instead of a fraction of a bin
   width just produced the same grid over and over.
3. **A broken n-step loop.** My first attempt updated the *current* state instead of the
   delayed `(S_tau, A_tau)`, forgot to discount the reward sum, and never flushed the last
   few updates after the episode ended.

## What "good" looks like

- Training returns start pinned at **-200** (MountainCar truncates at 200 steps).
- Once it learns to swing back and forth, the moving average climbs toward **-150 to -110**.
- Greedy evaluation should clearly beat random.

## Run

```bash
# Start here
python rl_learning/02_function_approximation/one_step_sarsa_mountaincar.py

# After that works
python rl_learning/02_function_approximation/n_step_sarsa_mountaincar.py
```

Set `RECORD_VIDEO = True` in the one-step script to save a greedy rollout under `videos/`,
and change `N_STEPS` in the n-step script to experiment.
