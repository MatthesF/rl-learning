# 01 · Tabular methods

The starting point: discrete states, discrete actions, and a simple table holding one
value `Q(state, action)` for every combination. No approximation, so you can literally
print the table and watch it learn.

## The idea

An agent estimates "how good is taking action `a` in state `s`" and nudges that estimate
toward what it observes. Over many episodes the good actions float to the top.

Two ways to do the update, and the difference matters:

- **Q-learning (off-policy)** bootstraps with the *best* next action, regardless of what
  the agent actually does next.
- **SARSA (on-policy)** bootstraps with the action it *actually takes* next, so it
  accounts for its own exploration.

## Files

| File | Environment | What it shows |
|------|-------------|---------------|
| `q_learning_frozenlake.py` | FrozenLake-v1 | A first working Q-table |
| `q_learning_vs_sarsa_cliffwalking.py` | CliffWalking-v0 | Off-policy vs on-policy, side by side |

## Key concepts

- **Epsilon-greedy**: explore randomly a fraction of the time, otherwise take `argmax Q`.
- **TD target**: `r + gamma * (value of the next state)`. On a terminal step there is no
  future, so the target is just `r`.
- **The update**: `Q(s,a) += alpha * (target - Q(s,a))` — a small step toward the target.

## Bugs I actually hit

1. **Table stuck at all zeros.** With a zero table and low epsilon, `argmax` always picked
   action 0 and the agent almost never reached the goal. Fix: explore more early.
2. **Breaking before the update.** I broke out of the loop on the terminal step *before*
   updating Q, so the one transition that carried the reward never trained anything.
3. **A plot squashed to nothing.** One disastrous early CliffWalking evaluation dragged
   the y-axis down to -10,000. Capping the axis made the real difference visible.

## What "good" looks like

- FrozenLake, deterministic: greedy success rate around **1.0**.
- FrozenLake, slippery: around **0.7-0.85** after enough episodes.
- CliffWalking: Q-learning often near **-13** (hugs the cliff), SARSA often safer near
  **-17** — the exact gap depends on epsilon.

## Run

```bash
python rl_learning/01_tabular/q_learning_frozenlake.py
python rl_learning/01_tabular/q_learning_vs_sarsa_cliffwalking.py
```

For slippery FrozenLake, set `SLIPPERY = True` at the top of the script.
