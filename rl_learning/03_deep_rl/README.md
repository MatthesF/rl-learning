# 03 · Deep RL (DQN and Double DQN)

Replace the hand-made features with a neural network that maps a state to Q-values for
every action. That one change unlocks harder problems but also introduces instability,
which is what most of the tricks here exist to tame.

## The idea

A network `Q(s)` outputs one value per action. We train it to satisfy the same TD target
as before, `r + gamma * max Q(s')`, using a batch of past experience. Two ingredients keep
it from diverging:

- **Replay buffer** — store transitions and sample random minibatches, so updates aren't
  correlated in time.
- **Target network** — a frozen copy used to compute the target, refreshed every so often,
  so we aren't chasing a target that moves on every gradient step.

## Files

| File | What it shows |
|------|---------------|
| `dqn_cartpole.py` | Full DQN: replay buffer, target net, warmup, Huber loss |
| `double_dqn_cartpole.py` | One-line change to the target that reduces overestimation |

`double_dqn_cartpole.py` imports the training loop from `dqn_cartpole.py` and only swaps in
a different target function, so the two stay honestly comparable.

## Key concepts

- **`gather`**: from the network's per-action outputs, pick `Q(s, a)` for the action that
  was actually taken in each transition.
- **Bootstrap mask**: mask the bootstrap only on `terminated`, not on `truncated`. A
  time-limit cutoff is not a real terminal state, so it should still bootstrap.
- **Step-based epsilon decay**: decay over environment steps, not episodes — early episodes
  are short and would otherwise burn through epsilon far too fast.
- **Warmup**: fill the buffer with some random experience before the first gradient step.
- **Huber loss**: less sensitive to the occasional huge TD error than plain MSE.
- **Double DQN**: the online net *selects* the next action, the target net *evaluates* it.
  Splitting selection from evaluation removes the built-in optimism of a single `max`.

## Bugs I actually hit

1. **`'module' object is not callable`.** I wrote `import tqdm` instead of
   `from tqdm import tqdm`.
2. **Sampling from an empty buffer.** Training started before there were `batch_size`
   transitions. Fix: wait for `len(buffer) >= max(warmup_steps, batch_size)`.
3. **A slow buffer.** `list.pop(0)` is O(n); `collections.deque(maxlen=...)` evicts in O(1).
4. **Performance collapse.** The agent hit 500 and then fell apart — a target-sync interval
   that was too large plus vanilla DQN's overestimation. Smaller sync intervals and Double
   DQN both help.

## What "good" looks like

- Training return climbs toward the **500** cap, usually with a wobble on the way.
- Greedy evaluation (epsilon = 0) should sit comfortably above **200**, often near 500.
- Give it enough episodes — five is only ever a smoke test, not a trained agent.

## Run

```bash
python rl_learning/03_deep_rl/dqn_cartpole.py
python rl_learning/03_deep_rl/double_dqn_cartpole.py
```

Set `RECORD_VIDEO = True` at the top to save a greedy rollout as an `.mp4` under `videos/`.
