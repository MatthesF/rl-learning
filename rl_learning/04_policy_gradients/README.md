# 04 · Policy gradients (REINFORCE)

Everything so far learned a *value* and derived a policy from it (`argmax Q`). REINFORCE
flips that around: the network **is** the policy. It outputs a probability for each action,
and training pushes up the probability of actions taken in episodes that went well.

## The idea

Play a whole episode, see how it went, then adjust. For every step we keep
`log pi(a_t|s_t)` — the log-probability of the action actually taken — and weight it by the
return that followed:

```text
L = -sum_t  gamma^t * G_t * log pi(a_t | s_t)
```

A high `G_t` makes that action more likely next time, a low one makes it less likely.
That is the whole algorithm: no replay buffer, no target network, no bootstrapping.

## Files

| File | Start here? | What it shows |
|------|-------------|---------------|
| `reinforce_cartpole.py` | **Yes** | Vanilla REINFORCE, Monte Carlo, one update per episode |
| `reinforce_baseline_cartpole.py` | after | A learned critic `V(s)` subtracted from the return |

`reinforce_baseline_cartpole.py` imports the policy, the episode runner and the helpers
from the vanilla script and only changes how the weight on each `log_prob` is computed.

## Key concepts

- **Stochastic policy**: the network outputs logits, `softmax` turns them into
  probabilities, and we sample. Exploration is built into the policy — no epsilon to decay.
- **`log_prob` stays on the graph**: it is the only path gradients flow through. Never call
  `.item()` or `.detach()` on it before the loss.
- **Only the chosen action counts**: store `log pi(a_t|s_t)` for the action taken, not the
  probabilities of every action.
- **Monte Carlo returns**: `G_t` is computed backwards over the finished episode, so the
  update has to wait for the whole trajectory.
- **Greedy for evaluation**: sample while training, take `argmax` when measuring or
  recording video.
- **Baseline / advantage**: subtracting `V(s_t)` gives `A_t = G_t - V(s_t)`. Actions are
  now judged on whether they beat expectations, which leaves the expected gradient
  unchanged but cuts its variance a lot.
- **Keep the two gradient paths apart**: the advantage uses `values.detach()`, so the
  policy loss never trains the critic. The critic learns only from its own regression onto
  `G_t`, with its own optimizer.
- **Compute both losses before either update**: if the critic steps first, the policy ends
  up using a baseline that has already seen this episode's answer.

## How it differs from DQN

| | DQN | REINFORCE |
|---|---|---|
| Learns | `Q(s,a)` | `pi(a|s)` directly |
| Data | replay buffer, off-policy | one episode, then discarded, on-policy |
| Target | TD bootstrap | full return `G_t` |
| Update | every step after warmup | once per episode |
| Exploration | epsilon-greedy | the policy's own randomness |

## Bugs I actually hit

1. **`state` undefined.** `env.reset()` returns `(state, info)` and I forgot to unpack it.
2. **Infinite episode.** I never set `done = terminated or truncated`, so the loop never ended.
3. **Softmax over the wrong axis.** A single CartPole state gives logits of shape `(2,)` with
   no batch dimension, so `dim=1` fails — it has to be `dim=-1`.
4. **`torch.sum()` on a Python list.** The loss terms are separate tensors and need
   `torch.stack(terms).sum()` to become one scalar.
5. **Plotting the wrong number.** I logged `sum(returns)` (the sum of all `G_t`) instead of
   the actual episode return, which made the learning curve meaningless.
6. **Missing `gamma^t`.** The textbook objective discounts each term by `gamma^t`. Leaving it
   out still learns, but it is not the same objective.
7. **Storing the wrong state for the baseline.** I appended the state *after* `env.step()`,
   so `states[t]` held `s_{t+1}` and every `V(s_t)` was off by one step.
8. **A baseline that changed nothing.** I computed the advantage but still fed raw `G_t`
   into the policy loss, so the critic trained while the policy ignored it.
9. **Gradients leaking into the critic.** Without `.detach()` on the values, the policy loss
   could lower its own loss by corrupting `V(s)` instead of improving the policy.
10. **One optimizer for two networks.** `Adam(policy.parameters())` never updates the critic;
    each network needs its own optimizer (or one built over both parameter sets).

## What "good" looks like

- The moving average climbs from ~20 toward **300-400**, with single episodes hitting the
  500 cap.
- It is **much noisier than DQN**, and dips are normal: every update comes from one episode,
  so a single unlucky trajectory moves the policy a lot.
- Don't read those dips as catastrophic forgetting. This is gradient variance, not the
  moving-target feedback loop that makes DQN collapse.

With the baseline, the same run is **visibly smoother**: the curve still wobbles, but the
deep collapses largely disappear and the recorded rollouts look far steadier. That is the
whole point of the critic.

From here the next step is PPO, which reuses the advantage idea and adds a clipped
objective plus several epochs over the same batch of rollouts.

## Run

```bash
# Start here
python rl_learning/04_policy_gradients/reinforce_cartpole.py

# After that works
python rl_learning/04_policy_gradients/reinforce_baseline_cartpole.py
```

Set `RECORD_VIDEO = True` in `reinforce_cartpole.py` to save a greedy rollout under
`videos/` (both scripts read that flag).
