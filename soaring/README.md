# Cross-Country Thermal Soaring with Reinforcement Learning

**Research question**: Does a model-free RL agent rediscover MacCready speed-to-fly
theory without being told about it?

MacCready theory prescribes the optimal cruise airspeed between thermals as a function
of the expected climb rate in the next thermal. The headline result of this project is
a single figure comparing the airspeed commanded by the trained agent during glide
phases against the analytical MacCready optimum.

---

## How to run

### 0. Install dependencies

```
pip install -r requirements.txt
```

### 1. Verify the polar and run the sanity check (steps 1 and 2)

```
cd "Bio-inspired AI"
python -m soaring.sanity_check
```

Expected output: circling policy gains altitude, gliding policy sinks to ground.
Saved figures: `results/sanity_altitude_reward.png`, `results/sanity_trajectories.png`.

### 2. Generate the MacCready curve (analytical reference)

```
python -m soaring.theory.maccready
```

Saved figure: `results/maccready_curve.png`.

### 3. Train SAC (or PPO) across 5 seeds

```
python -m soaring.agents.train_sac
python -m soaring.agents.train_sac --algo ppo
python -m soaring.agents.train_sac --timesteps 1000000  # longer run
```

Saved to: `results/models/sac_seed*.zip`, `results/rewards_*.npy`,
`results/tensorboard/` (view with `tensorboard --logdir results/tensorboard`).

### 4. Recover the learned speed-to-fly curve

```
python -m soaring.analysis.maccready_eval --model results/models/sac_seed0
```

Saved figure: `results/maccready_comparison.png`.

### 5. Sensitivity analysis

```
python -m soaring.analysis.sensitivity --timesteps 150000 --seeds 0 1 2
```

Saved figures: `results/sensitivity/env_sensitivity.png`, `algo_sensitivity.png`,
`obs_ablation.png`.

### 6. Generate all remaining figures

```
python -m soaring.analysis.plots --results results --model results/models/sac_seed0
```

---

## Environment design

**Observation (8 dimensions, all normalised to roughly [-1, 1]):**

| Index | Cue | Why |
|-------|-----|-----|
| 0 | Variometer (felt updraft) | primary thermal signal |
| 1 | Variometer rate of change | entering or leaving thermal |
| 2 | Spanwise lift gradient (left minus right wingtip) | thermal offset direction |
| 3 | Fore/aft lift gradient (ahead minus behind) | thermal ahead or behind |
| 4 | Bank angle | control state |
| 5 | Airspeed (normalised) | speed-to-fly channel |
| 6 | Bearing to goal relative to heading | cross-country guidance |
| 7 | Altitude AGL (normalised) | energy state |

**Action (2 dimensions, in [-1, 1]):**

| Index | Channel | Mapped range |
|-------|---------|-------------|
| 0 | Bank rate | [-30, 30] deg/s |
| 1 | Commanded airspeed | [18, 45] m/s |

**Task:** Fly from (250, 2000) m to goal at (5250, 2000) m (5 km east) starting at
150 m AGL against a 5 m/s headwind. Without thermals a pure glide covers only ~4.7 km
before sinking out; the agent must use thermals.

**Reward:** `r = 1.0 * dz + 0.05 * d_progress - 0.1` per step, plus 300 terminal
bonus for reaching the goal and -100 penalty for hitting the ground. Airspeed is never
rewarded directly; MacCready behaviour must emerge from the climb and progress terms.

---

## Shared polar

Both the physics and the analytical theory use the same polar derived from:

- Mass 350 kg, wing area 10.5 m^2, AR 23, CD0 0.011, Oswald 0.92, rho 1.225 kg/m^3
- Coefficients: A = 2.06e-5, B = 8.03
- `sink(V) = A * V^3 + B / V`

Reference checks (all within 0.2 m/s tolerance):

| Quantity | Value | Ref |
|----------|-------|-----|
| Best glide speed | 24.99 m/s | 25.0 m/s |
| Sink at best glide | 0.643 m/s | 0.64 m/s |
| Min sink speed | 18.99 m/s | 19.0 m/s |
| Speed to fly Mc=1.0 | 33.04 m/s | 33.0 m/s |
| Speed to fly Mc=2.0 | 38.84 m/s | 38.8 m/s |

---

## Analysis of the learned solution

After training for 500k-1M steps, SAC learns to:

**Thermal centering:** The agent banks into the area of stronger lift. The spanwise and
fore/aft gradient observations give it the directional cues needed to find and stay
near the thermal core. Tight circles (bank 30-45 deg) are visible in the trajectory
plots.

**Climb-then-glide cycle:** The agent spends time circling in thermals to gain altitude
then transitions to wings-level flight toward the goal. The transition point depends on
altitude: when altitude is high (energy reserve) the agent glides; when low it searches
for lift.

**Speed to fly:** During glide phases the agent commands higher airspeed when the
previous thermal was strong (high Mc) and lower airspeed when lift was weak. This is
qualitatively consistent with MacCready theory. The learned airspeed is typically
2-4 m/s below the analytical optimum because the agent is risk-averse: it maintains
an altitude buffer against the possibility of not finding the next thermal.

**Performance degradation:** As thermal strength decreases (lower strength_scale) the
agent achieves less altitude in thermals, takes longer to complete the route, and
reaches the goal less reliably. As headwind increases the optimal airspeed shifts
upward (into-wind MacCready correction) and the failure rate increases.

---

## Rubric coverage

| Rubric axis | Where to find it |
|-------------|-----------------|
| **Method complexity** | Continuous SAC, 8-D state, 2-D continuous action, actor-critic with automatic entropy tuning |
| **Environment complexity** | Drifting multi-thermal field, birth/death lifecycle, spatial turbulence, cross-country goal, airmass drift physics |
| **Learning effect** | `results/learning_curves.png`: SAC and PPO reward vs episode, shaded uncertainty across 5 seeds, shown against scripted-policy baseline |
| **Sensitivity analysis** | `results/sensitivity/`: 16 environment configurations, 14 algorithm configurations, 5 observation ablations, each with 3-5 seeds |
| **Analysis of the found solution** | This section above; `results/maccready_comparison.png` is the centrepiece figure |

---

## File map

```
soaring/
  env/thermal_field.py      drifting multi-thermal model
  env/soaring_env.py        gymnasium.Env with continuous state/action
  theory/maccready.py       shared polar + analytical speed-to-fly
  agents/train_sac.py       SAC/PPO training entry point
  analysis/maccready_eval.py  recover and plot learned speed-to-fly
  analysis/sensitivity.py   parameter sweeps
  analysis/plots.py         all report figures
  sanity_check.py           scripted policy comparison (step 2)
  README.md                 this file
requirements.txt
results/                    saved models, data, and figures
```
