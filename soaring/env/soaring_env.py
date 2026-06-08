"""
Cross-country soaring environment: gymnasium.Env with continuous state/action.

A glider starts at point A with low altitude and must reach point B several
kilometres away. The thermal field provides lift; the agent controls bank rate
and airspeed. Speed-to-fly behaviour (MacCready theory) must emerge from the
climb and progress rewards alone, never from the reward encoding it directly.

Observation (9-D, all clipped to [-1, 1]):
  0  updraft at current position (local variometer)
  1  bearing to nearest thermal relative to heading / pi  (left=-1, right=+1)
  2  distance to nearest thermal, normalised to [-1, 1]   (close=-1, far=+1)
  3  effective peak strength of nearest thermal / W_max
  4  current bank angle
  5  current airspeed (normalised)
  6  bearing to goal relative to heading
  7  glide margin: altitude surplus above min glide-path to goal (normalised)
  8  EMA of recent updraft / W_max  (thermal-richness memory for MacCready speed)

Action (2-D, in [-1, 1]):
  0  bank rate command (scaled to +/- PHI_RATE_MAX rad/s)
  1  airspeed command (scaled to [V_MIN, V_MAX] m/s)
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from soaring.env.thermal_field import ThermalField
from soaring.theory.maccready import POLAR, G


DT           = 0.5          # timestep [s]
V_MIN        = 18.0         # minimum airspeed [m/s]
V_MAX        = 45.0         # maximum airspeed [m/s]
PHI_MAX      = np.deg2rad(50.0)      # maximum bank angle [rad]
PHI_RATE_MAX = np.deg2rad(30.0)      # maximum bank rate [rad/s]

DOMAIN_X     = 15000.0      # domain width east-west [m]
DOMAIN_Y     = 7500.0       # domain height north-south [m]
START_X      = 250.0        # default start x [m]
START_Y      = 3750.0       # default start y [m]  (centred in domain)
GOAL_X       = 14750.0      # goal x [m]  (14500 m course, 250 m east buffer)
GOAL_Y       = 750.0       # goal y [m]
GOAL_RADIUS  = 100.0        # arrival radius [m]
INIT_ALT     = 300.0        # starting altitude AGL [m] (was 150; more margin → less risk-aversion)
ALT_NORM     = 500.0        # altitude normalisation reference [m]
MAX_STEPS    = 9000         # episode time limit (≈ 75 min at DT=0.5 s)


_MIN_NEAR_GOAL_DIST   = 2000.0  # near-thermal start must be ≥ this far from goal [m]

_W_ALT       =  1.00   # per metre of altitude gained
_W_PROGRESS  =  0.08   # per metre closer to goal
_W_STEP      = -0.20   # time pressure → MacCready speed signal
_R_SUCCESS   =  600.0  # goal bonus
_R_CRASH     = -500.0
_R_TIMEOUT   = -200.0

# Progress is suppressed when in significant lift so that circling beats
# flying straight through a thermal on a per-step basis.  Without this,
# flying east at V=25 through a 2 m/s thermal scores +0.875/step vs circling
# at +0.115/step — the agent never learns to thermal.
_LIFT_THRESHOLD = 0.5             # [m/s]  updraft above which progress is suppressed
_BANK_THRESHOLD = np.deg2rad(20)  # [rad]  min bank angle to earn the thermalling bonus

# When the agent is banking in significant lift it earns a bonus proportional
# to updraft strength.  This makes circling strongly preferred over flying
# straight through a thermal (+1.50 vs +0.48 per step in a 2 m/s thermal)
# while the suppression above keeps the routing clean.
_W_THERMAL = 1.0  # bonus weight: W_THERMAL * w_curr * DT, scaled by alt_scale

# Both the altitude reward and the thermalling bonus are scaled down as the
# agent's glide margin grows, decaying linearly to zero at margin = ALT_NORM.
# This provably prevents indefinite thermalling: for any updraft strength there
# exists a finite margin at which flying toward the goal beats circling
# (verified: ~230 m for 2 m/s thermals, <430 m for 7.5 m/s thermals).

# EMA decay for recent-updraft memory (obs channel 8).
# Time constant ≈ 1/alpha steps = 50 steps = 25 s.  After leaving a thermal
# the EMA stays elevated for ~150 steps, giving the agent a stable signal of
# thermal-environment richness to condition its inter-thermal airspeed on.
_W_EMA_ALPHA = 0.02


class SoaringEnv(gym.Env):
    """
    Continuous cross-country soaring with a drifting multi-thermal field.

    Parameters
    ----------
    field_kw        : kwargs forwarded to ThermalField (e.g. n_thermals,
                      strength_scale, wind, turbulence_amp)
    start_near_thermal : if True the glider starts within R/2 of the
                         strongest thermal (used for the sanity check)
    strength_scale_range : (lo, hi) tuple. When set, strength_scale is
                           drawn uniformly from this range at each reset.
                           Enables MacCready generalisation training.
    obs_mask        : boolean array of length 9. Zero entries are replaced
                      by zero in the observation (observation ablation).
    """

    def __init__(self, field_kw=None, start_near_thermal=False,
                 near_thermal_prob=0.0, strength_scale_range=None, obs_mask=None,
                 start_pos=None, goal_pos=None):
        super().__init__()

        fkw = dict(field_kw or {})
        fkw.setdefault("domain_x", DOMAIN_X)
        fkw.setdefault("domain_y", DOMAIN_Y)
        self._field_kw          = fkw
        self._near_thermal_prob = 1.0 if start_near_thermal else near_thermal_prob
        self._strength_scale_range = strength_scale_range
        self._obs_mask = (np.ones(9, dtype=bool) if obs_mask is None
                          else np.asarray(obs_mask, dtype=bool))
        self._start_x = float(start_pos[0]) if start_pos is not None else float(START_X)
        self._start_y = float(start_pos[1]) if start_pos is not None else float(START_Y)
        self._goal_x  = float(goal_pos[0])  if goal_pos  is not None else float(GOAL_X)
        self._goal_y  = float(goal_pos[1])  if goal_pos  is not None else float(GOAL_Y)

        _scale_max = (strength_scale_range[1] if strength_scale_range is not None
                      else fkw.get("strength_scale", 1.0))
        self._w_norm = fkw.get("W_range", (1.0, 3.0))[1] * _scale_max

        self.field = ThermalField(**fkw)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.x = self.y = self.z = 0.0
        self.psi = self.phi = 0.0
        self.V   = (V_MIN + V_MAX) / 2.0
        self._step_n  = 0
        self._prev_dist = 0.0
        self._w_ema   = 0.0

        self._traj: list[dict] = []


    def _glide_margin(self):
        """
        Altitude surplus above the minimum needed to glide to goal.

        Uses the agent's current airspeed but assumes level flight (phi=0).
        This makes the margin speed-honest — flying fast at V=35 shows a lower
        margin than V=25 — without the bank-angle distortion that would occur
        when using the actual phi: at phi=30 the forward groundspeed collapses
        to ~11 m/s, giving an artificially tiny margin that breaks the alt_scale
        altitude cap and lets the agent circle to 1 km+ unchecked.
        """
        dist = self._dist_to_goal()
        if dist < 1e-3:
            return self.z

        sink = POLAR.sink(self.V)          # level-flight sink at current speed

        dx, dy   = self._goal_x - self.x, self._goal_y - self.y
        wx, wy   = self.field.wind
        headwind = -(wx * dx + wy * dy) / dist

        V_ground    = max(self.V - headwind, 1.0)
        glide_ratio = V_ground / sink
        return self.z - dist / glide_ratio


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        fkw = dict(self._field_kw)
        if self._strength_scale_range is not None:
            lo, hi = self._strength_scale_range
            fkw["strength_scale"] = float(rng.uniform(lo, hi))

        self.field = ThermalField(rng=rng, **fkw)
        self._current_strength_scale = fkw.get("strength_scale", 1.0)

        if self._near_thermal_prob > 0.0 and rng.random() < self._near_thermal_prob:
            candidates = [t for t in self.field.thermals
                          if np.hypot(t.x - self._goal_x, t.y - self._goal_y) >= _MIN_NEAR_GOAL_DIST]
            target = (max(candidates, key=lambda t: t.effective_W)
                      if candidates else
                      max(self.field.thermals, key=lambda t: t.effective_W))
            angle  = rng.uniform(0, 2 * np.pi)
            dist   = target.R * rng.uniform(0.0, 0.4)
            self.x = float(np.clip(target.x + dist * np.cos(angle), 0, DOMAIN_X))
            self.y = float(np.clip(target.y + dist * np.sin(angle), 0, DOMAIN_Y))
        else:
            self.x = self._start_x
            self.y = self._start_y

        self.z        = INIT_ALT
        self.psi      = 0.0
        self.phi      = 0.0
        self.V        = 25.0
        self._w_ema   = 0.0
        self._step_n  = 0
        self._prev_dist = self._dist_to_goal()
        self._traj    = []

        return self._get_obs(), {}


    def step(self, action):
        action = np.asarray(action, dtype=float)

        # 1 Apply action
        bank_rate = float(action[0]) * PHI_RATE_MAX
        self.phi  = float(np.clip(
            self.phi + bank_rate * DT,
            -PHI_MAX, PHI_MAX
        ))
        V_norm = float(action[1])
        self.V = V_MIN + (V_norm + 1.0) / 2.0 * (V_MAX - V_MIN)
        self.V = float(np.clip(self.V, V_MIN, V_MAX))

        # 2 Kinematics
        dpsi     = G * np.tan(self.phi) / self.V
        self.psi += dpsi * DT
        wx, wy   = self.field.wind
        self.x   += (self.V * np.cos(self.psi) + wx) * DT
        self.y   += (self.V * np.sin(self.psi) + wy) * DT

        # 3 Advance thermal field
        self.field.step(DT)

        # 4 Aerodynamics
        w_curr = self.field.updraft_at(self.x, self.y)
        sink   = POLAR.sink_banked(self.V, self.phi)
        dz     = (w_curr - sink) * DT
        self.z = max(0.0, self.z + dz)

        # 5 Update thermal-richness memory
        self._w_ema += _W_EMA_ALPHA * (w_curr - self._w_ema)

        # 6 Reward
        dist_now = self._dist_to_goal()
        progress = self._prev_dist - dist_now
        self._prev_dist = dist_now

        in_lift         = w_curr > _LIFT_THRESHOLD
        banking_in_lift = in_lift and (abs(self.phi) > _BANK_THRESHOLD)

        # alt_scale decays linearly from 1 to 0 as glide margin grows to ALT_NORM.
        # Applying it to both altitude and thermalling bonus guarantees the agent
        # exits any thermal at a finite margin surplus (proven: <430 m for any w).
        margin    = self._glide_margin()
        alt_scale = max(0.0, 1.0 - max(0.0, margin) / ALT_NORM)

        thermal_bonus = _W_THERMAL * w_curr * DT if banking_in_lift else 0.0

        reward = ((_W_ALT * dz + thermal_bonus) * alt_scale
                + _W_PROGRESS * progress * (0.0 if in_lift else 1.0)
                + _W_STEP)

        # 7 Termination
        self._step_n += 1
        at_goal   = dist_now <= GOAL_RADIUS
        crashed   = self.z <= 0.0
        timed_out = self._step_n >= MAX_STEPS

        terminated = at_goal or crashed
        truncated  = timed_out and not terminated

        if at_goal:
            reward += _R_SUCCESS
        elif crashed:
            reward += _R_CRASH
        elif truncated:
            reward += _R_TIMEOUT

        # 8 Observation
        obs = self._get_obs(w_curr=w_curr)

        self._traj.append({
            "x": self.x, "y": self.y, "z": self.z,
            "psi": self.psi, "phi": self.phi, "V": self.V,
            "w": w_curr, "w_ema": self._w_ema, "sink": sink,
            "reward": float(reward),
        })

        info = {
            "dist_to_goal": dist_now,
            "updraft":       w_curr,
            "sink":          sink,
            "at_goal":       at_goal,
        }
        return obs, float(reward), terminated, truncated, info


    def _get_obs(self, w_curr=None):
        if w_curr is None:
            w_curr = self.field.updraft_at(self.x, self.y)

        nearest = min(self.field.thermals,
                      key=lambda t: np.hypot(self.x - t.x, self.y - t.y))
        dx_t = nearest.x - self.x
        dy_t = nearest.y - self.y
        bearing_thermal_abs = np.arctan2(dy_t, dx_t)
        bearing_thermal_rel = (bearing_thermal_abs - self.psi + np.pi) % (2 * np.pi) - np.pi
        _max_dist = np.hypot(DOMAIN_X, DOMAIN_Y)
        dist_thermal = np.hypot(dx_t, dy_t)

        bearing_goal_abs = np.arctan2(self._goal_y - self.y, self._goal_x - self.x)
        bearing_goal_rel = (bearing_goal_abs - self.psi + np.pi) % (2 * np.pi) - np.pi

        V_norm = 2.0 * (self.V - V_MIN) / (V_MAX - V_MIN) - 1.0

        obs = np.array([
            np.clip(w_curr / self._w_norm,                      -1.0, 1.0),  # 0
            np.clip(bearing_thermal_rel / np.pi,                -1.0, 1.0),  # 1
            np.clip(2.0 * dist_thermal / _max_dist - 1.0,      -1.0, 1.0),  # 2
            np.clip(nearest.effective_W / self._w_norm,          0.0, 1.0),  # 3
            np.clip(self.phi / PHI_MAX,                         -1.0, 1.0),  # 4
            np.clip(V_norm,                                     -1.0, 1.0),  # 5
            np.clip(bearing_goal_rel / np.pi,                   -1.0, 1.0),  # 6
            np.clip(self._glide_margin() / ALT_NORM,            -1.0, 1.0),  # 7
            np.clip(self._w_ema / self._w_norm,                  0.0, 1.0),  # 8 EMA
        ], dtype=np.float32)
        obs[~self._obs_mask] = 0.0
        return obs

    @property
    def obs_norms(self):
        return {
            "updraft [m/s]":               self._w_norm,
            "thermal bearing rel [rad/pi]": np.pi,
            "thermal distance [norm]":      np.hypot(DOMAIN_X, DOMAIN_Y),
            "thermal W_eff [norm]":         self._w_norm,
            "updraft EMA [norm]":           self._w_norm,
        }

    def _dist_to_goal(self):
        return float(np.hypot(self.x - GOAL_X, self.y - GOAL_Y))

    def render(self):
        """Return trajectory buffer for external plotting."""
        return self._traj

    def get_trajectory(self):
        """Return trajectory as dict of arrays."""
        if not self._traj:
            return {}
        keys = self._traj[0].keys()
        return {k: np.array([s[k] for s in self._traj]) for k in keys}

    def close(self):
        pass
