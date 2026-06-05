"""
Cross-country soaring environment: gymnasium.Env with continuous state/action.

A glider starts at point A with low altitude and must reach point B several
kilometres away. The thermal field provides lift; the agent controls bank rate
and airspeed. Speed-to-fly behaviour (MacCready theory) must emerge from the
climb and progress rewards alone, never from the reward encoding it directly.

Observation (8-D, all clipped to [-1, 1]):
  0  updraft at current position (local variometer)
  1  bearing to nearest thermal relative to heading / pi  (left=-1, right=+1)
  2  distance to nearest thermal, normalised to [-1, 1]   (close=-1, far=+1)
  3  effective peak strength of nearest thermal / W_max
  4  current bank angle
  5  current airspeed (normalised)
  6  bearing to goal relative to heading
  7  altitude (normalised)

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

DOMAIN_X     = 5500.0       # domain width east-west [m]
DOMAIN_Y     = 4000.0       # domain height north-south [m]
START_X      = 250.0        # default start x [m]
START_Y      = 2000.0       # default start y [m]
GOAL_X       = 5250.0       # goal x [m]
GOAL_Y       = 2000.0       # goal y [m]
GOAL_RADIUS  = 100.0        # arrival radius [m]
INIT_ALT     = 150.0        # starting altitude AGL [m]
ALT_NORM     = 500.0        # altitude normalisation reference [m]
MAX_STEPS    = 3600         # episode time limit (= 30 min at DT=0.5 s)



_ANCHOR_RADIUS        = 600.0   # guaranteed thermal within this distance of start [m]
_MIN_NEAR_GOAL_DIST   = 2000.0  # near-thermal start must be ≥ this far from goal [m]

_W_ALT       =  0.75         # per metre altitude gained
_W_PROGRESS  =  0.15        # per metre closer to goal
_W_STEP      = -0.15        # per timestep (efficiency pressure)
_R_SUCCESS   =  300.0       # terminal bonus for reaching goal
_R_CRASH     = -100.0       # terminal penalty for hitting ground


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
    obs_mask        : boolean array of length 8. Zero entries are replaced
                      by zero in the observation (observation ablation).
    """

    def __init__(self, field_kw=None, start_near_thermal=False,
                 near_thermal_prob=0.0, strength_scale_range=None, obs_mask=None):
        super().__init__()

        fkw = dict(field_kw or {})
        fkw.setdefault("domain_x", DOMAIN_X)
        fkw.setdefault("domain_y", DOMAIN_Y)
        # Guarantee one thermal near the start position every episode reset
        fkw.setdefault("anchor", (START_X, START_Y, _ANCHOR_RADIUS))
        self._field_kw             = fkw
        # start_near_thermal=True is legacy shorthand for near_thermal_prob=1.0
        self._near_thermal_prob    = 1.0 if start_near_thermal else near_thermal_prob
        self._strength_scale_range = strength_scale_range
        self._obs_mask = (np.ones(8, dtype=bool) if obs_mask is None
                          else np.asarray(obs_mask, dtype=bool))

        # W_max: used to normalise channels 0 and 3 to [-1, 1]
        _scale_max = (strength_scale_range[1] if strength_scale_range is not None
                      else fkw.get("strength_scale", 1.0))
        self._w_norm = fkw.get("W_range", (1.0, 3.0))[1] * _scale_max

        self.field = ThermalField(**fkw)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.x = self.y = self.z = 0.0
        self.psi = self.phi = 0.0
        self.V   = (V_MIN + V_MAX) / 2.0
        self._step_n  = 0
        self._prev_dist = 0.0

        self._traj: list[dict] = []


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Optionally randomise thermal strength each episode
        fkw = dict(self._field_kw)
        if self._strength_scale_range is not None:
            lo, hi = self._strength_scale_range
            fkw["strength_scale"] = float(rng.uniform(lo, hi))

        self.field = ThermalField(rng=rng, **fkw)
        self._current_strength_scale = fkw.get("strength_scale", 1.0)

        # Start position
        if self._near_thermal_prob > 0.0 and rng.random() < self._near_thermal_prob:
            # Only start near thermals far enough from the goal, so the agent
            # can't trivially complete the task without cross-country flying.
            candidates = [t for t in self.field.thermals
                          if np.hypot(t.x - GOAL_X, t.y - GOAL_Y) >= _MIN_NEAR_GOAL_DIST]
            target = (max(candidates, key=lambda t: t.effective_W)
                      if candidates else
                      max(self.field.thermals, key=lambda t: t.effective_W))
            angle  = rng.uniform(0, 2 * np.pi)
            dist   = target.R * rng.uniform(0.0, 0.4)
            self.x = float(np.clip(target.x + dist * np.cos(angle), 0, DOMAIN_X))
            self.y = float(np.clip(target.y + dist * np.sin(angle), 0, DOMAIN_Y))
        else:
            self.x = START_X
            self.y = START_Y

        self.z        = INIT_ALT
        self.psi      = 0.0              # heading east
        self.phi      = 0.0              # wings level
        self.V        = 25.0             # best-glide speed initially

        self._step_n  = 0
        self._prev_dist = self._dist_to_goal()
        self._traj    = []

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=float)

        # 1 Apply action
        bank_rate = float(action[0]) * PHI_RATE_MAX      # [rad/s]
        self.phi  = float(np.clip(
            self.phi + bank_rate * DT,
            -PHI_MAX, PHI_MAX
        ))
        V_norm = float(action[1])
        self.V = V_MIN + (V_norm + 1.0) / 2.0 * (V_MAX - V_MIN)
        self.V = float(np.clip(self.V, V_MIN, V_MAX))

        # 2 Kinematics
        # Glider position = airspeed vector + airmass drift (wind).
        # Both the glider and the thermals are carried by the same wind, so a
        # circling glider stays inside a drifting thermal as in real flight.
        dpsi    = G * np.tan(self.phi) / self.V
        self.psi += dpsi * DT
        wx, wy = self.field.wind          # airmass drift [m/s]
        self.x  += (self.V * np.cos(self.psi) + wx) * DT
        self.y  += (self.V * np.sin(self.psi) + wy) * DT

        # 3 Advance thermal field (thermals drift while glider moves)
        self.field.step(DT)

        # 4. Aerodynamics
        w_curr  = self.field.updraft_at(self.x, self.y)
        sink    = POLAR.sink_banked(self.V, self.phi)
        dz      = (w_curr - sink) * DT
        self.z  = max(0.0, self.z + dz)

        # 5. Reward
        dist_now    = self._dist_to_goal()
        progress    = self._prev_dist - dist_now        # positive = closer
        self._prev_dist = dist_now

        reward  = (_W_ALT * dz
                   + _W_PROGRESS * progress
                   + _W_STEP)

        # 6 Termination
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

        # 7 Observation
        obs = self._get_obs(w_curr=w_curr)

        # trajectory logging for render/analysis
        self._traj.append({
            "x": self.x, "y": self.y, "z": self.z,
            "psi": self.psi, "phi": self.phi, "V": self.V,
            "w": w_curr, "sink": sink,
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

        # nearest thermal: heading-relative bearing + distance + strength
        nearest = min(self.field.thermals,
                      key=lambda t: np.hypot(self.x - t.x, self.y - t.y))
        dx_t = nearest.x - self.x
        dy_t = nearest.y - self.y
        bearing_thermal_abs = np.arctan2(dy_t, dx_t)
        # heading-relative, wrapped [-pi, pi] — same convention as goal bearing
        bearing_thermal_rel = (bearing_thermal_abs - self.psi + np.pi) % (2 * np.pi) - np.pi
        _max_dist = np.hypot(DOMAIN_X, DOMAIN_Y)   # ~6800 m
        dist_thermal = np.hypot(dx_t, dy_t)

        # bearing to goal relative to heading, wrapped to [-pi, pi]
        bearing_goal_abs = np.arctan2(GOAL_Y - self.y, GOAL_X - self.x)
        bearing_goal_rel = (bearing_goal_abs - self.psi + np.pi) % (2 * np.pi) - np.pi

        V_norm = 2.0 * (self.V - V_MIN) / (V_MAX - V_MIN) - 1.0

        obs = np.array([
            np.clip(w_curr / self._w_norm,                      -1.0, 1.0),
            np.clip(bearing_thermal_rel / np.pi,                -1.0, 1.0),
            np.clip(2.0 * dist_thermal / _max_dist - 1.0,      -1.0, 1.0),
            np.clip(nearest.effective_W / self._w_norm,          0.0, 1.0),
            np.clip(self.phi / PHI_MAX,                         -1.0, 1.0),
            np.clip(V_norm,                                     -1.0, 1.0),
            np.clip(bearing_goal_rel / np.pi,                   -1.0, 1.0),
            np.clip(self.z / ALT_NORM,                           0.0, 1.0) * 2.0 - 1.0,
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
