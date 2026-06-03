"""
Thermal soaring baseline: single thermal centering with tabular Q-learning.
Pure numpy + matplotlib. This is the LOW-complexity baseline that the
continuous high-dimensional version (PPO/SAC) is later compared against.

Physics:
  - Glider flies at fixed airspeed V, controlled by bank angle phi.
  - Action increments bank: left/hold/right. phi is part of the state.
  - Gedeon thermal updraft (sinking ring at the edge forces centering).
  - Bank-dependent sink polar: steeper bank turns tighter but sinks faster.
  - Observation = local cues only (lift felt, spanwise lift gradient, bank).
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ---- physics constants ----
V = 10.0          # airspeed [m/s]
G = 9.81          # gravity [m/s^2]
VS0 = 0.6         # base sink rate, wings level [m/s]
W = 2.5           # thermal core peak updraft [m/s]
R = 40.0          # thermal radius [m]
XC, YC = 0.0, 0.0 # thermal core location
SPAN = 10.0       # sensing span for spanwise gradient [m]
DT = 0.5          # timestep [s]
DPHI = np.deg2rad(5.0)   # bank increment per action [rad]
PHI_MAX = np.deg2rad(50.0)
T_MAX = 120.0     # episode horizon [s]
R_LOST = 150.0    # lost the thermal if drift beyond this [m]


def updraft(x, y):
    r = np.hypot(x - XC, y - YC)
    rn2 = (r / R) ** 2
    return W * np.exp(-rn2) * (1.0 - rn2)


def sink(phi):
    return VS0 / np.cos(phi) ** 1.5


class ThermalEnv:
    def reset(self):
        ang = rng.uniform(0, 2 * np.pi)
        r0 = 30.0
        self.x = XC + r0 * np.cos(ang)
        self.y = YC + r0 * np.sin(ang)
        self.psi = rng.uniform(0, 2 * np.pi)
        self.phi = 0.0
        self.t = 0.0
        self.z = 0.0
        return self._obs()

    def _obs(self):
        w_here = updraft(self.x, self.y)
        # spanwise gradient: sample updraft at the two wingtips
        left = np.array([-np.sin(self.psi), np.cos(self.psi)])
        tipL = np.array([self.x, self.y]) + left * SPAN / 2
        tipR = np.array([self.x, self.y]) - left * SPAN / 2
        grad = updraft(tipL[0], tipL[1]) - updraft(tipR[0], tipR[1])
        return np.array([w_here, grad, self.phi])

    def step(self, action):
        # action: 0 = bank left, 1 = hold, 2 = bank right
        if action == 0:
            self.phi = max(self.phi - DPHI, -PHI_MAX)
        elif action == 2:
            self.phi = min(self.phi + DPHI, PHI_MAX)
        psidot = G * np.tan(self.phi) / V
        self.psi += psidot * DT
        self.x += V * np.cos(self.psi) * DT
        self.y += V * np.sin(self.psi) * DT
        climb = updraft(self.x, self.y) - sink(self.phi)
        self.z += climb * DT
        self.t += DT
        r = np.hypot(self.x - XC, self.y - YC)
        done = (self.t >= T_MAX) or (r > R_LOST)
        reward = climb * DT  # altitude gained this step
        return self._obs(), reward, done


# ---- discretisation for the tabular table ----
LIFT_BINS = np.linspace(-0.5, W, 7)
GRAD_BINS = np.linspace(-0.6, 0.6, 9)
BANK_BINS = np.linspace(-PHI_MAX, PHI_MAX, 11)
N_ACTIONS = 3


def discretise(obs):
    li = np.digitize(obs[0], LIFT_BINS)
    gi = np.digitize(obs[1], GRAD_BINS)
    bi = np.digitize(obs[2], BANK_BINS)
    return (li, gi, bi)


def train(episodes=5000, alpha=0.2, gamma=0.98,
          eps_start=1.0, eps_end=0.05):
    env = ThermalEnv()
    Q = {}

    def getQ(s):
        if s not in Q:
            Q[s] = np.zeros(N_ACTIONS)
        return Q[s]

    history = []
    for ep in range(episodes):
        eps = eps_end + (eps_start - eps_end) * max(0, 1 - ep / (0.8 * episodes))
        obs = env.reset()
        s = discretise(obs)
        total = 0.0
        done = False
        while not done:
            qa = getQ(s)
            a = rng.integers(N_ACTIONS) if rng.random() < eps else int(np.argmax(qa))
            obs2, r, done = env.step(a)
            s2 = discretise(obs2)
            target = r + (0.0 if done else gamma * np.max(getQ(s2)))
            qa[a] += alpha * (target - qa[a])
            s = s2
            total += r
        history.append(total)
    return Q, np.array(history)


def greedy_rollout(Q):
    env = ThermalEnv()
    obs = env.reset()
    xs, ys, zs, banks = [env.x], [env.y], [env.z], [np.rad2deg(env.phi)]
    done = False
    while not done:
        s = discretise(obs)
        a = int(np.argmax(Q[s])) if s in Q else 1
        obs, r, done = env.step(a)
        xs.append(env.x); ys.append(env.y)
        zs.append(env.z); banks.append(np.rad2deg(env.phi))
    return map(np.array, (xs, ys, zs, banks))


if __name__ == "__main__":
    Q, hist = train()
    print(f"states visited: {len(Q)}")
    print(f"mean altitude gained, last 200 eps: {hist[-200:].mean():.1f} m")
    print(f"mean altitude gained, first 200 eps: {hist[:200].mean():.1f} m")

    xs, ys, zs, banks = greedy_rollout(Q)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # smoothed learning curve
    k = 50
    sm = np.convolve(hist, np.ones(k) / k, mode="valid")
    ax[0].plot(sm)
    ax[0].set_xlabel("episode")
    ax[0].set_ylabel("altitude gained per episode [m]")
    ax[0].set_title("learning curve (50 ep moving avg)")

    # trajectory over the thermal field
    gx = np.linspace(-80, 80, 200)
    GXX, GYY = np.meshgrid(gx, gx)
    field = updraft(GXX, GYY)
    cs = ax[1].contourf(GXX, GYY, field, levels=20, cmap="RdBu_r")
    ax[1].plot(xs, ys, "k", lw=1.2)
    ax[1].plot(xs[0], ys[0], "go", label="start")
    ax[1].plot(xs[-1], ys[-1], "ks", label="end")
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("y [m]")
    ax[1].set_title("learned trajectory (greedy)")
    ax[1].legend(loc="upper right")
    fig.colorbar(cs, ax=ax[1], label="updraft [m/s]", fraction=0.046)

    # altitude and bank over time
    t = np.arange(len(zs)) * DT
    ax[2].plot(t, zs, label="altitude gain [m]")
    ax2b = ax[2].twinx()
    ax2b.plot(t, banks, "r", alpha=0.6, label="bank [deg]")
    ax[2].set_xlabel("time [s]")
    ax[2].set_ylabel("altitude gain [m]")
    ax2b.set_ylabel("bank angle [deg]", color="r")
    ax[2].set_title("climb and bank during greedy flight")

    plt.tight_layout()
    plt.savefig("baseline_result.png", dpi=110)
    print("saved figure")