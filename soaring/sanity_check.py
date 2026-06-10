"""
Sanity check: two scripted policies from an identical starting state.

Policy A (circling): constant 30 deg bank, 19 m/s (min-sink speed).
  Expected result: gains altitude inside the thermal.

Policy B (gliding):  wings level, 25 m/s (best-glide speed), heading east.
  Expected result: monotonically loses altitude and sinks to ground.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soaring.env.soaring_env import (SoaringEnv, DT, V_MIN, V_MAX, PHI_MAX, PHI_RATE_MAX,
                                     DOMAIN_X, DOMAIN_Y, GOAL_X, GOAL_Y)
from soaring.theory.maccready import POLAR
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED         = 42
MAX_STEPS    = 700          # enough to see glider sink out (~466 steps at 150 m)
PHI_TARGET_CIRC = np.deg2rad(30.0)
V_CIRC       = 19.0         # min-sink speed [m/s]
V_GLIDE      = 25.0         # best-glide speed [m/s]


FIELD_KW = dict(
    n_thermals     = 6,
    wind           = (-5.0, 0.0),
    W_range        = (1.5, 3.0),
    R_range        = (100.0, 180.0),
    lifetime_range = (2000, 4000),
    strength_scale = 1.0,
    turbulence_amp = 0.03,
)


def _draw_thermals(ax, xs, ys, Rs, label="Thermal cores"):
    """Hollow dashed rings + cross markers so the updraft color shows through."""
    for i, (x, y, R) in enumerate(zip(xs, ys, Rs)):
        circle = plt.Circle(
            (x, y), R,
            fill=False, linestyle="--", edgecolor="k",
            linewidth=1.0, alpha=0.6, zorder=4,
        )
        ax.add_patch(circle)
        # ax.plot(x, y, "+k", ms=7, mew=1.5, zorder=5,
        #         label=label if i == 0 else None)


def _v_action(V):
    return float(np.clip(2.0 * (V - V_MIN) / (V_MAX - V_MIN) - 1.0, -1.0, 1.0))


def circling_action(phi):
    """Maintain 30 deg right bank at min-sink speed."""
    err = PHI_TARGET_CIRC - phi
    # proportional control on bank rate, clipped to [-1, 1]
    bank_rate_norm = float(np.clip(err / (PHI_RATE_MAX * DT), -1.0, 1.0))
    return np.array([bank_rate_norm, _v_action(V_CIRC)], dtype=np.float32)


def gliding_action(phi):
    """Wings level, best-glide speed, no heading correction needed."""
    err = 0.0 - phi
    bank_rate_norm = float(np.clip(err / (PHI_RATE_MAX * DT), -1.0, 1.0))
    return np.array([bank_rate_norm, _v_action(V_GLIDE)], dtype=np.float32)


def run_policy(env, policy_fn, max_steps=MAX_STEPS):
    obs, _ = env.reset(seed=SEED)
    log = dict(
        t=[], z=[], phi=[], V=[], w=[], reward=[],
        x=[], y=[], sink=[],
    )
    total_reward = 0.0
    for step in range(max_steps):
        action = policy_fn(env.phi)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        log["t"].append(step * DT)
        log["z"].append(env.z)
        log["phi"].append(np.rad2deg(env.phi))
        log["V"].append(env.V)
        log["w"].append(info["updraft"])
        log["sink"].append(info["sink"])
        log["reward"].append(reward)
        log["x"].append(env.x)
        log["y"].append(env.y)
        if terminated or truncated:
            break
    for k in log:
        log[k] = np.array(log[k])
    log["total_reward"] = total_reward
    return log


def print_summary(circ, glide):
    print("\n" + "=" * 60)
    print("SANITY CHECK RESULTS")
    print("=" * 60)
    print(f"\nCircling policy  (phi=30 deg, V={V_CIRC} m/s):")
    print(f"  Steps run         : {len(circ['t'])}")
    print(f"  Final altitude    : {circ['z'][-1]:.1f} m")
    print(f"  Peak altitude     : {circ['z'].max():.1f} m")
    print(f"  Altitude change   : {circ['z'][-1] - circ['z'][0]:.1f} m")
    print(f"  Mean updraft      : {circ['w'].mean():.2f} m/s")
    print(f"  Mean sink         : {circ['sink'].mean():.2f} m/s")
    print(f"  Total reward      : {circ['total_reward']:.1f}")

    print(f"\nGliding policy   (phi=0 deg, V={V_GLIDE} m/s):")
    print(f"  Steps run         : {len(glide['t'])}")
    print(f"  Final altitude    : {glide['z'][-1]:.1f} m")
    print(f"  Altitude at crash : {glide['z'][-1]:.1f} m  (sank out: {glide['z'][-1] <= 1.0})")
    print(f"  Distance covered  : {glide['x'][-1] - glide['x'][0]:.0f} m east")
    print(f"  Mean sink         : {glide['sink'].mean():.2f} m/s")
    print(f"  Total reward      : {glide['total_reward']:.1f}")

    circ_climbed = circ["z"][-1] > circ["z"][0]
    glide_sank   = glide["z"][-1] <= 1.0
    print(f"\nCircling gained altitude : {circ_climbed} (expected True)")
    print(f"Gliding sank to ground   : {glide_sank} (expected True)")
    if circ_climbed and glide_sank:
        print("\nSanity check PASSED - reward shaping is physically consistent.\n")
    else:
        print("\nSanity check FAILED - check reward weights and thermal config.\n")



def make_figures(env: SoaringEnv, circ, glide):

    fig1, axes = plt.subplots(2, 2, figsize=(14, 7),
                              sharey="col", sharex="row")

    labels = ["Circling  (phi=30 deg, V=19 m/s)",
              "Gliding   (phi=0 deg,  V=25 m/s)"]
    colors = ["steelblue", "firebrick"]
    logs   = [circ, glide]

    for row, (log, label, col) in enumerate(zip(logs, labels, colors)):
        ax_alt = axes[row, 0]
        ax_rew = axes[row, 1]

        ax_alt.plot(log["t"], log["z"], color=col, lw=2)
        ax_alt.axhline(0, color="k", lw=0.8, ls="--", alpha=0.45)
        ax_alt.set_ylabel("Altitude AGL [m]", fontsize=10)
        ax_alt.set_title(label, fontsize=10, color=col)
        ax_alt.grid(True, alpha=0.3)

        ax_rew.plot(log["t"], log["reward"], color=col, lw=1.5, alpha=0.85)
        ax_rew.axhline(0, color="k", lw=0.8, ls="--", alpha=0.45)
        ax_rew.set_ylabel("Per-step reward", fontsize=10)
        ax_rew.set_title(label, fontsize=10, color=col)
        ax_rew.grid(True, alpha=0.3)

        if row == 1:
            ax_alt.set_xlabel("Time [s]", fontsize=10)
            ax_rew.set_xlabel("Time [s]", fontsize=10)

    fig1.suptitle("Scripted policy comparison: altitude and reward",
                  fontsize=12, y=1.01)
    plt.tight_layout()
    p1 = os.path.join(RESULTS_DIR, "sanity_altitude_reward.pdf")
    fig1.savefig(p1, dpi=120, bbox_inches="tight")
    print(f"Saved {p1}")
    plt.close(fig1)


    env.reset(seed=SEED)
    xs0, ys0, Ws0, Rs0 = env.field.get_positions()

    all_x = np.concatenate([circ["x"], glide["x"]])
    all_y = np.concatenate([circ["y"], glide["y"]])
    pad   = 300
    x_lo, x_hi = all_x.min() - pad, all_x.max() + pad
    y_lo, y_hi = all_y.min() - pad, all_y.max() + pad

    gx = np.linspace(x_lo, x_hi, 240)
    gy = np.linspace(y_lo, y_hi, 180)
    GX, GY = np.meshgrid(gx, gy)

    env.reset(seed=SEED)
    for _ in range(MAX_STEPS // 2):
        env.field.step(DT)
    xs_mid, ys_mid, _, Rs_mid = env.field.get_positions()
    U_mid = env.field.updraft_grid(GX, GY)

    fig2, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(15, 6))

    env.reset(seed=SEED)
    gx_f = np.linspace(0, DOMAIN_X, 200)
    gy_f = np.linspace(0, DOMAIN_Y, 150)
    GXf, GYf = np.meshgrid(gx_f, gy_f)
    U_init = env.field.updraft_grid(GXf, GYf)
    xs_init, ys_init, _, Rs_init = env.field.get_positions()

    cf = ax_full.contourf(GXf, GYf, U_init, levels=16, cmap="RdYlBu_r", alpha=0.75)
    fig2.colorbar(cf, ax=ax_full, label="Updraft [m/s]", shrink=0.75)
    _draw_thermals(ax_full, xs_init, ys_init, Rs_init, label="Thermals at t=0")
    ax_full.plot(circ["x"], circ["y"], "b-", lw=1.2, alpha=0.8,
                 label=f"Circling (z_end={circ['z'][-1]:.0f} m)")
    ax_full.plot(glide["x"], glide["y"], "r-", lw=1.5, alpha=0.9,
                 label="Gliding (sinks out)")
    ax_full.plot(circ["x"][0], circ["y"][0], "go", ms=9, zorder=6, label="Start")
    ax_full.plot(GOAL_X, GOAL_Y, "m*", ms=13, zorder=6, label="Goal")
    ax_full.annotate("", xy=(800, DOMAIN_Y * 0.55), xytext=(1500, DOMAIN_Y * 0.55),
                     arrowprops=dict(arrowstyle="->", color="k", lw=2))
    ax_full.text(1100, DOMAIN_Y * 0.60, "Wind 5 m/s\n(thermals drift west)",
                 ha="center", fontsize=8)
    ax_full.set_xlim(0, DOMAIN_X); ax_full.set_ylim(0, DOMAIN_Y)
    ax_full.set_aspect("equal")
    ax_full.set_xlabel("x [m]  (East)", fontsize=10)
    ax_full.set_ylabel("y [m]  (North)", fontsize=10)
    ax_full.set_title("Full domain: initial thermal field and trajectories",
                      fontsize=10)
    ax_full.legend(fontsize=8, loc="upper left")
    ax_full.grid(True, alpha=0.2)

    cf2 = ax_zoom.contourf(GX, GY, U_mid, levels=16, cmap="RdYlBu_r", alpha=0.75)
    fig2.colorbar(cf2, ax=ax_zoom, label="Updraft [m/s]", shrink=0.75)
    _draw_thermals(ax_zoom, xs_mid, ys_mid, Rs_mid, label="Thermals at t=mid")
    ax_zoom.plot(circ["x"], circ["y"], "b-", lw=1.5, alpha=0.9,
                 label="Circling")
    ax_zoom.plot(glide["x"], glide["y"], "r-", lw=1.5, alpha=0.9,
                 label="Gliding")
    ax_zoom.plot(circ["x"][0], circ["y"][0], "go", ms=9, zorder=6)
    ax_zoom.set_xlim(x_lo, x_hi); ax_zoom.set_ylim(y_lo, y_hi)
    ax_zoom.set_aspect("equal")
    ax_zoom.set_xlabel("x [m]  (East)", fontsize=10)
    ax_zoom.set_ylabel("y [m]  (North)", fontsize=10)
    ax_zoom.set_title(
        "Zoomed: flight region with mid-episode thermal positions\n"
        "(circling stays co-located with thermals because both drift at wind speed)",
        fontsize=9,
    )
    ax_zoom.legend(fontsize=8, loc = 'upper center', bbox_to_anchor=(0.5, -0.85), ncol=1)
    ax_zoom.grid(True, alpha=0.2)

    plt.tight_layout()
    p2 = os.path.join(RESULTS_DIR, "sanity_trajectories.pdf")
    fig2.savefig(p2, dpi=120, bbox_inches="tight")
    print(f"Saved {p2}")
    plt.close(fig2)


def main():
    print("=" * 60)
    print("Step 1: Polar verification")
    print("=" * 60)
    POLAR.verify()

    print("=" * 60)
    print("Step 2: Sanity check - scripted policies")
    print("=" * 60)

    env = SoaringEnv(field_kw=FIELD_KW, start_near_thermal=True)

    print("Running circling policy ...")
    circ  = run_policy(env, circling_action)

    print("Running gliding policy  ...")
    glide = run_policy(env, gliding_action)

    print_summary(circ, glide)
    make_figures(env, circ, glide)
    print("Figures saved to results/")


if __name__ == "__main__":
    main()
