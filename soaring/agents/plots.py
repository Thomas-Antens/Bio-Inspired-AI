"""
Generate all report figures from saved results.
"""

import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from soaring.theory.maccready import POLAR
from soaring.env.soaring_env import SoaringEnv, DT, GOAL_X, GOAL_Y, DOMAIN_X, DOMAIN_Y
from soaring.agents.train_sac import FIELD_KW


RESULTS_DIR = "results"


def _save(fig, path):
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")



def plot_maccready_theory(out_dir):
    Mc_ref = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    POLAR.plot_polar(ax1, Mc_ref)
    POLAR.plot_stf_curve(ax2, Mc_ref)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "maccready_curve.pdf"))


def smooth(x, k=50):
    return np.convolve(x, np.ones(k) / k, mode="valid")


def _load_reward_curves(out_dir, algo, k=50):
    files = sorted(glob.glob(os.path.join(out_dir, f"rewards_{algo}_seed*.npy")))
    curves = []
    for fp in files:
        r = np.load(fp)
        if len(r) > k:
            curves.append(smooth(r, k))
    return curves


def plot_learning_curves_sac(out_dir):
    """SAC-only learning curves, one line per seed + mean ± std band."""
    curves = _load_reward_curves(out_dir, "sac")
    if not curves:
        print("No SAC reward files found; skipping SAC learning curve.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    seed_colors = plt.get_cmap("Blues")(np.linspace(0.4, 0.85, len(curves)))

    for i, (c, col) in enumerate(zip(curves, seed_colors)):
        ax.plot(np.arange(len(c)), c, lw=1.2, color=col,
                alpha=0.7, label=f"seed {i}")

    if len(curves) > 1:
        min_len = min(len(c) for c in curves)
        stacked = np.stack([c[:min_len] for c in curves])
        mean = stacked.mean(0)
        std  = stacked.std(0)
        xs   = np.arange(min_len)
        ax.plot(xs, mean, lw=2.5, color="steelblue", label="mean")
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color="steelblue")

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode reward (50-ep moving avg)", fontsize=11)
    ax.set_title(f"SAC learning curves ({len(curves)} seed(s))", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "learning_curves_sac.pdf"))


def plot_learning_curves_comparison(out_dir):
    """SAC vs PPO: mean ± std across all available seeds for each algo."""
    colors = {"sac": "steelblue", "ppo": "darkorange"}
    fig, ax = plt.subplots(figsize=(10, 5))
    found = []

    for algo in ("sac", "ppo"):
        curves = _load_reward_curves(out_dir, algo)
        if not curves:
            continue
        found.append(algo)
        min_len = min(len(c) for c in curves)
        stacked = np.stack([c[:min_len] for c in curves])
        mean = stacked.mean(0)
        std  = stacked.std(0)
        xs   = np.arange(min_len)
        col  = colors[algo]
        ax.plot(xs, mean, lw=2, color=col,
                label=f"{algo.upper()} (n={len(curves)} seeds)")
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=col)

    if not found:
        print("No reward files found; skipping comparison learning curve.")
        plt.close(fig)
        return

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode reward (50-ep moving avg)", fontsize=11)
    ax.set_title("Learning curves: SAC vs PPO", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "learning_curves_comparison.pdf"))


def _load_model(model_path):
    """Try loading as SAC then PPO.

    SB3 2.8 passes a raw ZipExtFile (non-seekable for compressed entries)
    directly to torch.load.  PyTorch 2.4's PyTorchFileReader requires a
    seekable stream to navigate the inner archive, so we buffer it in BytesIO.
    """
    import io as _io
    import torch as _th
    _orig = _th.load

    def _seekable_load(f_or_path, *args, **kwargs):
        if hasattr(f_or_path, "read"):
            # Always buffer into BytesIO: ZipExtFile.seekable() returns True
            # in Python 3.12 for compressed entries, but PyTorchFileReader (C++)
            # still can't seek through it correctly.
            f_or_path = _io.BytesIO(f_or_path.read())
        kwargs["weights_only"] = False
        return _orig(f_or_path, *args, **kwargs)

    _th.load = _seekable_load
    try:
        try:
            from stable_baselines3 import SAC
            return SAC.load(model_path)
        except Exception:
            from stable_baselines3 import PPO
            return PPO.load(model_path)
    finally:
        _th.load = _orig

def rollout(model, seed=0, strength_scale=1.0, long_life=True):
    fkw = dict(FIELD_KW)
    if long_life:
        fkw["lifetime_range"] = (500, 1200)
    fkw["strength_scale"] = strength_scale
    env = SoaringEnv(field_kw=fkw, start_near_thermal=False, near_thermal_prob = 0.0)
    obs, _ = env.reset(seed=seed)
    done = False
    field_snapshots = []   # thermal state at each step, for snapshot plots
    while not done:
        field_snapshots.append([
            (t.x, t.y, t.R, float(t.effective_W)) for t in env.field.thermals
        ])
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env._field_snapshots = field_snapshots
    return env


def _thermal_field_bg(env, ax, gx, gy, snapshot=None):
    """
    Render thermal updraft grid in axes ax.

    snapshot : list of (x, y, R, W_eff) tuples from _field_snapshots, or None
               to use the current (final) field state.
    """
    GX, GY = np.meshgrid(gx, gy)
    if snapshot is None:
        U = env.field.updraft_grid(GX, GY)
        thermals_for_circles = list(zip(*env.field.get_positions()))  # (x,y,W,R)
    else:
        # Reconstruct updraft from snapshot tuples (x, y, R, W_eff)
        U = np.zeros_like(GX)
        for x_t, y_t, R_t, W_t in snapshot:
            r   = np.hypot(GX - x_t, GY - y_t)
            rn2 = (r / R_t) ** 2
            U  += W_t * np.exp(-rn2) * (1.0 - rn2)
        thermals_for_circles = snapshot  # (x, y, R, W_eff)
    cf = ax.contourf(GX, GY, U, levels=16, cmap="RdYlBu_r", alpha=0.8)
    # draw thermal circles from the snapshot
    for entry in thermals_for_circles:
        x_t, y_t, R_t = entry[0], entry[1], entry[2]
        ax.add_patch(plt.Circle((x_t, y_t), R_t, fill=False,
                                linestyle="--", edgecolor="k",
                                linewidth=0.8, alpha=0.5, zorder=4))
        ax.plot(x_t, y_t, "+k", ms=6, mew=1.2, zorder=5)
    return cf


def _add_trajectory(ax, x, y, z, lw=1.8, alt_ref=600):
    """Colour trajectory by altitude (blue=low, cyan=high)."""
    from matplotlib.collections import LineCollection
    pts  = np.stack([x, y], axis=1)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    cols = plt.cm.cool(np.clip(z[:-1] / alt_ref, 0, 1))
    lc   = LineCollection(segs, colors=cols, linewidths=lw, zorder=4)
    ax.add_collection(lc)
    return lc



def _traj_bounds(x, y, pad=300):
    """Axis limits that cover both the thermal domain and the full trajectory."""
    return (min(0.0, x.min() - pad), max(DOMAIN_X, x.max() + pad),
            min(0.0, y.min() - pad), max(DOMAIN_Y, y.max() + pad))


def _extended_grid(x, y, n_x=200, n_y=150, pad=300):
    x0, x1, y0, y1 = _traj_bounds(x, y, pad)
    return np.linspace(x0, x1, n_x), np.linspace(y0, y1, n_y), x0, x1, y0, y1


def _draw_domain_rect(ax):
    """Dashed rectangle showing where thermals exist."""
    ax.add_patch(mpatches.Rectangle(
        (0, 0), DOMAIN_X, DOMAIN_Y,
        fill=False, edgecolor="grey", linewidth=1.2,
        linestyle="--", zorder=3, label="Thermal domain",
    ))


def plot_trajectory_full(env, out_dir):
    """Full A-to-B trajectory map + altitude profile side by side."""
    traj = env.get_trajectory()
    if not traj:
        return
    x, y, z = traj["x"], traj["y"], traj["z"]
    w        = traj["w"]
    t_s      = np.arange(len(z)) * DT

    gx, gy, x0, x1, y0, y1 = _extended_grid(x, y)
    fig, (ax_map, ax_alt) = plt.subplots(
        1, 2, figsize=(18, 7),
        gridspec_kw={"width_ratios": [2.5, 1]},
    )

    snap0 = getattr(env, "_field_snapshots", [None])[0]
    cf = _thermal_field_bg(env, ax_map, gx, gy, snapshot=snap0)
    plt.colorbar(cf, ax=ax_map, label="Updraft [m/s]", shrink=0.8)
    _draw_domain_rect(ax_map)
    _add_trajectory(ax_map, x, y, z)
    ax_map.plot(x[0], y[0], "go", ms=12, zorder=6, label="Start A")
    ax_map.plot(GOAL_X, GOAL_Y, "m*", ms=15, zorder=6, label="Goal B")
    ax_map.set_xlabel("x [m]  (East)",  fontsize=11)
    ax_map.set_ylabel("y [m]  (North)", fontsize=11)
    ax_map.set_title("Cross-country flight — thermal field at t = 0 (colour = altitude)")
    ax_map.legend(fontsize=10)
    ax_map.set_aspect("equal")
    ax_map.set_xlim(x0, x1)
    ax_map.set_ylim(y0, y1)
    ax_map.grid(True, alpha=0.2)

    # altitude profile
    contacts = w > 0.3
    ax_alt.plot(t_s / 60, z, color="steelblue", lw=1.5)
    ax_alt.fill_between(t_s / 60, 0, z, where=contacts,
                        alpha=0.25, color="green", label="Thermal contact")
    ax_alt.set_xlabel("Time [min]",    fontsize=11)
    ax_alt.set_ylabel("Altitude [m]",  fontsize=11)
    ax_alt.set_title("Altitude profile")
    ax_alt.legend(fontsize=9)
    ax_alt.grid(True, alpha=0.3)
    ax_alt.set_ylim(bottom=0)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "trajectory_full.pdf"))


def plot_trajectory_sample(env, out_dir):
    """Sample trajectory showing the climb-then-glide cycle."""
    traj = env.get_trajectory()
    if not traj:
        return
    phi_arr = traj["phi"]
    x, y    = traj["x"], traj["y"]

    circling = np.abs(phi_arr) > np.deg2rad(20)

    gx, gy, x0, x1, y0, y1 = _extended_grid(x, y)
    snap0 = getattr(env, "_field_snapshots", [None])[0]
    fig, ax = plt.subplots(figsize=(14, 7))
    cf = _thermal_field_bg(env, ax, gx, gy, snapshot=snap0)
    plt.colorbar(cf, ax=ax, label="Updraft [m/s]", shrink=0.8)
    _draw_domain_rect(ax)

    ax.plot(x[circling],  y[circling],  "b.", ms=3, alpha=0.7,
            label="Thermalling (|phi|>20°)")
    ax.plot(x[~circling], y[~circling], "r.", ms=3, alpha=0.7,
            label="Gliding")
    ax.plot(x[0],    y[0],    "go", ms=10, zorder=6, label="Start")
    ax.plot(x[-1],   y[-1],   "ks", ms=10, zorder=6, label="End")
    ax.plot(GOAL_X, GOAL_Y,  "m*",  ms=14, zorder=6, label="Goal")

    ax.set_xlabel("x [m]  (East)",  fontsize=11)
    ax.set_ylabel("y [m]  (North)", fontsize=11)
    ax.set_title("Climb-then-glide cycle: blue=circling, red=gliding")
    ax.set_aspect("equal")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "trajectory_sample.pdf"))


def plot_trajectory_snapshots(env, out_dir):
    """2×2 panel: trajectory at 25/50/75/100 % with the correct thermal
    field state at each moment (thermals drift with wind, so each panel
    shows a different field snapshot captured during the rollout)."""
    traj = env.get_trajectory()
    if not traj:
        return
    x, y, z = traj["x"], traj["y"], traj["z"]
    psi_arr = traj["psi"]
    n       = len(x)

    snapshots        = getattr(env, "_field_snapshots", None)
    gx, gy, x0, x1, y0, y1 = _extended_grid(x, y, n_x=150, n_y=110)
    fracs    = [0.25, 0.50, 0.75, 1.00]
    labels   = ["t = 25 %", "t = 50 %", "t = 75 %", "t = 100 % (end)"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)

    for ax, frac, label in zip(axes.flat, fracs, labels):
        idx      = min(int(frac * n), n - 1)
        snapshot = snapshots[idx] if snapshots else None

        _thermal_field_bg(env, ax, gx, gy, snapshot=snapshot)
        _draw_domain_rect(ax)

        if idx > 0:
            _add_trajectory(ax, x[:idx+1], y[:idx+1], z[:idx+1], lw=1.8)

        if idx < n - 1:
            ax.plot(x[idx:], y[idx:], "--", color="grey",
                    lw=0.8, alpha=0.5, zorder=3)

        ax.plot(x[idx], y[idx], "o", color="white", ms=9,
                markeredgecolor="k", markeredgewidth=1.2, zorder=7)
        hdg = 180.0
        ax.annotate(
            "", zorder=8,
            xy=(x[idx] + hdg * np.cos(psi_arr[idx]),
                y[idx] + hdg * np.sin(psi_arr[idx])),
            xytext=(x[idx], y[idx]),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=2.0),
        )
        ax.plot(x[0],   y[0],   "go", ms=10, zorder=6)
        ax.plot(GOAL_X, GOAL_Y, "m*",  ms=13, zorder=6)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal")

    for ax in axes[:, 0]:
        ax.set_ylabel("y [m]", fontsize=10)
    for ax in axes[1, :]:
        ax.set_xlabel("x [m]", fontsize=10)

    fig.suptitle("Trajectory snapshots — thermal field at each moment "
                 "(colour = altitude: blue=low, cyan=high)", fontsize=12)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "trajectory_snapshots.pdf"))


def plot_episode_timeseries(env, out_dir):
    """Altitude, bank, airspeed over time for one episode."""
    traj = env.get_trajectory()
    if not traj:
        return
    t    = np.arange(len(traj["z"])) * DT
    z    = traj["z"]
    phi  = np.rad2deg(traj["phi"])
    V    = traj["V"]
    w    = traj["w"]

    contacts = w > 0.3

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(t, z, "b-", lw=1.5, label="Altitude AGL")
    axes[0].fill_between(t, 0, z, where=contacts, alpha=0.25,
                         color="green", label="Thermal contact")
    axes[0].set_ylabel("Altitude [m]", fontsize=10)
    axes[0].set_title("Episode flight: altitude, bank, airspeed")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, phi, "r-", lw=1.2)
    axes[1].axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    axes[1].fill_between(t, -50, 50, where=contacts, alpha=0.15, color="green")
    axes[1].set_ylabel("Bank angle [deg]", fontsize=10)
    axes[1].set_ylim(-55, 55)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, V, "g-", lw=1.5, label="Airspeed")
    axes[2].fill_between(t, 18, 45, where=contacts, alpha=0.15,
                         color="green", label="Thermal contact")
    axes[2].set_ylabel("Airspeed [m/s]", fontsize=10)
    axes[2].set_xlabel("Time [s]",       fontsize=10)
    axes[2].set_ylim(16, 48)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "episode_timeseries.pdf"))



def make_animation(env, out_dir, fps=12, dpi=90, n_frames=120):
    """
    Animated MP4 (GIF fallback) of the full episode.

    Shows the drifting thermal field at each moment, the trajectory so far
    coloured by altitude, glider marker + heading arrow, and the domain
    boundary rectangle.  Targets n_frames total regardless of episode length.
    """
    import matplotlib.animation as animation

    traj = env.get_trajectory()
    if not traj:
        return
    x, y, z   = traj["x"], traj["y"], traj["z"]
    psi_arr    = traj["psi"]
    w_arr      = traj["w"]
    snapshots  = getattr(env, "_field_snapshots", None)
    n          = len(x)
    step       = max(1, n // n_frames)
    frame_idx  = list(range(0, n, step))

    x0, x1, y0, y1 = _traj_bounds(x, y)
    GX, GY = np.meshgrid(np.linspace(x0, x1, 90),
                         np.linspace(y0, y1, 65))

    # pre-compute updraft grids so rendering is fast
    print(f"  Pre-computing {len(frame_idx)} animation frames ...")
    U_list = []
    for idx in frame_idx:
        snap = snapshots[idx] if snapshots else None
        if snap:
            U = np.zeros_like(GX)
            for xt, yt, Rt, Wt in snap:
                r   = np.hypot(GX - xt, GY - yt)
                rn2 = (r / Rt) ** 2
                U  += Wt * np.exp(-rn2) * (1.0 - rn2)
        else:
            U = env.field.updraft_grid(GX, GY)
        U_list.append(U)
    vmax = max(max(abs(U.max()), abs(U.min())) for U in U_list)
    vmax = max(vmax, 0.1)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    plt.tight_layout(pad=0.4)

    def _render(fi):
        ax.clear()
        idx = frame_idx[fi]
        ax.contourf(GX, GY, U_list[fi], levels=14, cmap="RdYlBu_r",
                    alpha=0.85, vmin=-vmax, vmax=vmax)

        # thermal circles from snapshot
        snap = snapshots[idx] if snapshots else None
        if snap:
            for xt, yt, Rt, _ in snap:
                ax.add_patch(plt.Circle(
                    (xt, yt), Rt, fill=False,
                    ls="--", ec="k", lw=0.7, alpha=0.45, zorder=4))

        _draw_domain_rect(ax)

        # trajectory so far
        if idx > 0:
            _add_trajectory(ax, x[:idx+1], y[:idx+1], z[:idx+1], lw=1.6)

        # glider marker + heading arrow
        ax.plot(x[idx], y[idx], "o", color="white", ms=9,
                mec="k", mew=1.2, zorder=7)
        hdg = 160.0
        ax.annotate("", zorder=8,
                    xy=(x[idx] + hdg * np.cos(psi_arr[idx]),
                        y[idx] + hdg * np.sin(psi_arr[idx])),
                    xytext=(x[idx], y[idx]),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=2.0))

        ax.plot(x[0],   y[0],   "go", ms=10, zorder=6)
        ax.plot(GOAL_X, GOAL_Y, "m*",  ms=13, zorder=6)

        t_min = idx * DT / 60
        in_lift = "▲ in lift" if w_arr[idx] > 0.3 else ""
        ax.set_title(f"t = {t_min:.1f} min   alt = {z[idx]:.0f} m   {in_lift}",
                     fontsize=11)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]", fontsize=9)
        ax.set_ylabel("y [m]", fontsize=9)
        ax.grid(True, alpha=0.2)
        return []

    ani = animation.FuncAnimation(
        fig, _render, frames=len(frame_idx),
        blit=False, interval=1000 / fps,
    )

    mp4_path = os.path.join(out_dir, "flight_animation.mp4")
    gif_path = os.path.join(out_dir, "flight_animation.gif")
    try:
        ani.save(mp4_path, writer="ffmpeg", fps=fps, dpi=dpi,
                 extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
        print(f"Saved {mp4_path}")
    except Exception:
        print("ffmpeg not found, saving GIF (larger file) ...")
        ani.save(gif_path, writer="pillow", fps=fps, dpi=dpi)
        print(f"Saved {gif_path}")
    plt.close(fig)


def main(
    results_dir     = RESULTS_DIR,
    model_path      = None,
    trajectory_seed = 12,
    algo            = "sac"
):
    """
    Generate all report figures.

    Parameters
    ----------
    results_dir     : root directory; shared figures (theory, curves) go here.
    model_path      : path to a trained SB3 model (without .zip suffix);
                      if None the trajectory figures are skipped.
    trajectory_seed : seed for the evaluation episode used in trajectory
                      and timeseries figures.
    algo            : algorithm name — used to name the per-model subfolder.
    model_seed      : seed index   — used to name the per-model subfolder.
    """
    os.makedirs(results_dir, exist_ok=True)

    model_name = os.path.basename(model_path) if model_path else "no_model"

    print("Generating theory figures ...")
    plot_maccready_theory(results_dir)

    print("Generating learning curves ...")
    plot_learning_curves_sac(results_dir)
    plot_learning_curves_comparison(results_dir)

    if model_path is not None:
        model_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        print(f"Generating trajectory figures → {model_dir}/")
        model = _load_model(model_path)
        env   = rollout(model, seed=trajectory_seed)
        plot_trajectory_full(env, model_dir)
        plot_trajectory_sample(env, model_dir)
        plot_trajectory_snapshots(env, model_dir)
        plot_episode_timeseries(env, model_dir)
        print("Generating animation ...")
        make_animation(env, model_dir)
    else:
        print("No model_path supplied; skipping trajectory figures.")

    print("Done. All available figures saved to", results_dir)


if __name__ == "__main__":
    main(model_path=os.path.join(RESULTS_DIR, "models/sac_seed0"),
         algo="sac")
