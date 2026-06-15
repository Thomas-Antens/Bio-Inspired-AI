"""
MacCready recovery analysis: compare the learned speed-to-fly curve
against the analytical MacCready optimum.

Procedure
---------
1. Find all trained models for a given algorithm (e.g. all sac_seed*.zip).
2. Evaluate each model at several fixed strength_scale settings.
3. Pool results across seeds: more data per strength point, stronger evidence.
4. Detect glide phases (low bank, between thermals) and log airspeed + climb rate.
5. Plot learned speed vs achieved Mc, overlaid with the analytical curve.

Entry point: main(algo, models_dir, out_dir)
"""

import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from soaring.env.soaring_env import SoaringEnv, DT, PHI_MAX
from soaring.theory.maccready import POLAR
from soaring.agents.train_sac import FIELD_KW

RESULTS_DIR = "results"

# Strength scale settings to sweep (controls expected climb rate)
STRENGTH_SCALES = [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5]

# Glide phase detection thresholds
GLIDE_BANK_DEG  = 10.0          # |phi| < this [deg] -> gliding
GLIDE_MIN_STEPS = 15            # minimum consecutive glide steps to count

N_EVAL_EPISODES = 30            # episodes per strength setting per seed
SEED_BASE       = 100           # evaluation seeds (separate from training)


def detect_phases(traj, bank_threshold_deg=GLIDE_BANK_DEG,
                  min_glide_steps=GLIDE_MIN_STEPS):
    """
    Split a trajectory into glide segments and thermal segments.

    Returns
    -------
    glide_V     : list of airspeed arrays for each glide segment
    climb_rates : mean climb rate for each thermal segment between glides
    """
    phi_arr  = traj["phi"]
    V_arr    = traj["V"]
    w_arr    = traj["w"]
    sink_arr = traj["sink"]

    n = len(phi_arr)
    is_glide = np.abs(phi_arr) < np.deg2rad(bank_threshold_deg)

    glide_segments = []
    thermal_rates  = []
    i = 0
    while i < n:
        if is_glide[i]:
            j = i
            while j < n and is_glide[j]:
                j += 1
            if (j - i) >= min_glide_steps:
                glide_segments.append(V_arr[i:j])
                if i > 0:
                    mask = ~is_glide[:i]
                    if mask.any():
                        net = w_arr[:i][mask] - sink_arr[:i][mask]
                        thermal_rates.append(float(net.mean()))
            i = j
        else:
            i += 1

    return glide_segments, thermal_rates


def evaluate_model(model, strength_scale, n_episodes=N_EVAL_EPISODES,
                   seed_base=SEED_BASE):
    """Run n_episodes and return (achieved_Mc, glide_airspeeds) arrays."""
    fkw = {**FIELD_KW, "strength_scale": strength_scale,
           "lifetime_range": (1500, 3000)}
    env = SoaringEnv(field_kw=fkw)

    achieved_Mc = []
    glide_speeds = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        traj = env.get_trajectory()
        if len(traj) == 0:
            continue

        segs, therm_rates = detect_phases(traj)
        if segs and therm_rates:
            ep_mc = float(np.mean([r for r in therm_rates if r > 0.0] or [0.0]))
            ep_V  = float(np.mean([seg.mean() for seg in segs]))
            achieved_Mc.append(ep_mc)
            glide_speeds.append(ep_V)

    return np.array(achieved_Mc), np.array(glide_speeds)


def load_model(model_path):
    """Try loading as SAC then PPO (seekable-stream workaround for SB3+PyTorch)."""
    import io as _io
    import torch as _th
    _orig = _th.load

    def _seekable_load(f_or_path, *args, **kwargs):
        if hasattr(f_or_path, "read"):
            f_or_path = _io.BytesIO(f_or_path.read())
        kwargs["weights_only"] = False
        return _orig(f_or_path, *args, **kwargs)

    _th.load = _seekable_load
    try:
        try:
            return SAC.load(model_path)
        except Exception:
            return PPO.load(model_path)
    finally:
        _th.load = _orig


def _find_model_paths(algo, models_dir):
    paths = sorted(glob.glob(os.path.join(models_dir, f"{algo}_seed*.zip")))
    # strip .zip — SB3 adds it automatically
    return [p[:-4] for p in paths]


def _plot_maccready(all_Mc, all_V, scales, title, out_dir, fig_name):
    """Shared figure: polar with tangents + learned vs analytical scatter."""
    os.makedirs(out_dir, exist_ok=True)

    # Mc at which V_stf hits V_MAX=45 m/s: Mc = 2*A*V^3 - 2*B/V
    _A, _B, _V_MAX = 2.06e-5, 8.03, 45.0
    Mc_limit = 2 * _A * _V_MAX**3 - 2 * _B / _V_MAX   # ≈ 3.40 m/s

    Mc_fine     = np.linspace(0.0, Mc_limit, 300)
    Mc_tangents = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    fig, (ax_polar, ax_comp) = plt.subplots(1, 2, figsize=(16, 6))
    POLAR.plot_polar(ax_polar, Mc_tangents)

    ax_comp.plot(Mc_fine, POLAR.stf_curve(Mc_fine), "k-", lw=2.5,
                 label="Analytical MacCready", zorder=5)

    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(scales)))
    for sc, Mc_arr, V_arr, col in zip(scales, all_Mc, all_V, colors):
        if len(Mc_arr) == 0:
            continue
        ax_comp.scatter(Mc_arr, V_arr, color=col, s=20, alpha=0.35,
                        label=f"scale={sc:.1f}")
        ax_comp.errorbar(Mc_arr.mean(), V_arr.mean(),
                         xerr=Mc_arr.std(), yerr=V_arr.std(),
                         fmt="o", color=col, ms=8, capsize=4, lw=1.8)

    ax_comp.axhline(_V_MAX, color="crimson", linestyle="--", lw=1.4,
                    label=f"$V_{{max}}$ = {_V_MAX:.0f} m/s (env limit)", zorder=4)

    ax_comp.set_xlabel("Achieved Climb Rate in Thermals  Mc [m/s]", fontsize=12)
    ax_comp.set_ylabel("Glide-phase Commanded Airspeed [m/s]",       fontsize=12)
    ax_comp.set_title(title, fontsize=12)
    ax_comp.legend(fontsize=8, ncol=2)
    ax_comp.grid(True, alpha=0.3)
    ax_comp.set_xlim(0, 3.5)
    ax_comp.set_ylim(15, 50)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, fig_name)
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


def _collect_data(model_paths):
    """Evaluate a list of models and return (all_Mc, all_V, scales, n_seeds)."""
    all_Mc = [[] for _ in STRENGTH_SCALES]
    all_V  = [[] for _ in STRENGTH_SCALES]

    for s_idx, model_path in enumerate(model_paths):
        print(f"\n  Model: {os.path.basename(model_path)}")
        model     = load_model(model_path)
        seed_base = SEED_BASE + s_idx * N_EVAL_EPISODES
        for sc_idx, sc in enumerate(STRENGTH_SCALES):
            print(f"    strength_scale={sc:.2f} ...", end=" ", flush=True)
            Mc_arr, V_arr = evaluate_model(model, sc, seed_base=seed_base)
            all_Mc[sc_idx].append(Mc_arr)
            all_V[sc_idx].append(V_arr)
            if len(Mc_arr):
                print(f"mean Mc={Mc_arr.mean():.2f}  mean V={V_arr.mean():.1f}")
            else:
                print("no glide phases detected")

    all_Mc = [np.concatenate(a) if a else np.array([]) for a in all_Mc]
    all_V  = [np.concatenate(a) if a else np.array([]) for a in all_V]
    return all_Mc, all_V, list(STRENGTH_SCALES), len(model_paths)


def run_single(model_path, out_dir):
    """Evaluate one model; save figure and cache inside out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    data_dir  = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    model_name = os.path.basename(model_path)
    data_path  = os.path.join(data_dir, f"maccready_eval_{model_name}.npz")

    if os.path.exists(data_path):
        print(f"Loading cached data from {data_path}")
        d      = np.load(data_path, allow_pickle=True)
        all_Mc = d["all_Mc"].tolist()
        all_V  = d["all_V"].tolist()
        scales = list(d["scales"])
    else:
        all_Mc, all_V, scales, _ = _collect_data([model_path])
        np.savez(data_path,
                 all_Mc=np.array(all_Mc, dtype=object),
                 all_V=np.array(all_V,   dtype=object),
                 scales=np.array(scales))
        print(f"Saved evaluation data to {data_path}")

    _plot_maccready(all_Mc, all_V, scales,
                    title=f"Learned Speed-to-Fly vs Analytical MacCready  ({model_name})",
                    out_dir=out_dir, fig_name="maccready_comparison.pdf")


def run(algo="sac", models_dir="results/models", out_dir=RESULTS_DIR):
    """Evaluate all seeds for algo, pool results, save in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    data_dir  = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, f"maccready_eval_{algo}.npz")

    if os.path.exists(data_path):
        print(f"Loading cached data from {data_path}")
        d       = np.load(data_path, allow_pickle=True)
        all_Mc  = d["all_Mc"].tolist()
        all_V   = d["all_V"].tolist()
        scales  = list(d["scales"])
        n_seeds = int(d["n_seeds"])
    else:
        model_paths = _find_model_paths(algo, models_dir)
        if not model_paths:
            raise FileNotFoundError(
                f"No {algo}_seed*.zip files found in {models_dir}")
        print(f"Found {len(model_paths)} {algo.upper()} model(s): "
              + ", ".join(os.path.basename(p) for p in model_paths))
        all_Mc, all_V, scales, n_seeds = _collect_data(model_paths)
        np.savez(data_path,
                 all_Mc=np.array(all_Mc, dtype=object),
                 all_V=np.array(all_V,   dtype=object),
                 scales=np.array(scales),
                 n_seeds=np.array(n_seeds))
        print(f"\nSaved evaluation data to {data_path}")

    _plot_maccready(all_Mc, all_V, scales,
                    title=(f"Learned Speed-to-Fly vs Analytical MacCready  "
                           f"({algo.upper()}, {n_seeds} seed(s) pooled)"),
                    out_dir=out_dir,
                    fig_name=f"maccready_comparison_{algo}.pdf")


def main(
    algo        = "sac",
    models_dir  = "results/models",
    out_dir     = RESULTS_DIR,
    model_path  = None,
):
    """
    If model_path is given: evaluate that single model, save in its per-model
    folder (results/{model_name}/).
    Otherwise: pool all seeds for algo and save in out_dir.
    """
    if model_path is not None:
        model_name  = os.path.basename(model_path)
        results_root = os.path.dirname(os.path.dirname(os.path.abspath(model_path)))
        model_out   = os.path.join(results_root, model_name)
        run_single(model_path, out_dir=model_out)
    else:
        run(algo=algo, models_dir=models_dir, out_dir=out_dir)


if __name__ == "__main__":
    # main(model_path = 'results/models/sac_seed0')
    main()
