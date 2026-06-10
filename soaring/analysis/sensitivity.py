"""
Sensitivity analysis: four focused experiments to probe
the robustness and generalisation of the learned soaring policy, and to
identify which observation channels are most important for performance.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from soaring.env.soaring_env import SoaringEnv, START_X, START_Y, GOAL_X, GOAL_Y
from soaring.agents.train_sac import train_one, FIELD_KW, STRENGTH_RANGE
from soaring.analysis.maccready_eval import load_model

RESULTS_DIR      = "results/sensitivity"
N_THERMALS_SWEEP = [5, 10, 20, 40, 60, 80]
N_EVAL_EPISODES  = 30

# Training start/goal — must match START_X/Y, GOAL_X/Y in soaring_env.py
_TRAIN_START = (START_X, START_Y)
_TRAIN_GOAL  = (GOAL_X, GOAL_Y)

# Actual training course bearing and distance
_TRAIN_BEARING = np.arctan2(_TRAIN_GOAL[1] - _TRAIN_START[1],
                             _TRAIN_GOAL[0] - _TRAIN_START[0])   # ≈ -20°
_TRAIN_DIST    = float(np.hypot(_TRAIN_GOAL[0] - _TRAIN_START[0],
                                _TRAIN_GOAL[1] - _TRAIN_START[1]))  # ≈ 15 400 m

# Course distance sweep: training bearing, sub- and full-training distances
_DIST_SWEEP = [5000, 8000, 12000, round(_TRAIN_DIST / 100) * 100]   # last ≈ training

# Course heading sweep: fixed 6 km course, courses centred in domain
_DIST_HEADING = 6000.0
_TRAIN_BEARING_DEG = round(np.rad2deg(_TRAIN_BEARING))   # ≈ -20°
_HEADING_SWEEP = {
    "NE 60°":                              np.deg2rad(60),
    "NE 30°":                              np.deg2rad(30),
    "East (0°)":                           np.deg2rad(0),
    f"SE {_TRAIN_BEARING_DEG}° (train)":   _TRAIN_BEARING,
    "SE −30°":                             np.deg2rad(-30),
    "SE −60°":                             np.deg2rad(-60),
}


def _course_in_domain(bearing, dist, domain_x=15000.0, domain_y=7500.0):
    """Centre a course of length `dist` along `bearing` inside the domain."""
    cx, cy = domain_x / 2.0, domain_y / 2.0
    hx, hy = float(np.cos(bearing)), float(np.sin(bearing))
    sx, sy = cx - (dist / 2.0) * hx, cy - (dist / 2.0) * hy
    gx, gy = sx + dist * hx, sy + dist * hy
    return (sx, sy), (gx, gy)


# ---------------------------------------------------------------------------
# Observation ablation
# ---------------------------------------------------------------------------

def obs_configs():
    full = [True] * 9
    def mask(*off):
        m = list(full)
        for i in off:
            m[i] = False
        return m
    return {
        "obs_full":           {"obs_mask": full},
        "obs_no_ema_margin":  {"obs_mask": mask(7, 8)},     # remove EMA + glide margin
        "obs_no_thermal":     {"obs_mask": mask(1, 2, 3)},  # remove thermal nav channels
    }


def _eval_with_mask(model, obs_mask, n_episodes=20, seed=0):
    env = SoaringEnv(
        field_kw={**FIELD_KW, "strength_scale": 1.0},
        obs_mask=obs_mask,
    )
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def run_obs_ablation(seed=0, timesteps=500_000, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    data_dir = os.path.join(results_dir, "data")
    npz_path = os.path.join(data_dir, "obs_ablation.npz")

    # Load fully cached result (mean + std both stored)
    _KEYS = ["obs_full", "obs_no_ema_margin", "obs_no_thermal"]
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        if all(f + "_std" in d.files for f in _KEYS):
            print("  Loading cached obs_ablation results.")
            mean_rewards = {k: float(d[k])          for k in _KEYS}
            std_rewards  = {k: float(d[k + "_std"]) for k in _KEYS}
            return mean_rewards, std_rewards

    mean_rewards = {}
    std_rewards  = {}

    for name, cfg in obs_configs().items():
        omask = cfg["obs_mask"]
        model_zip = os.path.join(results_dir, name, "models", f"sac_seed{seed}.zip")
        if os.path.exists(model_zip):
            print(f"\n  Loading cached model: {name}")
            model = load_model(os.path.join(results_dir, name, "models", f"sac_seed{seed}"))
        else:
            print(f"\n  Training: {name}")
            model, _ = train_one(
                algo="sac", seed=seed, timesteps=timesteps,
                results_dir=os.path.join(results_dir, name),
                obs_mask=omask,
            )
        mean_r, std_r = _eval_with_mask(model, omask, n_episodes=20, seed=seed * 100)
        mean_rewards[name] = mean_r
        std_rewards[name]  = std_r
        print(f"    mean reward = {mean_r:.1f} ± {std_r:.1f}")

    os.makedirs(data_dir, exist_ok=True)
    save_dict = {}
    for k in mean_rewards:
        save_dict[k]          = np.array(mean_rewards[k])
        save_dict[k + "_std"] = np.array(std_rewards[k])
    np.savez(npz_path, **save_dict)
    return mean_rewards, std_rewards


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_COLORS = ["steelblue", "darkorange", "mediumseagreen", "crimson"]

def _decode_str_array(arr):
    """Safely convert a numpy string/bytes array to a plain Python list of str."""
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr.tolist()]

def _cache_valid(d, expected_labels):
    """True if the npz file d contains matching seed_labels."""
    return ("seed_labels" in d.files and
            _decode_str_array(d["seed_labels"]) == list(expected_labels))

def _save_multi(npz_path, x_key, x_vals, labels, all_rates):
    """Save multi-seed sweep results to npz."""
    save = {x_key: np.array(x_vals), "seed_labels": np.array(labels)}
    for i, rates in enumerate(all_rates):
        save[f"rates_{i}"] = np.array(rates)
    np.savez(npz_path, **save)

def _load_multi_n(d, x_key):
    """Load {seed_label: {x_val: rate}} from npz; x_vals are ints."""
    labels = _decode_str_array(d["seed_labels"])
    xs     = [int(v) for v in d[x_key].tolist()]
    return {lbl: dict(zip(xs, d[f"rates_{i}"].tolist()))
            for i, lbl in enumerate(labels)}

def _load_multi_str(d, x_key):
    """Load {seed_label: {str_x_val: rate}} from npz; x_vals are strings."""
    labels = _decode_str_array(d["seed_labels"])
    xs     = _decode_str_array(d[x_key])
    return {lbl: dict(zip(xs, d[f"rates_{i}"].tolist()))
            for i, lbl in enumerate(labels)}


# ---------------------------------------------------------------------------
# N-thermals robustness (eval-only, multi-seed)
# ---------------------------------------------------------------------------

def run_n_thermals_eval(model_paths, labels, n_thermals_list=N_THERMALS_SWEEP,
                        n_episodes=N_EVAL_EPISODES, results_dir=RESULTS_DIR, seed=200):
    os.makedirs(results_dir, exist_ok=True)
    data_dir = os.path.join(results_dir, "data")
    npz_path = os.path.join(data_dir, "n_thermals_eval.npz")

    if os.path.exists(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        if _cache_valid(d, labels):
            print("  Loading cached n_thermals results.")
            return _load_multi_n(d, "n_thermals")

    all_rates = []
    result    = {}
    for path, lbl in zip(model_paths, labels):
        print(f"\n  [{lbl}]")
        model = load_model(path)
        rates = {}
        for n in n_thermals_list:
            fkw = {**FIELD_KW, "n_thermals": n, "strength_scale": 1.0,
                   "lifetime_range": (1500, 3000)}
            env = SoaringEnv(field_kw=fkw)
            successes = 0
            for ep in range(n_episodes):
                obs, _ = env.reset(seed=seed + ep)
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                if info["at_goal"]:
                    successes += 1
            rate = successes / n_episodes
            rates[n] = rate
            print(f"    n={n:3d}: {rate:.0%}")
        result[lbl] = rates
        all_rates.append([rates[n] for n in n_thermals_list])

    os.makedirs(data_dir, exist_ok=True)
    _save_multi(npz_path, "n_thermals", n_thermals_list, labels, all_rates)
    return result


# ---------------------------------------------------------------------------
# Course position sweeps (eval-only, multi-seed)
# ---------------------------------------------------------------------------

def _eval_success_rate(model, start_pos, goal_pos, fkw, n_episodes, seed):
    env = SoaringEnv(field_kw=fkw, start_pos=start_pos, goal_pos=goal_pos)
    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info["at_goal"]:
            successes += 1
    return successes / n_episodes


def run_distance_sweep(model_paths, labels, n_episodes=N_EVAL_EPISODES,
                       results_dir=RESULTS_DIR, seed=300):
    os.makedirs(results_dir, exist_ok=True)
    data_dir = os.path.join(results_dir, "data")
    npz_path = os.path.join(data_dir, "distance_sweep.npz")

    if os.path.exists(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        if _cache_valid(d, labels):
            print("  Loading cached distance_sweep results.")
            return _load_multi_str(d, "dist_labels")

    # Build dist_labels list
    sx, sy = _TRAIN_START
    courses = []
    for dist_m in _DIST_SWEEP:
        goal  = (sx + dist_m * np.cos(_TRAIN_BEARING),
                 sy + dist_m * np.sin(_TRAIN_BEARING))
        label = f"{dist_m/1000:.1f} km" + (" (train)" if abs(dist_m - _TRAIN_DIST) < 200 else "")
        courses.append((label, _TRAIN_START, goal))

    fkw = {**FIELD_KW, "strength_scale": 1.0, "lifetime_range": (1500, 3000)}
    all_rates = []
    result    = {}
    for path, lbl in zip(model_paths, labels):
        print(f"\n  [{lbl}]")
        model = load_model(path)
        rates = {}
        for course_lbl, start, goal in courses:
            rate = _eval_success_rate(model, start, goal, fkw, n_episodes, seed)
            rates[course_lbl] = rate
            print(f"    {course_lbl}: {rate:.0%}")
        result[lbl] = rates
        all_rates.append([rates[cl] for cl, _, _ in courses])

    os.makedirs(data_dir, exist_ok=True)
    dist_labels = [c[0] for c in courses]
    _save_multi(npz_path, "dist_labels", dist_labels, labels, all_rates)
    # dist_labels are strings — overwrite with correct array
    d2 = dict(np.load(npz_path, allow_pickle=True))
    d2["dist_labels"] = np.array(dist_labels)
    np.savez(npz_path, **d2)
    return result


def run_heading_sweep(model_paths, labels, n_episodes=N_EVAL_EPISODES,
                      results_dir=RESULTS_DIR, seed=400):
    os.makedirs(results_dir, exist_ok=True)
    data_dir = os.path.join(results_dir, "data")
    npz_path = os.path.join(data_dir, "heading_sweep.npz")

    if os.path.exists(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        if _cache_valid(d, labels):
            print("  Loading cached heading_sweep results.")
            return _load_multi_str(d, "heading_labels")

    fkw = {**FIELD_KW, "strength_scale": 1.0, "lifetime_range": (1500, 3000)}
    heading_labels = list(_HEADING_SWEEP.keys())

    all_rates = []
    result    = {}
    for path, lbl in zip(model_paths, labels):
        print(f"\n  [{lbl}]")
        model = load_model(path)
        rates = {}
        for hlbl, angle in _HEADING_SWEEP.items():
            start, goal = _course_in_domain(angle, _DIST_HEADING)
            rate = _eval_success_rate(model, start, goal, fkw, n_episodes, seed)
            rates[hlbl] = rate
            print(f"    {hlbl}: {rate:.0%}  "
                  f"start=({start[0]:.0f},{start[1]:.0f})  goal=({goal[0]:.0f},{goal[1]:.0f})")
        result[lbl] = rates
        all_rates.append([rates[hl] for hl in heading_labels])

    os.makedirs(data_dir, exist_ok=True)
    _save_multi(npz_path, "heading_labels", heading_labels, labels, all_rates)
    d2 = dict(np.load(npz_path, allow_pickle=True))
    d2["heading_labels"] = np.array(heading_labels)
    np.savez(npz_path, **d2)
    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_obs_ablation(mean_rewards, std_rewards, out_path):
    """Write a LaTeX table snippet to <out_path stem>_table.tex."""
    display = {
        "obs_full":           r"\texttt{obs\_full} (all 9 channels)",
        "obs_no_ema_margin":  r"\texttt{obs\_no\_ema\_margin} (no EMA, no glide margin)",
        "obs_no_thermal":     r"\texttt{obs\_no\_thermal} (no thermal nav channels)",
    }
    lines = [
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Configuration & Mean reward & Std \\",
        r"\hline",
    ]
    for name in mean_rewards:
        label = display.get(name, r"\texttt{" + name.replace("_", r"\_") + "}")
        m = mean_rewards[name]
        s = std_rewards.get(name, float("nan"))
        s_str = f"{s:.0f}" if not (s != s) else "---"   # NaN check
        lines.append(rf"{label} & {m:.0f} & {s_str} \\")
    lines += [r"\hline", r"\end{tabular}"]

    tex_path = out_path.replace(".pdf", "_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved LaTeX table to {tex_path}")


def plot_n_thermals(success_rates_by_seed, out_path):
    ns = sorted(next(iter(success_rates_by_seed.values())).keys())

    train_parts = [f"{lbl}: {rates.get(40, 0):.0%}"
                   for lbl, rates in success_rates_by_seed.items()]
    train_label = "Training  n=40  (" + ",  ".join(train_parts) + ")"

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (lbl, rates) in enumerate(success_rates_by_seed.items()):
        ax.plot(ns, [rates[n] * 100 for n in ns],
                color=_COLORS[i], marker="o", linewidth=2, markersize=7, label=lbl)
    ax.axvline(40, color="gray", linestyle="--", linewidth=1.2, label=train_label)
    ax.set_xlabel("Number of thermals", fontsize=11)
    ax.set_ylabel("Success rate [%]", fontsize=11)
    ax.set_title("Robustness to thermal density", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_xticks(ns)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_distance_sweep(success_rates_by_seed, out_path):
    first = next(iter(success_rates_by_seed.values()))
    dist_pairs = sorted([(float(lbl.split()[0]), lbl) for lbl in first])
    xs = [p[0] for p in dist_pairs]

    train_parts = []
    for lbl, rates in success_rates_by_seed.items():
        for _, dlbl in dist_pairs:
            if "(train)" in dlbl:
                train_parts.append(f"{lbl}: {rates[dlbl]:.0%}")
    train_km   = next((p[0] for p in dist_pairs if "(train)" in p[1]), None)
    train_label = ("Training dist  " + ",  ".join(train_parts)
                   if train_parts else "Training distance")

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (lbl, rates) in enumerate(success_rates_by_seed.items()):
        ys = [rates[dlbl] * 100 for _, dlbl in dist_pairs]
        ax.plot(xs, ys, color=_COLORS[i], marker="o", linewidth=2,
                markersize=7, label=lbl)
    if train_km is not None:
        ax.axvline(train_km, color="gray", linestyle="--", linewidth=1.2,
                   label=train_label)
    ax.set_xlabel("Course distance [km]", fontsize=11)
    ax.set_ylabel("Success rate [%]", fontsize=11)
    ax.set_title("Generalisation: course distance", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_xticks(xs)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_heading_sweep(success_rates_by_seed, out_path):
    angle_map = {lbl: np.rad2deg(a) for lbl, a in _HEADING_SWEEP.items()}
    first = next(iter(success_rates_by_seed.values()))
    pairs = sorted([(angle_map[lbl], lbl, "(train)" in lbl)
                    for lbl in first if lbl in angle_map])
    xs = [p[0] for p in pairs]

    train_parts = []
    for seed_lbl, rates in success_rates_by_seed.items():
        for _, hlbl, is_train in pairs:
            if is_train:
                train_parts.append(f"{seed_lbl}: {rates[hlbl]:.0%}")
    train_deg   = next((p[0] for p in pairs if p[2]), np.rad2deg(_TRAIN_BEARING))
    train_label = ("Training heading  " + ",  ".join(train_parts)
                   if train_parts else f"Training heading ({train_deg:.0f}°)")

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (seed_lbl, rates) in enumerate(success_rates_by_seed.items()):
        ys = [rates[hlbl] * 100 for _, hlbl, _ in pairs]
        ax.plot(xs, ys, color=_COLORS[i], marker="o", linewidth=2,
                markersize=7, label=seed_lbl)
    ax.axvline(train_deg, color="gray", linestyle="--", linewidth=1.2,
               label=train_label)
    ax.set_xlabel("Course heading [°]", fontsize=11)
    ax.set_ylabel("Success rate [%]", fontsize=11)
    ax.set_title("Generalisation: course heading", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_xticks(xs)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    seed        = 0,
    timesteps   = 500_000,
    results_dir = RESULTS_DIR,
    model_paths = ("results/models/sac_seed0", "results/models/sac_seed1"),
    labels      = ("SAC seed 0", "SAC seed 1"),
    groups      = ("all",),
):
    run_groups = set(groups)
    if "all" in run_groups:
        run_groups = {"obs", "n_thermals", "distance", "heading"}

    os.makedirs(results_dir, exist_ok=True)

    if "obs" in run_groups:
        print("\n=== Observation ablation ===")
        mean_r, std_r = run_obs_ablation(seed=seed, timesteps=timesteps,
                                         results_dir=results_dir)
        plot_obs_ablation(mean_r, std_r,
                          os.path.join(results_dir, "obs_ablation.pdf"))

    if "n_thermals" in run_groups:
        print("\n=== N-thermals robustness (eval-only) ===")
        sr = run_n_thermals_eval(model_paths=model_paths, labels=labels,
                                 results_dir=results_dir)
        plot_n_thermals(sr, os.path.join(results_dir, "n_thermals_robustness.pdf"))

    if "distance" in run_groups:
        print("\n=== Course distance sweep (eval-only) ===")
        sr = run_distance_sweep(model_paths=model_paths, labels=labels,
                                results_dir=results_dir)
        plot_distance_sweep(sr, os.path.join(results_dir, "distance_sweep.pdf"))

    if "heading" in run_groups:
        print("\n=== Course heading sweep (eval-only) ===")
        sr = run_heading_sweep(model_paths=model_paths, labels=labels,
                               results_dir=results_dir)
        plot_heading_sweep(sr, os.path.join(results_dir, "heading_sweep.pdf"))

    print("\nSensitivity analysis complete.")


if __name__ == "__main__":
    main()
