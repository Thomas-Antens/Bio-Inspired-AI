"""
Reward-structure analysis for the cross-country soaring environment.

Covers the full design rationale in four report-quality figures:

  fig1_fly_through_problem.pdf  -- original fly-through problem and combined fix
  fig2_parameter_sweep.pdf      -- W_thermal bonus sweep + W_prog speed-gradient sweep
  fig3_altitude_cap.pdf         -- proof that alt_scale decay prevents indefinite thermalling
  fig4_reward_structure.pdf     -- final reward landscape (in-thermal + inter-thermal)

Key design decisions reflected:
  - Progress suppressed when w > LIFT_THR (makes circling beat fly-through per-step)
  - Banking bonus W_THERMAL * w * DT when phi > BANK_THR (strongly prefers circling)
  - alt_scale = max(0, 1 - margin/ALT_NORM) applied to (W_alt*dz + bonus):
      provably exits any thermal at finite margin surplus (< ALT_NORM for any w)
  - W_prog = 0.08 creates clear inter-thermal speed gradient (MacCready speed signal)

Usage:
    from soaring.env.reward_analysis import main
    main()   # saves to results/reward_analysis/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from soaring.theory.maccready import POLAR

# ── Parameters (must match soaring_env.py) ───────────────────────────────────
DT           = 0.5
HEADWIND     = 5.0
ALT_NORM     = 500.0
V_CIRCLE     = 19.0
PHI_CIRCLE   = np.deg2rad(30)

W_ALT        = 1.00
W_PROG       = 0.08
W_THERMAL    = 1.00
W_STEP       = -0.20
LIFT_THR     = 0.5

# ── Palette ───────────────────────────────────────────────────────────────────
C = dict(blue="#1f77b4", orange="#ff7f0e", green="#2ca02c",
         red="#d62728", purple="#9467bd", grey="#7f7f7f")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

_SINK_CIRCLE = POLAR.sink_banked(V_CIRCLE, PHI_CIRCLE)   # ~0.61 m/s


# ── Reward helpers ────────────────────────────────────────────────────────────

def _alt_scale(margin):
    return max(0.0, 1.0 - max(0.0, margin) / ALT_NORM)


def _r_circle(w, margin=0.0, W_th=W_THERMAL):
    """Circling (phi=30, V=19) in updraft w. Progress suppressed. Bonus applied."""
    nc    = w - _SINK_CIRCLE
    scale = _alt_scale(margin)
    return (W_ALT * nc * DT + W_th * w * DT) * scale + W_STEP


def _r_through_sup(w, V=25.0, margin=0.0):
    """Straight flight (phi=0) in updraft w, progress suppressed, no bonus."""
    dz    = (w - POLAR.sink(V)) * DT
    scale = _alt_scale(margin)
    return W_ALT * dz * scale + W_STEP


def _r_through_unsup(w, V=25.0):
    """Straight flight in updraft w, progress NOT suppressed (no suppression design)."""
    dz = (w - POLAR.sink(V)) * DT
    return W_ALT * dz + W_PROG * (V - HEADWIND) * DT + W_STEP


def _r_outside(V=35.0, margin=0.0):
    """Outside thermals, flying at V. Progress active. alt_scale on altitude loss."""
    scale = _alt_scale(margin)
    return W_ALT * (-POLAR.sink(V)) * DT * scale + W_PROG * (V - HEADWIND) * DT + W_STEP


def _best_outside_reward(margin=0.0):
    V_grid = np.linspace(18, 50, 300)
    return max(_r_outside(V, margin) for V in V_grid)


def _exit_margin(w):
    """Glide-margin surplus at which r_outside_best first exceeds r_circle."""
    for margin in range(-300, 600):
        if _best_outside_reward(margin) >= _r_circle(w, margin):
            return margin
    return 600   # did not find (very strong thermal)



def fig1_fly_through_problem(out_dir):
    """
    Left:  without suppression the progress term makes flying straight at V=35
           worth ~7x more per step than circling — agent never thermals.
    Right: with suppression + banking bonus circling is strongly preferred.
    """
    w = 2.0
    labels = ["circle\nph=30, V=19", "straight\nV=25", "straight\nV=35", "straight\nV=40"]
    colors = [C["blue"], C["orange"], C["green"], C["red"]]

    def _panel(ax, rewards, title):
        bars = ax.bar(labels, rewards, color=colors, edgecolor="white", linewidth=0.8,
                      zorder=3)
        ax.axhline(0, color="black", linewidth=0.8, zorder=4)
        ax.set_title(title)
        ax.set_ylabel("Per-step reward")
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
        for bar, r in zip(bars, rewards):
            va = "bottom" if r >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    r + 0.0035 * np.sign(r), f"{r:+.3f}",
                    ha="center", va=va, fontsize=8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: original reward (no suppression, old weights)
    W_alt_orig, W_prog_orig = 0.70, 0.06
    r_circ_orig = W_alt_orig*(w-_SINK_CIRCLE)*DT + W_prog_orig*(-HEADWIND*DT) + W_STEP
    rewards_orig = [
        r_circ_orig,
        W_alt_orig*(w-POLAR.sink(25))*DT + W_prog_orig*20*DT*0.5 + W_STEP,
        W_alt_orig*(w-POLAR.sink(35))*DT + W_prog_orig*30*DT*0.5 + W_STEP,
        W_alt_orig*(w-POLAR.sink(40))*DT + W_prog_orig*35*DT*0.5 + W_STEP,
    ]
    _panel(axes[0], rewards_orig,
           "Without Suppression\n(W_alt=0.70, W_prog=0.06, no bonus)")

    # Right: new design (suppression + bonus, margin=0 so alt_scale=1)
    rewards_new = [
        _r_circle(w, margin=0),                 # circle: bonus + suppressed
        _r_through_sup(w, V=25, margin=0),       # straight V=25: suppressed, no bonus
        _r_through_sup(w, V=35, margin=0),       # straight V=35: suppressed, no bonus
        _r_through_sup(w, V=40, margin=0),       # straight V=40: suppressed, no bonus
    ]
    _panel(axes[1], rewards_new,
           "Suppression + Banking Bonus\n"
           f"(W_alt={W_ALT}, W_prog={W_PROG}, W_thermal={W_THERMAL})")

    fig.suptitle(f"Per-step Reward in a {w} m/s Thermal (margin = 0, alt_scale = 1)",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "fig1_fly_through_problem.pdf")


# ── Figure 2: W_thermal bonus sweep + W_prog speed-gradient sweep ─────────────

def fig2_parameter_sweep(out_dir):
    w = 2.0    # representative thermal updraft for bonus sweep
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ── Left: W_thermal sweep ────────────────────────────────────────────────
    ax = axes[0]
    W_th_vals = np.linspace(0, 2.0, 400)

    r_circ    = [(W_ALT*(w-_SINK_CIRCLE)*DT + wt*w*DT) * 1.0 + W_STEP
                 for wt in W_th_vals]                     # suppressed, alt_scale=1
    r_sup     = W_ALT * (w - POLAR.sink(25)) * DT + W_STEP  # suppressed fly-through (constant)
    r_unsup   = _r_through_unsup(w, V=25)                    # unsuppressed fly-through (constant)
    r_out_ref = _r_outside(V=35, margin=0)                   # outside reference (constant)

    ax.plot(W_th_vals, r_circ,  color=C["blue"],   label="circle (suppressed + bonus)")
    ax.axhline(r_sup,    color=C["orange"], linestyle="--",
               label=f"fly-through V=25 (suppressed)  {r_sup:+.3f}")
    ax.axhline(r_unsup,  color=C["red"],    linestyle="-.",
               label=f"fly-through V=25 (unsuppressed) {r_unsup:+.3f}")
    ax.axhline(r_out_ref,color=C["green"],  linestyle=":",
               label=f"outside V=35  {r_out_ref:+.3f}")
    ax.axvline(W_THERMAL, color=C["grey"], linestyle="--", linewidth=1.2,
               label=f"chosen W_thermal = {W_THERMAL}")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("W_thermal  (banking bonus weight)")
    ax.set_ylabel("Per-step reward")
    ax.set_title(f"W_thermal sweep  (w = {w} m/s, alt_scale = 1)")
    ax.legend(framealpha=0.85, loc="upper left", fontsize=8)
    ax.set_xlim(0, 2.0)
    ax.grid(linestyle=":", alpha=0.4)

    # ── Right: W_prog sweep — outside-thermal speed gradient ─────────────────
    ax  = axes[1]
    ax2 = ax.twinx()
    W_prog_vals = np.linspace(0.02, 0.14, 400)

    for V, col, ls in [(25, C["blue"], "-"), (30, C["orange"], "--"),
                       (35, C["green"], "-."), (40, C["red"], ":")]:
        rs = [W_ALT*(-POLAR.sink(V))*DT*1.0 + wp*(V-HEADWIND)*DT + W_STEP
              for wp in W_prog_vals]
        ax.plot(W_prog_vals, rs, color=col, linestyle=ls, label=f"V = {V} m/s")

    # MacCready break-even nc (where circling reward = best outside reward)
    # r_circle at alt_scale=1, w_at_breakeven: solved numerically
    nc_be = []
    for wp in W_prog_vals:
        best_r = max(W_ALT*(-POLAR.sink(V))*DT + wp*(V-HEADWIND)*DT + W_STEP
                     for V in np.linspace(18, 50, 200))
        # circle reward = (W_ALT*nc*DT + W_THERMAL*w_be*DT)*1.0 + W_STEP = best_r
        # approximate: nc ≈ w_be - SINK_CIRCLE, so (W_ALT*(w-SINK_CIRCLE) + W_THERMAL*w)*DT = best_r - W_STEP
        # (W_ALT + W_THERMAL)*w*DT - W_ALT*SINK_CIRCLE*DT = best_r - W_STEP
        nc_breakeven = (best_r - W_STEP) / (W_ALT * DT) - W_THERMAL * 1.5 / W_ALT
        nc_be.append(max(0, nc_breakeven))

    ax2.plot(W_prog_vals, nc_be, color=C["grey"], linewidth=1.5,
             linestyle=(0, (3, 1, 1, 1)), label="approx. break-even nc")
    ax2.set_ylabel("MacCready break-even  nc  [m/s]", color=C["grey"])
    ax2.tick_params(axis="y", labelcolor=C["grey"])

    ax.axvline(W_PROG, color=C["grey"], linestyle="--", linewidth=1.2,
               label=f"chosen W_prog = {W_PROG}")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("W_prog")
    ax.set_ylabel("Outside-thermal per-step reward  (alt_scale = 1)")
    ax.set_title(f"W_prog sweep  (W_alt = {W_ALT}, headwind = {HEADWIND} m/s)")
    ax.set_xlim(W_prog_vals[0], W_prog_vals[-1])
    ax.grid(linestyle=":", alpha=0.4)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, framealpha=0.85, loc="upper left")

    fig.suptitle("Parameter sweeps", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_parameter_sweep.pdf")


# ── Figure 3: altitude cap — proof of finite thermalling ─────────────────────

def fig3_altitude_cap(out_dir):
    """
    Shows that the alt_scale decay to zero guarantees the agent exits any
    thermal at a finite glide-margin surplus, regardless of updraft strength.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Precompute outside reward once for all integer margins — reused everywhere
    # in this figure so _best_outside_reward is called 900× instead of 900×200×.
    _margin_ints  = np.arange(-300, 601)
    _outside_lut  = np.array([_best_outside_reward(m) for m in _margin_ints])

    def _exit_margin_fast(w):
        r_circ = np.array([_r_circle(w, m) for m in _margin_ints])
        idx = np.argmax(_outside_lut >= r_circ)
        return float(_margin_ints[idx]) if _outside_lut[idx] >= r_circ[idx] else 600.0

    def _best_outside_at(m):
        # fast lookup for integer margin; fallback for non-integer (left panel)
        i = int(round(m)) - int(_margin_ints[0])
        if 0 <= i < len(_outside_lut):
            return float(_outside_lut[i])
        return _best_outside_reward(m)

    # ── Left: reward vs margin for circling and flying outside ────────────────
    ax = axes[0]
    margins = np.linspace(-50, ALT_NORM + 50, 400)

    thermals = [(1.5, C["blue"]), (2.0, C["orange"]),
                (3.0, C["green"]), (5.0, C["red"])]

    for w, col in thermals:
        rs = [_r_circle(w, m) for m in margins]
        ax.plot(margins, rs, color=col, label=f"circle  w = {w} m/s")

    r_out = [_best_outside_at(m) for m in margins]
    ax.plot(margins, r_out, color=C["grey"], linewidth=2.0,
            linestyle="--", label="outside  (best V per margin)")

    for w, col in thermals:
        em = _exit_margin_fast(w)
        er = _best_outside_at(em)
        ax.scatter([em], [er], color=col, zorder=5, s=50, marker="x", linewidths=2)

    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color=C["grey"], linewidth=0.6, linestyle=":")
    ax.set_xlabel("Glide-margin surplus  [m]")
    ax.set_ylabel("Per-step reward")
    ax.set_title("Reward vs altitude surplus\n"
                 "(x marks: exit point where outside beats circling)")
    ax.set_xlim(margins[0], margins[-1])
    ax.legend(framealpha=0.85, fontsize=8)
    ax.grid(linestyle=":", alpha=0.4)

    # ── Right: exit margin vs thermal strength ────────────────────────────────
    ax = axes[1]
    w_vals = np.linspace(1.3, 12.0, 200)
    exits  = [_exit_margin_fast(w) if w - _SINK_CIRCLE > 0 else np.nan
              for w in w_vals]

    ax.plot(w_vals, exits, color=C["blue"], linewidth=1.8)
    ax.axhline(ALT_NORM, color=C["grey"], linestyle="--", linewidth=1.0,
               label=f"ALT_NORM = {ALT_NORM:.0f} m  (hard ceiling)")
    ax.fill_between(w_vals, exits, ALT_NORM,
                    alpha=0.08, color=C["grey"], label="unreachable region")

    for w, col in thermals:
        em = _exit_margin_fast(w)
        ax.scatter([w], [em], color=col, zorder=5, s=40)
        ax.annotate(f"{em} m", (w, em), textcoords="offset points",
                    xytext=(5, 4), fontsize=8, color=col)

    ax.set_xlabel("Peak Updraft  w  [m/s]")
    ax.set_ylabel("Exit Margin  [m]")
    ax.set_title("Thermalling Exit Margin vs Thermal Strength\n"
                 "(agent leaves thermal once this gliding surplus is reached)")
    ax.set_xlim(w_vals[0], w_vals[-1])
    ax.axvline(12.0, color=C["red"], linestyle=":", linewidth=0.8,
               label="max possible w (scale=4.0)")
    ax.set_ylim(0, ALT_NORM + 100)
    ax.legend(framealpha=0.85)
    ax.grid(linestyle=":", alpha=0.4)

    fig.suptitle(
        "Altitude Cap Prevents Infinite Thermalling  "
        f"(alt_scale = max(0, 1 - margin / {ALT_NORM:.0f}))",
        fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "fig3_altitude_cap.pdf")


# ── Figure 4: final reward structure ─────────────────────────────────────────

def fig4_reward_structure(out_dir):
    """
    Left:  in-thermal reward vs updraft strength at alt_scale=1 (altitude needed).
           Shows the ~2 m/s MacCready break-even where circling starts to dominate.
    Right: outside-thermal reward vs airspeed. Shows the speed-to-fly gradient.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ── Left: in-thermal reward vs updraft w (alt_scale = 1) ─────────────────
    ax = axes[0]
    w_range = np.linspace(0, 6.0, 2000)

    r_circ       = [_r_circle(w, margin=0)              for w in w_range]
    r_sup25      = [_r_through_sup(w, V=25, margin=0)   for w in w_range]
    r_unsup35    = [_r_through_unsup(w, V=35)            for w in w_range]
    r_out35      = _r_outside(V=35, margin=0)

    ax.plot(w_range, r_circ,    color=C["blue"],   linewidth=1.8,
            label="circle  (phi=30, V=19, suppressed + bonus)")
    ax.plot(w_range, r_sup25,   color=C["orange"],  linestyle="--",
            label="fly-through V=25  (suppressed, no bonus)")
    ax.plot(w_range, r_unsup35, color=C["red"],     linestyle="-.",
            label="fly-through V=35  (no suppression) baseline")
    ax.axhline(r_out35, color=C["grey"], linestyle=":", linewidth=1.2,
               label=f"outside V=35  {r_out35:+.3f}")

    # Break-even 1: circling vs unsuppressed fly-through V=35 (~1.9 m/s)
    # This is the key threshold: above it, circling beats the best alternative
    # the agent could get WITHOUT any suppression at all.
    w_be = next((w for w, rc, ru in zip(w_range, r_circ, r_unsup35) if rc >= ru), None)
    if w_be is not None:
        r_be = _r_circle(w_be, margin=0)
        ax.scatter([w_be], [r_be], color=C["red"], zorder=6, s=60, marker="o")
        ax.annotate(f"break-even\nw = {w_be:.1f} m/s",
                    (w_be, r_be), textcoords="offset points",
                    xytext=(8, -28), fontsize=8, color=C["red"],
                    arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.8))

    # Break-even 2: circling vs flying outside (~0.95 m/s) — thermalling threshold
    w_be2 = next((w for w, rc in zip(w_range, r_circ) if rc >= r_out35), None)
    if w_be2 is not None:
        ax.axvline(w_be2, color=C["blue"], linestyle=":", linewidth=0.9, alpha=0.7)
        ax.text(w_be2 + 0.07, -0.55, f"w={w_be2:.2f}", fontsize=7.5,
                color=C["blue"], va="bottom")

    ax.axvspan(0, LIFT_THR, alpha=0.06, color=C["grey"])
    ax.axvline(LIFT_THR, color=C["grey"], linestyle=":", linewidth=0.8,
               label=f"lift threshold {LIFT_THR} m/s")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Updraft  w  [m/s]")
    ax.set_ylabel("Per-step Reward")
    ax.set_title("In-thermal Reward  (alt_scale = 1)")
    ax.set_xlim(0, 5.0)
    ax.set_ylim(-0.8, 5.0)
    ax.legend(framealpha=0.85, fontsize=8)
    ax.grid(linestyle=":", alpha=0.4)

    # ── Right: outside-thermal reward vs airspeed ─────────────────────────────
    ax = axes[1]
    V_range = np.linspace(18, 52, 500)
    r_out = [_r_outside(V, margin=0) for V in V_range]
    ax.plot(V_range, r_out, color=C["blue"], linewidth=1.8)

    V_opt = V_range[int(np.argmax(r_out))]
    r_opt = max(r_out)
    V_bg  = POLAR.best_glide_speed()

    ax.axvline(V_opt, color=C["green"],  linestyle="--", linewidth=1.2,
               label=f"reward-optimal  V = {V_opt:.0f} m/s")
    ax.axvline(V_bg,  color=C["orange"], linestyle=":",  linewidth=1.2,
               label=f"best-glide  V = {V_bg:.0f} m/s")
    ax.scatter([V_opt], [r_opt], color=C["green"], zorder=5, s=40)
    ax.axhline(0, color="black", linewidth=0.7)

    # Annotate MacCready break-even nc
    exit_at_typical = _exit_margin(2.0)
    scale_exit = _alt_scale(exit_at_typical)
    best_r_at_exit = _best_outside_reward(exit_at_typical)
    nc_at_exit = (best_r_at_exit - W_STEP) / (W_ALT * DT)
    ax.text(0.97, 0.05,
            f"At exit margin ({exit_at_typical} m):\n"
            f"best outside = {best_r_at_exit:+.3f}/step",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color=C["grey"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["grey"], alpha=0.8))

    ax.set_xlabel("Airspeed  V  [m/s]")
    ax.set_ylabel("Per-step Reward  (alt_scale = 1)")
    ax.set_title(f"Outside-thermal Reward\n"
                 f"(W_alt={W_ALT}, W_prog={W_PROG}, headwind={HEADWIND} m/s)")
    ax.set_xlim(18, 52)
    ax.legend(framealpha=0.85)
    ax.grid(linestyle=":", alpha=0.4)

    fig.suptitle(
        f"Final reward structure | W_alt={W_ALT},  W_prog={W_PROG},  "
        f"W_thermal={W_THERMAL},  W_step={W_STEP}",
        fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "fig4_reward_structure.pdf")


# ── Utility ───────────────────────────────────────────────────────────────────

def _save(fig, out_dir, filename):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(out_dir="results/reward_analysis"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Reward analysis — saving figures to {out_dir}/")
    fig1_fly_through_problem(out_dir)
    fig2_parameter_sweep(out_dir)
    fig3_altitude_cap(out_dir)
    fig4_reward_structure(out_dir)


if __name__ == "__main__":
    main()
