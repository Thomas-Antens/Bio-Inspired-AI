"""
Thermal field model: multiple drifting thermals with birth/death lifecycle.

Each thermal drifts with background wind, grows, holds, then decays.
Dead thermals are replaced at random positions inside the domain.
Total updraft uses the Gedeon formula (negative ring at edge included).
"""

import numpy as np


class Thermal:
    """Single drifting thermal with smooth birth/death envelope."""

    BIRTH_FRAC = 0.10   # fraction of lifetime spent growing
    HOLD_END   = 0.70   # fraction at which decay begins (hold = 60 %)
    DECAY_FRAC = 0.30   # fraction of lifetime spent decaying

    def __init__(self, x, y, W, R, lifetime_steps):
        self.x        = float(x)
        self.y        = float(y)
        self.W        = float(W)    # peak strength [m/s]
        self.R        = float(R)    # core radius [m]
        self.lifetime = int(lifetime_steps)
        self.age      = 0

    def _envelope(self):
        phase = self.age / self.lifetime
        if phase < self.BIRTH_FRAC:
            return phase / self.BIRTH_FRAC
        if phase < self.HOLD_END:
            return 1.0
        return max(0.0, (1.0 - phase) / self.DECAY_FRAC)

    @property
    def effective_W(self):
        return self.W * self._envelope()

    def updraft(self, px, py):
        """Gedeon updraft [m/s] at point (px, py). Negative outside the ring."""
        r   = np.hypot(px - self.x, py - self.y)
        rn2 = (r / self.R) ** 2
        return self.W * np.exp(-rn2) * (1.0 - rn2) * self._envelope()

    def step(self, wx, wy, dt):
        """Advance one timestep. Returns True when the thermal dies."""
        self.x   += wx * dt
        self.y   += wy * dt
        self.age += 1
        return self.age >= self.lifetime


class ThermalField:
    """
    N drifting thermals over a rectangular domain.

    Parameters
    ----------
    n_thermals      : number of concurrent thermals
    domain_x/y      : domain size [m]
    wind            : (wx, wy) background wind [m/s]; thermals drift with it
    W_range         : (min, max) peak updraft strength [m/s]
    R_range         : (min, max) core radius [m]
    lifetime_range  : (min, max) lifetime [steps]
    strength_scale  : global multiplier applied to all W values; used to
                      sweep expected climb rate without changing field structure
    turbulence_amp  : std-dev of white-noise added to each updraft sample [m/s]
    anchor          : (x, y, radius) — if given, one thermal is always placed
                      within `radius` metres of (x, y) at initialisation.
                      Guarantees a reachable thermal near the start position.
    """

    def __init__(
        self,
        n_thermals     = 6,
        domain_x       = 15000.0,
        domain_y       = 7500.0,
        wind           = (-5.0, 0.0),
        W_range        = (1.0, 3.0),
        R_range        = (100.0, 200.0),
        lifetime_range = (400, 900),
        strength_scale = 1.0,
        turbulence_amp = 0.05,
        anchor         = None,
        rng            = None,
    ):
        self.n_thermals      = n_thermals
        self.domain_x        = domain_x
        self.domain_y        = domain_y
        self.wind            = np.array(wind, dtype=float)
        self.W_range         = W_range
        self.R_range         = R_range
        self.lifetime_range  = lifetime_range
        self.strength_scale  = strength_scale
        self.turbulence_amp  = turbulence_amp
        self._anchor         = anchor          # (x, y, radius) or None
        self._rng            = rng if rng is not None else np.random.default_rng()
        self.thermals: list[Thermal] = []
        self._init_field()


    def _spawn(self, stagger=False):
        x  = self._rng.uniform(0.0, self.domain_x)
        y  = self._rng.uniform(0.0, self.domain_y)
        W  = self._rng.uniform(*self.W_range) * self.strength_scale
        R  = self._rng.uniform(*self.R_range)
        lt = int(self._rng.uniform(*self.lifetime_range))
        t  = Thermal(x, y, W, R, lt)
        if stagger:
            t.age = int(self._rng.uniform(lt * 0.12, lt * 0.45))
        return t

    def _init_field(self):
        self.thermals = [self._spawn(stagger=True) for _ in range(self.n_thermals)]
        if self._anchor is not None:
            ax, ay, ar = self._anchor
            t      = self.thermals[0]
            angle  = self._rng.uniform(0, 2 * np.pi)
            r      = self._rng.uniform(0, ar)
            t.x    = float(np.clip(ax + r * np.cos(angle), 0, self.domain_x))
            t.y    = float(np.clip(ay + r * np.sin(angle), 0, self.domain_y))


    def reset(self, rng=None):
        """Reinitialise all thermals (call at the start of every episode)."""
        if rng is not None:
            self._rng = rng
        self._init_field()

    def step(self, dt):
        """Advance all thermals one timestep; replace dead ones."""
        new = []
        for t in self.thermals:
            if t.step(self.wind[0], self.wind[1], dt):
                new.append(self._spawn())      # born fresh inside domain
            else:
                new.append(t)
        self.thermals = new


    def updraft_at(self, px, py):
        """Total updraft [m/s] at point (px, py) with turbulence noise."""
        total = sum(t.updraft(px, py) for t in self.thermals)
        if self.turbulence_amp > 0.0:
            total += float(self._rng.normal(0.0, self.turbulence_amp))
        return float(total)

    def updraft_grid(self, gx, gy):
        """
        Updraft over a meshgrid (no turbulence, for visualisation).
        gx, gy: 2-D arrays from np.meshgrid.
        """
        out = np.zeros_like(gx, dtype=float)
        for t in self.thermals:
            r   = np.hypot(gx - t.x, gy - t.y)
            rn2 = (r / t.R) ** 2
            out += t.W * np.exp(-rn2) * (1.0 - rn2) * t._envelope()
        return out


    def get_positions(self):
        """
        Return (xs, ys, W_effs, Rs) arrays for all live thermals.
        W_eff includes the birth/death envelope.
        """
        xs    = np.array([t.x          for t in self.thermals])
        ys    = np.array([t.y          for t in self.thermals])
        W_eff = np.array([t.effective_W for t in self.thermals])
        Rs    = np.array([t.R          for t in self.thermals])
        return xs, ys, W_eff, Rs

    def nearest_thermal(self, px, py):
        """Return (x, y, R) of the thermal whose core is nearest to (px, py)."""
        best_d = np.inf
        best   = self.thermals[0]
        for t in self.thermals:
            d = np.hypot(px - t.x, py - t.y)
            if d < best_d:
                best_d = d
                best   = t
        return best.x, best.y, best.R

    def strongest_thermal(self):
        """Return (x, y, R) of the thermal with highest effective peak strength."""
        best = max(self.thermals, key=lambda t: t.effective_W)
        return best.x, best.y, best.R

    def plot(self, resolution=150, save_path=None):
        """
        Three-panel verification figure.

        Left  : 2-D updraft contour map with thermal centres (dots) and
                core-radius circles (dashed).  Confirms domain layout, Gedeon
                superposition, and the negative ring outside each core.
        Top-right  : radial profile through the strongest thermal.
                     Should show positive peak at r=0, zero crossing near r=R,
                     and a shallow negative wing beyond.
        Bot-right  : birth / hold / decay envelope over a synthetic lifetime.
                     Verifies the BIRTH_FRAC / HOLD_END / DECAY_FRAC constants.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig = plt.figure(figsize=(13, 7))
        gs  = fig.add_gridspec(2, 2, width_ratios=[2, 1], hspace=0.4, wspace=0.3)
        ax_map     = fig.add_subplot(gs[:, 0])
        ax_profile = fig.add_subplot(gs[0, 1])
        ax_life    = fig.add_subplot(gs[1, 1])

        # 2-D updraft map ------------------------------------------------
        xi = np.linspace(0, self.domain_x, resolution)
        yi = np.linspace(0, self.domain_y, resolution)
        gx, gy = np.meshgrid(xi, yi)
        Z = self.updraft_grid(gx, gy)

        vmax = max(abs(Z.max()), abs(Z.min()), 0.1)
        cf = ax_map.contourf(gx, gy, Z, levels=40, cmap="RdBu_r",
                             vmin=-vmax, vmax=vmax)
        fig.colorbar(cf, ax=ax_map, label="Updraft [m/s]", fraction=0.03, pad=0.01)

        xs, ys, W_eff, Rs = self.get_positions()
        for x, y, _, r in zip(xs, ys, W_eff, Rs):
            # ax_map.scatter(x, y, c="k", s=20, zorder=5)
            ax_map.add_patch(mpatches.Circle(
                (x, y), r, fill=False, edgecolor="k",
                linewidth=0.8, linestyle="--"
            ))

        ax_map.set_xlim(0, self.domain_x)
        ax_map.set_ylim(0, self.domain_y)
        ax_map.set_xlabel("x [m]")
        ax_map.set_ylabel("y [m]")
        ax_map.set_title("Updraft field (current snapshot)")
        ax_map.set_aspect("equal")

        # Gedeon radial profile ------------------------------------------
        ref   = max(self.thermals, key=lambda t: t.effective_W)
        r_arr = np.linspace(0, 3.5 * ref.R, 400)
        rn2   = (r_arr / ref.R) ** 2
        w_arr = ref.W * np.exp(-rn2) * (1.0 - rn2) * ref._envelope()

        ax_profile.plot(r_arr, w_arr, color="C0")
        ax_profile.axhline(0, color="k", linewidth=0.6)
        ax_profile.axvline(ref.R, color="grey", linestyle=":", linewidth=0.9,
                           label=f"R = {ref.R:.0f} m")
        ax_profile.set_xlabel("r [m]")
        ax_profile.set_ylabel("Updraft [m/s]")
        ax_profile.set_title("Gedeon profile\n(strongest thermal)")
        ax_profile.legend(fontsize=8)

        # Birth / hold / decay envelope ----------------------------------
        lt   = 1000
        ages = np.arange(lt + 1)
        env  = np.empty(lt + 1)
        for i, age in enumerate(ages):
            phase = age / lt
            if phase < Thermal.BIRTH_FRAC:
                env[i] = phase / Thermal.BIRTH_FRAC
            elif phase < Thermal.HOLD_END:
                env[i] = 1.0
            else:
                env[i] = max(0.0, (1.0 - phase) / Thermal.DECAY_FRAC)

        frac = ages / lt
        ax_life.plot(frac, env, color="C1")
        ax_life.axvspan(0,                Thermal.BIRTH_FRAC, alpha=0.12,
                        color="green", label="birth")
        ax_life.axvspan(Thermal.BIRTH_FRAC, Thermal.HOLD_END, alpha=0.10,
                        color="blue",  label="hold")
        ax_life.axvspan(Thermal.HOLD_END,   1.0,              alpha=0.12,
                        color="red",   label="decay")
        ax_life.set_xlabel("Fractional lifetime")
        ax_life.set_ylabel("Envelope")
        ax_life.set_title("Birth / hold / decay envelope")
        ax_life.set_ylim(-0.05, 1.15)
        ax_life.legend(fontsize=8, loc="upper right")

        fig.suptitle("ThermalField verification", fontsize=13)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)
        return fig
    
if __name__ == "__main__":
    rng = np.random.default_rng(33)
    field = ThermalField(rng=rng)
    field.plot(save_path="results/thermal_field_verification.pdf")
