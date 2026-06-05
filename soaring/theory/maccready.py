"""
MacCready speed-to-fly theory from a parameterised glider polar.

The polar form is: sink(V) = A * V**3 + B / V
derived from a parabolic drag polar with physical parameters.

Default parameters correspond to a standard class glider:
  mass 350 kg, wing area 10.5 m^2, AR 23, CD0 0.011,
  Oswald 0.92, rho 1.225 kg/m^3.

This module is the single source of truth for the polar. The environment
imports the GliderPolar class from here so the theory and physics are guaranteed
to use the same coefficients.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

G = 9.81   


class GliderPolar:
    """
    Glider sink polar and MacCready speed-to-fly construction.

    Parameters match the physical parameters of the glider; all derived
    quantities (polar coefficients, characteristic speeds) are computed
    from them in __init__.
    """

    _VERIFY_REFS = [
        ("Best glide speed [m/s]",     lambda p: p.best_glide_speed(),          25.0),
        ("Sink at best glide [m/s]",   lambda p: p.sink(p.best_glide_speed()),  0.64),
        ("Min sink speed [m/s]",       lambda p: p.min_sink_speed(),            19.0),
        ("Speed to fly Mc=1.0 [m/s]", lambda p: p.speed_to_fly(1.0),          33.0),
        ("Speed to fly Mc=2.0 [m/s]", lambda p: p.speed_to_fly(2.0),          38.8),
    ]

    def __init__(
        self,
        mass      = 350.0,
        wing_area = 10.5,
        AR        = 23.0,
        CD0       = 0.011,
        oswald    = 0.92,
        rho       = 1.225,
        g         = G,
    ):
        self.mass      = mass
        self.wing_area = wing_area
        self.AR        = AR
        self.CD0       = CD0
        self.oswald    = oswald
        self.rho       = rho
        self.g         = g

        # Polar coefficients derived from physical parameters
        self.A = CD0 * rho * wing_area / (2.0 * mass * g)
        self.B = 2.0 * mass * g / (np.pi * oswald * AR * rho * wing_area)



    def sink(self, V):
        """
        Level-flight sink rate [m/s, positive = sinking] at airspeed V [m/s].
        V may be a scalar or a numpy array.
        """
        V = np.asarray(V, dtype=float)
        return self.A * V**3 + self.B / V

    def sink_banked(self, V, phi):
        """
        Sink rate in a coordinated banked turn at bank angle phi [rad].
        Derived from the drag polar with load factor n = 1/cos(phi):
          sink_banked = cos(phi)*A*V^3 + B/(V*cos(phi))
        Equals sink(V) at phi=0.
        """
        c = np.cos(phi)
        return c * self.A * V**3 + self.B / (V * c)

    def min_sink_speed(self):
        """Speed of minimum sink [m/s]: (B / (3A))^0.25."""
        return (self.B / (3.0 * self.A)) ** 0.25

    def best_glide_speed(self):
        """Best glide speed = MacCready Mc=0 speed [m/s]: (B/A)^0.25."""
        return (self.B / self.A) ** 0.25



    def speed_to_fly(self, Mc, V_lo=15.0, V_hi=55.0):
        """
        Optimal cruise airspeed [m/s] for expected thermal climb rate Mc [m/s].

        Classical tangent construction: maximise V / (sink(V) + Mc).
        Tangent condition: V * sink'(V) = sink(V) + Mc
        which simplifies to: 2*A*V^4 - Mc*V - 2*B = 0.
        """
        Mc = max(0.0, float(Mc))
        if Mc == 0.0:
            return self.best_glide_speed()

        def residual(V):
            return 2.0 * self.A * V**4 - Mc * V - 2.0 * self.B

        root = brentq(residual, V_lo, V_hi, xtol=1e-6)
        return float(root)  # type: ignore[arg-type]

    def stf_curve(self, Mc_values):
        """Return optimal airspeed for each value in Mc_values."""
        return np.array([self.speed_to_fly(mc) for mc in Mc_values])



    def verify(self, verbose=True, tol=0.2):
        """
        Assert the polar reproduces five reference values within tol [m/s].
        Raises AssertionError if any check fails.
        """
        all_ok = True
        for name, fn, ref in self._VERIFY_REFS:
            val = fn(self)
            ok  = abs(val - ref) < tol
            if not ok:
                all_ok = False
            if verbose:
                tag = "OK  " if ok else "FAIL"
                print(f"  [{tag}] {name}: {val:.3f}  (ref {ref:.3f})")
        assert all_ok, "Polar verification failed. Fix coefficients before continuing."
        if verbose:
            print("Polar verification PASSED.\n")
        return True


    def plot_polar(self, ax, Mc_values=None):
        """
        Draw the sink polar with MacCready tangent lines onto ax.
        Parameters
        ----------
        ax         : matplotlib Axes
        Mc_values  : array-like of Mc values [m/s] for tangent lines;
                     None skips tangents.
        """
        V_arr = np.linspace(15.0, 50.0, 300)

        ax.plot(V_arr, -self.sink(V_arr), "b-", lw=2.5, label="Sink polar")
        ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.35)

        V_bg = self.best_glide_speed()
        V_ms = self.min_sink_speed()
        ax.plot(V_bg, -self.sink(V_bg), "ro", ms=9, zorder=5,
                label=f"Best glide  V={V_bg:.0f} m/s")
        ax.plot(V_ms, -self.sink(V_ms), "gs", ms=9, zorder=5,
                label=f"Min sink    V={V_ms:.0f} m/s")

        if Mc_values is not None:
            cmap = plt.get_cmap("plasma")
            cols = cmap(np.linspace(0.2, 0.8, len(Mc_values)))
            for Mc, col in zip(Mc_values, cols):
                if Mc == 0.0:
                    continue   
                V_opt = self.speed_to_fly(Mc)
                s_opt = self.sink(V_opt)

                slope = (-s_opt - Mc) / V_opt      
                Vl    = np.array([0.0, V_opt * 1.2])
                ax.plot(Vl, Mc + slope * Vl, "--", color=col, alpha=0.7, lw=1.3)

                ax.plot(0, Mc, "v", color=col, ms=7, zorder=6,
                        clip_on=False)
                ax.plot(V_opt, -s_opt, "^", color=col, ms=7, zorder=6,
                        label=f"Mc={Mc:.1f} → V*={V_opt:.0f} m/s")

        ax.set_xlabel("Airspeed V [m/s]", fontsize=11)
        ax.set_ylabel("Vertical speed [m/s]", fontsize=11)
        ax.set_title(
            "Glider polar",
            fontsize=10,
        )
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    def plot_stf_curve(self, ax, Mc_ref=None):
        """
        Draw the speed-to-fly curve onto ax.

        Parameters
        ----------
        ax      : matplotlib Axes
        Mc_ref  : array of reference Mc values to scatter; None skips.
        """
        Mc_fine = np.linspace(0.0, 3.5, 200)
        ax.plot(Mc_fine, self.stf_curve(Mc_fine), "b-", lw=2.5,
                label="Analytical MacCready")
        if Mc_ref is not None:
            ax.scatter(Mc_ref, self.stf_curve(Mc_ref), c="red", s=70,
                       zorder=5, label="Reference points")
        ax.set_xlabel("Expected climb rate Mc [m/s]", fontsize=11)
        ax.set_ylabel("Optimal cruise speed [m/s]",   fontsize=11)
        ax.set_title("MacCready speed-to-fly curve")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    def plot(self, Mc_values=None, save_path=None):
        """
        Combined figure: polar with tangents and speed-to-fly curve.

        Parameters
        ----------
        Mc_values : Mc values for tangent lines and STF scatter; uses
                    [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] by default.
        save_path : if given, save the figure here and return None;
                    otherwise return the Figure object.
        """
        if Mc_values is None:
            Mc_values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        self.plot_polar(ax1, Mc_values)
        self.plot_stf_curve(ax2, Mc_values)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {save_path}")
            return None
        return fig


POLAR   = GliderPolar()

if __name__ == "__main__":
    POLAR.verify()
    POLAR.plot(save_path="results/maccready_curve_theory.pdf")
