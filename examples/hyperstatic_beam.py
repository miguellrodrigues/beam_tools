#!/usr/bin/env python3
"""
Example: Hyperstatic beam with mixed supports, a point load, and a moment.

Demonstrates how to set up a beam, solve reactions (including hyperstatic),
and plot shear-force / bending-moment / slope / deflection diagrams.
"""

import numpy as np
import sympy as sp

try:
    import scienceplots  # noqa: F401 — activates matplotlib styles
except ImportError:
    pass

import matplotlib.pyplot as plt

from beam_tools import (
    Beam,
    Moment,
    PointLoad,
    Support,
    SupportType,
    UniformLoad,
    LoadOrientation,
)


def main():
    # Symbolic parameters
    L, p_1, p_2, w_1, w_2, m_1 = sp.symbols(
        "L p_1 p_2 w_1 w_2 m_1", real=True, positive=True
    )

    # Supports
    s1 = Support(0, SupportType.FIXED, "A")
    s3 = Support(L / 2, SupportType.FIXED, "B")
    s2 = Support(L, SupportType.HINGED, "C")

    # Loads
    p1 = PointLoad(p_1, 0.65*L, LoadOrientation.VERTICAL)
    m1 = Moment(m_1, L)

    # Build beam
    beam = Beam(L, [s1, s2, s3], [m1, p1])

    theta, deflection = beam.get_slope_and_deflection_equations()
    shear_force, bending_moment = beam.get_shear_force_and_bending_moment_equations()

    print("V(x) =", sp.nsimplify(shear_force))
    print("M(x) =", sp.nsimplify(bending_moment))
    print("θ(x) =", sp.nsimplify(theta))
    print("y(x) =", sp.nsimplify(deflection))

    # Numerical substitution
    subs_dict = {
        L: 10,
        p_1: 50_000,
        p_2: -50_000,
        w_1: -1_000,
        w_2: -50_000,
        m_1: 50_000,
        "E": 100_000,
        "I": 1_000,
    }

    for support in beam.supports:
        if isinstance(support.location, sp.Basic):
            support.location = support.location.subs(subs_dict)

    for reaction in beam.reactions:
        beam.reactions[reaction] = beam.reactions[reaction].subs(subs_dict)

    print("\nReactions:", beam.reactions)

    shear_force = shear_force.subs(subs_dict)
    bending_moment = bending_moment.subs(subs_dict)
    theta_num = theta.subs(subs_dict)
    deflection_num = deflection.subs(subs_dict)

    # Lambdify for plotting
    V_fn = sp.lambdify("x", shear_force)
    M_fn = sp.lambdify("x", bending_moment)
    y_fn = sp.lambdify("x", deflection_num)
    theta_fn = sp.lambdify("x", theta_num)

    x = np.linspace(0, 10, 1000)

    V = np.array([V_fn(xi) for xi in x])
    M = np.array([M_fn(xi) for xi in x])
    y = np.array([y_fn(xi) for xi in x]) * 1e3          # → mm
    slope = np.array([theta_fn(xi) for xi in x]) * 180 / np.pi  # → degrees

    # ---- Slope & Deflection plot ----
    try:
        plt.style.use(["science", "grid", "notebook"])
    except OSError:
        pass

    fig, axs = plt.subplots(2, 1, tight_layout=True)

    axs[0].set_title("Slope")
    axs[0].plot(x, slope, "orange", lw=3)
    axs[0].set_ylabel("Degrees")

    max_def_idx = np.argmax(np.abs(y))
    axs[1].set_title("Deflection")
    axs[1].plot([0, 10], [0, 0], "k", lw=3, label="Beam")
    axs[1].plot(
        [x[max_def_idx], x[max_def_idx]],
        [0, y[max_def_idx]],
        "g",
        linestyle="-.",
        label="Max Deflection",
    )
    axs[1].plot(x, y, "-r", lw=3)
    axs[1].set_ylabel("mm")

    for support in beam.supports:
        loc = float(support.location)
        marker_map = {
            SupportType.HINGED: ("go", "Hinged"),
            SupportType.ROLLER: ("bo", "Roller"),
            SupportType.FIXED: ("ko", "Fixed"),
        }
        fmt, label = marker_map[support.support_type]
        axs[1].plot(loc, 0, fmt, ms=10, label=f"{label} support")

    # ---- V & M diagram ----
    V /= 1000.0
    M /= 1000.0

    fig2, (ax_v, ax_m) = plt.subplots(2, 1, tight_layout=True)

    ax_v.plot(x, V, lw=3, color="black")
    ax_v.fill_between(x, V, where=(V > 0), color="skyblue", alpha=0.4, label="V > 0")
    ax_v.fill_between(x, V, where=(V < 0), color="orange", alpha=0.4, label="V < 0")
    ax_v.set_ylabel("V(x) [kN]")
    ax_v.legend()

    ax_m.invert_yaxis()
    ax_m.plot(x, M, lw=3, color="black")
    ax_m.fill_between(x, M, where=(M > 0), color="red", alpha=0.4, label="M > 0")
    ax_m.fill_between(x, M, where=(M < 0), color="green", alpha=0.4, label="M < 0")
    ax_m.set_xlabel("x [m]")
    ax_m.set_ylabel("M(x) [kN·m]")
    ax_m.legend()

    plt.show()


if __name__ == "__main__":
    main()

