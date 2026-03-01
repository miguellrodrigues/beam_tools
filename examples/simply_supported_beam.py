#!/usr/bin/env python3
"""
Example: Simple simply-supported beam with a uniform load.

A minimal example that shows how to define a beam, compute the
structural equations, and print the results.
"""

import sympy as sp

from beam_tools import (
    Beam,
    UniformLoad,
    Support,
    SupportType,
    LoadOrientation,
)


def main():
    L, w = sp.symbols("L w", positive=True)

    # Supports
    A = Support(0, SupportType.HINGED, "A")
    B = Support(L, SupportType.ROLLER, "B")

    # Loads
    q = UniformLoad(w=w, start=0, end=L, orientation=LoadOrientation.VERTICAL)

    # Build beam
    beam = Beam(L, [A, B], [q])

    V, M = beam.get_shear_force_and_bending_moment_equations()
    theta, y = beam.get_slope_and_deflection_equations()

    print("Reactions:", beam.reactions)
    print()
    print("V(x) =", sp.simplify(V))
    print("M(x) =", sp.simplify(M))
    print("θ(x) =", sp.simplify(theta))
    print("y(x) =", sp.simplify(y))


if __name__ == "__main__":
    main()

