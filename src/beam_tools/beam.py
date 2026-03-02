"""
Core beam analysis engine.

Solves for reaction forces, shear force, bending moment, slope, and
deflection of statically determinate **and** indeterminate (hyperstatic)
beams using Macaulay singularity functions.
"""

from enum import Enum
from typing import List

import numpy as np
import sympy as sp

from .load import (
    DummyLoad,
    Load,
    LoadOrientation,
    LoadType,
    Moment,
    PointLoad,
    UniformLoad,
)
from .support import Support, SupportType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MomentArm(Enum):
    POSITIVE = 1
    NEGATIVE = -1


def get_moment_arm_sign(load_location, load_value, support_location):
    return (
        MomentArm.POSITIVE
        if (load_location < support_location) == (load_value < 0)
        else MomentArm.NEGATIVE
    )


def moment_area_theorem(x_a, x_b, bending_moment_func=None, gamma=1.0):
    """
    Apply the moment-area theorem between two points.

    Parameters
    ----------
    x_a, x_b : float
        Start and end positions (x_a < x_b).
    bending_moment_func : callable
        Function M(x) returning the bending moment at position x.
    gamma : float
        Flexural rigidity factor (default 1.0).

    Returns
    -------
    list[float]
        [rotation_in_degrees, tangential_deviation_in_mm]
    """
    assert x_a < x_b, "x_a must be less than x_b"
    assert bending_moment_func is not None, "bending_moment_func must be callable"

    x_m = np.linspace(x_a, x_b, 1000)
    M = np.array([bending_moment_func(xi) for xi in x_m])

    m_integral = np.trapezoid(M, x_m)
    x_centroid = np.trapezoid(x_m * M, x_m) / m_integral

    theta_ba = m_integral / gamma
    lc = x_b - x_a
    lf = abs(lc - x_centroid)

    t_ba = lf * theta_ba
    return [float(theta_ba) * (180 / np.pi), 1e3 * float(t_ba)]


# ---------------------------------------------------------------------------
# Equation labelling
# ---------------------------------------------------------------------------

class EquationEnum(Enum):
    SupportMoment = "SupportPositiveMoment"
    VerticalForces = "VerticalForces"
    HorizontalForces = "HorizontalForces"
    PositiveMoment = "PositiveMoment"
    NegativeMoment = "NegativeMoment"
    MomentsEquations = "MomentsEquations"


moment_arm_map_dict = {
    MomentArm.POSITIVE: EquationEnum.PositiveMoment,
    MomentArm.NEGATIVE: EquationEnum.NegativeMoment,
}


# ---------------------------------------------------------------------------
# Beam class
# ---------------------------------------------------------------------------

class Beam:
    """
    Euler–Bernoulli beam model with symbolic analysis.

    Automatically solves reactions, and derives closed-form expressions for
    shear force V(x), bending moment M(x), slope θ(x), and deflection y(x).

    Parameters
    ----------
    length : float or sympy expression
        Total span of the beam.
    supports : list[Support]
        Support conditions.
    loads : list[Load]
        Applied loads and moments.
    E : sympy expression, optional
        Young's modulus (kept symbolic if omitted).
    I : sympy expression, optional
        Second moment of area (kept symbolic if omitted).
    boundary_conditions : dict, optional
        Custom boundary conditions of the form
        ``{"slope": [(x, val), ...], "deflection": [(x, val), ...]}``.
    """

    def __init__(
        self,
        length: float,
        supports: List[Support],
        loads: List[Load],
        E=None,
        I=None,
        boundary_conditions=None,
    ):
        self.length = length
        self.supports = supports
        self.loads = loads
        self.boundary_conditions = boundary_conditions

        if E is None or I is None:
            E, I = sp.symbols("E I")

        self.e = E
        self.i = I

        self.reactions, self.equilibrium_equations, self.is_hyper_static = (
            self.solve_reactions()
        )

        self.shear_force_equation = self.get_shear_force_equation().get_equation()
        self.bending_moment_equation = self.get_bending_moment_equation().get_equation()

        self.slope_equation, self.deflection_equation = (
            self.solve_deflection_equation(boundary_conditions)
        )

        if self.is_hyper_static:
            hyperstatic_solution = self.solve_hyperstatic(
                boundary_conditions if boundary_conditions else self.boundary_conditions
            )

            self.slope_equation = self.slope_equation.subs(hyperstatic_solution)
            self.deflection_equation = self.deflection_equation.subs(
                hyperstatic_solution
            )
            self.shear_force_equation = self.shear_force_equation.subs(
                hyperstatic_solution
            )
            self.bending_moment_equation = self.bending_moment_equation.subs(
                hyperstatic_solution
            )

            for reaction in self.reactions:
                if reaction in hyperstatic_solution:
                    self.reactions[reaction] = hyperstatic_solution[reaction]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_shear_force_and_bending_moment_equations(self):
        """Return ``(V(x), M(x))``."""
        return self.shear_force_equation, self.bending_moment_equation

    def get_slope_and_deflection_equations(self):
        """Return ``(θ(x), y(x))``."""
        return self.slope_equation, self.deflection_equation

    def get_loads(self) -> List[Load]:
        return self.loads

    def get_supports(self) -> List[Support]:
        return self.supports

    # ------------------------------------------------------------------
    # Shear force
    # ------------------------------------------------------------------

    def _add_reaction_loads_to_shear(self, shear_force):
        for support in self.supports:
            if support.location >= self.length:
                continue
            reaction = self.reactions.get(f"{support.name}_Ry")
            if reaction:
                load = PointLoad(reaction, support.location, LoadOrientation.VERTICAL)
                load.n = -1
                load.integrate()
                shear_force += load
        return shear_force

    def _add_loads_to_shear(self, shear_force):
        for load in self.loads:
            if (
                load.get_concentrated_location()
                and load.get_concentrated_location() >= self.length
            ):
                continue
            if load.get_orientation() == LoadOrientation.HORIZONTAL:
                continue
            if load.load_type in (LoadType.POINT, LoadType.UNIFORM, LoadType.TRIANGULAR):
                load_copy = load.copy()
                
                if load.load_type == LoadType.POINT:
                    load_copy.n = -1
                    
                load_copy.integrate()
                shear_force += load_copy
        return shear_force

    def get_shear_force_equation(self):
        shear_force = DummyLoad()
        shear_force = self._add_reaction_loads_to_shear(shear_force)
        shear_force = self._add_loads_to_shear(shear_force)
        return shear_force

    # ------------------------------------------------------------------
    # Bending moment
    # ------------------------------------------------------------------

    def _add_reaction_moments_to_bending(self, bending_moment):
        for support in self.supports:
            if support.location >= self.length:
                continue
            reaction_moment = self.reactions.get(f"{support.name}_Mz")
            if reaction_moment:
                bending_moment += Moment(reaction_moment, support.location)
        return bending_moment

    def _add_load_moments_to_bending(self, bending_moment):
        for load in self.loads:
            if load.load_type == LoadType.MOMENT:
                if load.get_concentrated_location() >= self.length:
                    continue
                bending_moment += load
        return bending_moment

    def get_bending_moment_equation(self):
        shear_force = self.get_shear_force_equation()
        bending_moment = shear_force.copy().integrate()
        bending_moment = self._add_reaction_moments_to_bending(bending_moment)
        bending_moment = self._add_load_moments_to_bending(bending_moment)
        return bending_moment

    # ------------------------------------------------------------------
    # Reactions
    # ------------------------------------------------------------------

    def solve_reactions(self):
        equations_dict = {
            EquationEnum.VerticalForces: 0,
            EquationEnum.HorizontalForces: 0,
            EquationEnum.MomentsEquations: [],
        }

        for load in self.loads:
            if load.load_type == LoadType.MOMENT:
                continue
            orientation = load.get_orientation()
            equivalent_load = load.compute_equivalent_load()

            if orientation == LoadOrientation.VERTICAL:
                equations_dict[EquationEnum.VerticalForces] += equivalent_load
            elif orientation == LoadOrientation.HORIZONTAL:
                equations_dict[EquationEnum.HorizontalForces] += equivalent_load
            else:
                components = load.get_components()
                equations_dict[EquationEnum.HorizontalForces] += components[0]
                equations_dict[EquationEnum.VerticalForces] += components[1]

        for support in self.supports:
            moment_equations_dict = {
                EquationEnum.SupportMoment: 0,
                EquationEnum.PositiveMoment: 0,
                EquationEnum.NegativeMoment: 0,
            }

            other_supports = [s for s in self.supports if s != support]

            for variable in support.reaction_variables:
                identification = str(variable)
                if "Ry" in identification:
                    equations_dict[EquationEnum.VerticalForces] += variable
                elif "Rx" in identification:
                    equations_dict[EquationEnum.HorizontalForces] += variable
                elif "Mz" in identification:
                    moment_equations_dict[EquationEnum.SupportMoment] += variable

            for load in self.loads:
                equivalent_load = load.compute_equivalent_load()
                concentrated_location = load.get_concentrated_location()

                moment_arm_sign = get_moment_arm_sign(
                    load.get_concentrated_location(),
                    equivalent_load,
                    support.location,
                )

                if load.load_type == LoadType.MOMENT:
                    # A pure couple has no moment arm — it enters the
                    # equilibrium equation directly.  We always place it
                    # in NegativeMoment so that the equation
                    #   Pos − Neg = 0  ⟹  Pos − M = 0
                    # correctly requires the reactions to balance M.
                    # Using a fixed bucket avoids evaluating the symbolic
                    # sign (which fails when symbols are positive=True).
                    moment_equations_dict[
                        EquationEnum.NegativeMoment
                    ] += equivalent_load
                else:
                    if load.get_orientation() == LoadOrientation.HORIZONTAL:
                        continue
                    length = abs(concentrated_location - support.location)
                    moment_equations_dict[
                        moment_arm_map_dict[moment_arm_sign]
                    ] += equivalent_load * length

            for other_support in other_supports:
                for reaction_variable in other_support.reaction_variables:
                    identification = str(reaction_variable).split("_")[1]

                    if "Mz" in identification:
                        moment_equations_dict[
                            EquationEnum.SupportMoment
                        ] += reaction_variable
                    else:
                        orientation = (
                            LoadOrientation.HORIZONTAL
                            if "Rx" in identification
                            else LoadOrientation.VERTICAL
                        )
                        if orientation == LoadOrientation.HORIZONTAL:
                            continue
                        moment_arm_sign = get_moment_arm_sign(
                            other_support.location, 1, support.location
                        )
                        length = abs(support.location - other_support.location)
                        moment_equations_dict[
                            moment_arm_map_dict[moment_arm_sign]
                        ] += reaction_variable * length

            equations_dict[EquationEnum.MomentsEquations].append(
                sp.Eq(
                    moment_equations_dict[EquationEnum.PositiveMoment]
                    - moment_equations_dict[EquationEnum.NegativeMoment],
                    moment_equations_dict[EquationEnum.SupportMoment],
                )
            )

        vertical_eq = sp.Eq(equations_dict[EquationEnum.VerticalForces], 0)
        horizontal_eq = sp.Eq(equations_dict[EquationEnum.HorizontalForces], 0)

        all_equations = [vertical_eq, horizontal_eq]
        all_equations.extend(equations_dict[EquationEnum.MomentsEquations])

        all_variables = []
        n_constraints = 0
        constraints = {
            SupportType.FIXED: 4,
            SupportType.HINGED: 2,
            SupportType.ROLLER: 1,
        }

        for support in self.supports:
            all_variables += support.reaction_variables
            n_constraints += constraints[support.support_type]

        if n_constraints > 3:
            return_dict = {}
            for support in self.supports:
                for variable in support.reaction_variables:
                    return_dict[str(variable)] = variable
            return return_dict, all_equations, True

        solution = sp.solve(all_equations, all_variables, dict=True)
        if isinstance(solution, list) and len(solution) > 0:
            solution = solution[0]
        elif isinstance(solution, list):
            solution = {}
        solution = {str(key): value for key, value in solution.items()}
        return solution, all_equations, False

    # ------------------------------------------------------------------
    # Deflection
    # ------------------------------------------------------------------

    def _stipulate_boundary_conditions(self):
        boundary_conditions = {"slope": [], "deflection": []}

        for support in self.supports:
            if support.support_type == SupportType.FIXED:
                boundary_conditions["slope"].append((support.location, 0))
                boundary_conditions["deflection"].append((support.location, 0))
            elif support.support_type in (SupportType.ROLLER, SupportType.HINGED):
                boundary_conditions["deflection"].append((support.location, 0))

        self.boundary_conditions = boundary_conditions
        return boundary_conditions

    def solve_deflection_equation(self, boundary_conditions=None):
        x = sp.symbols("x")
        gamma = 1 / (self.e * self.i)

        bending_moment_equation = self.get_bending_moment_equation()
        C1, C2 = sp.symbols("C_1 C_2")

        theta = bending_moment_equation.integrate()
        deflection = theta.integrate()

        theta_equation = theta.get_equation() + C1
        deflection_equation = deflection.get_equation() + C1 * x + C2

        theta_equation = sp.nsimplify(theta_equation)
        deflection_equation = sp.nsimplify(deflection_equation)

        if not boundary_conditions:
            boundary_conditions = self._stipulate_boundary_conditions()

        slope_bcs = boundary_conditions.get("slope", [])
        deflection_bcs = boundary_conditions.get("deflection", [])

        equations = []
        for bc in slope_bcs:
            equations.append(sp.Eq(theta_equation.subs("x", bc[0]), bc[1]))
        for bc in deflection_bcs:
            equations.append(sp.Eq(deflection_equation.subs("x", bc[0]), bc[1]))

        sol = sp.solve(equations, [C1, C2], dict=True)
        if isinstance(sol, list) and len(sol) > 0:
            sol = sol[0]
        elif isinstance(sol, list):
            sol = {}

        theta_equation = theta_equation.subs(sol)
        deflection_equation = deflection_equation.subs(sol)

        return theta_equation * gamma, deflection_equation * gamma

    def solve_hyperstatic(self, boundary_conditions=None):
        theta, deflection = self.slope_equation, self.deflection_equation

        slope_bcs = boundary_conditions.get("slope", [])
        deflection_bcs = boundary_conditions.get("deflection", [])

        equations = []
        for bc in slope_bcs:
            equations.append(sp.Eq(theta.subs("x", bc[0]), bc[1]))
        for bc in deflection_bcs:
            equations.append(sp.Eq(deflection.subs("x", bc[0]), bc[1]))

        all_variables = []
        for support in self.supports:
            all_variables.extend(support.get_reaction_variables())
        all_variables.extend(sp.symbols("C_1, C_2"))

        equations += self.equilibrium_equations

        solution = sp.solve(equations, all_variables, dict=True)
        if isinstance(solution, list) and len(solution) > 0:
            solution = solution[0]
        elif isinstance(solution, list):
            solution = {}
        solution = {str(key): value for key, value in solution.items()}
        return solution

