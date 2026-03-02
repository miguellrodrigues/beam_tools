"""
Load models for structural beam analysis.

Provides point loads, uniform (distributed) loads, triangular loads,
applied moments, and combinable load containers — all expressed
symbolically with SymPy using Macaulay singularity functions.
"""

import copy
from enum import Enum

import sympy as sp

from .singularity import symb_singular_function


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class LoadType(Enum):
    """Classification of load shapes."""

    POINT = "point"
    MOMENT = "moment"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    COMBINED = "combined"


class LoadOrientation(Enum):
    """Direction in which a load acts."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    BOTH = "both"


# ---------------------------------------------------------------------------
# Load Vector
# ---------------------------------------------------------------------------

class LoadVector:
    """
    Magnitude and direction of a load.

    Parameters
    ----------
    magnitude : sympy expression
        Scalar magnitude of the load.
    orientation : LoadOrientation
        Direction the load acts in.
    theta : sympy expression, optional
        Angle (radians) from horizontal — required when *orientation* is BOTH.
    load_type : LoadType
        Shape of the load.
    length : float
        Span of a distributed load (ignored for point loads / moments).
    """

    def __init__(
        self,
        magnitude: sp.Number,
        orientation: LoadOrientation = LoadOrientation.VERTICAL,
        theta: sp.Number = None,
        load_type: LoadType = LoadType.POINT,
        length: float = 1,
    ):
        self.magnitude = magnitude
        self.orientation = orientation
        self.theta = theta
        self.length = length
        self.load_type = load_type

        if orientation == LoadOrientation.BOTH or theta:
            if load_type != LoadType.POINT:
                raise ValueError(
                    "Loads with both orientations are only allowed for point loads"
                )
            if theta is None:
                raise ValueError(
                    "Theta must be provided for loads with both orientations"
                )
            if orientation != LoadOrientation.BOTH:
                raise ValueError(
                    "Orientation must be BOTH for loads with both orientations"
                )

            self.horizontal_component = magnitude * sp.cos(theta)
            self.vertical_component = magnitude * sp.sin(theta)
        else:
            self.horizontal_component = (
                magnitude if orientation == LoadOrientation.HORIZONTAL else 0
            )
            self.vertical_component = (
                magnitude if orientation == LoadOrientation.VERTICAL else 0
            )

        self.force = sp.Matrix([self.horizontal_component, self.vertical_component])

    def get_component(self, orientation: LoadOrientation):
        return {
            LoadOrientation.HORIZONTAL: self.horizontal_component,
            LoadOrientation.VERTICAL: self.vertical_component,
            LoadOrientation.BOTH: sp.sqrt(
                self.horizontal_component ** 2 + self.vertical_component ** 2
            ),
        }[orientation]

    def get_equivalent_load(self):
        component = self.get_component(self.orientation)
        return {
            LoadType.POINT: lambda: component,
            LoadType.MOMENT: lambda: component,
            LoadType.UNIFORM: lambda: component * self.length,
            LoadType.TRIANGULAR: lambda: (component * self.length ** 2) * 0.5,
        }[self.load_type]()

    def get_magnitude(self):
        return self.magnitude

    def get_components(self):
        return self.horizontal_component, self.vertical_component


# ---------------------------------------------------------------------------
# Load Location
# ---------------------------------------------------------------------------

class LoadLocation:
    """Position descriptor for a load along the beam."""

    def __init__(
        self,
        load_type: LoadType = LoadType.POINT,
        location: float = None,
        start: float = None,
        end: float = None,
    ):
        self.load_type = load_type

        if load_type in (LoadType.POINT, LoadType.MOMENT):
            self.location = location
            self.start = None
            self.end = None
        elif load_type in (LoadType.UNIFORM, LoadType.TRIANGULAR):
            self.start = start
            self.end = end
            self.location = None
        else:
            raise ValueError("Invalid load type")

    def get_load_length(self):
        return {
            LoadType.POINT: lambda: 0,
            LoadType.MOMENT: lambda: 0,
            LoadType.UNIFORM: lambda: abs(self.end - self.start),
            LoadType.TRIANGULAR: lambda: abs(self.end - self.start),
        }[self.load_type]()

    def get_concentrated_location(self):
        return {
            LoadType.POINT: lambda: self.location,
            LoadType.MOMENT: lambda: self.location,
            LoadType.UNIFORM: lambda: (self.start + self.end) / 2,
            LoadType.TRIANGULAR: lambda: (2 / 3) * (self.end - self.start),
        }[self.load_type]()


# ---------------------------------------------------------------------------
# Abstract Load base class
# ---------------------------------------------------------------------------

class Load:
    """
    Base class for all load types.

    Subclasses must implement :meth:`compute_for` and :meth:`get_equation`.
    """

    def __init__(self, load_type: LoadType):
        self.load_type = load_type

        self.load_vector: LoadVector = None
        self.load_location: LoadLocation = None

        self.coef = 1.0
        self.n = 0

    def get_orientation(self):
        return self.load_vector.orientation if self.load_vector else None

    def get_concentrated_location(self):
        return self.load_location.get_concentrated_location()

    def get_magnitude(self):
        return self.load_vector.get_magnitude()

    def get_components(self):
        return self.load_vector.get_components()

    def integrate(self):
        self.n += 1.0
        if self.n > 0:
            self.coef /= self.n
        return self

    def differentiate(self):
        self.coef *= self.n
        self.n -= 1.0

    def compute_for(self, x):
        raise NotImplementedError

    def compute_equivalent_load(self):
        return self.load_vector.get_equivalent_load()

    def get_equation(self):
        raise NotImplementedError

    def copy(self):
        return copy.deepcopy(self)

    def __add__(self, other):
        if not isinstance(other, Load):
            raise TypeError("Only instances of Load can be combined.")
        return CombinedLoad(self, other)

    def __str__(self):
        return f"Load({self.load_type})"


# ---------------------------------------------------------------------------
# Concrete Load types
# ---------------------------------------------------------------------------

class DummyLoad(Load):
    """Zero-valued placeholder load."""

    def __init__(self):
        super().__init__(LoadType.POINT)

    def compute_for(self, x):
        return 0

    def compute_equivalent_load(self):
        return 0

    def get_equation(self):
        return 0

    def __str__(self):
        return "DummyLoad"


class CombinedLoad(Load):
    """Container that aggregates multiple :class:`Load` objects."""

    def __init__(self, *loads):
        super().__init__(LoadType.COMBINED)
        self.loads = []
        for load in loads:
            if isinstance(load, CombinedLoad):
                self.loads.extend(load.loads)
            elif isinstance(load, Load):
                self.loads.append(load)
            else:
                raise TypeError("All components must be instances of Load.")

    def compute_for(self, x):
        return sum(load.compute_for(x) for load in self.loads)

    def integrate(self):
        integrated_loads = [load.copy() for load in self.loads]
        for load in integrated_loads:
            load.integrate()
        return CombinedLoad(*integrated_loads)

    def compute_equivalent_load(self):
        return sum(load.compute_equivalent_load() for load in self.loads)

    def get_equation(self) -> sp.Expr:
        loads = [l for l in self.loads if not isinstance(l, DummyLoad)]
        combined = loads[0].get_equation()
        for i in range(1, len(loads)):
            combined += loads[i].get_equation()
        return combined

    def __str__(self):
        loads = [l for l in self.loads if not isinstance(l, DummyLoad)]
        return " + ".join(str(l.get_equation()) for l in loads)


class PointLoad(Load):
    """
    Concentrated force applied at a single point.

    Parameters
    ----------
    magnitude : sympy expression
        Force magnitude (negative = downward by convention).
    location : float or sympy expression
        Position along the beam.
    orientation : LoadOrientation
        Direction of the force.
    theta : sympy expression, optional
        Angle from horizontal (required when *orientation* is BOTH).
    """

    def __init__(
        self,
        magnitude: sp.Number,
        location: float,
        orientation: LoadOrientation = LoadOrientation.VERTICAL,
        theta: sp.Number = None,
    ):
        super().__init__(LoadType.POINT)
        self.load_vector = LoadVector(magnitude, orientation, theta, LoadType.POINT)
        self.load_location = LoadLocation(LoadType.POINT, location)

    def compute_for(self, x):
        return self.get_equation().subs('x', x)

    def get_equation(self) -> sp.Expr:
        return (
            self.coef
            * self.get_magnitude()
            * symb_singular_function(self.get_concentrated_location(), self.n)
        )

    def __str__(self):
        return (
            f"{self.get_magnitude() * self.coef}"
            f"⟨x-({self.get_concentrated_location()})⟩^{self.n}"
        )


class UniformLoad(Load):
    """
    Uniformly distributed load over a span.

    Parameters
    ----------
    w : sympy expression
        Load intensity (force per unit length).
    start, end : float or sympy expression
        Span endpoints.
    orientation : LoadOrientation
        Direction of the distributed force.
    """

    def __init__(
        self,
        w: float,
        start: float,
        end: float,
        orientation: LoadOrientation = LoadOrientation.VERTICAL,
    ):
        super().__init__(LoadType.UNIFORM)
        self.load_location = LoadLocation(
            load_type=LoadType.UNIFORM, start=start, end=end
        )
        self.load_vector = LoadVector(
            magnitude=w,
            orientation=orientation,
            load_type=LoadType.UNIFORM,
            length=self.load_location.get_load_length(),
        )

    def compute_for(self, x):
        return self.get_equation().subs('x', x)

    def get_equation(self) -> sp.Expr:
        return self.coef * self.get_magnitude() * (
            symb_singular_function(self.load_location.start, self.n)
            - symb_singular_function(self.load_location.end, self.n)
        )

    def __str__(self):
        return (
            f"{self.compute_equivalent_load() * self.coef:.2f}"
            f"(⟨x-({self.load_location.start})⟩^{self.n}"
            f" - ⟨x-({self.load_location.end})⟩^{self.n})"
        )


class TriangularLoad(Load):
    """
    Linearly varying (triangular) distributed load.

    Parameters
    ----------
    alpha : sympy expression
        Peak load intensity.
    start, end : float or sympy expression
        Span endpoints.
    orientation : LoadOrientation
        Direction of the distributed force.
    """

    def __init__(
        self,
        alpha: float,
        start: float,
        end: float,
        orientation: LoadOrientation = LoadOrientation.VERTICAL,
    ):
        super().__init__(LoadType.TRIANGULAR)
        self.load_location = LoadLocation(
            load_type=LoadType.TRIANGULAR, start=start, end=end
        )
        self.load_vector = LoadVector(
            magnitude=alpha,
            orientation=orientation,
            load_type=LoadType.TRIANGULAR,
            length=self.load_location.get_load_length(),
        )
        self.n = 1.0

    def compute_for(self, x):
        return self.get_equation().subs('x', x)

    def get_equation(self) -> sp.Expr:
        return self.coef * self.get_magnitude() * (
            symb_singular_function(self.load_location.start, self.n)
            - symb_singular_function(self.load_location.end, self.n)
        )

    def __str__(self):
        return (
            f"{self.get_magnitude() * self.coef:.2f}"
            f"(⟨x-({self.load_location.start})⟩^{int(self.n)}"
            f" - ⟨x-({self.load_location.end})⟩^{int(self.n)})"
        )


class Moment(Load):
    """
    Applied moment (couple) at a point.

    Parameters
    ----------
    load : sympy expression
        Moment magnitude (positive = counter-clockwise).
    location : float or sympy expression
        Position along the beam.
    """

    def __init__(self, load: float, location: float):
        super().__init__(LoadType.MOMENT)
        self.load_vector = LoadVector(magnitude=load, load_type=LoadType.MOMENT)
        self.load_location = LoadLocation(LoadType.MOMENT, location)

    def compute_for(self, x):
        return self.get_equation().subs('x', x)

    def get_equation(self) -> sp.Expr:
        return (
            self.coef
            * self.get_magnitude()
            * symb_singular_function(self.get_concentrated_location(), self.n)
        )

    def __str__(self):
        return (
            f"{self.get_magnitude() * self.coef}"
            f"⟨x-({self.get_concentrated_location()})⟩^{self.n}"
        )

