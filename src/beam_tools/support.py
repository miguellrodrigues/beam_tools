"""Support types and reaction variable generation for beam analysis."""

from enum import Enum

import sympy as sp


class SupportType(Enum):
    """Enumeration of supported beam support types."""

    FIXED = "fixed"
    ROLLER = "roller"
    HINGED = "hinged"


def generate_support_id(name, location, support_type):
    """Generate a unique identifier for a support."""
    return f"{location}_{support_type}" if name is None else name


class Support:
    """
    Represents a structural support on a beam.

    Parameters
    ----------
    location : float or sympy expression
        Position of the support along the beam.
    support_type : SupportType
        Type of support (FIXED, ROLLER, or HINGED).
    name : str, optional
        A human-readable label for the support (e.g. "A", "B").

    Attributes
    ----------
    reaction_variables : list[sympy.Symbol]
        Symbolic variables representing the unknown reaction forces/moments.
    """

    def __init__(
        self,
        location: float,
        support_type: SupportType,
        name: str = None,
    ):
        self.location = location
        self.support_type = support_type
        self.name = name

        self.support_id = generate_support_id(name, location, support_type)
        self.reaction_variables = self.get_reaction_variables()

    def get_reaction_variables(self):
        """Return the symbolic reaction variables for this support type."""
        sid = self.support_id

        reactions = {
            SupportType.FIXED: [
                sp.symbols(f"{sid}_Ry"),
                sp.symbols(f"{sid}_Rx"),
                sp.symbols(f"{sid}_Mz"),
            ],
            SupportType.ROLLER: [
                sp.symbols(f"{sid}_Ry"),
            ],
            SupportType.HINGED: [
                sp.symbols(f"{sid}_Ry"),
                sp.symbols(f"{sid}_Rx"),
            ],
        }

        return reactions[self.support_type]

