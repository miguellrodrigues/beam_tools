"""
beam_tools — Symbolic beam structural analysis.

Analyse Euler–Bernoulli beams with singularity (Macaulay) functions
and SymPy.  Computes shear force, bending moment, slope, and deflection
for statically determinate **and** indeterminate (hyperstatic) beams.
"""

from .beam import Beam, moment_area_theorem
from .load import (
    CombinedLoad,
    DummyLoad,
    Load,
    LoadOrientation,
    LoadType,
    Moment,
    PointLoad,
    TriangularLoad,
    UniformLoad,
)
from .support import Support, SupportType

__all__ = [
    # Core
    "Beam",
    "moment_area_theorem",
    # Loads
    "Load",
    "PointLoad",
    "UniformLoad",
    "TriangularLoad",
    "Moment",
    "DummyLoad",
    "CombinedLoad",
    "LoadType",
    "LoadOrientation",
    # Supports
    "Support",
    "SupportType",
]

__version__ = "0.1.0"

