"""Unit-operation models built on stateprop's thermodynamic core.

Modules
-------
* amine_column — counter-current reactive absorber/regenerator for
  CO2 capture in alkanolamine solutions (v0.9.104).
"""
from .amine_column import (
    AmineAbsorber, AmineRegenerator, AmineColumnResult,
)

__all__ = [
    "AmineAbsorber", "AmineRegenerator", "AmineColumnResult",
]
