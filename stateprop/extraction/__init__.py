"""Multi-stage liquid-liquid extraction.

Two-phase countercurrent extraction columns.  Both phases are liquids,
in mutual equilibrium via gamma_i^R x_i^R = gamma_i^E x_i^E
(activity-coefficient equilibrium; no vapor pressures involved).

The solver is structurally similar to the reactive-distillation
Naphtali-Sandholm solver in ``stateprop.reaction.reactive_column``
but with two key differences:

1. **No vapor phase**.  K-values come from gamma ratios only, not from
   gamma * p^sat / p, so pressure has no role.

2. **Two simultaneous activity calls per stage**.  The two phases have
   different compositions, so gamma must be evaluated independently on
   x^R and x^E at the same T.

The standard reference is Treybal "Mass Transfer Operations" (1980),
Ch. 10; or Henley & Seader "Separation Process Principles" Ch. 8.
"""

from .extraction_column import (
    ExtractionColumnResult,
    lle_extraction_column,
)

__all__ = [
    "ExtractionColumnResult",
    "lle_extraction_column",
]
