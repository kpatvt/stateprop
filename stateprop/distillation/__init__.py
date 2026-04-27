"""Multi-stage distillation column solvers.

A non-reactive companion to ``stateprop.reaction.reactive_column``.
The underlying numerics (Wang-Henke / Naphtali-Sandholm with optional
energy balance) are shared with the reactive solver -- this module
just exposes a cleaner API for the standard separation case where no
chemistry is involved.
"""

from .column import (
    distillation_column,
    DistillationColumnResult,
    FeedSpec,
    PumpAround,
    Spec,
)
from stateprop.reaction.reactive_column import SideStripper
from .tray_hydraulics import (
    TrayDesign,
    StageHydraulics,
    TrayHydraulicsResult,
    tray_hydraulics,
    size_tray_diameter,
    flooding_velocity,
)

__all__ = [
    "distillation_column",
    "DistillationColumnResult",
    "FeedSpec",
    "PumpAround",
    "Spec",
    "SideStripper",
    # Tray hydraulics (v0.9.113)
    "TrayDesign",
    "StageHydraulics",
    "TrayHydraulicsResult",
    "tray_hydraulics",
    "size_tray_diameter",
    "flooding_velocity",
]
