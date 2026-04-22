"""stateprop.chemdb: optional interface to the `chemicals` library.

This is a convenience module that re-exports the chemicals-library
interface from :mod:`stateprop.cubic.from_chemicals` under shorter names
suitable for a top-level import::

    from stateprop.chemdb import PR_from_name, lookup, cubic_mixture_from_names

    eos = PR_from_name("propane")
    info = lookup("methane")   # {"T_c":..., "p_c":..., "omega":..., "M":...}
    mix = cubic_mixture_from_names(
        ["methane", "ethane", "propane"],
        composition=[0.9, 0.07, 0.03],
        family="pr",
    )

All actual logic lives in :mod:`stateprop.cubic.from_chemicals`.

Install the `chemicals` package (``pip install stateprop[chemdb]``) to
enable lookup of ~26,000 chemicals. Without `chemicals`, a built-in
fallback table covering the 21 GERG-2008 components is used -- enough
for natural-gas workflows.

The name ``chemdb`` comes from "chemical database"; the interface was
added in stateprop v0.8.0.
"""
from __future__ import annotations

from typing import List, Sequence

from .cubic.from_chemicals import (
    lookup_pure_component as lookup,
    cubic_from_name,
    PR_from_name,
    PR78_from_name,
    SRK_from_name,
    RK_from_name,
    VDW_from_name,
    cubic_mixture_from_names,
    chemicals_available,
)
from .cubic.eos import CubicEOS


def components_from_names(
    identifiers: Sequence[str],
    family: str = "pr",
    **eos_kwargs,
) -> List[CubicEOS]:
    """Build a list of per-component CubicEOS instances from identifiers.

    Useful when you want to hold the components yourself rather than
    letting :func:`cubic_mixture_from_names` wrap them into a
    :class:`~stateprop.cubic.CubicMixture`.

    Parameters
    ----------
    identifiers : sequence of str
        Chemical names/formulas/CAS numbers.
    family : str
        Cubic family: "pr" (default), "pr78", "srk", "rk", or "vdw".
    **eos_kwargs
        Passed through to the per-component EOS factory.

    Returns
    -------
    list of CubicEOS, one per identifier, in the same order.

    Examples
    --------
    >>> from stateprop.chemdb import components_from_names
    >>> comps = components_from_names(["methane", "ethane", "propane"])
    >>> [round(c.T_c, 2) for c in comps]
    [190.56, 305.32, 369.82]
    """
    return [cubic_from_name(name, family=family, **eos_kwargs)
            for name in identifiers]


__all__ = [
    "lookup",
    "cubic_from_name",
    "PR_from_name", "PR78_from_name", "SRK_from_name",
    "RK_from_name", "VDW_from_name",
    "cubic_mixture_from_names",
    "components_from_names",
    "chemicals_available",
]
