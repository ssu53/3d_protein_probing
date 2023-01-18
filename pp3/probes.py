"""Contains 3D geometric probes for proteins."""
from typing import Any, Callable

from Bio.PDB import Structure


PROBE_TYPE = Callable[[Structure], Any]
PROBE_REGISTRY = {}


def register_probe(probe: PROBE_TYPE) -> None:
    """Register a probe function."""
    PROBE_REGISTRY[probe.__name__] = probe


def get_probe(probe: str) -> PROBE_TYPE:
    """Get a probe class by name.

    :param probe: The name of the probe.
    """
    return PROBE_REGISTRY[probe]


@register_probe
def residue_pair_distances(structure: Structure) -> dict[str, dict[str, float]]:
    """Get the distances between residue pairs.

    :param structure: The protein structure.
    :return: The distances between residue pairs as a dictionary of dictionaries.
    """
    residues = [residue for residue in structure.get_residues()]
    breakpoint()
