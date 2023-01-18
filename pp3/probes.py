"""Contains 3D geometric probes for proteins."""
from itertools import combinations
from typing import Any, Callable

import numpy as np
from Bio.PDB import Structure
from tqdm import tqdm


PROBE_TYPE = Callable[[Structure], Any]
RESIDUE_ID_TYPE = tuple[str, int, str]
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
def residue_pair_distances(pdb_structure: Structure) -> dict[RESIDUE_ID_TYPE, dict[RESIDUE_ID_TYPE, float]]:
    """Get the distances between residue pairs.

    :param pdb_structure: The protein structure.
    :return: The distances between residue pairs as a dictionary of dictionaries.
    """
    # Get all residues
    residues = [residue for residue in pdb_structure.get_residues()]

    # Set up residue distances dictionary
    residue_distances = {residue.get_id(): {} for residue in residues}

    # Compute distances between all residue pairs
    for residue_i, residue_j in tqdm(combinations(residues, 2)):
        # Get residue IDs
        residue_i_id = residue_i.get_id()
        residue_j_id = residue_j.get_id()

        # Get residue coordinates
        residue_i_coord = residue_i['CA'].get_coord()
        residue_j_coord = residue_j['CA'].get_coord()

        # Compute distance between residues
        residue_distance = np.linalg.norm(residue_i_coord - residue_j_coord)

        # Add distance to dictionary
        residue_distances[residue_i_id][residue_j_id] = residue_distances[residue_j_id][residue_i_id] = residue_distance

    return residue_distances
