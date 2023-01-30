"""Contains 3D geometric concepts for proteins."""
from typing import Any, Callable

import numpy as np
import torch
from biotite.structure import (
    annotate_sse,
    AtomArray,
    get_chains,
    get_residue_count,
    index_angle,
    sasa
)

from pp3.utils.constants import SS_LETTER_TO_INDEX
from pp3.utils.pdb import get_pdb_residue_coordinates


CONCEPT_FUNCTION_TYPE = Callable[[AtomArray], Any]
CONCEPT_TO_FUNCTION = {}
CONCEPT_TO_LEVEL = {}


def register_concept(concept_level: str) -> Callable[[CONCEPT_FUNCTION_TYPE], None]:
    """Register a concept function with a specified level."""

    def _register_concept(concept: CONCEPT_FUNCTION_TYPE) -> None:
        """Register a concept function."""
        CONCEPT_TO_FUNCTION[concept.__name__] = concept
        CONCEPT_TO_LEVEL[concept.__name__] = concept_level

    return _register_concept


def get_concept_names() -> list[str]:
    """Get all concept names."""
    return sorted(CONCEPT_TO_FUNCTION)


def get_concept_function(concept: str) -> CONCEPT_FUNCTION_TYPE:
    """Get a concept class by name.

    :param concept: The name of the concept.
    """
    return CONCEPT_TO_FUNCTION[concept]


def get_concept_level(concept: str) -> str:
    """Get the level of a concept.

    :param concept: The name of the concept.
    """
    return CONCEPT_TO_LEVEL[concept]


def compute_all_concepts(structure: AtomArray) -> dict[str, Any]:
    """Compute all concepts for a protein structure.

    :param structure: The protein structure.
    :return: A dictionary of concept names and values.
    """
    return {
        concept_name: concept_function(structure)
        for concept_name, concept_function in CONCEPT_TO_FUNCTION.items()
    }


@register_concept('residue_pair')
def residue_distances(structure: AtomArray) -> torch.Tensor:
    """Get the distances between residue pairs.

    :param structure: The protein structure.
    :return: A PyTorch tensor with the distances between residue pairs.
    """
    # Get residue coordinates
    residue_coordinates = get_pdb_residue_coordinates(structure=structure)

    # Compute pairwise distances
    return torch.cdist(residue_coordinates, residue_coordinates, p=2)


@register_concept('protein')
def protein_sasa(structure: AtomArray) -> float:
    """Get the solvent accessible surface area of a protein.

    :param structure: The protein structure.
    :return: The solvent accessible surface area of the protein.
    """
    return float(np.sum(sasa(structure)))


@register_concept('protein')
def protein_sasa_normalized(structure: AtomArray) -> float:
    """Get the solvent accessible surface area of a protein, normalized by protein length.

    :param structure: The protein structure.
    :return: The solvent accessible surface area of the protein, normalized by protein length
    """
    return protein_sasa(structure) / get_residue_count(structure)


@register_concept('residue')
def residue_sasa(structure: AtomArray) -> torch.Tensor:
    """Get the solvent accessible surface area of all residues.

    :param structure: The protein structure.
    :return: The solvent accessible surface area of all residues.
    """
    return torch.from_numpy(sasa(structure))


# TODO: need to handle multi-class classification concepts
@register_concept('residue')
def secondary_structure(structure: AtomArray) -> torch.Tensor:
    """Get the secondary structure of all residues.

    :param structure: The protein structure.
    :return: The secondary structure of all residues as indices (0 = alpha helix, 1 = beta sheet, 2 = coil).
    """
    # Get secondary structure
    sse = annotate_sse(structure, get_chains(structure)[0])

    # Convert letters to indices
    sse = np.vectorize(SS_LETTER_TO_INDEX.get)(sse)

    return torch.from_numpy(sse)


@register_concept('residue_triplet')
def bond_angles(structure: AtomArray) -> torch.Tensor:
    """Get the angle between residue triplets.

    :param structure: The protein structure.
    :return: A PyTorch tensor with the angles between residue triplets.
    """
    # Get CA indices
    indices = np.arange(0, len(structure))[structure.atom_name == 'CA']

    # Set up index triples
    index = np.stack([indices[:-2], indices[1:-1], indices[2:]]).T

    # Get bond angles
    angles = index_angle(structure, index)

    return torch.from_numpy(angles)
