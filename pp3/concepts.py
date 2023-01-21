"""Contains 3D geometric concepts for proteins."""
from typing import Any, Callable

import torch
from Bio.PDB import Structure
from Bio.PDB.SASA import ShrakeRupley

from pp3.utils.pdb import get_pdb_residue_coordinates


CONCEPT_TYPE = Callable[[Structure], Any]
RESIDUE_ID_TYPE = tuple[str, int, str]
CONCEPT_REGISTRY = {}


def register_concept(concept: CONCEPT_TYPE) -> None:
    """Register a concept function."""
    CONCEPT_REGISTRY[concept.__name__] = concept


def get_concept(concept: str) -> CONCEPT_TYPE:
    """Get a concept class by name.

    :param concept: The name of the concept.
    """
    return CONCEPT_REGISTRY[concept]


def compute_all_concepts(structure: Structure) -> dict[str, Any]:
    """Compute all concepts for a protein structure.

    :param structure: The protein structure.
    :return: A dictionary of concept names and values.
    """
    return {
        concept_name: concept_function(structure)
        for concept_name, concept_function in CONCEPT_REGISTRY
    }


@register_concept
def residue_pair_distances(structure: Structure) -> torch.Tensor:
    """Get the distances between residue pairs.

    :param structure: The protein structure.
    :return: A PyTorch tensor with the distances between residue pairs.
    """
    # Get residue coordinates
    residue_coordinates = get_pdb_residue_coordinates(structure=structure)

    # Compute pairwise distances
    return torch.cdist(residue_coordinates, residue_coordinates, p=2)


@register_concept
def protein_sasa(structure: Structure) -> float:
    """Get the solvent accessible surface area of a protein.

    :param structure: The protein structure.
    :return: The solvent accessible surface area of the protein.
    """
    return ShrakeRupley().compute(structure, level='S')
