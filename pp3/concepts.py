"""Contains 3D geometric concepts for proteins."""
from typing import Any, Callable

import numpy as np
import torch
from Bio.PDB import Structure
from Bio.PDB.SASA import ShrakeRupley

from pp3.utils.pdb import get_pdb_residue_coordinates


CONCEPT_FUNCTION_TYPE = Callable[[Structure], Any]
CONCEPT_TO_FUNCTION = {}
CONCEPT_TO_LEVEL = {}
np.int = np.int32  # Fix for SASA


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


def compute_all_concepts(structure: Structure) -> dict[str, Any]:
    """Compute all concepts for a protein structure.

    :param structure: The protein structure.
    :return: A dictionary of concept names and values.
    """
    return {
        concept_name: concept_function(structure)
        for concept_name, concept_function in CONCEPT_TO_FUNCTION.items()
    }


@register_concept('residue_pair')
def residue_pair_distances(structure: Structure) -> torch.Tensor:
    """Get the distances between residue pairs.

    :param structure: The protein structure.
    :return: A PyTorch tensor with the distances between residue pairs.
    """
    # Get residue coordinates
    residue_coordinates = get_pdb_residue_coordinates(structure=structure)

    # Compute pairwise distances
    return torch.cdist(residue_coordinates, residue_coordinates, p=2)


@register_concept('protein')
def protein_sasa(structure: Structure) -> float:
    """Get the solvent accessible surface area of a protein.

    :param structure: The protein structure.
    :return: The solvent accessible surface area of the protein.
    """
    ShrakeRupley().compute(structure, level='S')

    return float(structure.sasa)


@register_concept('protein')
def protein_sasa_normalized(structure: Structure) -> float:
    """Get the solvent accessible surface area of a protein, normalized by protein length.

    :param structure: The protein structure.
    :return: The solvent accessible surface area of the protein, normalized by protein length
    """
    ShrakeRupley().compute(structure, level='S')

    return float(structure.sasa) / len(list(structure.get_residues()))


@register_concept('residue')
def residue_sasa(structure: Structure) -> torch.Tensor:
    """Get the solvent accessible surface area of all residues.

    :param structure: The protein structure.
    :return: The solvent accessible surface area of all residues.
    """
    ShrakeRupley().compute(structure, level='R')

    return torch.FloatTensor([residue.sasa for residue in structure.get_residues()])


@register_concept('residue_triplet')
def residue_triplet_angles(structure: Structure) -> torch.Tensor:
    """Get the angle between residue triplets.

    :param structure: The protein structure.
    :return: A PyTorch tensor with the angles between residue triplets.
    """
    # Get residue coordinates
    residue_coordinates = get_pdb_residue_coordinates(structure=structure)
    v1 = residue_coordinates[:-2] - residue_coordinates[1:-1]
    v2 = residue_coordinates[2:] - residue_coordinates[1:-1]
    # Compute pairwise angles
    return torch.acos(
        torch.einsum('ij,ij->i', v1, v2) # N x 3, N x 3 -> N
        / torch.norm(v1, dim=1)
        / torch.norm(v2, dim=1)
    )