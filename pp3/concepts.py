"""Contains 3D geometric concepts for proteins."""
from typing import Any, Callable

import numpy as np
import torch
from biotite.structure import (
    annotate_sse,
    apply_residue_wise,
    AtomArray,
    get_chains,
    get_residue_count,
    filter_backbone,
    index_angle,
    sasa
)
from biotite.structure.info import standardize_order

from pp3.utils.constants import SS_LETTER_TO_INDEX
from pp3.utils.pdb import get_pdb_residue_coordinates


CONCEPT_FUNCTION_TYPE = Callable[[AtomArray], Any]
CONCEPT_TO_FUNCTION = {}
CONCEPT_TO_LEVEL = {}
CONCEPT_TO_TYPE = {}
CONCEPT_TO_OUTPUT_DIM = {}


def register_concept(concept_level: str, concept_type: str, output_dim: int) -> Callable[[CONCEPT_FUNCTION_TYPE], CONCEPT_FUNCTION_TYPE]:
    """Register a concept function with associated characteristics."""

    def _register_concept(concept: CONCEPT_FUNCTION_TYPE) -> CONCEPT_FUNCTION_TYPE:
        """Register a concept function."""
        CONCEPT_TO_FUNCTION[concept.__name__] = concept
        CONCEPT_TO_LEVEL[concept.__name__] = concept_level
        CONCEPT_TO_TYPE[concept.__name__] = concept_type
        CONCEPT_TO_OUTPUT_DIM[concept.__name__] = output_dim

        return concept

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


def get_concept_type(concept: str) -> str:
    """Get the type of a concept.

    :param concept: The name of the concept.
    """
    return CONCEPT_TO_TYPE[concept]


def get_concept_output_dim(concept: str) -> int:
    """Get the output dimension of a concept.

    :param concept: The name of the concept.
    """
    return CONCEPT_TO_OUTPUT_DIM[concept]


def compute_all_concepts(structure: AtomArray) -> dict[str, Any]:
    """Compute all concepts for a protein structure.

    :param structure: The protein structure.
    :return: A dictionary of concept names and values.
    """
    return {
        concept_name: concept_function(structure)
        for concept_name, concept_function in CONCEPT_TO_FUNCTION.items()
    }


def get_backbone_residues(structure: AtomArray) -> AtomArray:
    """Get the backbone atoms of a protein structure in canonical order.

    :param structure: The protein structure.
    :return: The backbone atoms of the protein structure.
    """
    # Standardize atom order
    structure = structure[standardize_order(structure)]

    # Filter to only backbone residues
    structure = structure[filter_backbone(structure)]

    return structure


@register_concept(concept_level='residue_pair', concept_type='regression', output_dim=1)
def residue_distances(structure: AtomArray) -> torch.Tensor:
    """Get the distances between residue pairs.

    :param structure: The protein structure.
    :return: A PyTorch tensor with the distances between residue pairs.
    """
    # Get residue coordinates
    residue_coordinates = get_pdb_residue_coordinates(structure=structure)

    # Compute pairwise distances
    return torch.cdist(residue_coordinates, residue_coordinates, p=2)


@register_concept(concept_level='protein', concept_type='regression', output_dim=1)
def protein_sasa(structure: AtomArray) -> float:
    """Get the solvent accessible surface area of a protein.

    :param structure: The protein structure.
    :return: The solvent accessible surface area of the protein.
    """
    return float(np.nansum(sasa(structure)))


@register_concept(concept_level='protein', concept_type='regression', output_dim=1)
def protein_sasa_normalized(structure: AtomArray) -> float:
    """Get the solvent accessible surface area of a protein, normalized by protein length.

    :param structure: The protein structure.
    :return: The solvent accessible surface area of the protein, normalized by protein length
    """
    return protein_sasa(structure) / get_residue_count(structure)


@register_concept(concept_level='residue', concept_type='regression', output_dim=1)
def residue_sasa(structure: AtomArray) -> torch.Tensor:
    """Get the solvent accessible surface area of all residues.

    :param structure: The protein structure.
    :return: The solvent accessible surface area of all residues.
    """
    atom_sasa = sasa(structure)
    res_sasa = apply_residue_wise(structure, atom_sasa, np.nansum)

    return torch.from_numpy(res_sasa)


# TODO: need to handle multi-class classification concepts
@register_concept(concept_level='residue', concept_type='multi_classification', output_dim=3)
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


@register_concept(concept_level='residue', concept_type='regression', output_dim=1)
def bond_angles(structure: AtomArray, first_last_nan: bool = True) -> torch.Tensor:
    """Get the angle between residue triplets.

    :param structure: The protein structure.
    :param first_last_nan: If True, set the first and last angles to NaN. Otherwise, exclude them (length = N - 2).
    :return: A PyTorch tensor with the angles between residue triplets.
    """
    # Get CA indices
    indices = np.arange(0, len(structure))[structure.atom_name == 'CA']

    # Set up index triples
    index = np.stack([indices[:-2], indices[1:-1], indices[2:]]).T

    # Get bond angles
    angles = index_angle(structure, index)

    # Optionally, set first and last angles to NaN
    if first_last_nan:
        angles = np.concatenate([np.full(1, np.nan), angles, np.full(1, np.nan)])

    return torch.from_numpy(angles)
