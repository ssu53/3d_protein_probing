"""Contains 3D geometric concepts for proteins."""
from typing import Any, Callable

import numpy as np
import torch
import einops
from biotite.structure import (
    annotate_sse,
    apply_residue_wise,
    AtomArray,
    get_chains,
    get_residue_count,
    filter_peptide_backbone,
    index_angle,
    index_dihedral,
    residue_iter,
    sasa
)
from biotite.structure.info import standardize_order

from pp3.utils.constants import SS_LETTER_TO_INDEX
from pp3.utils.pdb import get_residue_coordinates


CONCEPT_FUNCTION_TYPE = Callable[[AtomArray], Any]
CONCEPT_TO_FUNCTION = {}
CONCEPT_TO_LEVEL = {
    'solubility': 'protein',
    'enzyme_commission': 'protein',
    'gene_ontology': 'protein',
    'scop': 'protein'
}
CONCEPT_TO_TYPE = {
    'solubility': 'binary_classification',
    'enzyme_commission': 'binary_classification',
    'gene_ontology': 'binary_classification',
    'scop': 'multi_classification'
}
CONCEPT_TO_OUTPUT_DIM = {
    'solubility': 1,
    'enzyme_commission': 10,
    'gene_ontology': 364,
    'scop': 13
}


def register_concept(
        concept_level: str,
        concept_type: str,
        output_dim: int
) -> Callable[[CONCEPT_FUNCTION_TYPE], CONCEPT_FUNCTION_TYPE]:
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
    structure = structure[filter_peptide_backbone(structure)]

    return structure


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


@register_concept(concept_level='residue', concept_type='multi_classification', output_dim=3)
def secondary_structure(structure: AtomArray) -> torch.Tensor:
    """Get the secondary structure of all residues.

    :param structure: The protein structure.
    :return: The secondary structure of all residues as indices (0 = alpha helix, 1 = beta sheet, 2 = coil).
    """
    # Get secondary structure
    sse = annotate_sse(structure)

    # Convert letters to indices
    sse = np.vectorize(SS_LETTER_TO_INDEX.get)(sse)

    return torch.from_numpy(sse)


@register_concept(concept_level='residue_triplet_1', concept_type='regression', output_dim=1)
def bond_angles(structure: AtomArray, residue_distance: int = 1) -> torch.Tensor:
    """Get the angle between residue triplets.

    :param structure: The protein structure.
    :param residue_distance: The distance (# of amino acids) between residues in the triplet.
    :return: A PyTorch tensor with the angles between residue triplets (length N - 2 * residue_distance).
    """
    # Get CA indices
    indices = np.arange(0, len(structure))[structure.atom_name == 'CA']

    # Set up index triples
    index = np.stack([
        indices[:-2 * residue_distance],
        indices[residue_distance:-residue_distance],
        indices[2 * residue_distance:]]
    ).T

    # Get bond angles
    angles = index_angle(structure, index)

    return torch.from_numpy(angles)


@register_concept(concept_level='residue_triplet_24', concept_type='regression', output_dim=1)
def bond_angles_distant(structure: AtomArray) -> torch.Tensor:
    """Get the angle between residue triplets that are 24 AA apart.

    :param structure: The protein structure.
    :return: A PyTorch tensor with the angles between residue triplets (length N - 48).
    """
    return bond_angles(structure, residue_distance=24)


def residue_angles_(
    structure: AtomArray | None = None, 
    residue_coordinates: torch.Tensor | None = None, 
    num_neighbs: int = None,
) -> torch.Tensor:
    """Get angles formed by residue backbone and the num_neighbs nearest neighbors.
    :param structure: The protein structure.
    :param num_neighbs: Number of nearest neighbors.
    :returns: A PyTorch tensor of shape (num_residues, 1 + 2 * num_neighbs) with cosine of angles.
    Angles are: 
        - backbone angle at residue
        - angle between left-adjacent residue and num_neighbs nearest residues
        - angle between right-ajacent residue and num_neighbs nearest residues
    """

    # Get alpha carbon residue coordinates
    if residue_coordinates is None:
        residue_coordinates = get_residue_coordinates(structure=structure)[:, 1, :]
    else:
        residue_coordinates = residue_coordinates[:, 1, :]
    num_residues = residue_coordinates.size(0)
    assert residue_coordinates.shape == (num_residues, 3)


    # Compute pairwise distances

    distances = torch.cdist(residue_coordinates, residue_coordinates, p=2, compute_mode='donot_use_mm_for_euclid_dist')


    # Get argsort indices for the residues in order of distance away

    indices = torch.argsort(distances, dim=1)

    if indices.shape[1] == 0:
        print(f"{residue_coordinates.shape=}")
        print(f"{distances.shape=}")
        print(f"{indices.shape=}")

    # closest residue is self
    assert all(indices[:,0] == torch.arange(len(indices))) # itself is the closest

    # next closest residue is either left or right along backbone
    # assert all((indices[:,1] == torch.arange(len(indices)) -1) | (indices[:,1] == torch.arange(len(indices)) + 1))

    # next closest residue is either left or right along backbone, unless is at either end of protien
    # assert all((indices[1:-1,2] == torch.arange(len(indices)-2)) | (indices[1:-1,2] == torch.arange(len(indices)-2) + 2))

    # exclude self
    indices_top = indices[:,1:num_neighbs+1]


    # Obtain noramlised vectors to nearest neighbours

    indices_top_flat = einops.rearrange(indices_top, "r s -> (r s)")
    residue_coordinates_top = residue_coordinates[indices_top_flat]
    residue_coordinates_top = einops.rearrange(residue_coordinates_top, "(r s) d -> r s d", r=num_residues)
    if indices_top.shape[1] < num_neighbs:
        print(f"Warning: num_neighbs = {num_neighbs} but protein has {num_residues} residues.")
        assert residue_coordinates_top.shape == (num_residues, num_residues-1, 3)
        residue_coordinates_top = torch.cat((residue_coordinates_top, torch.empty(num_residues, num_neighbs-num_residues+1, 3) * torch.nan), dim=1)
    vec_local = residue_coordinates_top - einops.rearrange(residue_coordinates, "r d -> r 1 d")
    vec_local = vec_local / vec_local.norm(p=2, dim=-1, keepdim=True)


    # Obtain normalised backbone vectors. 

    residue_left = residue_coordinates[:-2]    # truncated index (residues at end ignored)
    residue_centre = residue_coordinates[1:-1]
    residue_right = residue_coordinates[2:]

    vec_left = residue_left - residue_centre
    vec_right = residue_right - residue_centre
    vec_left = vec_left / vec_left.norm(p=2, dim=-1, keepdim=True)
    vec_right = vec_right / vec_right.norm(p=2, dim=-1, keepdim=True)

    
    # Get their dot products i.e. cosine angles
    
    angle_backbone = torch.einsum("r d, r d -> r", vec_left, vec_right)
    angle_backbone = einops.rearrange(angle_backbone, "r -> r 1")
    angle_left = torch.einsum("r d, r s d -> r s", vec_left, vec_local[1:-1])
    angle_right = torch.einsum("r d, r s d -> r s", vec_right, vec_local[1:-1])

    angle_local = torch.cat((angle_backbone, angle_left, angle_right), dim=1)
    # angle_local = torch.cat([                   # pad to native index (incl. residues at end)
    #     torch.empty(1,angle_local.shape[-1]) * torch.nan, 
    #     angle_local, 
    #     torch.empty(1,angle_local.shape[-1]) * torch.nan], dim=0)

    assert angle_local.shape == (num_residues-2, 1 + num_neighbs * 2)

    return angle_local


@register_concept(concept_level='residue_triplet_multivariate_1', concept_type='regression', output_dim=17)
def residue_angles_8(structure: AtomArray, residue_coordinates: torch.Tensor) -> torch.Tensor:
    return residue_angles_(structure, residue_coordinates, num_neighbs=8)


@register_concept(concept_level='residue_triplet_multivariate_1', concept_type='regression', output_dim=11)
def residue_angles_5(structure: AtomArray, residue_coordinates: torch.Tensor) -> torch.Tensor:
    return residue_angles_(structure, residue_coordinates, num_neighbs=5)


@register_concept(concept_level='residue_triplet_multivariate_1', concept_type='regression', output_dim=1)
def residue_angles_0(structure: AtomArray, residue_coordinates: torch.Tensor) -> torch.Tensor:
    return residue_angles_(structure, residue_coordinates, num_neighbs=0)


@register_concept(concept_level='residue', concept_type='regression', output_dim=1)
def bond_angles_basic(structure: AtomArray, residue_coordinates: torch.Tensor) -> torch.Tensor:
    """
    Cosine of the bond angle, as single-variate, non-triplet
    """
    residue_bond_angle = residue_angles_(structure, residue_coordinates, num_neighbs=0)
    num_residues = len(residue_bond_angle) + 2
    assert residue_bond_angle.shape == (num_residues-2, 1)
    residue_bond_angle_ = torch.full((num_residues,), torch.nan)
    residue_bond_angle_[1:-1] = residue_bond_angle.squeeze(dim=1)
    return residue_bond_angle_


@register_concept(concept_level='residue', concept_type='regression', output_dim=1)
def foldseek_distance() -> None:
    """
    This is pre-computed from Foldseek. Just reigster the concept here.
    """
    pass


@register_concept(concept_level='residue', concept_type='regression', output_dim=1)
def foldseek_near_seqdist() -> None:
    """
    This is pre-computed from Foldseek. Just reigster the concept here.
    """
    pass


@register_concept(concept_level='residue', concept_type='regression', output_dim=1)
def foldseek_log_seqdist() -> None:
    """
    This is pre-computed from Foldseek. Just reigster the concept here.
    """
    pass


def residue_seqdist_(
    structure: AtomArray | None = None, 
    residue_coordinates: torch.Tensor | None = None, 
    num_neighbs: int = None,
) -> torch.Tensor:
    """
    """

    # Get alpha carbon residue coordinates
    if residue_coordinates is None:
        residue_coordinates = get_residue_coordinates(structure=structure)[:, 1, :]
    else:
        residue_coordinates = residue_coordinates[:, 1, :]
    num_residues = residue_coordinates.size(0)
    assert residue_coordinates.shape == (num_residues, 3)


    # Compute pairwise distances
    distances = torch.cdist(residue_coordinates, residue_coordinates, p=2, compute_mode='donot_use_mm_for_euclid_dist')

    # no pairing with first or last residue
    distances[:, 0] = float('inf')
    # distances[0, :] = float('inf')
    distances[:, -1] = float('inf')
    # distances[-1, :] = float('inf')

    # Get argsort indices for the residues in order of distance away

    indices = torch.argsort(distances, dim=1)
    assert indices.shape == (num_residues, num_residues), indices.shape

    # closest residue is self
    # assert all(indices[:,0] == torch.arange(len(indices))) # itself is the closest

    # exclude self
    indices_top = indices[:,1:num_neighbs+1]

    seq_dist = indices_top - torch.range(start=0, end=num_residues-1, step=1, dtype=torch.int).unsqueeze(1)

    if indices_top.shape[1] < num_neighbs:
        print(f"Warning: num_neighbs = {num_neighbs} but protein has {num_residues} residues.")
        assert seq_dist.shape == (num_residues, num_residues-1)
        seq_dist = torch.cat((seq_dist, torch.empty(num_residues, num_neighbs-num_residues+1) * torch.nan), dim=1)

    return seq_dist


@register_concept(concept_level='residue_multivariate', concept_type='regression', output_dim=8)
def residue_seqdist_8(structure: AtomArray, residue_coordinates: torch.Tensor) -> torch.Tensor:
    return residue_seqdist_(structure, residue_coordinates, num_neighbs=8).to(torch.float32)


@register_concept(concept_level='residue_multivariate', concept_type='regression', output_dim=5)
def residue_seqdist_5(structure: AtomArray, residue_coordinates: torch.Tensor) -> torch.Tensor:
    return residue_seqdist_(structure, residue_coordinates, num_neighbs=5).to(torch.float32)


@register_concept(concept_level='residue_multivariate', concept_type='regression', output_dim=1)
def residue_seqdist_1(structure: AtomArray, residue_coordinates: torch.Tensor) -> torch.Tensor:
    return residue_seqdist_(structure, residue_coordinates, num_neighbs=1).to(torch.float32)


def residue_logseqdist_(
    structure: AtomArray | None = None, 
    residue_coordinates: torch.Tensor | None = None, 
    num_neighbs: int = None,
) -> torch.Tensor:

    seq_dist = residue_seqdist_(structure, residue_coordinates, num_neighbs)
    seq_dist = torch.sign(seq_dist) * torch.log(torch.abs(seq_dist) + 1)
    
    # valid_mask = torch.full((len(residue_coordinates),), True)
    # valid_mask[0] = False
    # valid_mask[-1] = False
    # from pp3.utils.foldseek_parsers import find_nearest_residues
    # partner_idx = find_nearest_residues(residue_coordinates.numpy(), valid_mask.numpy())
    # seq_dist = (partner_idx - np.arange(len(partner_idx)))
    # seq_dist = torch.tensor(seq_dist, dtype=torch.float32)
    # seq_dist = torch.sign(seq_dist) * torch.log(torch.abs(seq_dist) + 1)

    return seq_dist


@register_concept(concept_level='residue_multivariate', concept_type='regression', output_dim=8)
def residue_logseqdist_8(structure: AtomArray, residue_coordinates: torch.Tensor) -> torch.Tensor:
    return residue_logseqdist_(structure, residue_coordinates, num_neighbs=8)


@register_concept(concept_level='residue_multivariate', concept_type='regression', output_dim=5)
def residue_logseqdist_5(structure: AtomArray, residue_coordinates: torch.Tensor) -> torch.Tensor:
    return residue_logseqdist_(structure, residue_coordinates, num_neighbs=5)


@register_concept(concept_level='residue_multivariate', concept_type='regression', output_dim=1)
def residue_logseqdist_1(structure: AtomArray, residue_coordinates: torch.Tensor) -> torch.Tensor:
    return residue_logseqdist_(structure, residue_coordinates, num_neighbs=1)


@register_concept(concept_level='residue_quadruplet_1', concept_type='regression', output_dim=1)
def dihedral_angles(structure: AtomArray, residue_distance: int = 1) -> torch.Tensor:
    """Get the dihedral angles between residue quadruplets.

    :param structure: The protein structure.
    :param residue_distance: The distance (# of amino acids) between residues in the quadruplet.
    :return: A PyTorch tensor with the dihedral angles between residue quadruplets (length N - 3 * residue_distance).
    """
    # Get CA indices
    indices = np.arange(0, len(structure))[structure.atom_name == 'CA']

    # Set up index quadruples
    index = np.stack([
        indices[:-3 * residue_distance],
        indices[residue_distance:-2 * residue_distance],
        indices[2 * residue_distance:-residue_distance],
        indices[3 * residue_distance:]]
    ).T

    # Get dihedral angles
    angles = index_dihedral(structure, index)

    return torch.from_numpy(angles)


@register_concept(concept_level='residue_quadruplet_24', concept_type='regression', output_dim=1)
def dihedral_angles_distant(structure: AtomArray) -> torch.Tensor:
    """Get the dihedral angles between residue quadruplets that are 24 AA apart.

    :param structure: The protein structure.
    :return: A PyTorch tensor with the dihedral angles between residue quadruplets (length N - 72).
    """
    return dihedral_angles(structure, residue_distance=24)


def residue_neighb_distances_(
    structure: AtomArray | None,
    residue_coordinates: torch.Tensor | None = None,
    num_neighbs: int | None = None,
) -> torch.Tensor:
    """Get the distance between alpha carbons of each residue and its num_neighbs nearest neighbors, per residue.

    :param structure: The protein structure.
    :param num_neighbs: The number of nearest neighbors.
    :return: A PyTorch tensor of shape (num_residues, num_neighbs) with the distances.
    """

    # Get alpha carbon residue coordinates
    if residue_coordinates is None:
        residue_coordinates = get_residue_coordinates(structure=structure)[:, 1, :]
    else:
        residue_coordinates = residue_coordinates[:, 1, :]
    num_residues = residue_coordinates.size(0)
    assert residue_coordinates.shape == (num_residues, 3)

    # Compute pairwise distances
    distances = torch.cdist(residue_coordinates, residue_coordinates, p=2, compute_mode='donot_use_mm_for_euclid_dist')

    # Get argsort indices for the residues in order of distance away
    distances_sorted, _ = torch.sort(distances, dim=1)
    distances_sorted = distances_sorted[:,1:num_neighbs+1]
    distances_top = torch.empty((num_residues, num_neighbs)) * torch.nan
    if distances_sorted.size(1) < num_neighbs: print(f"Warning: num_neighbs = {num_neighbs} but protein has {num_residues} residues.")
    distances_top[:,:distances_sorted.size(1)] = distances_sorted

    return distances_top


@register_concept(concept_level='residue_multivariate', concept_type='regression', output_dim=16)
def residue_neighb_distances_16(structure: AtomArray, residue_coordinates: torch.Tensor) -> torch.Tensor:
    return residue_neighb_distances_(structure, residue_coordinates, num_neighbs=16)


@register_concept(concept_level='residue_multivariate', concept_type='regression', output_dim=8)
def residue_neighb_distances_8(structure: AtomArray, residue_coordinates: torch.Tensor) -> torch.Tensor:
    return residue_neighb_distances_(structure, residue_coordinates, num_neighbs=8)


@register_concept(concept_level='residue_multivariate', concept_type='regression', output_dim=5)
def residue_neighb_distances_5(structure: AtomArray, residue_coordinates: torch.Tensor) -> torch.Tensor:
    return residue_neighb_distances_(structure, residue_coordinates, num_neighbs=5)


@register_concept(concept_level='residue_pair', concept_type='regression', output_dim=1)
def residue_distances(
        structure: AtomArray,
        max_distance: float | None = 25.0
) -> torch.Tensor:
    """Get the distances between the alpha carbons of residue pairs within a maximum distance window.

    :param structure: The protein structure.
    :param max_distance: The maximum distance between residue pairs (in Angstroms) to include (NaN otherwise).
    :return: A PyTorch tensor with the distances between the alpha carbons of residue pairs.
    """
    # Get alpha carbon residue coordinates
    # residue_coordinates = get_residue_coordinates(structure=structure)[:, :, 1] # bug!!!
    residue_coordinates = get_residue_coordinates(structure=structure)[:, 1, :]

    # Compute pairwise distances
    distances = torch.cdist(residue_coordinates, residue_coordinates, p=2, compute_mode='donot_use_mm_for_euclid_dist')

    # Set self distances to NaN
    distances[torch.eye(distances.shape[0], dtype=torch.bool)] = torch.nan

    # Set distances above max_distance to NaN
    if max_distance is not None:
        distances[distances > max_distance] = torch.nan

    return distances


@register_concept(concept_level='residue_pair', concept_type='regression', output_dim=1)
def residue_distances_all(structure: AtomArray) -> torch.Tensor:
    """Get the distances between the alpha carbons of all residue pairs.

    :param structure: The protein structure.
    :return: A PyTorch tensor with the distances between the alpha carbons of all residue pairs.
    """
    return residue_distances(structure=structure, max_distance=None)


@register_concept(concept_level='residue', concept_type='regression', output_dim=1)
def residue_distances_by_residue(
        structure: AtomArray,
        max_distance: float | None = 25.0
) -> torch.Tensor:
    """Get the average distance from a residue to all other residues (using alpha carbons) within a maximum distance window.

    :param structure: The protein structure.
    :param max_distance: The maximum distance between residue pairs (in Angstroms) to include.
    :return: A PyTorch tensor with average distance from each residue to all other residues (using alpha carbons) within max_distance.
    """
    # Get residue distances
    distances = residue_distances(
        structure=structure,
        max_distance=max_distance
    )

    # Get average distance from each residue to all other residues
    distances = torch.nanmean(distances, dim=1)

    return distances


@register_concept(concept_level='residue_pair', concept_type='regression', output_dim=1)
def residue_distances_8(structure: AtomArray) -> torch.Tensor:
    """Get the distances between the alpha carbons of all residue pairs.

    :param structure: The protein structure.
    :return: A PyTorch tensor with the distances between the alpha carbons of all residue pairs.
    """
    return residue_distances(structure=structure, max_distance=8.)


@register_concept(concept_level='residue', concept_type='regression', output_dim=1)
def residue_distances_by_residue_8(
        structure: AtomArray,
) -> torch.Tensor:
    """Get the average distance from a residue to all other residues (using alpha carbons) within 8.0 A window.

    :param structure: The protein structure.
    :param max_distance: The maximum distance between residue pairs (in Angstroms) to include.
    :return: A PyTorch tensor with average distance from each residue to all other residues (using alpha carbons) within max_distance.
    """
    # Get residue distances
    distances = residue_distances(
        structure=structure,
        max_distance=8.
    )

    # Get average distance from each residue to all other residues
    distances = torch.nanmean(distances, dim=1)

    return distances


@register_concept(concept_level='residue_pair', concept_type='binary_classification', output_dim=1)
def residue_contacts(
        structure: AtomArray,
        contact_threshold: float = 8.0,
        long_range_threshold: int = 24
) -> torch.Tensor:
    """Get the long-range contacts between residue pairs.

    Note: Sets residue pairs within long_range_threshold residues of each other as NaN so only long-range contacts are used.

    :param structure: The protein structure.
    :param contact_threshold: The distance threshold for a contact (below this threshold is a contact).
    :param long_range_threshold: The distance threshold for a long-range contact (below this threshold is short-range).
    :return: A PyTorch tensor with the contacts between residue pairs (type: bool).
    """
    # Get residue distances using the residue_distances concept function above
    distances = residue_distances(structure, max_distance=None)

    # Get contacts
    contacts = (distances < contact_threshold).float()

    # Get short-range mask (all residues less than long_range_threshold of each other)
    ones = torch.ones_like(contacts)
    short_range_mask = ~(ones.triu(diagonal=long_range_threshold) + ones.tril(diagonal=-long_range_threshold)).bool()

    # Set short-range contacts to NaN
    contacts[short_range_mask] = torch.nan

    return contacts


@register_concept(concept_level='residue', concept_type='binary_classification', output_dim=1)
def residue_contacts_by_residue(
        structure: AtomArray,
        contact_threshold: float = 8.0,
        long_range_threshold: int = 24
) -> torch.Tensor:
    """Get whether each residue is involved in a long-range contact.

    :param structure: The protein structure.
    :param contact_threshold: The distance threshold for a contact (below this threshold is a contact).
    :param long_range_threshold: The distance threshold for a long-range contact (below this threshold is short-range).
    :return: A PyTorch tensor showing which residues have long range contacts (type: bool).
    """
    # Get residue contacts
    contacts = residue_contacts(
        structure=structure,
        contact_threshold=contact_threshold,
        long_range_threshold=long_range_threshold
    )

    # Sum over the second dimension to get which residues have contacts
    contacts = torch.nansum(contacts, dim=1) > 0

    return contacts


@register_concept(concept_level='residue', concept_type='regression', output_dim=1)
def residue_locations(
        structure: AtomArray
) -> torch.Tensor:
    """Get the relative location of each residue in the protein.

    :param structure: The protein structure.
    :return: A PyTorch tensor with the relative location of each residue (type: float).
    """
    num_residues = get_residue_count(structure)
    locations = torch.arange(0, num_residues) / num_residues

    return locations


@register_concept(concept_level='residue', concept_type='regression', output_dim=1)
def b_factors(
        structure: AtomArray
) -> torch.Tensor:
    """Get the average B-factors of each residue.

    :param structure: The protein structure.
    :return: A PyTorch tensor with the average B-factors of each residue (type: float).
    """
    return torch.from_numpy(np.array([
        np.mean(residue.b_factor) for residue in residue_iter(structure)
    ]))
