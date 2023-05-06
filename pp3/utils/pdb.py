"""PDB helper functions."""
from pathlib import Path

import torch
from biotite.structure import (
    AtomArray,
    check_bond_continuity,
    check_duplicate_atoms,
    filter_canonical_amino_acids,
    get_chain_count,
    get_residue_count,
    get_residues,
    residue_iter
)
from biotite.structure.info import standardize_order
from biotite.structure.io.pdb import PDBFile

from pp3.utils.constants import AA_3_TO_1, AA_ATOM_NAMES, BACKBONE_ATOM_NAMES


def get_pdb_path_experimental(pdb_id: str, pdb_dir: Path, simple_format: bool = False) -> Path:
    """Get the path of an experimental PDB file.

    :param pdb_id: The PDB ID of the protein structure.
    :param pdb_dir: The directory containing the experimental PDB structures.
    :param simple_format: Whether to use the simple PDB format.
    :return: The path of the experimental PDB file.
    """
    if simple_format:
        path = pdb_dir / f"{pdb_id}.pdb"
    else:
        path = pdb_dir / pdb_id[1:3].lower() / f'pdb{pdb_id.lower()}.ent'

    return path


def pdb_id_is_alphafold(pdb_id: str) -> bool:
    """Check if a PDB ID is an AlphaFold PDB ID.

    :param pdb_id: The PDB ID of the protein structure.
    :return: True if the PDB ID is an AlphaFold PDB ID, False otherwise.
    """
    return pdb_id.startswith('AF_AF') and pdb_id.endswith('F1')


def convert_pdb_id_computational(pdb_id: str, model_name: str = 'model_v4') -> str:
    """Converts a PDB ID to an AlphaFold download ID.

    Ex. AF_AFQ17898F1 ==> AF-Q17898-F1

    :param pdb_id: The PDB ID of the protein structure.
    :param model_name: The name of the AlphaFold model.
    :return: The AlphaFold download ID.
    """
    if not pdb_id_is_alphafold(pdb_id):
        raise ValueError(f'Invalid AlphaFold PDB ID: {pdb_id}')

    return f'AF-{pdb_id[5:-2]}-F1-{model_name}'


def verify_residues(structure: AtomArray) -> None:
    """Verify that the residues in the structure contain the correct atoms.

    Specifically, enforce that the residue contains all three backbone atoms
    and a subset of the atoms in the canonical residue.

    :raises ValueError: If the structure contains invalid residues.

    :param structure: The PDB structure.
    :return: True if the structure contains all valid residues, False otherwise.
    """
    # Iterate over residues
    for residue in residue_iter(structure):
        # Get residue name and atom names
        residue_name = residue.res_name[0]
        residue_atom_names = set(residue.atom_name)

        # Check if residue contains all backbone atoms
        if not BACKBONE_ATOM_NAMES.issubset(residue_atom_names):
            raise ValueError('Residue does not contain all backbone atoms')

        # Check if residue contains subset of canonical atoms for that residue
        if not residue_atom_names.issubset(AA_ATOM_NAMES[residue_name]):
            raise ValueError('Residue contains invalid atoms')


def load_structure(
        pdb_id: str,
        pdb_dir: Path,
        one_chain_only: bool = False,
        chain_id: str | None = None
) -> AtomArray:
    """Load the protein structure from a PDB file.

    Note: Only keeps amino acids, standardizes atom order, and checks quality of structure.

    :raises FileNotFoundError: If the PDB file does not exist.
    :raises InvalidFileError: If the PDB file is invalid.
    :raises BadStructureError: If the PDB structure is invalid.
    :raises ValueError: If the PDB file contains errors.
    :raises TypeError: If the PDB file contains type errors.

    :param pdb_id: The PDB ID of the protein structure.
    :param pdb_dir: The directory containing the PDB structures.
    :param one_chain_only: Whether to only allow proteins with one chain.
    :param chain_id: The chain ID of the protein structure to extract. Used if one_chain_only is False.
    :return: The loaded (and cleaned) protein structure.
    """
    # Check if chain ID is given when allowing multichain proteins
    if not one_chain_only and chain_id is None:
        raise ValueError('Need chain_id if not restricting to one chain proteins')

    # Get PDB file path
    pdb_path = get_pdb_path_experimental(pdb_id=pdb_id, pdb_dir=pdb_dir)

    # Check if PDB file exists
    if not pdb_path.exists():
        raise FileNotFoundError('PDB file does not exist')

    # Parse PDB structure
    structure = PDBFile.read(pdb_path).get_structure()

    # Get first model
    structure = structure[0]

    # Keep only amino acid residues
    structure = structure[filter_canonical_amino_acids(structure)]

    # Restrict to one chain, either by enforcing one chain only or by selecting a chain
    if one_chain_only:
        if get_chain_count(structure) != 1:
            raise ValueError('Structure contains more than one chain')
    else:
        # Restrict to chain
        structure = structure[structure.chain_id == chain_id]

    # Check if there are no residues
    if len(structure) == 0:
        raise ValueError('Structure does not contain any residues after cleaning')

    # Standardize atom order
    structure = structure[standardize_order(structure)]

    # Check if residues are valid
    verify_residues(structure)

    # Check for duplicate atoms
    if len(check_duplicate_atoms(structure)) > 0:
        raise ValueError('Structure contains duplicate atoms')

    # Check for backbone bond continuity
    if len(check_bond_continuity(structure)) > 0:
        raise ValueError('Structure contains invalid backbone bonds')

    return structure


def get_residue_coordinates(structure: AtomArray) -> torch.Tensor:
    """Get the residue coordinates from a protein structure.

    :raises ValueError: If the protein structure does not contain coordinates for all residues.

    :param structure: The protein structure.
    :return: A numpy array with the coordinates of the residues (CA atoms) in the protein structure.
    """
    residue_coords = structure[structure.atom_name == 'CA'].coord

    if len(residue_coords) != get_residue_count(structure):
        raise ValueError('Structure does not contain coordinates for all residues')

    return torch.from_numpy(residue_coords)


def get_sequence_from_structure(structure: AtomArray) -> str:
    """Get the sequence from a protein structure.

    :param structure: The protein structure.
    :return: The sequence of the protein structure.
    """
    res_ids, res_names = get_residues(structure)
    sequence = ''.join(AA_3_TO_1[res_name] for res_name in res_names)

    return sequence
