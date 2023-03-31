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


def get_pdb_path(pdb_id: str, pdb_dir: Path, simple_format: bool = False) -> Path:
    """Get the path of a PDB file.

    :param pdb_id: The PDB ID of the protein structure.
    :param pdb_dir: The directory containing the PDB structures.
    :return: The path of the PDB file.
    """
    if simple_format:
        path = pdb_dir / f"{pdb_id}.pdb"
    else:
        path = pdb_dir / pdb_id[1:3].lower() / f'pdb{pdb_id.lower()}.ent'
    return path


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


def load_structure(pdb_id: str, pdb_dir: Path) -> AtomArray:
    """Load the protein structure from a PDB file.

    Note: Only keeps amino acids, standardizes atom order, and checks quality of structure.

    :raises FileNotFoundError: If the PDB file does not exist.
    :raises InvalidFileError: If the PDB file is invalid.
    :raises BadStructureError: If the PDB structure is invalid.
    :raises ValueError: If the PDB file contains errors.
    :raises TypeError: If the PDB file contains type errors.

    :param pdb_id: The PDB ID of the protein structure.
    :param pdb_dir: The directory containing the PDB structures.
    :return: The loaded (and cleaned) protein structure.
    """
    # Get PDB file path
    pdb_path = get_pdb_path(pdb_id=pdb_id, pdb_dir=pdb_dir)

    # Check if PDB file exists
    if not pdb_path.exists():
        raise FileNotFoundError('PDB file does not exist')

    # Parse PDB structure
    structure = PDBFile.read(pdb_path).get_structure()

    # Get first model
    structure = structure[0]

    # Keep only amino acid residues
    structure = structure[filter_canonical_amino_acids(structure)]

    # Check if there are no residues
    if len(structure) == 0:
        raise ValueError('Structure does not contain any residues after cleaning')

    # Ensure only one chain
    if get_chain_count(structure) != 1:
        raise ValueError('Structure contains more than one chain')

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
