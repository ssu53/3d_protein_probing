"""PDB helper functions."""
from pathlib import Path

import torch
from biotite.structure import AtomArray, filter_canonical_amino_acids, get_residue_count, get_residues
from biotite.structure.io.pdb import PDBFile
from Bio import SeqIO

from pp3.utils.constants import AA_3_TO_1


def get_pdb_path(pdb_id: str, pdb_dir: Path) -> Path:
    """Get the path of a PDB file.

    :param pdb_id: The PDB ID of the protein structure.
    :param pdb_dir: The directory containing the PDB structures.
    :return: The path of the PDB file.
    """
    return pdb_dir / pdb_id[1:3].lower() / f'pdb{pdb_id.lower()}.ent'


def load_pdb_structure(pdb_id: str, pdb_dir: Path) -> AtomArray:
    """Load the structure from a PDB file.

    Note: Removes hetero residues.

    :raises FileNotFoundError: If the PDB file does not exist.
    :raises ValueError: If the PDB file does not contain any residues after cleaning.

    :param pdb_id: The PDB ID of the protein structure.
    :param pdb_dir: The directory containing the PDB structures.
    :return: The loaded PDB structure.
    """
    # Get PDB file path
    pdb_path = get_pdb_path(pdb_id=pdb_id, pdb_dir=pdb_dir)

    # Check if PDB file exists
    if not pdb_path.exists():
        raise FileNotFoundError(f'PDB file {pdb_path} does not exist')

    # Parse PDB structure
    structure = PDBFile.read(pdb_path).get_structure()

    # Ensure only one model
    if len(structure) != 1:
        raise ValueError('PDB file must contain only one model')

    # Get model
    structure = structure[0]

    # Keep only amino acid residues
    structure = structure[filter_canonical_amino_acids(structure)]

    # Check if there are no residues
    if len(structure) == 0:
        raise ValueError(f'PDB file {pdb_path} does not contain any residues after cleaning')

    return structure


def get_pdb_residue_coordinates(structure: AtomArray) -> torch.Tensor:
    """Get the residue coordinates from a PDB structure.

    :raises ValueError: If the PDB structure does not contain coordinates for all residues.

    :param structure: The PDB structure.
    :return: A numpy array with the coordinates of the residues (CA atoms) in the structure.
    """
    residue_coords = structure[structure.atom_name == 'CA'].coords

    if len(residue_coords) != get_residue_count(structure):
        raise ValueError('PDB structure does not contain coordinates for all residues')

    return torch.from_numpy(residue_coords)


def validate_pdb_residue_indices(structure: AtomArray) -> bool:
    """Validate if the residue indices from a PDB structure are sequential.

    :param structure: The PDB structure.
    :return: True if the residue indices are sequential, False otherwise.
    """
    # Get residue indices
    residue_indices = structure.res_id.tolist()

    # Check if residue indices are sequential
    if residue_indices != list(range(min(residue_indices), max(residue_indices) + 1)):
        return False

    return True


def get_pdb_sequence_from_structure(structure: AtomArray) -> str:
    """Get the sequence from a PDB structure.

    :param structure: The PDB structure.
    :return: The sequence of the structure.
    """
    res_ids, res_names = get_residues(structure)
    sequence = ''.join(AA_3_TO_1[res_name] for res_name in res_names)

    return sequence


def load_pdb_sequence(pdb_id: str, pdb_dir: Path) -> str:
    """Load the sequence from a PDB file.

    :raises ValueError: If the PDB file contains multiple sequence records.

    :param pdb_id: The PDB ID of the protein structure.
    :param pdb_dir: The directory containing the PDB structures.
    :return: The loaded PDB sequence.
    """
    # Get PDB file path
    pdb_path = get_pdb_path(pdb_id=pdb_id, pdb_dir=pdb_dir)

    # Parse PDB sequence
    records = list(SeqIO.parse(pdb_path, 'pdb-seqres'))

    # Ensure only one sequence record
    if len(records) != 1:
        raise ValueError(f'PDB file {pdb_path} contains multiple sequence records')

    # Get sequence
    sequence = str(records[0].seq)

    return sequence
