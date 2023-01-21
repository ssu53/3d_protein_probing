"""PDB helper functions."""
from pathlib import Path

import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser, Structure

from pp3.utils.constants import AA_3_TO_1


def clean_pdb_structure(structure: Structure) -> None:
    """Clean the PDB structure by removing empty chains and hetero residues.

    :param structure: The PDB structure to clean.
    """
    for model in structure:
        residues_to_remove, chains_to_remove = [], []

        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':
                    residues_to_remove.append((chain.id, residue.id))

            if len(chain) == 0:
                chains_to_remove.append(chain.id)

        for chain_id, residue_id in residues_to_remove:
            model[chain_id].detach_child(residue_id)

        for chain in chains_to_remove:
            model.detach_child(chain)


def get_pdb_path(pdb_id: str, pdb_dir: Path) -> Path:
    """Get the path of a PDB file.

    :param pdb_id: The PDB ID of the protein structure.
    :param pdb_dir: The directory containing the PDB structures.
    :return: The path of the PDB file.
    """
    return pdb_dir / pdb_id[1:3].lower() / f'pdb{pdb_id.lower()}.ent'


def load_pdb_structure(pdb_id: str, pdb_dir: Path) -> Structure:
    """Load the structure from a PDB file.

    Note: Removes empty chains and hetero residues.

    :raises FileNotFoundError: If the PDB file does not exist.
    :raises PDBConstructionException: If the PDB file is invalid.
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
    structure = PDBParser(PERMISSIVE=False).get_structure(id=pdb_id, file=pdb_path)

    # Clean the PDB structure (remove empty chains and hetero residues)
    clean_pdb_structure(structure)

    # Check if there are no residues
    if len(list(structure.get_residues())) == 0:
        raise ValueError(f'PDB file {pdb_path} does not contain any residues after cleaning')

    return structure


def get_pdb_residue_coordinates(structure: Structure) -> np.ndarray:
    """Get the residue coordinates from a PDB structure.

    :raises ValueError: If any residue does not contain a CA atom.

    :param structure: The PDB structure.
    :return: The coordinates of the residues (CA atoms) in the structure.
    """
    # Get coordinates
    residue_coordinates = []
    for residue in structure.get_residues():
        if 'CA' not in residue:
            raise ValueError(f'Residue {residue} does not contain a CA atom')
        residue_coordinates.append(residue['CA'].get_coord())

    return np.array(residue_coordinates)


def validate_pdb_residue_indices(structure: Structure) -> bool:
    """Validate if the residue indices from a PDB structure are sequential.

    :param structure: The PDB structure.
    :return: True if the residue indices are sequential, False otherwise.
    """
    # Get residue indices
    residue_indices = [residue.get_id()[1] for residue in structure.get_residues()]

    # Check if residue indices are sequential
    if residue_indices != list(range(min(residue_indices), max(residue_indices) + 1)):
        return False

    return True


def get_pdb_sequence_from_structure(structure: Structure) -> str:
    """Get the sequence from a PDB structure.

    :raises: ValueError: If the PDB structure contains any non-standard residues.

    :param structure: The PDB structure.
    :return: The sequence of the structure.
    """
    # Get sequence
    sequence = []
    for residue in structure.get_residues():
        resname = residue.get_resname()
        if resname not in AA_3_TO_1:
            raise ValueError(f'Invalid residue name {resname}')
        sequence.append(AA_3_TO_1[resname])

    return ''.join(sequence)


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
