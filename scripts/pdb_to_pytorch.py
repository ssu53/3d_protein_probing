"""Parses PDB files and saves coordinates and sequence in PyTorch format while removing invalid structures."""
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch
from Bio.PDB.PDBExceptions import PDBConstructionException
from tap import Tap
from tqdm import tqdm

sys.path.append(Path(__file__).parent.parent.as_posix())

from pp3.utils.pdb import (
    get_pdb_residue_coordinates,
    get_pdb_sequence_from_structure,
    load_pdb_sequence,
    load_pdb_structure,
    validate_pdb_residue_indices
)


class Args(Tap):
    ids_path: Path  # Path to a TXT file containing PDB IDs.
    pdb_dir: Path  # Path to a directory containing PDB structures.
    structure_save_dir: Path  # Path to a directory where PyTorch files with structures will be saved.
    ids_save_path: Path  # Path to CSV file where PDB IDs of converted structures will be saved.

    def process_args(self) -> None:
        self.structure_save_dir.mkdir(parents=True, exist_ok=True)
        self.ids_save_path.parent.mkdir(parents=True, exist_ok=True)


def convert_pdb_to_pytorch(pdb_id: str, pdb_dir: Path, save_dir: Path) -> bool:
    """Parses PDB file and saves coordinates and sequence in PyTorch format while removing invalid structures.

    :param pdb_id: The PDB ID of the protein structure.
    :param pdb_dir: The directory containing the PDB structures.
    :param save_dir: The directory where PyTorch file with coordinates and sequence will be saved.
    :return: Whether the PDB file was successfully converted.
    """
    # Load PDB structure
    try:
        structure = load_pdb_structure(pdb_id=pdb_id, pdb_dir=pdb_dir)
    except (FileNotFoundError, PDBConstructionException, ValueError):
        return False

    # Get residue coordinates
    try:
        residue_coordinates = get_pdb_residue_coordinates(structure=structure)
    except ValueError:
        return False

    # Check the residue indices
    if not validate_pdb_residue_indices(structure=structure):
        return False

    # Get sequence from structure residues
    try:
        structure_sequence = get_pdb_sequence_from_structure(structure=structure)
    except ValueError:
        return False

    # Load PDB sequence
    try:
        sequence = load_pdb_sequence(pdb_id=pdb_id, pdb_dir=pdb_dir)
    except ValueError:
        return False

    # Ensure the structure's sequence matches a subsequence of the full PDB sequence
    if structure_sequence not in sequence:
        return False

    # Get the start and end indices of the structure's sequence in the full PDB sequence
    start_index = sequence.index(structure_sequence)
    end_index = start_index + len(structure_sequence)

    # Save PyTorch file
    torch.save({
        'residue_coords': residue_coordinates,
        'sequence': sequence,
        'start_index': start_index,
        'end_index': end_index
    }, save_dir / f'{pdb_id}.pt')

    return True


def pdb_to_pytorch(args: Args) -> None:
    """Parses PDB files and saves coordinates and sequence in PyTorch format while removing invalid structures."""
    # Load PDB IDs
    with open(args.ids_path) as f:
        pdb_ids = f.read().strip().split(',')

    print(f'Loaded {len(pdb_ids):,} PDB IDs')

    # Check which PDB IDs have structures
    with Pool() as pool:
        convert_pdb_to_pytorch_fn = partial(convert_pdb_to_pytorch, pdb_dir=args.pdb_dir, save_dir=args.structure_save_dir)
        convert_success = list(
            tqdm(pool.imap(convert_pdb_to_pytorch_fn, pdb_ids), total=len(pdb_ids))
        )

    # Get PDB IDs of successfully converted structures
    converted_pdb_ids = sorted(pdb_id for pdb_id, success in zip(pdb_ids, convert_success) if success)

    print(f'Converted {len(converted_pdb_ids):,} PDB files successfully')

    # Save PDB IDs of successfully converted structures
    pd.DataFrame({'pdb_id': converted_pdb_ids}).to_csv(args.ids_save_path, index=False)


if __name__ == '__main__':
    pdb_to_pytorch(Args().parse_args())
