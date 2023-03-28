"""Parses PDB files and saves coordinates and sequence in PyTorch format while removing invalid structures."""
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch
from biotite import InvalidFileError
from biotite.structure import BadStructureError
from tqdm import tqdm

from pp3.utils.pdb import (
    get_residue_coordinates,
    get_sequence_from_structure,
    load_structure
)


def convert_pdb_to_pytorch(
        pdb_id: str,
        pdb_dir: Path,
        max_protein_length: int | None = None
) -> dict[str, torch.Tensor | str] | None:
    """Parses PDB file and converts structure and sequence to PyTorch format while removing invalid structures.

    :param pdb_id: The PDB ID of the protein structure.
    :param pdb_dir: The directory containing the PDB structures.
    :param max_protein_length: The maximum length of a protein structure.
    :return: A dictionary containing the structure and sequence or an error message if the structure is invalid.
    """
    # Load PDB structure
    try:
        structure = load_structure(pdb_id=pdb_id, pdb_dir=pdb_dir)
    except (BadStructureError, FileNotFoundError, InvalidFileError, ValueError, TypeError) as e:
        return {'error': repr(e)}

    # Check if structure is too long
    if max_protein_length is not None and len(structure) > max_protein_length:
        return {'error': f'Structure is too long (> {max_protein_length} residues)'}

    # Get residue coordinates
    try:
        residue_coordinates = get_residue_coordinates(structure=structure)
    except ValueError as e:
        return {'error': repr(e)}

    # Get sequence from structure residues
    sequence = get_sequence_from_structure(structure=structure)

    # Return dictionary containing structure and sequence
    return {
        'structure': residue_coordinates,
        'sequence': sequence
    }


def pdb_to_pytorch(
        ids_path: Path,
        pdb_dir: Path,
        proteins_save_path: Path,
        ids_save_path: Path,
        max_protein_length: int | None = None
) -> None:
    """Parses PDB files and saves coordinates and sequence in PyTorch format while removing invalid structures.

    :param ids_path: Path to a TXT file containing PDB IDs.
    :param pdb_dir: Path to a directory containing PDB structures.
    :param proteins_save_path: Path to a directory where PyTorch files with coordinates and sequences will be saved.
    :param ids_save_path: Path to CSV file where PDB IDs of converted structures will be saved.
    :param max_protein_length: The maximum length of a protein structure.
    """
    # Load PDB IDs
    with open(ids_path) as f:
        pdb_ids = f.read().strip().split(',')

    print(f'Loaded {len(pdb_ids):,} PDB IDs')

    # Set up conversion function
    convert_pdb_to_pytorch_fn = partial(
        convert_pdb_to_pytorch,
        pdb_dir=pdb_dir,
        max_protein_length=max_protein_length
    )

    # Convert PDB files to PyTorch format
    pdb_id_to_protein = {}
    error_counter = Counter()

    with Pool() as pool:
        for pdb_id, protein in tqdm(zip(pdb_ids, pool.imap(convert_pdb_to_pytorch_fn, pdb_ids)), total=len(pdb_ids)):
            if 'error' in protein:
                error_counter[protein['error']] += 1
            else:
                pdb_id_to_protein[pdb_id] = protein

    # Print errors
    for error, count in error_counter.most_common():
        print(f'{count:,} errors: {error}')

    print(f'\nConverted {len(pdb_id_to_protein):,} PDB files successfully')

    # Save protein structures and sequences of successfully converted structures
    proteins_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pdb_id_to_protein, proteins_save_path)

    # Save PDB IDs of successfully converted structures
    ids_save_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_ids = sorted(pdb_id_to_protein)
    pd.DataFrame({'pdb_id': pdb_ids}).to_csv(ids_save_path, index=False)


if __name__ == '__main__':
    from tap import tapify

    tapify(pdb_to_pytorch)
