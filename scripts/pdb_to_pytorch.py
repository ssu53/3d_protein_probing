"""Parses PDB files and saves coordinates and sequence in PyTorch format while removing invalid structures."""
from typing import Literal
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import einops
from biotite import InvalidFileError
from biotite.structure import BadStructureError, get_residue_count
from tqdm import tqdm

from pp3.utils.constants import MAX_SEQ_LEN
from pp3.utils.pdb import (
    get_pdb_path_experimental,
    get_residue_coordinates,
    get_sequence_from_structure,
    load_structure
)
from pp3.utils import foldseek_parsers


def convert_pdb_to_pytorch(
        pdb_path: Path,
        max_protein_length: int,
        one_chain_only: bool = False,
        first_chain_only: bool = False,
        chain_id: str | None = None,
        domain_start: int | None = None,
        domain_end: int | None = None,
        discard_discontinuous_backbone: bool = True,
) -> dict[str, torch.Tensor | str] | None:
    """Parses PDB file and converts structure and sequence to PyTorch format while removing invalid structures.

    :param pdb_path: The path to the PDB structure.
    :param max_protein_length: The maximum length of a protein structure.
    :param one_chain_only: Whether to only allow proteins with one chain.
    :param chain_id: The chain ID of the protein structure to extract. Used if one_chain_only is False.
    :param domain_start: The start of the domain to extract.
    :param domain_end: The end of the domain to extract.
    :return: A dictionary containing the structure and sequence or an error message if the structure is invalid.
    """
    # Load PDB structure
    try:
        structure = load_structure(
            pdb_path=pdb_path,
            one_chain_only=one_chain_only,
            first_chain_only=first_chain_only,
            chain_id=chain_id,
            domain_start=domain_start,
            domain_end=domain_end,
            discard_discontinuous_backbone=discard_discontinuous_backbone,
        )
    except (BadStructureError, FileNotFoundError, InvalidFileError, ValueError, TypeError) as e:
        return {'error': repr(e)}

    # Check if structure is too long
    if max_protein_length is not None and get_residue_count(structure) > max_protein_length:
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


def convert_pdb_to_pytorch_foldseek_style(
    pdb_path: Path,
    max_protein_length: int,
    one_chain_only: bool = False,
    first_chain_only: bool = False,
    chain_id: str | None = None,
    domain_start: int | None = None,
    domain_end: int | None = None,
    discard_discontinuous_backbone: bool = True,
) -> dict[str, torch.Tensor | str] | None:

    assert one_chain_only, "one_chain_only must be true for Folseek compatibility"
    assert first_chain_only, "first_chain_only must be true for Foldseek compatibility"
    assert chain_id is None, "must select first chain for Foldseek compatibility"
    if domain_start is not None: raise NotImplementedError
    if domain_end is not None: raise NotImplementedError
    assert ~discard_discontinuous_backbone, "no backbone checking for Foldseek compatibility"

    # Get the CA, CB, N, C coordinates
    try:
        coords, valid_mask, sequence = foldseek_parsers.get_coords_from_pdb(pdb_path, full_backbone=True)
    except (FileNotFoundError, ValueError, TypeError, KeyError) as e:
        # KeyError: presumably for indexing invalid 3-letter residue for sequence
        return {'error': repr(e)}
    if not np.any(valid_mask): 
        return {'error': 'No valid residues.'}
    coords = torch.tensor(coords)
    valid_mask = torch.tensor(valid_mask)
    assert coords.ndim == 2
    assert coords.size(1) == 4 * 3

    # Check if structure is too long
    if max_protein_length is not None and coords.size(0) > max_protein_length:
        return {'error': f'Structure is too long (> {max_protein_length} residues)'}

    # Reshape to N, CA, C coordinates for downstream compatibility
    residue_coordinates = torch.empty((coords.size(0), 3, 3))
    residue_coordinates[:, 0, :] = coords[:, 6:9] # N
    residue_coordinates[:, 1, :] = coords[:, 0:3] # CA
    residue_coordinates[:, 2, :] = coords[:, 9:12] # C

    # Return dictionary containing structure and sequence
    return {
        'structure': residue_coordinates, # N, CA, C coordinates
        'valid_mask': valid_mask,
        'sequence': sequence,
    }


def pdb_to_pytorch_legacy(
        ids_path: Path,
        pdb_dir: Path,
        proteins_save_path: Path,
        ids_save_path: Path,
        max_protein_length: int | None = MAX_SEQ_LEN
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

    # Create PDB paths
    pdb_paths = [
        get_pdb_path_experimental(
            pdb_id=pdb_id,
            pdb_dir=pdb_dir
        )
        for pdb_id in pdb_ids
    ]

    # Set up conversion function
    convert_pdb_to_pytorch_fn = partial(
        convert_pdb_to_pytorch,
        max_protein_length=max_protein_length,
        one_chain_only=True
    )

    # Convert PDB files to PyTorch format
    pdb_id_to_protein = {}
    error_counter = Counter()

    with Pool() as pool:
        for pdb_id, protein in tqdm(zip(pdb_ids, pool.imap(convert_pdb_to_pytorch_fn, pdb_paths)), total=len(pdb_ids)):
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


def get_pdb_scop_path(
    pdb_id: str,
    pdb_dir: Path,
):
    return pdb_dir / pdb_id


def pdb_to_pytorch(
    ids_path: Path,
    pdb_dir: Path,
    proteins_save_path: Path,
    ids_save_path: Path,
    max_protein_length: int | None = MAX_SEQ_LEN,
    load_mode: Literal['default', 'scop'] = 'default',
    parse_mode: Literal['default', 'foldseek'] = 'default',
) -> None:
    """Parses PDB files and saves coordinates and sequence in PyTorch format while removing invalid structures.

    :param ids_path: Path to a TXT file containing PDB IDs.
    :param pdb_dir: Path to a directory containing PDB structures.
    :param proteins_save_path: Path to a directory where PyTorch files with coordinates and sequences will be saved.
    :param ids_save_path: Path to CSV file where PDB IDs of converted structures will be saved.
    :param max_protein_length: The maximum length of a protein structure.
    :param load_mode: Mode for getting PDB IDs and PDB paths, depending on layout of ID file and .pdb directories.
    :param parse_mode: Mode for reading .pdb files into PyTorch.
        'default' does strict structure checking
        'foldseek' is more permissive and the same pipeline as Foldseek
    """

    # Load PDB IDs
    print(f"{ids_path=}")

    with open(ids_path) as f:
        if load_mode == 'default':
            pdb_ids = f.read().strip().split(',')
        elif load_mode == 'scop':
            pdb_ids = f.read().split('\n')
        else:
            raise NotImplementedError

    print(f'Loaded {len(pdb_ids):,} PDB IDs')


    # Create PDB paths
    if load_mode == 'default':
        pdb_paths = [
            get_pdb_path_experimental(
                pdb_id=pdb_id,
                pdb_dir=pdb_dir
            )
            for pdb_id in pdb_ids
        ]
    elif load_mode == 'scop':
        pdb_paths = [
            get_pdb_scop_path(
                pdb_id=pdb_id,
                pdb_dir=pdb_dir
            ) 
            for pdb_id in pdb_ids
        ]
    else:
        raise NotImplementedError


    # Convert PDB files to PyTorch format
    pdb_id_to_protein = {}
    error_counter = Counter()

    if parse_mode == 'default':
        print("Running default pdb_to_pytorch!")

        # Set up conversion function
        convert_pdb_to_pytorch_fn = partial(
            convert_pdb_to_pytorch,
            max_protein_length=max_protein_length,
            one_chain_only=True
        )

        with Pool() as pool:
            for pdb_id, protein in tqdm(zip(pdb_ids, pool.imap(convert_pdb_to_pytorch_fn, pdb_paths)), total=len(pdb_ids)):
                if 'error' in protein:
                    error_counter[protein['error']] += 1
                else:
                    pdb_id_to_protein[pdb_id] = protein

        # for pdb_id, pdb_path in tqdm(zip(pdb_ids, pdb_paths), total=len(pdb_ids)):
        #     protein = convert_pdb_to_pytorch(
        #         pdb_path,
        #         max_protein_length=max_protein_length, 
        #         one_chain_only=True, 
        #     )
        #     if pdb_id == '7WMW':
        #         print(pdb_path)
        #         print(protein)
        #     if 'error' in protein:
        #         error_counter[protein['error']] += 1
        #     else:
        #         pdb_id_to_protein[pdb_id] = protein
    
    elif parse_mode == 'foldseek':
        print("Running scop pdb_to_pytorch, with Foldseek compatibility!")

        convert_pdb_to_pytorch_fn = partial(
            convert_pdb_to_pytorch_foldseek_style,
            max_protein_length=max_protein_length, 
            one_chain_only=True, 
            first_chain_only=True, 
            discard_discontinuous_backbone=False,
        )

        with Pool() as pool:
            for pdb_id, protein in tqdm(zip(pdb_ids, pool.imap(convert_pdb_to_pytorch_fn, pdb_paths)), total=len(pdb_ids)):
                if 'error' in protein:
                    error_counter[protein['error']] += 1
                else:
                    pdb_id_to_protein[pdb_id] = protein

        # Non-multithreaded version, since above is mysteriously hanging
        # for pdb_id, pdb_path in tqdm(zip(pdb_ids, pdb_paths), total=len(pdb_ids)):
        #     protein = convert_pdb_to_pytorch_foldseek_style(
        #         pdb_path,
        #         max_protein_length=max_protein_length, 
        #         one_chain_only=True, 
        #         first_chain_only=True, 
        #         discard_discontinuous_backbone=False,
        #     )
        #     if 'error' in protein:
        #         error_counter[protein['error']] += 1
        #     else:
        #         pdb_id_to_protein[pdb_id] = protein
    
    else:
        raise NotImplementedError

    # Print errors
    for error, count in error_counter.most_common():
        print(f'{count:,} errors: {error}')

    if parse_mode == 'default':
        print(f'\nConverted {len(pdb_id_to_protein):,} PDB files successfully')
    elif parse_mode == 'foldseek':
        print(f'\nConverted {len(pdb_id_to_protein):,} PDB files successfully. Discontinuous backbones are retained!')
    else:
        raise NotImplementedError

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


    """
    e.g.

    python pdb_to_pytorch.py \
        --ids_path data/scope40_foldseek_compatible/pdbs_train.txt \
        --pdb_dir /oak/stanford/groups/jamesz/shiye/scope40 \
        --proteins_save_path data/scope40_foldseek_compatible/proteins_train.pt \
        --ids_save_path data/scope40_foldseek_compatible/valid_pdb_ids_train.csv 
        --load_mode scop
        --parse_mode foldseek

    """
