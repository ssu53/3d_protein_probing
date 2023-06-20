"""Process SCOP data and convert to PyTorch tensors and concepts."""
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from pp3.utils.pdb import get_pdb_path_experimental
from pp3.utils.constants import (
    MAX_SEQ_LEN,
    SCOP_CLASS_COLUMN,
    SCOP_SF_PDBID_COLUMN,
    SCOP_SF_PDBREG_COLUMN,
    SCOP_HEADER
)
from pdb_to_pytorch import convert_pdb_to_pytorch


def scop_to_pytorch(
        scop_path: Path,
        pdb_dir: Path,
        save_dir: Path,
        num_super_families: int = 10
) -> None:
    """Process SCOP data and convert to PyTorch tensors.

    :param scop_path: The path to the SCOP data TXT file.
    :param pdb_dir: The directory where PDB files are stored.
    :param save_dir: The directory where the PyTorch tensors and concept values will be saved.
    :param num_super_families: The number of super families to include.
    """
    # Load SCOP data
    data = pd.read_csv(scop_path, sep=' ', comment='#', names=SCOP_HEADER)

    print(f'Loaded {len(data):,} SCOP entries')

    # Filter to only include proteins with a PDB file
    data = data[[
        get_pdb_path_experimental(pdb_id=pdb_id, pdb_dir=pdb_dir).exists()
        for pdb_id in data[SCOP_SF_PDBID_COLUMN]
    ]]

    print(f'Filtered to {len(data):,} SCOP entries with PDB files')

    # Determine super families
    data['super_family'] = data[SCOP_CLASS_COLUMN].apply(lambda x: str(x)[:3])

    # Get most common super families
    top_super_families = data['super_family'].value_counts().index.tolist()[:num_super_families]
    breakpoint()

    # Extract concepts
    # TODO
    pdb_id_to_concept = {}

    # Convert PDB files to PyTorch format, along with filtering for quality
    pdb_id_to_protein = {}
    error_counter = Counter()

    for pdb_id in tqdm(pdb_id_to_concept):
        protein_id, chain_id = pdb_id.split('.')

        protein = convert_pdb_to_pytorch(
            pdb_path=pdb_dir / f'{protein_id}.pdb',
            max_protein_length=MAX_SEQ_LEN,
            one_chain_only=False,
            chain_id=chain_id
        )

        if 'error' in protein:
            error_counter[protein['error']] += 1
        else:
            pdb_id_to_protein[pdb_id] = protein

    # Print errors
    for error, count in error_counter.most_common():
        print(f'{count:,} errors: {error}')

    print(f'Converted {len(pdb_id_to_protein):,} PDB files successfully')

    # Filter out proteins that failed to convert
    pdb_id_to_concept = {
        pdb_id: concept
        for pdb_id, concept in pdb_id_to_concept.items()
        if pdb_id in pdb_id_to_protein
    }

    # Save proteins and concepts
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pdb_id_to_protein, save_dir / f'scop_proteins.pt')
    torch.save(pdb_id_to_concept, save_dir / f'scop.pt')
