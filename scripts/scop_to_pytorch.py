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
        min_proteins_per_super_family: int = 250
) -> None:
    """Process SCOP data and convert to PyTorch tensors.

    :param scop_path: The path to the SCOP data TXT file.
    :param pdb_dir: The directory where PDB files are stored.
    :param save_dir: The directory where the PyTorch tensors and concept values will be saved.
    :param min_proteins_per_super_family: The minimum number of proteins per super family.
    """
    # Load SCOP data
    data = pd.read_csv(scop_path, sep=' ', comment='#', names=SCOP_HEADER)

    print(f'Loaded {len(data):,} SCOP entries')

    # Filter to only include proteins with a PDB file
    data = data[[
        get_pdb_path_experimental(pdb_id=pdb_id, pdb_dir=pdb_dir).exists()
        for pdb_id in tqdm(data[SCOP_SF_PDBID_COLUMN])
    ]]

    print(f'Filtered to {len(data):,} SCOP entries with PDB files')

    # Determine super families
    data['super_family'] = data[SCOP_CLASS_COLUMN].apply(lambda scop_class: scop_class.split(',')[-2])

    # Get most common super families
    super_family_counts = Counter(data['super_family'])
    keep_super_families = [
        super_family
        for super_family, count in super_family_counts.most_common()
        if count >= min_proteins_per_super_family
    ]

    print(f'Keeping {len(keep_super_families):,} super families with at least {min_proteins_per_super_family:,} proteins')

    # Restrict data to those super families
    data = data[data['super_family'].isin(keep_super_families)]

    print(f'Filtered to {len(data):,} SCOP entries in those {len(keep_super_families):,} super families '
          f'with {data[SCOP_SF_PDBID_COLUMN].nunique():,} unique PDB entries')

    # Map super family to index
    super_family_to_index = {
        super_family: index
        for index, super_family in enumerate(keep_super_families)
    }

    # Extract concepts
    pdb_id_to_concept = {
        f'{pdb_id}.{pdb_reg}': super_family_to_index[super_family]
        for pdb_id, pdb_reg, super_family in zip(
            data[SCOP_SF_PDBID_COLUMN],
            data[SCOP_SF_PDBREG_COLUMN],
            data['super_family']
        )
    }

    # Convert PDB files to PyTorch format, along with filtering for quality
    pdb_id_to_protein = {}
    error_counter = Counter()

    for pdb_id in tqdm(pdb_id_to_concept):
        # Get protein ID, chain ID, and domain range
        protein_id, reg_id = pdb_id.split('.')

        # Skip examples with multiple domains
        if ',' in reg_id:
            continue

        chain_id, domain_range = reg_id.split(':')
        domain_start, domain_end = domain_range.rsplit('-', maxsplit=1)  # maxsplit to allow negative indices

        protein = convert_pdb_to_pytorch(
            pdb_path=get_pdb_path_experimental(pdb_id=pdb_id, pdb_dir=pdb_dir),
            max_protein_length=MAX_SEQ_LEN,
            one_chain_only=False,
            chain_id=chain_id,
            domain_start=domain_start,
            domain_end=domain_end
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


if __name__ == '__main__':
    from tap import tapify

    tapify(scop_to_pytorch)
