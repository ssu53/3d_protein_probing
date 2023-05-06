"""Converts a downstream task dataset in JSON form to a concept dataset in PyTorch form."""
import json
from collections import Counter
from pathlib import Path
from typing import Literal

import requests
import torch
from pp3.utils.constants import MAX_SEQ_LEN
from pp3.utils.pdb import convert_pdb_id_computational
from tqdm import tqdm

from pp3.utils.pdb import get_pdb_path_experimental
from pdb_to_pytorch import convert_pdb_to_pytorch


def downstream_to_concept(
        data_dir: Path,
        data_name: str,
        concept_name: str,
        pdb_dir: Path,
        save_dir: Path,
        structure_type: Literal['experimental', 'computational'],
        max_protein_length: int | None = MAX_SEQ_LEN
) -> None:
    """Converts a downstream task dataset in JSON form to a concept dataset in PyTorch form.

    :param data_dir: The directory containing the downstream task dataset.
    :param data_name: The name of the downstream task dataset.
    :param concept_name: The name of the concept.
    :param pdb_dir: The directory containing the PDB structures.
    :param save_dir: The directory where the concept dataset will be saved.
    :param structure_type: The type of PDB structure to use.
    :param max_protein_length: The maximum length of a protein structure.
    """
    # Load downstream task dataset, merging train, val, and test splits
    data = []
    for split in ['train', 'valid', 'test']:
        with open(data_dir / f'{data_name}_{split}_chains_{structure_type}.json') as f:
            data += json.load(f)

    print(f'Number of {structure_type} proteins in downstream task dataset: {len(data):,}')

    # Extract mapping from PDB ID to concept value
    pdb_id_to_concept = {
        protein['pdb_id']: protein[concept_name]
        for protein in data
        if protein is not None and 'pdb_id' in protein
    }

    print(f'Number of {structure_type} proteins with unique PDB IDs: {len(pdb_id_to_concept):,}')

    # Only keep PDB IDs where we have the corresponding PDB file
    if structure_type == 'experimental':
        # If experimental, check if we've already downloaded the PDB file
        pdb_id_to_concept = {
            pdb_id: concept
            for pdb_id, concept in tqdm(pdb_id_to_concept.items(), total=len(pdb_id_to_concept))
            if get_pdb_path_experimental(pdb_id=pdb_id.split('.')[0], pdb_dir=pdb_dir).exists()
        }
    elif structure_type == 'computational':
        # If computational, first check if we've downloaded the PDB file, otherwise download it
        for pdb_id, concept in tqdm(pdb_id_to_concept.items(), total=len(pdb_id_to_concept)):
            pdb_id, chain = pdb_id.split('.')
            pdb_id = convert_pdb_id_computational(pdb_id=pdb_id)
            pdb_path = pdb_dir / f'{pdb_id}.pdb'

            # If we haven't downloaded the AlphaFold PDB file, download it
            if not pdb_path.exists():
                response = requests.get(f'https://alphafold.ebi.ac.uk/files/{pdb_id}.pdb')
                if response.status_code == 200:
                    pdb_path.write_text(response.text)

            # If we failed to download the AlphaFold PDB file, remove it from the mapping
            if not pdb_path.exists():
                del pdb_id_to_concept[f'{pdb_id}.{chain}']
    else:
        raise ValueError(f'Invalid structure type: {structure_type}')

    print(f'Number of {structure_type} proteins with corresponding PDB files: {len(pdb_id_to_concept):,}')

    # Convert PDB files to PyTorch format, along with filtering for quality
    pdb_id_to_protein = {}
    error_counter = Counter()

    for pdb_id in tqdm(pdb_id_to_concept):
        protein_id, chain_id = pdb_id.split('.')

        protein = convert_pdb_to_pytorch(
            pdb_id=protein_id,
            pdb_dir=pdb_dir,
            max_protein_length=max_protein_length,
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

    print(f'\nConverted {len(pdb_id_to_protein):,} {structure_type} PDB files successfully')

    # Filter out proteins that failed to convert
    pdb_id_to_concept = {
        pdb_id: concept
        for pdb_id, concept in pdb_id_to_concept.items()
        if pdb_id in pdb_id_to_protein
    }

    # Save proteins and concepts
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pdb_id_to_protein, save_dir / f'{concept_name}_{structure_type}_proteins.pt')
    torch.save(pdb_id_to_concept, save_dir / f'{concept_name}_{structure_type}.pt')


if __name__ == '__main__':
    from tap import tapify

    tapify(downstream_to_concept)
