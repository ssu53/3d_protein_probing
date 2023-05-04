"""Converts a downstream task dataset in JSON form to a concept dataset in PyTorch form."""
import json
from collections import Counter
from pathlib import Path

import torch
from pp3.utils.constants import MAX_SEQ_LEN
from tqdm import tqdm

from pp3.utils.pdb import get_pdb_path
from pdb_to_pytorch import convert_pdb_to_pytorch


def downstream_to_concept(
        data_dir: Path,
        data_name: str,
        concept_name: str,
        pdb_dir: Path,
        save_dir: Path,
        max_protein_length: int | None = MAX_SEQ_LEN
) -> None:
    """Converts a downstream task dataset in JSON form to a concept dataset in PyTorch form.

    :param data_dir: The directory containing the downstream task dataset.
    :param data_name: The name of the downstream task dataset.
    :param concept_name: The name of the concept.
    :param pdb_dir: The directory containing the PDB structures.
    :param save_dir: The directory where the concept dataset will be saved.
    :param max_protein_length: The maximum length of a protein structure.
    """
    # Load downstream task dataset, merging train, val, and test splits
    data = {}
    for split in ['train', 'valid', 'test']:
        with open(data_dir / f'{data_name}_{split}.json') as f:
            data |= json.load(f)

    print(f'Number of proteins in downstream task dataset: {len(data):,}')

    # Extract mapping from PDB ID to concept value
    pdb_id_to_concept = {
        protein['pdb_id'].split('_')[0]: protein[concept_name]
        for protein in data.values()
        if protein is not None
    }

    print(f'Number of proteins with unique PDB IDs: {len(pdb_id_to_concept):,}')

    # Only keep PDB IDs where we have the corresponding PDB file
    pdb_id_to_concept = {
        pdb_id: concept
        for pdb_id, concept in tqdm(pdb_id_to_concept.items(), total=len(pdb_id_to_concept))
        if get_pdb_path(pdb_id=pdb_id, pdb_dir=pdb_dir).exists()
    }

    print(f'Number of proteins with corresponding PDB files: {len(pdb_id_to_concept):,}')

    # Convert PDB files to PyTorch format, along with filtering for quality
    pdb_id_to_protein = {}
    error_counter = Counter()

    for pdb_id in tqdm(pdb_id_to_concept):
        protein_id, chain_id = pdb_id.split('_')

        protein = convert_pdb_to_pytorch(
            pdb_id=protein_id,
            pdb_dir=pdb_dir,
            max_protein_length=max_protein_length,
            one_chain_only=False,
            chain_id=pdb_id
        )

        if 'error' in protein:
            error_counter[protein['error']] += 1
        else:
            pdb_id_to_protein[pdb_id] = protein

    # Print errors
    for error, count in error_counter.most_common():
        print(f'{count:,} errors: {error}')

    print(f'\nConverted {len(pdb_id_to_protein):,} PDB files successfully')

    # Filter out proteins that failed to convert
    pdb_id_to_concept = {
        pdb_id: concept
        for pdb_id, concept in pdb_id_to_concept.items()
        if pdb_id in pdb_id_to_protein
    }

    # Save proteins and concepts
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pdb_id_to_protein, save_dir / f'{concept_name}_proteins.pt')
    torch.save(pdb_id_to_concept, save_dir / f'{concept_name}.pt')


if __name__ == '__main__':
    from tap import tapify

    tapify(downstream_to_concept)
