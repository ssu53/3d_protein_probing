"""Converts a downstream task dataset in JSON form to a concept dataset in PyTorch form."""
import json
from pathlib import Path

import torch

from pp3.utils.pdb import get_pdb_path


def downstream_to_concept(
        data_dir: Path,
        data_name: str,
        concept_name: str,
        pdb_dir: Path,
        save_dir: Path
) -> None:
    """Converts a downstream task dataset in JSON form to a concept dataset in PyTorch form.

    :param data_dir: The directory containing the downstream task dataset.
    :param data_name: The name of the downstream task dataset.
    :param concept_name: The name of the concept.
    :param pdb_dir: The directory containing the PDB structures.
    :param save_dir: The directory where the concept dataset will be saved.
    """
    # Load downstream task dataset, merging train, val, and test splits
    data = {}
    for split in ['train', 'valid', 'test']:
        with open(data_dir / f'{data_name}_{split}.json') as f:
            data |= json.load(f)

    print(f'Number of proteins in downstream task dataset: {len(data):,}')

    # Extract mapping from PDB ID to concept value
    pdb_to_concept = {
        protein['pdb_id']: protein[concept_name]
        for protein in data.values()
        if protein is not None
    }

    print(f'Number of proteins with unique PDB IDs: {len(pdb_to_concept):,}')

    # Only keep PDB IDs where we have the corresponding PDB file
    pdb_to_concept = {
        pdb_id: concept
        for pdb_id, concept in pdb_to_concept.items()
        if get_pdb_path(pdb_id=pdb_id, pdb_dir=pdb_dir).exists()
    }

    print(f'Number of proteins with corresponding PDB files: {len(pdb_to_concept):,}')

    # Save mapping from PDB ID to concept value as PyTorch file
    torch.save(pdb_to_concept, save_dir / f'{concept_name}.pt')


if __name__ == '__main__':
    from tap import tapify

    tapify(downstream_to_concept)
