"""Process torchdrug proteins for GeneOntology or EnzymeComission datasets.

Assumes zip folder with data is already downloaded.

GeneOntology: https://torchdrug.ai/docs/_modules/torchdrug/datasets/gene_ontology.html#GeneOntology
GeneOntology: https://zenodo.org/record/6622158/files/GeneOntology.zip

EnzymeCommission: https://torchdrug.ai/docs/_modules/torchdrug/datasets/enzyme_commission.html#EnzymeCommission
EnzymeCommission: https://zenodo.org/record/6622158/files/EnzymeCommission.zip
"""
from collections import Counter
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm

from pp3.utils.constants import MAX_SEQ_LEN
from pdb_to_pytorch import convert_pdb_to_pytorch


def torchdrug_process_proteins(
        dataset_dir: Path,
        dataset_name: Literal['enzyme_commission', 'gene_ontology'],
        save_dir: Path
) -> None:
    # Load PDB paths
    train = list((dataset_dir / 'train').glob('*.pdb'))
    valid = list((dataset_dir / 'valid').glob('*.pdb'))
    test = list((dataset_dir / 'test').glob('*.pdb'))
    pdb_id_to_path = {
        path.stem.split('_')[0]: path
        for path in train + valid + test
    }

    # Convert PDB to PyTorch
    pdb_id_to_protein = {}
    error_counter = Counter()

    for pdb_id, pdb_path in tqdm(pdb_id_to_path.items()):
        protein = convert_pdb_to_pytorch(
            pdb_path=pdb_path,
            max_protein_length=MAX_SEQ_LEN,
            one_chain_only=True
        )

        if 'error' in protein:
            error_counter[protein['error']] += 1
        else:
            pdb_id_to_protein[pdb_id] = protein

    # Print errors
    for error, count in error_counter.most_common():
        print(f'{count:,} errors: {error}')

    print(f'\nConverted {len(pdb_id_to_protein):,} PDB files successfully')

    # Save proteins
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pdb_id_to_protein, save_dir / f'{dataset_name}_proteins.pt')


if __name__ == '__main__':
    from tap import tapify

    tapify(torchdrug_process_proteins)
