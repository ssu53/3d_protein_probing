"""Compute 3D geometric concepts from protein structures."""
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tap import Tap
from tqdm import tqdm

sys.path.append(Path(__file__).parent.parent.as_posix())

from pp3.concepts import compute_all_concepts
from pp3.utils.pdb import load_pdb_structure


class Args(Tap):
    ids_path: Path  # Path to a CSV file containing PDB IDs.
    pdb_dir: Path  # Path to a directory containing PDB structures.
    save_dir: Path  # Path to a directory where PyTorch files with computed concepts will be saved.

    def process_args(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)


def compute_concepts_for_structure(pdb_id: str, pdb_dir: Path) -> dict[str, Any]:
    """Computes 3D geometric concepts from a protein structure.

    :param pdb_id: The PDB ID of the protein structure.
    :param pdb_dir: The directory containing the PDB structures.
    :return: A dictionary mapping concept names to values.
    """
    # Load PDB structure
    structure = load_pdb_structure(pdb_id=pdb_id, pdb_dir=pdb_dir)

    # Set up concept dictionary
    concepts = compute_all_concepts(structure=structure)

    return concepts


def compute_concepts(args: Args) -> None:
    """Compute 3D geometric concepts from protein structures."""
    # Load PDB IDs
    pdb_ids = pd.read_csv(args.ids_path)['pdb_id'].tolist()

    print(f'Loaded {len(pdb_ids):,} PDB IDs')

    # Check which PDB IDs have structures
    with Pool() as pool:
        compute_concepts_for_structure_fn = partial(compute_concepts_for_structure, pdb_dir=args.pdb_dir)
        structure_concepts = list(
            tqdm(pool.imap(compute_concepts_for_structure_fn, pdb_ids), total=len(pdb_ids))
        )

    # Map PDB IDs to concepts
    pdb_id_to_concepts = dict(zip(pdb_ids, structure_concepts))

    # Save each concept separately
    for concept_name in structure_concepts[0].keys():
        concept_pdb_to_value = {
            pdb_id: concepts[concept_name] for pdb_id, concepts in pdb_id_to_concepts.items()
        }
        torch.save(concept_pdb_to_value, args.save_dir / f'{concept_name}.pt')


if __name__ == '__main__':
    compute_concepts(Args().parse_args())
