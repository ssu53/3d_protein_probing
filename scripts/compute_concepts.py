"""Compute 3D geometric concepts from protein structures."""
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from tqdm import tqdm

from pp3.concepts import compute_all_concepts, get_concept_names, get_concept_function
from pp3.utils.pdb import load_structure


def compute_concepts_for_structure(
        pdb_id: str,
        pdb_dir: Path,
        concepts: Optional[list[str]] = None
) -> dict[str, Any]:
    """Computes 3D geometric concepts from a protein structure.

    :param pdb_id: The PDB ID of the protein structure.
    :param pdb_dir: The directory containing the PDB structures.
    :param concepts: List of concepts to compute. If None, all concepts will be computed.
    :return: A dictionary mapping concept names to values.
    """
    # Load PDB structure
    structure = load_structure(pdb_id=pdb_id, pdb_dir=pdb_dir)

    # Set up concept dictionary
    if concepts is None:
        concept_name_to_value = compute_all_concepts(structure=structure)
    else:
        concept_name_to_value = {
            concept_name: get_concept_function(concept_name)(structure)
            for concept_name in concepts
        }

    return concept_name_to_value


def compute_concepts(
        ids_path: Path,
        pdb_dir: Path,
        save_dir: Path,
        concepts: Optional[list[str]] = None
) -> None:
    """Compute 3D geometric concepts from protein structures.

    :param ids_path: Path to a CSV file containing PDB IDs.
    :param pdb_dir: Path to a directory containing PDB structures.
    :param save_dir: Path to a directory where PyTorch files with computed concepts will be saved.
    :param concepts: List of concepts to compute. If None, all concepts will be computed.
    """
    # Load PDB IDs
    pdb_ids = pd.read_csv(ids_path)['pdb_id'].tolist()

    print(f'Loaded {len(pdb_ids):,} PDB IDs')

    # Set up save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set up concept function
    compute_concepts_for_structure_fn = partial(
        compute_concepts_for_structure,
        pdb_dir=pdb_dir,
        concepts=concepts
    )

    # Check which PDB IDs have structures
    with Pool() as pool:
        concept_dicts = list(
            tqdm(pool.imap(compute_concepts_for_structure_fn, pdb_ids), total=len(pdb_ids))
        )

    # Map PDB IDs to concepts
    pdb_id_to_concepts = dict(zip(pdb_ids, concept_dicts))

    # Save each concept separately
    for concept_name in concept_dicts[0].keys():
        concept_pdb_to_value = {
            pdb_id: concepts[concept_name] for pdb_id, concepts in pdb_id_to_concepts.items()
        }
        torch.save(concept_pdb_to_value, save_dir / f'{concept_name}.pt')


if __name__ == '__main__':
    from tap import Tap

    class Args(Tap):
        ids_path: Path
        """Path to a CSV file containing PDB IDs."""
        pdb_dir: Path
        """Path to a directory containing PDB structures."""
        save_dir: Path
        """Path to a directory where PyTorch files with dictionaries mapping PDB ID to concept values will be saved."""
        concepts: Optional[list[str]] = None
        """List of concepts to compute. If None, all concepts will be computed."""

        def configure(self) -> None:
            self.add_argument('--concepts', choices=get_concept_names())

    compute_concepts(**Args().parse_args().as_dict())
