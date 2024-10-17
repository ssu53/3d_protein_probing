"""Compute 3D geometric concepts from protein structures."""
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from tqdm import tqdm

from pp3.concepts import compute_all_concepts, get_concept_function
from pp3.utils.pdb import load_structure, get_pdb_path_experimental


def compute_concepts_for_structure(
        pdb_path: Path,
        concepts: list[str] | None = None
    ) -> dict[str, Any]:
    """Computes 3D geometric concepts from a protein structure.

    :param pdb_path: The path to the PDB structure.
    :param concepts: List of concepts to compute. If None, all concepts will be computed.
    :return: A dictionary mapping concept names to values.
    """
    # Load PDB structure
    structure = load_structure(pdb_path=pdb_path, one_chain_only=True)

    # Set up concept dictionary
    if concepts is None:
        concept_name_to_value = compute_all_concepts(structure=structure)
    else:
        concept_name_to_value = {
            concept_name: get_concept_function(concept_name)(structure=structure)
            for concept_name in concepts
        }

    return concept_name_to_value


def compute_concepts_for_residue_coordinates(
        pdb_id: str,
        proteins: Dict[str, torch.Tensor],
        concepts: list[str] | None = None
    ) -> dict[str, Any]:
    """Computes 3D geometric concepts from a protein structure.
    Only compatible with certain concepts!

    :param pdb_id: The PDB ID.
    :param proteins: The dictionary of proteins, indexed by pdb_id key.
    :param concepts: List of concepts to compute. If None, all concepts will be computed.
    :return: A dictionary mapping concept names to values.
    """
    # Load (valid) residue coordinates
    residue_coordinates = proteins[pdb_id]['structure']
    residue_coordinates = residue_coordinates[proteins[pdb_id]['valid_mask']]

    # Set up concept dictionary
    concept_name_to_value = {
        concept_name: get_concept_function(concept_name)(structure=None, residue_coordinates=residue_coordinates)
        for concept_name in concepts
    }

    return concept_name_to_value


def compute_concepts(
        ids_path: Path,
        save_dir: Path,
        pdb_dir: Path | None = None,
        proteins_path: Path | None = None,
        concepts: list[str] | None = None,
) -> None:
    """Compute 3D geometric concepts from protein structures.
    
    Specify either pdb_dir or proteins_path.

    :param ids_path: Path to a CSV file containing PDB IDs.
    :param pdb_dir: Path to a directory containing PDB structures.
    :param proteins_path: Path to torch.load-able file containing proteins (from pdb_to_pytorch processing).
    :param save_dir: Path to a directory where PyTorch files with computed concepts will be saved.
    :param concepts: List of concepts to compute. If None, all concepts will be computed.
    """
    # Load PDB IDs
    pdb_ids = pd.read_csv(ids_path)['pdb_id'].tolist()

    print(f'Loaded {len(pdb_ids):,} PDB IDs')

    # Set up save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    if proteins_path is not None:
        assert pdb_dir is None

        proteins = torch.load(proteins_path)
        pdb_ids = proteins.keys()

        # Set up concept function
        compute_concepts_for_residue_coordinates_fn = partial(
            compute_concepts_for_residue_coordinates,
            proteins=proteins,
            concepts=concepts
        )
    
        # Single-threaded
        concept_dicts = []
        for pdb_id in tqdm(pdb_ids):
            concept_dicts.append(compute_concepts_for_residue_coordinates_fn(pdb_id=pdb_id))
        
    else:
        assert proteins_path is None

        # Create PDB paths
        pdb_paths = [
            get_pdb_path_experimental(
                pdb_id=pdb_id,
                pdb_dir=pdb_dir
            )
            for pdb_id in pdb_ids
        ]

        # Set up concept function
        compute_concepts_for_structure_fn = partial(
            compute_concepts_for_structure,
            concepts=concepts
        )

        # Check which PDB IDs have structures

        # Multi-threaded
        with Pool() as pool:
            concept_dicts = list(
                tqdm(pool.imap(compute_concepts_for_structure_fn, pdb_paths), total=len(pdb_paths))
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
    from tap import tapify

    tapify(compute_concepts)
