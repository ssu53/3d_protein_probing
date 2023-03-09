"""Data classes and functions."""
from itertools import product
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from pp3.baseline_embeddings import (
    get_baseline_protein_embedding,
    get_baseline_residue_embedding
)
from pp3.concepts import get_concept_level, get_concept_type


def collate_protein(batch: list[tuple[torch.Tensor, float]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate a batch of items at the protein level.

    :param batch: A batch of items at the protein level, where embeddings are 1D tensors and concepts are floats.
    :return: A collated batch.
    """
    embeddings, concept_values = zip(*batch)

    return torch.stack(embeddings), torch.Tensor(concept_values)


def collate_residue(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate a batch of items at the residue level.

    :param batch: A batch of items at the residue level, where embeddings are 2D tensors and concepts are 1D tensors.
    :return: A collated batch.
    """
    embeddings, concept_values = zip(*batch)

    return torch.cat(embeddings), torch.cat(concept_values)


class ProteinConceptDataset(Dataset):
    """A dataset of protein structures and 3D geometric concepts."""

    def __init__(
            self,
            pdb_ids: list[str],
            pdb_id_to_protein: dict[str, dict[str, torch.Tensor | str]],
            pdb_id_to_embeddings: dict[str, torch.Tensor],
            pdb_id_to_concept_value: dict[str, torch.Tensor | float],
            concept_level: str,
            concept_type: str,
            protein_embedding_method: str,
    ) -> None:
        """Initialize the dataset.

        :param pdb_ids: The PDB IDs in this dataset.
        :param pdb_id_to_protein: A dictionary mapping PDB ID  to protein dictionary with sequence and structure.
        :param pdb_id_to_embeddings: A dictionary mapping PDB ID to sequence embeddings.
        :param pdb_id_to_concept_value: A dictionary mapping PDB ID to concept values.
        :param concept_level: The level of the concept (e.g., protein or residue).
        :param concept_type: The type of the concept (e.g., regression or classification).
        :param protein_embedding_method: The method to use to compute the protein embedding from the residue embeddings.
        """
        self.pdb_ids = pdb_ids
        self.pdb_id_to_protein = pdb_id_to_protein
        self.pdb_id_to_embeddings = pdb_id_to_embeddings
        self.pdb_id_to_concept_value = pdb_id_to_concept_value
        self.concept_level = concept_level
        self.concept_type = concept_type
        self.protein_embedding_method = protein_embedding_method

        self.max_residues_for_pairs = 25
        self.rng = np.random.default_rng(seed=0)

        if self.concept_level == 'protein':
            self.collate_fn = collate_protein
        elif self.concept_level.startswith('residue'):
            self.collate_fn = collate_residue
        else:
            raise ValueError(f'Invalid concept level: {self.concept_level}')

    @property
    def embedding_dim(self) -> int:
        """Get the embedding size."""
        embeddings, concept_value = self[0]
        embedding_dim = embeddings.shape[-1]

        return embedding_dim

    @property
    def targets(self) -> torch.Tensor:
        """Get the concept values across the entire dataset, leaving out NaN values."""
        # Get target array
        target_array = [
            self.pdb_id_to_concept_value[pdb_id]
            for pdb_id in self.pdb_ids
        ]

        # Get target type
        target_type = type(target_array[0])

        # Convert to PyTorch Tensor
        if target_type == float:
            target_array = torch.Tensor(target_array)
        elif target_type == torch.Tensor:
            target_array = torch.cat([
                targets.flatten()
                for targets in target_array
            ])
        else:
            raise ValueError(f'Invalid concept value type: {target_type}')

        # Remove NaN values
        target_array = target_array[~torch.isnan(target_array)]

        return target_array

    @property
    def target_mean(self) -> float | None:
        """Get the mean of the concept values if regression, None otherwise."""
        return float(torch.mean(self.targets)) if self.concept_type == 'regression' else None

    @property
    def target_std(self) -> float | None:
        """Get the standard deviation of the concept values if regression, None otherwise."""
        return float(torch.std(self.targets)) if self.concept_type == 'regression' else None

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return len(self.pdb_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor | float]:
        """Get an item from the dataset.

        :param index: The index of the item.
        :return: A tuple of sequence embeddings and concept value.
        """
        # Get PDB ID
        pdb_id = self.pdb_ids[index]

        # Get embeddings
        embeddings = self.pdb_id_to_embeddings[pdb_id]

        # Get concept value
        concept_value = self.pdb_id_to_concept_value[pdb_id]

        # If needed, modify embedding structure based on concept level
        if self.concept_level == 'residue_pair':
            # Get number of residues
            num_residues = len(embeddings)

            # Get residue pair indices
            pair_indices = np.array(list(product(range(num_residues), repeat=2)))  # (num_residues * num_residues, 2)

            # Randomly sample pairs of residues (too much memory to use all of them)
            if num_residues > self.max_residues_for_pairs:
                num_pairs_to_sample = self.max_residues_for_pairs ** 2
                pair_indices = self.rng.choice(pair_indices, size=num_pairs_to_sample, replace=False)

            # Create pair embeddings
            embedding_dim = embeddings.shape[-1]
            pair_embeddings = torch.zeros((len(pair_indices), 2 * embedding_dim))  # (num_pairs, 2 * embedding_dim)
            pair_embeddings[:, :embedding_dim] = embeddings[pair_indices[:, 0]]  # (num_pairs, embedding_dim)
            pair_embeddings[:, embedding_dim:] = embeddings[pair_indices[:, 1]]  # (num_pairs, embedding_dim)
            embeddings = pair_embeddings

            # Get concept values
            concept_value = concept_value[pair_indices[:, 0], pair_indices[:, 1]]  # (num_pairs,)
        elif self.concept_level == 'residue_triplet':
            # Create adjacent triples of residue embeddings
            embeddings = torch.cat([
                embeddings[:-2],  # (num_residues - 2, embedding_dim)
                embeddings[1:-1],  # (num_residues - 2, embedding_dim)
                embeddings[2:]  # (num_residues - 2, embedding_dim)
            ], dim=1)

            # Select corresponding concept values
            concept_value = concept_value[1:-1]  # (num_residues - 2,)

        return embeddings, concept_value


class ProteinConceptDataModule(pl.LightningDataModule):
    """A data module of protein structures and 3D geometric concepts."""

    def __init__(
            self,
            proteins_path: Path,
            embeddings_path: Path,
            concepts_dir: Path,
            concept: str,
            protein_embedding_method: str,
            plm_residue_to_protein_method: str,
            batch_size: int,
            num_workers: int = 8,
            split_seed: int = 0
    ) -> None:
        """Initialize the data module.

        :param proteins_path: Path to PT file containing a dictionary mapping PDB ID to structure and sequence.
        :param embeddings_path: Path to PT file containing a dictionary mapping PDB ID to embeddings.
        :param concepts_dir: Path to a directory containing PT files with dictionaries mapping PDB ID to concept values.
        :param concept: The concept to learn.
        :param batch_size: The batch size.
        :param protein_embedding_method: The method to use to compute the protein embedding from the residue embeddings.
        :param plm_residue_to_protein_method: The method to use to compute the PLM protein embedding from the residue embeddings for protein concepts.
        :param num_workers: The number of workers to use for data loading.
        :param split_seed: The random seed to use for the train/val/test split.
        """
        super().__init__()
        self.proteins_path = proteins_path
        self.embeddings_path = embeddings_path
        self.concepts_dir = concepts_dir
        self.concept = concept
        self.concept_level = get_concept_level(concept)
        self.concept_type = get_concept_type(concept)
        self.protein_embedding_method = protein_embedding_method
        self.plm_residue_to_protein_method = plm_residue_to_protein_method
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_seed = split_seed

        self.train_pdb_ids: list[str] | None = None
        self.val_pdb_ids: list[str] | None = None
        self.test_pdb_ids: list[str] | None = None
        self.train_dataset: ProteinConceptDataset | None = None
        self.val_dataset: ProteinConceptDataset | None = None
        self.test_dataset: ProteinConceptDataset | None = None
        self.is_setup = False

    def get_embeddings(self, pdb_id_to_proteins: dict[str, dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor]:
        """Load or compute embeddings for proteins or residues.

        :param pdb_id_to_proteins: A dictionary mapping PDB ID to protein dictionary with sequence and structure
        :return: A dictionary mapping PDB ID to embeddings.
        """
        # PLM embeddings
        if self.protein_embedding_method == 'plm':
            # Load PDB ID to PLM embeddings dictionary
            pdb_id_to_embeddings = torch.load(self.embeddings_path)

            # If concept is protein level, aggregate residue embeddings to protein embeddings
            if self.concept_level == 'protein':
                # Set up aggregation function
                aggregate_fn = getattr(torch, self.plm_residue_to_protein_method)

                # Aggregate residue embeddings to protein embeddings
                pdb_id_to_embeddings = {
                    pdb_id: aggregate_fn(embeddings, dim=0)
                    for pdb_id, embeddings in pdb_id_to_embeddings.items()
                }
            elif not self.concept_level.startswith('residue'):
                raise ValueError(f'Invalid concept level: {self.concept_level}')

        # Baseline embeddings
        elif self.protein_embedding_method == 'baseline':
            # Get appropriate baseline embedding method for concept level
            if self.concept_level == 'protein':
                baseline_embedding_fn = get_baseline_protein_embedding
            elif self.concept_level.startswith('residue'):
                baseline_embedding_fn = get_baseline_residue_embedding
            else:
                raise ValueError(f'Invalid concept level: {self.concept_level}')

            # Compute baseline embeddings
            pdb_id_to_embeddings = {
                pdb_id: baseline_embedding_fn(protein['sequence'])
                for pdb_id, protein in pdb_id_to_proteins.items()
            }

        # Other embedding methods
        else:
            raise ValueError(f'Invalid protein embedding method: {self.protein_embedding_method}')

        return pdb_id_to_embeddings

    def setup(self, stage: str | None = None) -> None:
        """Prepare the data module by loading the data and splitting into train, val, and test."""
        if self.is_setup:
            return

        print('Loading data')

        # Load PDB ID to protein dictionary with sequence and structure
        pdb_id_to_proteins: dict[str, dict[str, torch.Tensor | str]] = torch.load(self.proteins_path)

        # Split PDB IDs into train and test sets
        pdb_ids = sorted(pdb_id_to_proteins)
        self.train_pdb_ids, val_test_pdb_ids = train_test_split(
            pdb_ids,
            test_size=0.2,
            random_state=self.split_seed
        )
        self.val_pdb_ids, self.test_pdb_ids = train_test_split(
            val_test_pdb_ids,
            test_size=0.5,
            random_state=self.split_seed
        )

        # Load or compute embeddings for proteins or residues
        pdb_id_to_embeddings = self.get_embeddings(pdb_id_to_proteins)

        # Load PDB ID to concept value dictionary
        pdb_id_to_concept_value: dict[str, torch.Tensor | float] = torch.load(self.concepts_dir / f'{self.concept}.pt')

        # Ensure that the PDB IDs are the same across dictionaries
        assert set(pdb_id_to_proteins) == set(pdb_id_to_embeddings) == set(pdb_id_to_concept_value)

        # Create train dataset
        self.train_dataset = ProteinConceptDataset(
            pdb_ids=self.train_pdb_ids,
            pdb_id_to_protein=pdb_id_to_proteins,
            pdb_id_to_embeddings=pdb_id_to_embeddings,
            pdb_id_to_concept_value=pdb_id_to_concept_value,
            concept_level=self.concept_level,
            concept_type=self.concept_type,
            protein_embedding_method=self.protein_embedding_method
        )
        print(f'Train dataset size: {len(self.train_dataset):,}')

        # Create val dataset
        self.val_dataset = ProteinConceptDataset(
            pdb_ids=self.val_pdb_ids,
            pdb_id_to_protein=pdb_id_to_proteins,
            pdb_id_to_embeddings=pdb_id_to_embeddings,
            pdb_id_to_concept_value=pdb_id_to_concept_value,
            concept_level=self.concept_level,
            concept_type=self.concept_type,
            protein_embedding_method=self.protein_embedding_method
        )
        print(f'Val dataset size: {len(self.val_dataset):,}')

        # Create test dataset
        self.test_dataset = ProteinConceptDataset(
            pdb_ids=self.test_pdb_ids,
            pdb_id_to_protein=pdb_id_to_proteins,
            pdb_id_to_embeddings=pdb_id_to_embeddings,
            pdb_id_to_concept_value=pdb_id_to_concept_value,
            concept_level=self.concept_level,
            concept_type=self.concept_type,
            protein_embedding_method=self.protein_embedding_method
        )
        print(f'Test dataset size: {len(self.test_dataset):,}')

        self.is_setup = True

    def train_dataloader(self) -> DataLoader:
        """Get the train data loader."""
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation data loader."""
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        """Get the test data loader."""
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.collate_fn
        )

    predict_dataloader = test_dataloader

    @property
    def embedding_dim(self) -> int:
        """Get the embedding size."""
        return self.train_dataset.embedding_dim
