"""Data classes and functions."""
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from pp3.concepts import get_concept_level


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
            pdb_id_to_protein: dict[str, dict[str, torch.Tensor | str | int]],
            pdb_id_to_embeddings: dict[str, torch.Tensor],
            pdb_id_to_concept_value: dict[str, torch.Tensor],
            concept_level: str,
            protein_embedding_method: str,
    ) -> None:
        """Initialize the dataset.

        :param pdb_ids: The PDB IDs in this dataset.
        :param pdb_id_to_protein: A dictionary mapping PDB IDs to protein dictionaries.
        :param pdb_id_to_embeddings: A dictionary mapping PDB ID to sequence embeddings.
        :param pdb_id_to_concept_value: A dictionary mapping PDB ID to concept values.
        :param concept_level: The level of the concept.
        :param protein_embedding_method: The method to use to compute the protein embedding from the residue embeddings.
        """
        self.pdb_ids = pdb_ids
        self.pdb_id_to_protein = pdb_id_to_protein
        self.pdb_id_to_embeddings = pdb_id_to_embeddings
        self.pdb_id_to_concept_value = pdb_id_to_concept_value
        self.concept_level = concept_level
        self.protein_embedding_method = protein_embedding_method

        if self.concept_level == 'protein':
            self.collate_fn = collate_protein
        elif self.concept_level == 'residue':
            self.collate_fn = collate_residue
        else:
            raise ValueError(f'Invalid concept level: {self.concept_level}')

    @property
    def embedding_dim(self) -> int:
        """Get the embedding size."""
        return self.pdb_id_to_embeddings[self.pdb_ids[0]].shape[-1]

    @property
    def targets(self) -> np.ndarray:
        """Get the concept values across the entire dataset."""
        target_array = [self.pdb_id_to_concept_value[pdb_id] for pdb_id in self.pdb_ids]

        if self.concept_level == 'protein':
            target_array = np.array(target_array)
        else:
            target_array = np.concatenate(target_array)

        return target_array

    @property
    def target_mean(self) -> float:
        """Get the mean of the concept values."""
        # TODO: enable for non-scalar values
        return float(np.mean(self.targets))

    @property
    def target_std(self) -> float:
        """Get the standard deviation of the concept values."""
        # TODO: enable for non-scalar values
        return float(np.std(self.targets))

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

        # TODO: precompute this once or cache this?
        # If residue level, then get residue embeddings for residues in structure
        if 'residue' in self.concept_level:
            # TODO: modify MLP or this preloading to handle residue-level embeddings
            start_index = self.pdb_id_to_protein[pdb_id]['start_index']
            end_index = self.pdb_id_to_protein[pdb_id]['end_index']

            embeddings = self.pdb_id_to_embeddings[pdb_id][start_index:end_index]
        # If protein level, then get average embedding across all residues
        elif self.concept_level == 'protein':
            if self.protein_embedding_method == 'sum':
                embeddings = self.pdb_id_to_embeddings[pdb_id].sum(dim=0)
            elif self.protein_embedding_method == 'mean':
                embeddings = self.pdb_id_to_embeddings[pdb_id].mean(dim=0)
            else:
                raise ValueError(f'Invalid protein embedding method: {self.protein_embedding_method}')
        else:
            raise ValueError(f'Invalid concept level: {self.concept_level}')

        # Get concept value
        concept_value = self.pdb_id_to_concept_value[pdb_id]

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
        :param num_workers: The number of workers to use for data loading.
        :param split_seed: The random seed to use for the train/val/test split.
        """
        super().__init__()
        self.proteins_path = proteins_path
        self.embeddings_path = embeddings_path
        self.concepts_dir = concepts_dir
        self.concept = concept
        self.concept_level = get_concept_level(concept)
        self.protein_embedding_method = protein_embedding_method
        self.batch_size = batch_size
        self.train_dataset = self.val_dataset = self.test_dataset = None
        self.is_setup = False
        self.num_workers = num_workers
        self.split_seed = split_seed

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare the data module by loading the data and splitting into train, val, and test."""
        if self.is_setup:
            return

        print('Loading data')

        # Load PDB ID to proteins dictionary
        pdb_id_to_proteins = torch.load(self.proteins_path)

        # Load PDB ID to embeddings dictionary
        pdb_id_to_embeddings = torch.load(self.embeddings_path)

        # Load PDB ID to concept value dictionary
        pdb_id_to_concept_value = torch.load(self.concepts_dir / f'{self.concept}.pt')

        # Ensure that the PDB IDs are the same across dictionaries
        assert set(pdb_id_to_proteins) == set(pdb_id_to_embeddings) == set(pdb_id_to_concept_value)

        # Split PDB IDs into train and test sets
        pdb_ids = sorted(pdb_id_to_proteins)
        train_pdb_ids, test_pdb_ids = train_test_split(pdb_ids, test_size=0.2, random_state=self.split_seed)
        val_pdb_ids, test_pdb_ids = train_test_split(test_pdb_ids, test_size=0.5, random_state=self.split_seed)

        # Create train dataset
        self.train_dataset = ProteinConceptDataset(
            pdb_ids=train_pdb_ids,
            pdb_id_to_protein=pdb_id_to_proteins,
            pdb_id_to_embeddings=pdb_id_to_embeddings,
            pdb_id_to_concept_value=pdb_id_to_concept_value,
            concept_level=self.concept_level,
            protein_embedding_method=self.protein_embedding_method
        )
        print(f'Train dataset size: {len(self.train_dataset):,}')

        # Create val dataset
        self.val_dataset = ProteinConceptDataset(
            pdb_ids=val_pdb_ids,
            pdb_id_to_protein=pdb_id_to_proteins,
            pdb_id_to_embeddings=pdb_id_to_embeddings,
            pdb_id_to_concept_value=pdb_id_to_concept_value,
            concept_level=self.concept_level,
            protein_embedding_method=self.protein_embedding_method
        )
        print(f'Val dataset size: {len(self.val_dataset):,}')

        # Create test dataset
        self.test_dataset = ProteinConceptDataset(
            pdb_ids=test_pdb_ids,
            pdb_id_to_protein=pdb_id_to_proteins,
            pdb_id_to_embeddings=pdb_id_to_embeddings,
            pdb_id_to_concept_value=pdb_id_to_concept_value,
            concept_level=self.concept_level,
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
