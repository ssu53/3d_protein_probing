"""Data classes and functions."""
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from pp3.baseline_embeddings import get_baseline_residue_embedding
from pp3.concepts import get_concept_level, get_concept_type
from pp3.utils.constants import BATCH_TYPE, ONE_EMBEDDING_SIZE


def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> BATCH_TYPE:
    """Collate a batch of items at the residue level.

    :param batch: A batch of items at the residue level, where each element of the batch is a tuple containing:
                    - Embeddings (num_residues, embedding_dim)
                    - Coordinates (num_residues, 3)
                    - Y value (num_residues,)
    :return: A collated batch with:
                - Embeddings (batch_size, max_num_residues, embedding_dim)
                - Coordinates (batch_size, max_num_residues, 3)
                - Y value (batch_size, max_num_residues)
                - Padding mask (batch_size, max_num_residues)
    """
    embeddings, coords, y = zip(*batch)

    # Apply padding
    lengths = [embedding.shape[0] for embedding in embeddings]
    max_seq_len = max(lengths)
    padding_mask = torch.tensor([[1] * length + [0] * (max_seq_len - length) for length in lengths])
    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    coords = torch.nn.utils.rnn.pad_sequence(coords, batch_first=True)

    # Flatten y if needed
    if isinstance(y[0], float):
        y = torch.tensor(y)
    elif y[0].ndim == 2:
        padded_y = torch.zeros((len(y), max_seq_len, max_seq_len))

        for i, y_i in enumerate(y):
            padded_y[i, :y_i.shape[0], :y_i.shape[1]] = y_i

        y = padded_y.view(len(y), -1)
    else:
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)

    return embeddings, coords, y, padding_mask


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
            pdb_id_to_coordinates: dict[str, torch.Tensor],
    ) -> None:
        """Initialize the dataset.

        :param pdb_ids: The PDB IDs in this dataset.
        :param pdb_id_to_protein: A dictionary mapping PDB ID  to protein dictionary with sequence and structure.
        :param pdb_id_to_embeddings: A dictionary mapping PDB ID to sequence embeddings.
        :param pdb_id_to_concept_value: A dictionary mapping PDB ID to concept values.
        :param concept_level: The level of the concept (e.g., protein or residue).
        :param concept_type: The type of the concept (e.g., regression or classification).
        :param embedding_method: The method to use to compute the protein embedding from the residue embeddings.
        """
        self.pdb_ids = pdb_ids
        self.pdb_id_to_protein = pdb_id_to_protein
        self.pdb_id_to_embeddings = pdb_id_to_embeddings
        self.pdb_id_to_concept_value = pdb_id_to_concept_value
        self.concept_level = concept_level
        self.concept_type = concept_type
        self.pdb_id_to_coordinates = pdb_id_to_coordinates

        self.max_pairs = 25 ** 2
        self.rng = np.random.default_rng(seed=0)

    @property
    def embedding_dim(self) -> int:
        """Get the embedding size."""
        embeddings = next(iter(self.pdb_id_to_embeddings.values()))
        embedding_dim = embeddings.shape[-1]

        return embedding_dim

    @property
    def targets(self) -> torch.Tensor:
        """Get the concept values across the entire dataset, removing NaNs."""
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

        # Remove NaNs
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | float]:
        """Get an item from the dataset.

        :param index: The index of the item.
        :return: A tuple of sequence embeddings and concept value.
        """
        # Get PDB ID
        pdb_id = self.pdb_ids[index]

        # Get embeddings
        embeddings = self.pdb_id_to_embeddings[pdb_id]

        # Get coordinates
        coordinates = self.pdb_id_to_coordinates[pdb_id]

        # Get concept value
        concept_value = self.pdb_id_to_concept_value[pdb_id]

        return embeddings, coordinates, concept_value


class ProteinConceptDataModule(pl.LightningDataModule):
    """A data module of protein structures and 3D geometric concepts."""

    def __init__(
            self,
            proteins_path: Path,
            embeddings_path: Path,
            concepts_dir: Path,
            concept: str,
            embedding_method: str,
            batch_size: int,
            num_workers: int = 4,
            split_seed: int = 0,
            split_path: Path | None = None
    ) -> None:
        """Initialize the data module.

        :param proteins_path: Path to PT file containing a dictionary mapping PDB ID to structure and sequence.
        :param embeddings_path: Path to PT file containing a dictionary mapping PDB ID to embeddings.
        :param concepts_dir: Path to a directory containing PT files with dictionaries mapping PDB ID to concept values.
        :param concept: The concept to learn.
        :param batch_size: The batch size.
        :param embedding_method: The method to use to compute the protein embedding from the residue embeddings.
        :param num_workers: The number of workers to use for data loading.
        :param split_seed: The random seed to use for the train/val/test split.
        :param split_path: Optional path to a JSON file containing a dictionary mapping split to list of PDB IDs.
                           If provided, used in place of split seed.
        """
        super().__init__()
        self.proteins_path = proteins_path
        self.embeddings_path = embeddings_path
        self.concepts_dir = concepts_dir
        self.concept = concept
        self.concept_level = get_concept_level(concept)
        self.concept_type = get_concept_type(concept)
        self.embedding_method = embedding_method
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_seed = split_seed
        self.split_path = split_path

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
        if self.embedding_method == 'plm':
            # Load PDB ID to PLM embeddings dictionary
            pdb_id_to_embeddings = torch.load(self.embeddings_path)

        # Baseline embeddings
        elif self.embedding_method == 'baseline':
            # Compute baseline embeddings
            pdb_id_to_embeddings = {
                pdb_id: get_baseline_residue_embedding(protein['sequence'])
                for pdb_id, protein in pdb_id_to_proteins.items()
            }
        
        elif self.embedding_method == 'one':
            pdb_id_to_embeddings = {
                pdb_id: torch.ones(len(protein['sequence']), ONE_EMBEDDING_SIZE)
                for pdb_id, protein in pdb_id_to_proteins.items()
            }

        # Other embedding methods
        else:
            raise ValueError(f'Invalid embedding method: {self.embedding_method}')

        return pdb_id_to_embeddings

    def setup(self, stage: str | None = None) -> None:
        """Prepare the data module by loading the data and splitting into train, val, and test."""
        if self.is_setup:
            return

        print('Loading data')

        # Load PDB ID to protein dictionary with sequence and structure
        pdb_id_to_proteins: dict[str, dict[str, torch.Tensor | str]] = torch.load(self.proteins_path)

        # Get PDB IDs
        pdb_ids = sorted(pdb_id_to_proteins)

        # Split PDB IDs into train and test sets
        if self.split_path is not None:
            with open(self.split_path) as f:
                split_to_pdb_ids: dict[str, list[str]] = json.load(f)

            pdb_ids_set = set(pdb_ids)
            self.train_pdb_ids = sorted(set(split_to_pdb_ids['train']) & pdb_ids_set)
            self.val_pdb_ids = sorted(set(split_to_pdb_ids['valid']) & pdb_ids_set)
            self.test_pdb_ids = sorted(set(split_to_pdb_ids['test']) & pdb_ids_set)
        else:
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

        # Load id to coordinate dictionary
        pdb_id_to_coordinates = {k: v["structure"] for k, v in pdb_id_to_proteins.items()}

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
            pdb_id_to_coordinates=pdb_id_to_coordinates
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
            pdb_id_to_coordinates=pdb_id_to_coordinates
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
            pdb_id_to_coordinates=pdb_id_to_coordinates
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
            collate_fn=collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation data loader."""
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        """Get the test data loader."""
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

    predict_dataloader = test_dataloader

    @property
    def embedding_dim(self) -> int:
        """Get the embedding size."""
        return self.train_dataset.embedding_dim
