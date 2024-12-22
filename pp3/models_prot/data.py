# %%

from typing import List
from pathlib import Path
from functools import partial
import random

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl



def collate_fn_single_pos(
    batch: list[tuple[str, str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    num_negatives: int = 16,
    pdb_ids: list[str] = None,
    pdb_id_to_embeddings: dict[str, torch.Tensor] = None,
    pdb_id_to_coordinates: dict[str, torch.Tensor] = None,
):
    """
    :param batch: A batch of items, where each item contains
        - pdb_ids_1 first protein of paired pdb_ids
        - pdb_ids_2 second protein of paired pdb_ids
        - embedding_1 (num_residues_1, embedding_dim) of first protein
        - embedding_2 (num_residues_2, embedding_dim) of second protein
    :param num_negatives: Number of negative contrastive examples in batch
    :param pdb_id_to_embeddings: Embeddings for entire protein dataset.
        Draw randomly from this for negative examples.
    """

    assert len(batch) == 1 # Only one positive pair

    pdb_id_1, pdb_id_2, embedding_1, embedding_2, coord_1, coord_2, target = batch[0]

    include_coords = coord_1 is not None

    negative_keys = random.sample(pdb_ids, num_negatives)
    negative_embeddings = [pdb_id_to_embeddings[key] for key in negative_keys]
    if include_coords:
        negative_coords = [pdb_id_to_coordinates[key] for key in negative_keys]

    pdb_ids = [
        pdb_id_1,
        pdb_id_2,
        *negative_keys,
    ]

    embeddings = [
        embedding_1,
        embedding_2,
        *negative_embeddings,
    ]

    if include_coords:
        coords = [
            coord_1,
            coord_2,
            *negative_coords,
        ]

    lengths = [embedding.shape[0] for embedding in embeddings]
    max_seq_len = max(lengths)
    valid_positions = torch.tensor([[1] * length + [0] * (max_seq_len - length) for length in lengths])
    padding_mask = ~valid_positions.bool() # True where padding token
    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    if include_coords:
        coords = torch.nn.utils.rnn.pad_sequence(coords, batch_first=True)

    return {
        'pdb_ids': pdb_ids, 
        'embeddings': embeddings,
        'coords': coords,
        'padding_mask': padding_mask,
        'targets': None,
    }



def collate_fn_pairs(
    batch: list[tuple[str, str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
):
    """
    """

    pdb_ids_1, pdb_ids_2, embeddings_1, embeddings_2, coords_1, coords_2, targets = zip(*batch)

    include_coords = coords_1[0] is not None

    pdb_ids = [
        *pdb_ids_1,
        *pdb_ids_2,
    ]
    
    embeddings = [
        *embeddings_1,
        *embeddings_2,
    ]

    if include_coords:
        coords = [
            *coords_1,
            *coords_2,
        ]

    if targets[0] is not None:
        targets = torch.tensor(targets, dtype=torch.float32)
    else:
        targets = None

    lengths = [embedding.shape[0] for embedding in embeddings]
    max_seq_len = max(lengths)
    valid_positions = torch.tensor([[1] * length + [0] * (max_seq_len - length) for length in lengths])
    padding_mask = ~valid_positions.bool() # True where padding token
    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    if include_coords:
        coords = torch.nn.utils.rnn.pad_sequence(coords, batch_first=True)

    return {
        'pdb_ids': pdb_ids, 
        'embeddings': embeddings,
        'coords': coords,
        'padding_mask': padding_mask,
        'targets': targets,
    }




class ProteinPairDataset(Dataset):

    def __init__(
            self,
            pdb_ids_1: list[str],
            pdb_ids_2: list[str],
            targets: list[float],
            pdb_id_to_embeddings: dict[str, torch.Tensor],
            pdb_id_to_coordinates: dict[str, torch.Tensor],
    ) -> None:

        assert len(pdb_ids_1) == len(pdb_ids_2)

        self.pdb_ids_1 = pdb_ids_1
        self.pdb_ids_2 = pdb_ids_2
        self.targets = targets
        self.pdb_id_to_embeddings = pdb_id_to_embeddings
        self.pdb_id_to_coordinates = pdb_id_to_coordinates

        self.rng = np.random.default_rng(seed=0)

    @property
    def embedding_dim(self) -> int:
        """Get the embedding size."""
        embeddings = next(iter(self.pdb_id_to_embeddings.values()))
        embedding_dim = embeddings.shape[-1]

        return embedding_dim

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return len(self.pdb_ids_1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get an item from the dataset.

        :param index: The index of the item.
        :return: A tuple pair of the two proteins' residue-level embeddings.
        """

        pdb_id_1 = self.pdb_ids_1[index]
        pdb_id_2 = self.pdb_ids_2[index]

        if self.targets is not None:
            target = self.targets[index]
        else:
            target = None

        embedding_1 = self.pdb_id_to_embeddings[pdb_id_1]
        embedding_2 = self.pdb_id_to_embeddings[pdb_id_2]

        if self.pdb_id_to_coordinates is not None:
            coord_1 = self.pdb_id_to_coordinates[pdb_id_1]
            coord_2 = self.pdb_id_to_coordinates[pdb_id_2]
        else:
            coord_1, coord_2 = None, None

        return pdb_id_1, pdb_id_2, embedding_1, embedding_2, coord_1, coord_2, target


class ProteinPairDataModule(pl.LightningDataModule):

    def __init__(
        self,
        embeddings_path: Path,
        proteins_path: Path,
        pdb_ids_train_path: Path,
        pdb_ids_val_path: Path,
        pairfile_train_path: Path,
        pairfile_val_path: Path,
        is_supervised: bool,
        batch_size: int,
        num_workers: int = 4,
    ) -> None:
        """
        specify proteins_path if require coordinates for pre-encoding with an EGNN. 
        set to None to omit this field from datasets.
        """

        super().__init__()
        print(f"{is_supervised=}")
        if not is_supervised:
            assert batch_size > 2, "batch size must be greater than 2 to include one positive pair"

        self.embeddings_path = embeddings_path
        self.proteins_path = proteins_path
        self.pdb_ids_train_path = pdb_ids_train_path
        self.pdb_ids_val_path = pdb_ids_val_path
        self.pairfile_train_path = pairfile_train_path
        self.pairfile_val_path = pairfile_val_path
        self.is_supervised = is_supervised
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.batch_of_pairs = True

        self.train_dataset: ProteinPairDataset | None = None
        self.val_dataset: ProteinPairDataset | None = None
        self.val_dataset_1: ProteinPairDataset | None = None
        self.val_dataset_2: ProteinPairDataset | None = None
        self.val_dataset_3: ProteinPairDataset | None = None


    def setup(self, stage: str | None = None) -> None:


        self.pdb_id_to_embeddings = torch.load(self.embeddings_path)
        if self.proteins_path is not None:
            pdb_id_to_proteins = torch.load(self.proteins_path)
            self.pdb_id_to_coordinates = {k: v["structure"] for k, v in pdb_id_to_proteins.items()}
        self.pdb_ids_train = pd.read_csv(self.pdb_ids_train_path, header=None)[0].tolist()
        self.pdb_ids_val = pd.read_csv(self.pdb_ids_val_path, header=None)[0].tolist()
        assert set(self.pdb_ids_train + self.pdb_ids_val) == set(self.pdb_id_to_embeddings)
        # self.pdb_ids = list(self.pdb_id_to_embeddings)

        
        pairfile_train = pd.read_csv(self.pairfile_train_path, header=None, sep=' ')
        pairfile_val = pd.read_csv(self.pairfile_val_path, header=None, sep=' ')

        path_local = Path(__file__).parent
        pairfile_val_1 = pd.read_csv(path_local / '../../data/embed_for_retrieval/train_data/tmaln-06_data_val.csv', header=None, sep=' ')
        pairfile_val_2 = pd.read_csv(path_local / '../../data/embed_for_retrieval/train_data/tmaln_data_val_within_fold.csv', header=None, sep=' ')
        pairfile_val_3 = pd.read_csv(path_local / '../../data/embed_for_retrieval/train_data/tmaln_data_val_random_pairs.csv', header=None, sep=' ')


        # Create train dataset
        self.train_dataset = ProteinPairDataset(
            pdb_ids_1=pairfile_train[0].tolist(),
            pdb_ids_2=pairfile_train[1].tolist(),
            targets=pairfile_train[2].tolist() if self.is_supervised else None,
            pdb_id_to_embeddings=self.pdb_id_to_embeddings,
            pdb_id_to_coordinates=self.pdb_id_to_coordinates if self.proteins_path is not None else None
        )
        print(f'Train dataset size: {len(self.train_dataset):,}')


        # Create val dataset
        self.val_dataset = ProteinPairDataset(
            pdb_ids_1=pairfile_val[0].tolist(),
            pdb_ids_2=pairfile_val[1].tolist(),
            targets=pairfile_val[2].tolist() if self.is_supervised else None,
            pdb_id_to_embeddings=self.pdb_id_to_embeddings,
            pdb_id_to_coordinates=self.pdb_id_to_coordinates if self.proteins_path is not None else None
        )
        print(f'Val dataset size: {len(self.val_dataset):,}')

        self.val_dataset_1 = ProteinPairDataset(
            pdb_ids_1=pairfile_val_1[0].tolist(),
            pdb_ids_2=pairfile_val_1[1].tolist(),
            targets=pairfile_val_1[2].tolist() if self.is_supervised else None,
            pdb_id_to_embeddings=self.pdb_id_to_embeddings,
            pdb_id_to_coordinates=self.pdb_id_to_coordinates if self.proteins_path is not None else None
        )
        print(f'Val 1 dataset size: {len(self.val_dataset_1):,}')

        self.val_dataset_2 = ProteinPairDataset(
            pdb_ids_1=pairfile_val_2[0].tolist(),
            pdb_ids_2=pairfile_val_2[1].tolist(),
            targets=pairfile_val_2[2].tolist() if self.is_supervised else None,
            pdb_id_to_embeddings=self.pdb_id_to_embeddings,
            pdb_id_to_coordinates=self.pdb_id_to_coordinates if self.proteins_path is not None else None
        )
        print(f'Val 2 dataset size: {len(self.val_dataset_2):,}')

        self.val_dataset_3 = ProteinPairDataset(
            pdb_ids_1=pairfile_val_3[0].tolist(),
            pdb_ids_2=pairfile_val_3[1].tolist(),
            targets=pairfile_val_3[2].tolist() if self.is_supervised else None,
            pdb_id_to_embeddings=self.pdb_id_to_embeddings,
            pdb_id_to_coordinates=self.pdb_id_to_coordinates if self.proteins_path is not None else None
        )
        print(f'Val 3 dataset size: {len(self.val_dataset_3):,}')

        self.is_setup = True
    

    def train_dataloader(self) -> DataLoader:
        """Get the train data loader."""
        if self.is_supervised or self.batch_of_pairs:
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn_pairs,
            )
        else:
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=1,  # one positive pair per batch
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=partial(
                    collate_fn_single_pos, 
                    num_negatives=self.batch_size-2, 
                    pdb_ids=self.pdb_ids_train, 
                    pdb_id_to_embeddings=self.pdb_id_to_embeddings,
                    pdb_id_to_coordinates=self.pdb_id_to_coordinates,
                ),
            )

    def val_dataloader_(self) -> DataLoader:
        """Get the validation data loader."""
        if self.is_supervised or self.batch_of_pairs:
            return DataLoader(
                dataset=self.val_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn_pairs,
            )
        else:
            return DataLoader(
                dataset=self.val_dataset,
                batch_size=1,  # one positive pair per batch
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=partial(
                    collate_fn_single_pos, 
                    num_negatives=self.batch_size-2, 
                    pdb_ids=self.pdb_ids_val, 
                    pdb_id_to_embeddings=self.pdb_id_to_embeddings,
                    pdb_id_to_coordinates=self.pdb_id_to_coordinates,
                ),
            )
    
    def val_dataloader(self) -> List[DataLoader]:
        """Get the validation data loaders."""
        assert self.is_supervised and self.batch_of_pairs 
        dl0 = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn_pairs,
            )
        dl1 = DataLoader(
                dataset=self.val_dataset_1,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn_pairs,
            )
        dl2 = DataLoader(
                dataset=self.val_dataset_2,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn_pairs,
            )
        dl3 = DataLoader(
                dataset=self.val_dataset_3,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn_pairs,
            )
        return [dl0, dl1, dl2, dl3]


    @property
    def embedding_dim(self) -> int:
        """Get the embedding size."""
        return self.train_dataset.embedding_dim


# %%



def get_train_dataloader():

    pairfile_train = pd.read_csv(
        '/home/groups/jamesz/shiye/foldseek-analysis/metrics/pairfile_train.out',
        header=None,
        sep=' ',
        )

    pdb_ids_1 = pairfile_train[0].tolist()
    pdb_ids_2 = pairfile_train[1].tolist()

    pdb_id_to_embeddings = torch.load('/home/groups/jamesz/shiye/foldseek-analysis/training/data_dev/encodings_v6.pt')
    pdb_ids_all = list(pdb_id_to_embeddings)
    

    train_dataset = ProteinPairDataset(
        pdb_ids_1=pdb_ids_1,
        pdb_ids_2=pdb_ids_2,
        pdb_id_to_embeddings=pdb_id_to_embeddings,
    )


    num_workers = 0
    batch_size = 16
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=partial(
            collate_fn_single_pos, 
            num_negatives=batch_size-2, 
            pdb_ids=pdb_ids_all, 
            pdb_id_to_embeddings=pdb_id_to_embeddings)
    )

    return train_dataloader





def test():

    pairfile_train = pd.read_csv(
        '/home/groups/jamesz/shiye/foldseek-analysis/metrics/pairfile_train.out',
        header=None,
        sep=' ',
        )

    pdb_ids_1 = pairfile_train[0].tolist()
    pdb_ids_2 = pairfile_train[1].tolist()

    pdb_id_to_embeddings = torch.load('/home/groups/jamesz/shiye/foldseek-analysis/training/data_dev/encodings_v6.pt')
    pdb_ids_all = list(pdb_id_to_embeddings)

    pairfile_train_set = {f"{pdb_id_1} {pdb_id_2}" for pdb_id_1, pdb_id_2 in zip(pdb_ids_1, pdb_ids_2)}

    # %%


    train_dataset = ProteinPairDataset(
        pdb_ids_1=pdb_ids_1,
        pdb_ids_2=pdb_ids_2,
        pdb_id_to_embeddings=pdb_id_to_embeddings,
    )

    # %%



    num_workers = 0
    batch_size = 16
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=partial(
            collate_fn_single_pos, 
            num_negatives=batch_size-2, 
            pdb_ids=pdb_ids_all, 
            pdb_id_to_embeddings=pdb_id_to_embeddings)
    )


    foo = 0
    for batch in train_dataloader:
        assert len(batch) == 3
        print(f"{len(batch[0])} {batch[1].shape} {batch[2].shape}")
        batch_ids = batch[0]
        # print(batch_ids)
        for i in range(len(batch_ids)):
            for j in range(len(batch_ids)):
                key = f"{batch_ids[i]} {batch_ids[j]}"
                if i != 0 and j != 1 and key in pairfile_train_set:
                    print("accidental positive pair", key)
        foo += 1
        if foo == 10: break

    # %%



    num_workers = 0
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_paired_pos,
    )


    foo = 0
    for batch in train_dataloader:
        assert len(batch) == 3
        print(f"{len(batch[0])} {batch[1].shape} {batch[2].shape}")
        batch_ids = batch[0]
        print(batch_ids)
        for i in range(len(batch_ids)):
            for j in range(len(batch_ids)):
                key = f"{batch_ids[i]} {batch_ids[j]}"
                if i+len(batch_ids)//2 != j and key in pairfile_train_set:
                    print("accidental positive pair", key)
        foo += 1
        if foo == 1: break


    # %%


def test():

    import pp3.models_prot.default_paths as default_paths

    embeddings_path = '/home/groups/jamesz/shiye/3d_protein_probing/data/embed_for_retrieval/encodings/encodings_aa_onehot_v6.pt'
    valid_pdb_ids_train_path = default_paths.get_valid_pdb_ids_train_path()
    valid_pdb_ids_val_path = default_paths.get_valid_pdb_ids_val_path()
    pairfile_train_path = default_paths.get_tmaln_data_train_path() 
    pairfile_val_path = default_paths.get_tmaln_data_val_path()

    data_module = ProteinPairDataModule(
        embeddings_path=embeddings_path,
        pdb_ids_train_path=valid_pdb_ids_train_path,
        pdb_ids_val_path=valid_pdb_ids_val_path,
        pairfile_train_path=pairfile_train_path,
        pairfile_val_path=pairfile_val_path,
        is_supervised=True,
        batch_size=16,
        num_workers=0,
    )
    data_module.setup()
    
    for batch in data_module.train_dataloader():
        pdb_ids, embeddings, padding_mask, targets = batch
        break
    
    print(len(pdb_ids))
    print(embeddings.shape)
    print(padding_mask.shape)
    print(targets.shape)
# %%
