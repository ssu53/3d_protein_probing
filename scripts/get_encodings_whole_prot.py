# %%

from pathlib import Path
from tqdm import tqdm

import pandas as pd
import einops
import torch
from torch.utils.data import Dataset, DataLoader
from pp3.models_prot.model import ModelProt
from pp3.models_prot import default_paths


def compute_encodings(model, dataloader, save_path):

    pdb_id_to_encodings_whole_prot = {}

    # Progress bar
    pbar = tqdm(total=len(dataloader))

    # Forward pass on each batch
    for i,batch in enumerate(dataloader):

        pdb_ids = batch['pdb_ids']
        embeddings = batch['embeddings']
        coords = batch['coords']
        padding_mask = batch['padding_mask']

        embeddings = embeddings.to(model.device)
        coords = coords.to(model.device)
        padding_mask = padding_mask.to(model.device)

        with torch.no_grad():
            batch_encodings = model.forward(embeddings, coords, padding_mask, mode='val')
        
        # Extract the encodings per protein
        for j in range(len(batch_encodings)):

            pdb_id = pdb_ids[j]
            prot_encoding = batch_encodings[j]
            pdb_id_to_encodings_whole_prot[pdb_id] = prot_encoding.cpu()
        
        pbar.update(len(batch_encodings))

    # One encoding at a time... too slow
    # for pdb_id,encoding in tqdm(pdb_id_to_encodings.items()):

    #     encoding = encoding.to(model.device)
    #     num_res = encoding.size(0)
    #     encoding = einops.rearrange(encoding, 's d -> 1 s d')

    #     padding_mask = torch.full((1,num_res), fill_value=False)
    #     padding_mask = padding_mask.to(model.device)
    #     # print(encoding.shape, padding_mask.shape)

    #     with torch.no_grad():
    #         encoding_whole_prot = model.encoder(x=encoding, pad_mask=padding_mask)
    #     # print(encoding_whole_prot.shape)

    #     pdb_id_to_encodings_whole_prot[pdb_id] = encoding_whole_prot.cpu()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pdb_id_to_encodings_whole_prot, save_path)

    return pdb_id_to_encodings_whole_prot




class ProteinEncodingDataset(Dataset):

    def __init__(
        self,
        pdb_ids,
        pdb_id_to_embeddings: dict[str, torch.Tensor],
        pdb_id_to_coordinates: dict[str, torch.Tensor],
    ) -> None:

        self.pdb_ids = pdb_ids
        self.pdb_id_to_embeddings = pdb_id_to_embeddings
        self.pdb_id_to_coordinates = pdb_id_to_coordinates

    @property
    def embedding_dim(self) -> int:
        """Get the embedding size."""
        embeddings = next(iter(self.pdb_id_to_embeddings.values()))
        embedding_dim = embeddings.shape[-1]

        return embedding_dim

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return len(self.pdb_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get an item from the dataset.

        :param index: The index of the item.
        :return: A tuple pair of the two proteins' residue-level embeddings.
        """

        pdb_id = self.pdb_ids[index]
        embedding = self.pdb_id_to_embeddings[pdb_id]
        if self.pdb_id_to_coordinates is not None:
            coord = self.pdb_id_to_coordinates[pdb_id]
        else:
            coord = None

        return pdb_id, embedding, coord


def collate_fn(batch):
    
    pdb_ids, embeddings, coords = zip(*batch)

    lengths = [embedding.shape[0] for embedding in embeddings]
    max_seq_len = max(lengths)
    valid_positions = torch.tensor([[1] * length + [0] * (max_seq_len - length) for length in lengths])
    padding_mask = ~valid_positions.bool() # True where padding token
    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    if coords[0] is not None:
        coords = torch.nn.utils.rnn.pad_sequence(coords, batch_first=True)

    return {
        'pdb_ids': pdb_ids, 
        'embeddings': embeddings,
        'coords': coords,
        'padding_mask': padding_mask,
    }



# %%

def main():

    root_dir = Path(__file__).parent / '..'

    encoding_name = 'encodings_foldseek_sinusoid'
    # checkpoint_path = root_dir / f'results/embed_for_retrieval/{encoding_name}_3L_0.0001lr_32bs_0.1temp/epoch=8-step=185481.ckpt'
    # checkpoint_path = root_dir / f'results/embed_for_retrieval/{encoding_name}_3L_0.0001lr_32bs_0.01temp/epoch=8-step=185481.ckpt'
    # checkpoint_path = root_dir / f'results/embed_for_retrieval/{encoding_name}_l1_3L_0.0001lr_32bs/epoch=25-step=32994.ckpt'
    # checkpoint_path = root_dir / f'results/embed_for_retrieval/{encoding_name}_l1_3L_0.0001lr_32bs/epoch=49-step=32250.ckpt'
    # checkpoint_path = root_dir / f'results/embed_for_retrieval_egnn/{encoding_name}_l1_2L_0.0001lr_32bs_egnn/epoch=49-step=63450.ckpt'
    checkpoint_path = root_dir / f'results/embed_for_retrieval_egnn/{encoding_name}_l1_4L_0.0001lr_16bs_egnn_large/epoch=44-step=114210.ckpt'
    load_path = root_dir / f'data/embed_for_retrieval/encodings/{encoding_name}.pt'
    save_path = root_dir / f'data/embed_for_retrieval/encodings_whole_prot/{encoding_name}_v2.pt'

    print(f"{checkpoint_path=}")
    print(f"{load_path=}")
    print(f"{save_path=}")

    # Load residue-level encodings
    pdb_id_to_embeddings = torch.load(load_path)


    proteins_path = default_paths.get_proteins_path()
    valid_pdb_ids_train_path = default_paths.get_valid_pdb_ids_train_path()
    valid_pdb_ids_val_path = default_paths.get_valid_pdb_ids_val_path()

    pdb_id_to_proteins = torch.load(proteins_path)
    pdb_id_to_coordinates = {k: v["structure"] for k, v in pdb_id_to_proteins.items()}
    pdb_ids_train = pd.read_csv(valid_pdb_ids_train_path, header=None)[0].tolist()
    pdb_ids_val = pd.read_csv(valid_pdb_ids_val_path, header=None)[0].tolist()
    pdb_ids = pdb_ids_train + pdb_ids_val
    print(f"{len(pdb_id_to_embeddings)} {len(pdb_id_to_coordinates)} {len(pdb_ids)}")

    dataset = ProteinEncodingDataset(pdb_ids, pdb_id_to_embeddings, pdb_id_to_coordinates)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Load model
    model = ModelProt.load_from_checkpoint(checkpoint_path)
    model.freeze()
    model.eval()

    # %%

    # Compute and save encodings
    pdb_id_to_encodings_whole_prot = compute_encodings(model, dataloader, save_path)

    print(f"Computed encodings and saved to {save_path}")



if __name__ == '__main__':
    main()

# %%
