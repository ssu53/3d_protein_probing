# %%

from pathlib import Path
from tqdm import tqdm

import einops
import torch
from torch.utils.data import Dataset, DataLoader
from pp3.models_prot.model import ModelProt



def compute_encodings(model, dataloader, save_path):

    pdb_id_to_encodings_whole_prot = {}

    # Progress bar
    pbar = tqdm(total=len(dataloader))

    # Forward pass on each batch
    for i,batch in enumerate(dataloader):

        pdb_ids, embeddings, padding_mask = batch
        embeddings = embeddings.to(model.device)
        padding_mask = padding_mask.to(model.device)

        with torch.no_grad():
            batch_encodings = model.encoder(x=embeddings, pad_mask=padding_mask)
        
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
        pdb_id_to_embeddings: dict[str, torch.Tensor],
    ) -> None:

        self.pdb_id_to_embeddings = pdb_id_to_embeddings
        self.pdb_ids = list(pdb_id_to_embeddings.keys())

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

        return pdb_id, embedding


def collate_fn(batch):
    
    pdb_ids, embeddings = zip(*batch)

    lengths = [embedding.shape[0] for embedding in embeddings]
    max_seq_len = max(lengths)
    valid_positions = torch.tensor([[1] * length + [0] * (max_seq_len - length) for length in lengths])
    padding_mask = ~valid_positions.bool() # True where padding token
    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)

    return pdb_ids, embeddings, padding_mask



# %%

def main():

    root_dir = Path(__file__).parent / '..'

    encoding_name = 'encodings_aa_onehot_v6'
    checkpoint_path = root_dir / f'results/embed_for_retrieval/{encoding_name}_3L_0.0001lr_32bs_0.1temp/epoch=8-step=185481.ckpt'
    load_path = root_dir / f'data/embed_for_retrieval/encodings/{encoding_name}.pt'
    save_path = root_dir / f'data/embed_for_retrieval/encodings_whole_prot/{encoding_name}.pt'


    # Load residue-level encodings
    pdb_id_to_encodings = torch.load(load_path)

    # %%

    dataset = ProteinEncodingDataset(pdb_id_to_encodings)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
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
