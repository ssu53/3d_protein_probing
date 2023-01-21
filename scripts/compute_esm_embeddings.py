"""Compute protein residue embeddings using an ESM2 model from https://github.com/facebookresearch/esm."""
from time import time
from pathlib import Path

import torch
from esm import Alphabet, BatchConverter, ESM2
from tqdm import trange


def load_esm_model(
        hub_dir: str,
        esm_model: str
) -> tuple[ESM2, Alphabet, BatchConverter]:
    """Load an ESM2 model and batch converter.

    :param hub_dir: Path to directory where torch hub models are saved.
    :param esm_model: Pretrained ESM2 model to use. See options at https://github.com/facebookresearch/esm.
    :return: A tuple of a pretrained ESM2 model and a BatchConverter for preparing protein sequences as input.
    """
    torch.hub.set_dir(hub_dir)
    model, alphabet = torch.hub.load('facebookresearch/esm:main', esm_model)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    return model, alphabet, batch_converter


def compute_esm_embeddings(
        data_dir: Path,
        hub_dir: Path,
        esm_model: str,
        last_layer: int,
        save_path: Path,
        batch_size: int
) -> None:
    """Compute protein residue embeddings using an ESM2 model from https://github.com/facebookresearch/esm.

    :param data_dir: Path to directory containing protein structure/sequence PyTorch files.
    :param hub_dir: Path to directory where torch hub models are saved.
    :param esm_model: Pretrained ESM2 model to use. See options at https://github.com/facebookresearch/esm.
    :param last_layer: Last layer of the ESM2 model, which will be used to extract embeddings.
    :param save_path: Path to PT file where a dictionary mapping protein name to embeddings will be saved.
    :param batch_size: The batch size.
    """
    # Load map from PDB ID to protein sequence
    pdb_ids_and_sequences = []
    for protein_path in data_dir.glob('*.pt'):
        protein = torch.load(protein_path)
        pdb_ids_and_sequences.append((protein['pdb_id'], protein['sequence']))

    print(f'Loaded {len(pdb_ids_and_sequences):,} proteins')

    # Load ESM2 model
    model, alphabet, batch_converter = load_esm_model(
        hub_dir=str(hub_dir),
        esm_model=esm_model
    )

    # Move model to device
    model = model.cuda()

    # Compute ESM2 embeddings
    start = time()
    pdb_id_to_embedding = {}

    with torch.no_grad():
        # Iterate over batches of sequences
        for i in trange(0, len(pdb_ids_and_sequences), batch_size):
            # Get batch of sequences
            batch_pdb_ids_and_sequences = pdb_ids_and_sequences[i:i + batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_pdb_ids_and_sequences)
            batch_tokens = batch_tokens.cuda()

            # Compute embeddings
            results = model(batch_tokens, repr_layers=[last_layer], return_contacts=False)

            # Get per-residue embeddings
            batch_embeddings = results['representations'][last_layer].cpu()

            # Map sequence name to embedding
            for (pdb_id, sequence), embedding in zip(batch_pdb_ids_and_sequences, batch_embeddings):
                # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1
                pdb_id_to_embedding[pdb_id] = embedding[1:len(sequence) + 1]

    print(f'Time = {time() - start} seconds for {len(pdb_ids_and_sequences):,} sequences')

    assert len(pdb_ids_and_sequences) == len(pdb_id_to_embedding)

    # Save embeddings
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pdb_id_to_embedding, save_path)


if __name__ == '__main__':
    from tap import Tap

    class Args(Tap):
        data_dir: Path
        """"Path to directory containing protein structure/sequence PyTorch files."""
        hub_dir: Path
        """Path to directory where torch hub models are saved."""
        esm_model: str
        """Pretrained ESM2 model to use. See options at https://github.com/facebookresearch/esm."""
        last_layer: int
        """Last layer of the ESM2 model, which will be used to extract embeddings."""
        save_path: Path
        """Path to PT file where a dictionary mapping PDB ID to embeddings will be saved."""
        batch_size: int = 100
        """The number of sequences to process at once."""

    compute_esm_embeddings(**Args().parse_args().as_dict())
