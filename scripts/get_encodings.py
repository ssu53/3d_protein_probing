# %%

from pathlib import Path
from functools import partial
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from pp3.data import ProteinConceptDataset, collate_fn
from pp3.baseline_embeddings import get_baseline_residue_embedding
from pp3.concepts import get_concept_level, get_concept_type, get_concept_output_dim
from pp3.models.model import Model



def get_data(
    proteins_path: Path,
    concept: str,
    concepts_dir: Path, # not used
    batch_size: int,
    num_workers: int = 4,
):

    # Load PDB ID to protein dictionary with sequence and structure
    pdb_id_to_proteins: dict[str, dict[str, torch.Tensor | str]] = torch.load(proteins_path)

    # Get PDB IDs
    pdb_ids = sorted(pdb_id_to_proteins)

    # Load or compute embeddings for proteins or residues
    pdb_id_to_embeddings = {
        pdb_id: get_baseline_residue_embedding(protein['sequence'], identify_residue=False)
        for pdb_id, protein in pdb_id_to_proteins.items()
    }

    # Load id to coordinate dictionary
    pdb_id_to_coordinates = {k: v["structure"] for k, v in pdb_id_to_proteins.items()}

    # Load PDB ID to concept value dictionary
    # We won't need this to get the encodings, but should compute and check the performance on these in future.
    # pdb_id_to_concept_value: dict[str, torch.Tensor | float] = torch.load(concepts_dir / f'{self.concept}.pt')
    # for now, use dummy concept tensors
    print(f"{get_concept_level(concept)=}")
    if get_concept_level(concept) == 'residue_multivariate':
        pdb_id_to_concept_value: dict[str, torch.Tensor | float] = {
            pdb_id: torch.zeros((len(pdb_id_to_embeddings[pdb_id]), 1)) for pdb_id in pdb_ids
        }
    elif get_concept_level(concept) == 'residue_pair':
        pdb_id_to_concept_value: dict[str, torch.Tensor | float] = {
            pdb_id: torch.zeros((len(pdb_id_to_embeddings[pdb_id]), len(pdb_id_to_embeddings[pdb_id]))) for pdb_id in pdb_ids
        }
    else:
        pdb_id_to_concept_value: dict[str, torch.Tensor | float] = {
            pdb_id: torch.zeros((len(pdb_id_to_embeddings[pdb_id]),)) for pdb_id in pdb_ids
        }

    # Ensure that the PDB IDs are the same across dictionaries
    assert set(pdb_id_to_proteins) == set(pdb_id_to_embeddings) == set(pdb_id_to_concept_value)


    concept_level = get_concept_level(concept)
    concept_type = get_concept_type(concept)

    dataset = ProteinConceptDataset(
        pdb_ids=pdb_ids,
        pdb_id_to_protein=pdb_id_to_proteins,
        pdb_id_to_embeddings=pdb_id_to_embeddings,
        pdb_id_to_concept_value=pdb_id_to_concept_value,
        concept_level=concept_level,
        concept_type=concept_type,
        pdb_id_to_coordinates=pdb_id_to_coordinates
    )
    print(f'Dataset size: {len(dataset):,}')


    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, concept_level=concept_level)
    )

    return dataset, dataloader



def get_model_(
    concept = None,
    checkpoint_path = None,
    encoder_type = 'egnn',
    input_dim = 5,
    encoder_num_layers = 3,
    encoder_hidden_dim = 16,
    predictor_num_layers = 2,
    predictor_hidden_dim = 100,
    dropout = 0.,
    max_neighbors = 24,
):
    """
    Requires model hyperparameters.
    Use Model.load_from_checkpoint instead if the pl.LightningModule model was run with save_hyperparameters().
    """


    model = Model(
        encoder_type=encoder_type,
        input_dim=input_dim,
        output_dim=get_concept_output_dim(concept),
        encoder_num_layers=encoder_num_layers,
        encoder_hidden_dim=encoder_hidden_dim,
        predictor_num_layers=predictor_num_layers,
        predictor_hidden_dim=predictor_hidden_dim,
        concept_level=get_concept_level(concept),
        target_type=get_concept_type(concept),
        target_mean=None,       # not used if only extracting embeddings
        target_std=None,        # not used if only extracting embeddings
        learning_rate=None,     # not used if only extracting embeddings
        weight_decay=None,      # not used if only extracting embeddings
        dropout=dropout,
        max_neighbors=max_neighbors
    )

    model.eval()
    state_dict_loaded = torch.load(
        checkpoint_path, 
        # map_location=torch.device('cpu'),
    )
    model.load_state_dict(state_dict_loaded['state_dict'])

    return model



def compute_encodings(
    model,
    dataset,
    dataloader,
    encodings_save_path: Path,
    ) -> dict[str, torch.Tensor]:

    pdb_id_to_encodings = {}

    # Progress bar
    pbar = tqdm(total=len(dataset))

    # Forward pass on each batch
    for i,batch in enumerate(dataloader):

        embeddings, coords, y, padding_mask = batch
        embeddings = embeddings.to(model.device)
        coords = coords.to(model.device)
        padding_mask = padding_mask.to(model.device)

        with torch.no_grad():
            batch_encodings = model.encoder(embeddings, coords, padding_mask)
        
        # Extract the encodings per protein
        for j in range(len(batch_encodings)):

            prot_encoding = batch_encodings[j][padding_mask[j].bool(),:]
            
            pdb_id = dataset.pdb_ids[i*dataloader.batch_size + j]
            num_residues = dataset.pdb_id_to_coordinates[pdb_id].shape[0]
            assert len(prot_encoding) == num_residues

            pdb_id_to_encodings[pdb_id] = prot_encoding.cpu()

        pbar.update(len(batch_encodings))
    
    
    encodings_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pdb_id_to_encodings, encodings_save_path)

    return pdb_id_to_encodings




def main(
    proteins_path: Path,
    concept: str, 
    concepts_dir: Path = None,
    batch_size: int = 8, 
    checkpoint_path: Path = None, 
    encodings_save_path: Path = None,
    ):

    # Load data
    dataset, dataloader = get_data(
        proteins_path=proteins_path,
        concept=concept, 
        concepts_dir=concepts_dir,
        batch_size=batch_size,
    )

    # Load model
    model = Model.load_from_checkpoint(checkpoint_path)
    model.freeze()
    model.eval()

    # Compute and save encodings
    pdb_id_to_encodings = compute_encodings(model, dataset, dataloader, encodings_save_path)
    print(f"Computed encodings and saved to {encodings_save_path}")


if __name__ == '__main__':
    from tap import tapify

    tapify(main)

    """
    e.g.

    python scripts/get_encodings.py \
        --proteins_path data/scope40_foldseek_compatible/proteins.pt \
        --concept residue_neighb_distances_8 \
        --checkpoint_path results/scope40_foldseek_compatible/basic_embed_dim2/residue_neighb_distances_8_baseline-basic_egnn_3L_mlp_2L_split_0/epoch\=99-step\=88300.ckpt \
        --encodings_save_path data/scope40_foldseek_compatible/encodings/residue_neighb_distances_8.pt

    """
