# %%

from tqdm import tqdm

import torch
from pp3.models.egnn import SinusoidalEmbeddings


pdb_id_to_embeddings = {}


pdb_id_to_foldseek_feat = torch.load('/home/groups/jamesz/shiye/foldseek-analysis/training/data_dev/encodings_foldseek.pt')

pos_dim = 32
se = SinusoidalEmbeddings(pos_dim)

# %%

for pdb_id in tqdm(pdb_id_to_foldseek_feat):

    foldseek_feat = pdb_id_to_foldseek_feat[pdb_id].to(torch.float32)
    num_res = len(foldseek_feat)
    assert foldseek_feat.shape == (num_res, 10), foldseek_feat.shape

    # fill terminal residues (which have nan features) with the average
    foldseek_feat_mean = foldseek_feat.nanmean(dim=0)
    foldseek_feat[0,:] = foldseek_feat_mean
    foldseek_feat[-1,:] = foldseek_feat_mean

    pos_feat = se(torch.arange(1, num_res + 1))

    pdb_id_to_embeddings[pdb_id] = torch.cat((foldseek_feat, pos_feat), dim=-1)

# %%

torch.save(pdb_id_to_embeddings, '/home/groups/jamesz/shiye/3d_protein_probing/data/embed_for_retrieval/encodings/encodings_foldseek_sinusoid.pt')
# %%
