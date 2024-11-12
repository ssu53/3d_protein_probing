# %%

import os
import numpy as np
import pandas as pd
import torch

encoding_name = 'encodings_aa_onehot_v6_v1'
load_path = f'/home/groups/jamesz/shiye/3d_protein_probing/data/embed_for_retrieval/encodings_whole_prot/{encoding_name}.pt'
save_dir = f'/home/groups/jamesz/shiye/protein_vector_retrieve/{encoding_name}_train'

# save_dir.parent.mkdir(parents=True, exist_ok=True)

pdb_ids = pd.read_csv(
    '/home/groups/jamesz/shiye/foldseek-analysis/training/data_dev/valid_pdb_ids_train.csv',
    header=None)
print(pdb_ids)

pdb_id_to_encodings = torch.load(load_path)

encodings = []

for pdb_id in pdb_ids[0]:
    encodings.append(pdb_id_to_encodings[pdb_id])


# %%
encodings = np.stack(encodings, axis=0)
print(encodings.shape)

# %%

pdb_ids.to_csv(os.path.join(save_dir, 'pdb_ids.csv'), header=None, index=False)
np.save(os.path.join(save_dir, 'seq_reps'), encodings)


# %%
