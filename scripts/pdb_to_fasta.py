# %%

import pandas as pd
from tqdm import tqdm

from Bio.PDB import PDBParser

from pp3.utils.constants import AA_3_TO_1
AA_3_TO_1['UNK'] = 'X'



pdb_ids = pd.read_csv('/home/groups/jamesz/shiye/3d_protein_probing/data/scope40_foldseek_compatible/valid_pdb_ids_val.csv')
pdb_ids = pdb_ids.pdb_id.tolist()



for pdb_id in tqdm(pdb_ids):

    pdbfile = f'/scratch/groups/jamesz/shiye/scope40/{pdb_id}'

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('None', pdbfile)

    model = structure[0]  # take only first model
    chain = list(model.get_chains())[0]  # take only first chain

    sequence = []
    for residue in chain: 
        het_flag,_ ,_ = residue.id
        if het_flag != ' ':
            continue
        sequence.append(AA_3_TO_1[residue.resname])
    sequence = ''.join(sequence)

    print(f">{pdb_id}")
    print(sequence)

# %%
