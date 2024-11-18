# %%

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio import pairwise2 as pw2
from Bio.pairwise2 import format_alignment

# %%
def pairwise_sequence_align_util(sequence1, sequence2):
    # this definition of sequence identity is from PLMalign
    global_align = pw2.align.globalxx(sequence1, sequence2)
    best_sequence_identity = -1
    align = "None"
    for i in global_align:
        sequence_identity = i[2]/(i[4]-i[3])
        if (sequence_identity > best_sequence_identity):
            align = i
            best_sequence_identity = sequence_identity
    return best_sequence_identity, format_alignment(*align)

# %%

with open('/home/groups/jamesz/shiye/3d_protein_probing/data/scope40_foldseek_compatible/scope40_valid_seqs_val.fasta','r') as f:
    fastas = f.read().splitlines()

pdb_id_to_fasta = {}

for i in range(len(fastas)//2):
    pdb_id = fastas[2*i]
    assert pdb_id[0] == '>'
    pdb_id = pdb_id.strip('>')
    fasta = fastas[2*i+1]
    pdb_id_to_fasta[pdb_id] = fasta

# %%

scop_lookup = pd.read_csv(
    '/home/groups/jamesz/shiye/foldseek-analysis/training/data/scop_lookup.tsv',
    sep='\t', header=None)
scop_lookup.columns = ['sid', 'family']
scop_lookup.set_index('sid', inplace=True)
scop_lookup['superfamily'] = scop_lookup['family'].apply(lambda x: x[:x.rfind('.')])
scop_lookup['fold'] = scop_lookup['superfamily'].apply(lambda x: x[:x.rfind('.')])
scop_lookup['class'] = scop_lookup['fold'].apply(lambda x: x[:x.rfind('.')])
scop_lookup = scop_lookup.loc[pdb_id_to_fasta.keys()]

# %%

def get_seq_ident():

    import itertools


    all_pairs = []

    for fold in scop_lookup.fold.unique():
        pdb_ids = scop_lookup[scop_lookup.fold == fold].index.tolist()
        pairs = list(itertools.permutations(pdb_ids, 2))
        assert len(pairs) == len(pdb_ids)**2-len(pdb_ids)
        all_pairs.extend(pairs)


    pbar = tqdm(total=len(all_pairs))

    df = pd.DataFrame(index=range(len(all_pairs)))

    for i, (pdb_id_1, pdb_id_2) in enumerate(all_pairs):
        seq1 = pdb_id_to_fasta[pdb_id_1]
        seq2 = pdb_id_to_fasta[pdb_id_2]
        best_sequence_identity, _ = pairwise_sequence_align_util(seq1, seq2)
        df.loc[i,'pdb_id_1'] = pdb_id_1
        df.loc[i,'pdb_id_2'] = pdb_id_2
        df.loc[i,'seq_ident'] = best_sequence_identity
        pbar.update(1)

    pbar.close()

    df.to_csv('/home/groups/jamesz/shiye/3d_protein_probing/data/seq_ident.csv', index=False, header=False, sep=' ')


# %%

def get_seq_ident_for_hits():

    # hits = pd.read_csv('/home/groups/jamesz/shiye/tm-vec/tabular_scope40_val_128.txt', sep='\t')
    hits = pd.read_csv('/home/groups/jamesz/shiye/tm-vec/tabular_scope40_val.txt', sep='\t')
    lookup = pd.read_csv('/home/groups/jamesz/shiye/3d_protein_probing/hits_.csv', sep=' ') # already computed some sequence identities

    lookup_dict = {f"{x.query_id} {x.database_id}": x.seq_ident for i,x in lookup.iterrows()}

    num_new_pairs_computed = 0
    for i in tqdm(hits.index):
        query_id = hits.query_id[i]
        database_id = hits.database_id[i]
        seq1 = pdb_id_to_fasta[query_id]
        seq2 = pdb_id_to_fasta[database_id]

        pair_name = f"{query_id} {database_id}"
        if pair_name in lookup_dict:
            hits.loc[i,'seq_ident'] = lookup_dict[pair_name]
        else:
            num_new_pairs_computed += 1
            hits.loc[i,'seq_ident'], _ = pairwise_sequence_align_util(seq1, seq2)

        if i % 10000 == 0:
            hits.to_csv('hits.csv', index=False, sep=' ')

    print(f"{num_new_pairs_computed=}")

    hits.to_csv('hits.csv', index=False, sep=' ')


# %%

get_seq_ident_for_hits()
exit


# %%

hits = pd.read_csv('/home/groups/jamesz/shiye/3d_protein_probing/hits.csv', sep=' ')
hits = hits.dropna()
# hits = hits[hits.query_id != hits.database_id]
hits


# %%

pdb_ids = hits.query_id.unique()
print(pdb_ids)


# %%


within_fold_pairs = pd.read_csv('/home/groups/jamesz/shiye/3d_protein_probing/data/seq_ident.csv', header=None, sep=' ')
within_fold_pairs.columns = ['pdb_id_1', 'pdb_id_2', 'seq_ident']

# add fam, sfam, fold annotations for the database hits
within_fold_pairs['fam_2'] = within_fold_pairs.apply(lambda x: scop_lookup.family[x.pdb_id_2], axis=1)
within_fold_pairs['sfam_2'] = within_fold_pairs.apply(lambda x: scop_lookup.superfamily[x.pdb_id_2], axis=1)
within_fold_pairs['fold_2'] = within_fold_pairs.apply(lambda x: scop_lookup.fold[x.pdb_id_2], axis=1)
within_fold_pairs
# %%


for thresh in [0.3, 0.4, 0.5]:

    within_fold_pairs_thresh = within_fold_pairs[within_fold_pairs.seq_ident < thresh]
    hits_thresh = hits[hits.seq_ident < thresh]

    print(f"{thresh=}")
    print(f"{len(within_fold_pairs_thresh)}/{len(within_fold_pairs)}")
    print(f"{len(hits_thresh)}/{len(hits)}")

    fam_counts = []
    sfam_counts = []
    fold_counts = []

    fam_hits = []
    sfam_hits = []
    fold_hits = []

    for pdb_id in tqdm(pdb_ids):

        within_fold_pairs_ = within_fold_pairs_thresh[(within_fold_pairs_thresh.pdb_id_1 == pdb_id)]

        if len(within_fold_pairs_) == 0:
            fam_counts.append(0)
            sfam_counts.append(0)
            fold_counts.append(0)
            fam_hits.append(0)
            sfam_hits.append(0)
            fold_hits.append(0)
            continue

        # count the number of fam, sfam, fold
        query_scop_fam = scop_lookup.family[pdb_id]
        query_scop_sfam = scop_lookup.superfamily[pdb_id]
        query_scop_fold = scop_lookup.fold[pdb_id]
        num_fam = sum(within_fold_pairs_.fam_2 == query_scop_fam)
        num_sfam = sum(within_fold_pairs_.sfam_2 == query_scop_sfam) 
        num_fold = sum(within_fold_pairs_.fold_2 == query_scop_fold)
        fam_counts.append(num_fam)
        sfam_counts.append(num_sfam)
        fold_counts.append(num_fold)
        

        # count the number of hits
        hits_ = hits_thresh[(hits_thresh.query_id == pdb_id)]
        
        fam_hit = 0
        sfam_hit = 0
        fold_hit = 0
        reached_end = True
        for _,row in hits_.iterrows():
            if query_scop_fam == scop_lookup.family[row.database_id]:
                fam_hit += 1
            elif query_scop_sfam == scop_lookup.superfamily[row.database_id]:
                sfam_hit += 1
            elif query_scop_fold == scop_lookup.fold[row.database_id]:
                fold_hit += 1
            else:
                reached_end = False
                break
        if reached_end and fam_hit + sfam_hit + fold_hit < len(within_fold_pairs_):
            print(query_scop_fold, pdb_id, len(within_fold_pairs_), len(hits_), "Reached end of hits, possibly more beyond these top-n")
        fam_hits.append(fam_hit)
        sfam_hits.append(sfam_hit)
        fold_hits.append(fold_hit)


    print(fam_counts)
    print(sfam_counts)
    print(fold_counts)

    print(fam_hits)
    print(sfam_hits)
    print(fold_hits)


    fam_counts = np.array(fam_counts)
    sfam_counts = np.array(sfam_counts)
    fold_counts = np.array(fold_counts)
    fam_hits = np.array(fam_hits)
    sfam_hits = np.array(sfam_hits)
    fold_hits = np.array(fold_hits)

    fam_denom = fam_counts
    sfam_denom = sfam_counts - fam_counts
    fold_denom = fold_counts - sfam_counts

    print(np.nanmean(fam_hits / fam_denom))
    print(np.nanmean(sfam_hits / sfam_denom))
    print(np.nanmean(fold_hits / fold_denom))


    # plt.figure()
    # plt.hist(hits_.seq_ident, bins=20)
    # plt.show()

    # plt.figure()
    # plt.hist(within_fold_pairs_.seq_ident, bins=20)
    # plt.show()


    np.save(
        f'thresh{thresh}',
        np.stack((fam_counts, sfam_counts, fold_counts, fam_hits, sfam_hits, fold_hits)))
# %%

results = pd.DataFrame()

for thresh in [0.5, 0.4, 0.3, 0.25, 0.24, 0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17]:

    fam_counts, sfam_counts, fold_counts, fam_hits, sfam_hits, fold_hits = np.array_split(np.load(f'thresh{thresh}.npy'), 6, axis=0)
    
    fam_denom = fam_counts
    sfam_denom = sfam_counts - fam_counts
    fold_denom = fold_counts - sfam_counts

    with np.errstate(invalid='ignore'):
        results.loc[thresh, 'fam_micro'] = np.nanmean(fam_hits / fam_denom)
        results.loc[thresh, 'sfam_micro'] = np.nanmean(sfam_hits / sfam_denom)
        results.loc[thresh, 'fold_micro'] = np.nanmean(fold_hits / fold_denom)

        results.loc[thresh, 'fam_macro'] = np.nanmean(fam_hits) / np.nanmean(fam_denom)
        results.loc[thresh, 'sfam_macro'] = np.nanmean(sfam_hits) / np.nanmean(sfam_denom)
        results.loc[thresh, 'fold_macro'] = np.nanmean(fold_hits) / np.nanmean(fold_denom)

    results.loc[thresh, 'fam_hits'] = np.sum(fam_hits)
    results.loc[thresh, 'sfam_hits'] = np.sum(sfam_hits)
    results.loc[thresh, 'fold_hits'] = np.sum(fold_hits)

    results.loc[thresh, 'fam_counts'] = np.sum(fam_counts)
    results.loc[thresh, 'sfam_counts'] = np.sum(sfam_counts)
    results.loc[thresh, 'fold_counts'] = np.sum(fold_counts)

results
# %%

plt.figure(figsize=(7,7))
for col in [
    # 'fam_micro', 'sfam_micro', 'fold_micro', 
    'fam_macro', 'sfam_macro', 'fold_macro',
]:
    plt.plot(results.index, results[col], '-o', label=col, markersize=5)
plt.legend()
plt.grid()
plt.title('tm-vec (macro-averaged)')
plt.xlabel('sequence identity thresh')
plt.ylabel('sensitivity up to the 1st FP')
plt.show()
# %%
