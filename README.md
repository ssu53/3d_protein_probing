# 3D Protein Probing

## Install dependencies

Install conda environment.
```bash
conda env create -f environment.yml
conda activate 3d_protein_probing
```

## Download PDB

Download all PDB files in PDB format.
```bash
rsync -rlpt -v -z --delete --port=873 pdbjsnap.protein.osaka-u.ac.jp::20230102/pub/pdb/data/structures/divided/pdb/ /oak/stanford/groups/jamesz/swansonk/pdb
```

Search for single chain proteins with 30% sequence clustering on 1/16/23.
1. Go to https://www.rcsb.org/search/advanced
2. Under Structure Attributes, add "Total Number of Polymer Instances (Chains)" = 1 AND "Entry Polymer Types" is "Protein (only)"
3. Return "Polymer Entities" groupd by "Sequence Identity 30%" displaying as "Representatives"
4. Click Search
5. Click the download button
6. Copy and paste the 16,653 PDB IDs into "data/single_chain_protein_pdb_ids.txt"
