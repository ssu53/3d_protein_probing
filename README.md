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
