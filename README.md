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

## Get diverse set of PDB IDs

[Search](https://www.rcsb.org/search?request=%7B%22query%22%3A%7B%22type%22%3A%22group%22%2C%22logical_operator%22%3A%22and%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C%22logical_operator%22%3A%22and%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C%22logical_operator%22%3A%22and%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22operator%22%3A%22equals%22%2C%22negation%22%3Afalse%2C%22value%22%3A1%2C%22attribute%22%3A%22rcsb_entry_info.deposited_polymer_entity_instance_count%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22operator%22%3A%22exact_match%22%2C%22negation%22%3Afalse%2C%22value%22%3A%22Protein%20(only)%22%2C%22attribute%22%3A%22rcsb_entry_info.selected_polymer_entity_types%22%7D%7D%5D%7D%5D%2C%22label%22%3A%22text%22%7D%5D%2C%22label%22%3A%22query-builder%22%7D%2C%22return_type%22%3A%22polymer_entity%22%2C%22request_options%22%3A%7B%22group_by_return_type%22%3A%22representatives%22%2C%22group_by%22%3A%7B%22aggregation_method%22%3A%22sequence_identity%22%2C%22ranking_criteria_type%22%3A%7B%22sort_by%22%3A%22rcsb_entry_info.resolution_combined%22%2C%22direction%22%3A%22asc%22%7D%2C%22similarity_cutoff%22%3A30%7D%2C%22paginate%22%3A%7B%22start%22%3A0%2C%22rows%22%3A25%7D%2C%22results_content_type%22%3A%5B%22experimental%22%5D%2C%22sort%22%3A%5B%7B%22sort_by%22%3A%22score%22%2C%22direction%22%3A%22desc%22%7D%5D%2C%22scoring_strategy%22%3A%22combined%22%7D%2C%22request_info%22%3A%7B%22query_id%22%3A%229a92da7eb5793e2431f93d7028e06a47%22%7D%7D) for single chain proteins with 30% sequence clustering on 1/16/23.
1. Go to https://www.rcsb.org/search/advanced
2. Under Structure Attributes, add "Total Number of Polymer Instances (Chains)" = 1 AND "Entry Polymer Types" is "Protein (only)"
3. Return "Polymer Entities" groupd by "Sequence Identity 30%" displaying as "Representatives"
4. Click Search (this finds 68,394 polymer entities of which 68,081 are in 16,653 groups)
5. Click the download button
6. Copy and paste the 16,653 PDB IDs into `data/pdb_single_chain_protein_30_identity_ids.txt`

## Subset PDB IDs to those with structures

Subset the PDB IDs to those for which we have structures.
```bash
python scripts/select_pdb_ids_with_structures.py \
    --ids_path data/pdb_single_chain_protein_30_identity_ids.txt \
    --pdb_dir /oak/stanford/groups/jamesz/swansonk/pdb \
    --save_path data/pdb_single_chain_protein_30_identity_ids_with_structures.csv
```
