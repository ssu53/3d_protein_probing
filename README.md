# 3D Protein Probing

## Install dependencies

Install the conda environment. If using a GPU, first open `environment.yml` and uncomment the line with `cudatoolkit=11.3`.
```bash
conda env create -f environment.yml
```

Activate the conda environment.
```bash
conda activate 3d_protein_probing
```

## Set up PDB data

### Download PDB

Download all PDB files in PDB format.
```bash
rsync -rlpt -v -z --delete --port=873 pdbjsnap.protein.osaka-u.ac.jp::20230102/pub/pdb/data/structures/divided/pdb pdb
```

Unzip all PDB files.
```bash
gunzip -r pdb
```


### Get diverse set of PDB IDs

Search for single chain proteins with 30% sequence clustering on 2/4/23.
1. Go to https://www.rcsb.org/search/advanced
2. Under Structure Attributes, add "Total Number of Polymer Instances (Chains)" = 1 AND "Entry Polymer Types" is "Protein (only)"
3. Select return "Polymer Entities" grouped by "Sequence Identity 30%" displaying as "Representatives"
4. Click Search (this finds 68,599 polymer entities of which 68,284 are in 16,691 groups with 315 not grouped) [search link](https://www.rcsb.org/search?request=%7B%22query%22%3A%7B%22type%22%3A%22group%22%2C%22logical_operator%22%3A%22and%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C%22logical_operator%22%3A%22and%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_entry_info.deposited_polymer_entity_instance_count%22%2C%22operator%22%3A%22equals%22%2C%22negation%22%3Afalse%2C%22value%22%3A1%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_entry_info.selected_polymer_entity_types%22%2C%22operator%22%3A%22exact_match%22%2C%22negation%22%3Afalse%2C%22value%22%3A%22Protein%20(only)%22%7D%7D%5D%2C%22logical_operator%22%3A%22and%22%7D%2C%7B%22type%22%3A%22group%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_entry_info.resolution_combined%22%2C%22value%22%3A0.5%2C%22operator%22%3A%22less%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_entry_info.resolution_combined%22%2C%22value%22%3A%7B%22from%22%3A0.5%2C%22to%22%3A1%2C%22include_lower%22%3Atrue%2C%22include_upper%22%3Afalse%7D%2C%22operator%22%3A%22range%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_entry_info.resolution_combined%22%2C%22value%22%3A%7B%22from%22%3A1%2C%22to%22%3A1.5%2C%22include_lower%22%3Atrue%2C%22include_upper%22%3Afalse%7D%2C%22operator%22%3A%22range%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_entry_info.resolution_combined%22%2C%22value%22%3A%7B%22from%22%3A1.5%2C%22to%22%3A2%2C%22include_lower%22%3Atrue%2C%22include_upper%22%3Afalse%7D%2C%22operator%22%3A%22range%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_entry_info.resolution_combined%22%2C%22value%22%3A%7B%22from%22%3A2%2C%22to%22%3A2.5%2C%22include_lower%22%3Atrue%2C%22include_upper%22%3Afalse%7D%2C%22operator%22%3A%22range%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_entry_info.resolution_combined%22%2C%22value%22%3A%7B%22from%22%3A2.5%2C%22to%22%3A3%2C%22include_lower%22%3Atrue%2C%22include_upper%22%3Afalse%7D%2C%22operator%22%3A%22range%22%7D%7D%5D%2C%22logical_operator%22%3A%22or%22%2C%22label%22%3A%22rcsb_entry_info.resolution_combined%22%7D%5D%2C%22logical_operator%22%3A%22and%22%7D%5D%2C%22label%22%3A%22text%22%7D%5D%7D%2C%22return_type%22%3A%22polymer_entity%22%2C%22request_options%22%3A%7B%22group_by_return_type%22%3A%22representatives%22%2C%22group_by%22%3A%7B%22aggregation_method%22%3A%22sequence_identity%22%2C%22ranking_criteria_type%22%3A%7B%22sort_by%22%3A%22rcsb_entry_info.resolution_combined%22%2C%22direction%22%3A%22asc%22%7D%2C%22similarity_cutoff%22%3A30%7D%2C%22paginate%22%3A%7B%22start%22%3A0%2C%22rows%22%3A25%7D%2C%22results_content_type%22%3A%5B%22experimental%22%5D%2C%22sort%22%3A%5B%7B%22sort_by%22%3A%22score%22%2C%22direction%22%3A%22desc%22%7D%5D%2C%22scoring_strategy%22%3A%22combined%22%7D%2C%22request_info%22%3A%7B%22query_id%22%3A%2211e14ab552aca419c879acde7d8d4da0%22%7D%7D)
5. Under Refinements, select all the Refinement Resolution (Å) filters <= 3.0 Å
6. Click the green right arrow next to Refinement (this finds 55,785 polymer entities of which 55,616 are in 10,986 groups with 169 not grouped)
7. Click the download icon (top right of the search results table)
8. Copy and paste the 10,986 PDB IDs into `data/pdb_single_chain_protein_30_identity/pdb_ids.txt`


### Convert PDB to PyTorch

TODO: remove large proteins? Perhaps use length cutoff of 1500?

Parse PDB files and save coordinates and sequence in PyTorch format while removing invalid structures.
```bash
python scripts/pdb_to_pytorch.py \
    --ids_path data/pdb_single_chain_protein_30_identity/pdb_ids.txt \
    --pdb_dir pdb \
    --proteins_save_path data/pdb_single_chain_protein_30_identity/proteins.pt \
    --ids_save_path data/pdb_single_chain_protein_30_identity/valid_pdb_ids.csv
```

This successfully converts 4,757 structures.


### Compute concepts from PDB structures

Compute concepts from PDB structures.
```bash
python scripts/compute_concepts.py \
    --ids_path data/pdb_single_chain_protein_30_identity/valid_pdb_ids.csv \
    --pdb_dir pdb \
    --save_dir data/pdb_single_chain_protein_30_identity/concepts
```

TODO: plot concept value distribution


## Compute ESM2 embeddings

Compute ESM2 embeddings for all PDB structures.
```bash
python scripts/compute_esm_embeddings.py \
    --proteins_path data/pdb_single_chain_protein_30_identity/proteins.pt \
    --hub_dir pretrained_models \
    --esm_model esm2_t33_650M_UR50D \
    --last_layer 33 \
    --save_path data/pdb_single_chain_protein_30_identity/embeddings/esm2_t33_650M_UR50D.pt
```


## Probe ESM2 embeddings for concepts

Probe sequence embeddings for protein SASA concept.
```bash
python scripts/probe_sequence_embeddings.py \
    --proteins_path data/pdb_single_chain_protein_30_identity/proteins.pt \
    --embeddings_path data/pdb_single_chain_protein_30_identity/embeddings/esm2_t33_650M_UR50D.pt \
    --save_dir results/pdb_single_chain_protein_30_identity/esm2_t33_650M_UR50D/protein_sasa \
    --concepts_dir data/pdb_single_chain_protein_30_identity/concepts \
    --concept protein_sasa
```
