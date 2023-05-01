#!/bin/bash

for CONCEPT in residue_sasa # bond_angles secondary_structure residue_distances residue_contacts  # 
do
    for PROTEIN_EMBEDDING_METHOD in baseline # plm (currently experiencing memory issues)
    do
        python probe.py \
            --project_name probing \
            --proteins_path /storage/jwohlwend/prob_pdb/proteins.pt \
            --embeddings_path /storage/jwohlwend/prob_pdb/esm2_t33_650M_UR50D.pt \
            --save_dir /Mounts/rbg-storage1/users/jwohlwend/prob_results \
            --concepts_dir /storage/jwohlwend/prob_pdb/concepts \
            --concept $CONCEPT \
            --embedding_method $PROTEIN_EMBEDDING_METHOD \
            --encoder_num_layers 3 \
            --encoder_type egnn \
            --batch_size 16 \
            --num_workers 8 \
            --max_neighbors 24 \
            --encoder_hidden_dim 256 \
            --predictor_num_layers 2 \
            --predictor_hidden_dim 128
    done
done
