# First set of experiments
for EMBEDDING_METHOD in plm baseline
do
    for CONCEPT in residue_distances residue_sasa secondary_structure bond_angles dihedral_angles residue_distances_by_residue residue_contacts residue_contacts_by_residue residue_locations
    do
        for ENCODER_NUM_LAYERS in 3
        do
            for PREDICTOR_NUM_LAYERS in 2
            do
                python scripts/probe.py \
                    --project_name probing-egnn-max-neighbors \
                    --proteins_path data/pdb_single_chain_protein_30_identity/proteins.pt \
                    --embeddings_path data/pdb_single_chain_protein_30_identity/embeddings/esm2_t33_650M_UR50D.pt \
                    --save_dir results/pdb_single_chain_protein_30_identity \
                    --concepts_dir data/pdb_single_chain_protein_30_identity/concepts \
                    --concept $CONCEPT \
                    --embedding_method $EMBEDDING_METHOD \
                    --encoder_type egnn \
                    --encoder_num_layers $ENCODER_NUM_LAYERS \
                    --encoder_hidden_dim 16 \
                    --predictor_num_layers $PREDICTOR_NUM_LAYERS \
                    --predictor_hidden_dim 100 \
                    --batch_size 16 \
                    --max_neighbors 24
            done
        done
    done
done

# First set of experiments
for EMBEDDING_METHOD in plm baseline
do
    for CONCEPT in residue_distances residue_sasa secondary_structure bond_angles dihedral_angles residue_distances_by_residue residue_contacts residue_contacts_by_residue residue_locations
    do
        for ENCODER_NUM_LAYERS in 3
        do
            for PREDICTOR_NUM_LAYERS in 2
            do
                python scripts/probe.py \
                    --project_name probing-egnn-all-neighbors \
                    --proteins_path data/pdb_single_chain_protein_30_identity/proteins.pt \
                    --embeddings_path data/pdb_single_chain_protein_30_identity/embeddings/esm2_t33_650M_UR50D.pt \
                    --save_dir results/pdb_single_chain_protein_30_identity \
                    --concepts_dir data/pdb_single_chain_protein_30_identity/concepts \
                    --concept $CONCEPT \
                    --embedding_method $EMBEDDING_METHOD \
                    --encoder_type egnn \
                    --encoder_num_layers $ENCODER_NUM_LAYERS \
                    --encoder_hidden_dim 16 \
                    --predictor_num_layers $PREDICTOR_NUM_LAYERS \
                    --predictor_hidden_dim 100 \
                    --batch_size 16
            done
        done
    done
done