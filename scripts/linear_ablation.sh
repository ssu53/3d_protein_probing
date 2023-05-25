# Protein function
for CONCEPT in enzyme_commission gene_ontology
do
    # Sequence
    python scripts/probe.py \
              --project_name probe_ablation \
              --proteins_path data/torchdrug/${CONCEPT}_proteins.pt \
              --embeddings_path data/torchdrug/${CONCEPT}_esm2_t33_650M_UR50D.pt \
              --save_dir results/torchdrug \
              --concepts_dir data/torchdrug \
              --concept $CONCEPT \
              --embedding_method plm \
              --encoder_type mlp \
              --encoder_num_layers 0 \
              --encoder_hidden_dim 100 \
              --predictor_num_layers 1 \
              --predictor_hidden_dim 100 \
              --batch_size 100 \
              --split_path data/torchdrug/${CONCEPT}_split_to_pdb_id_30.json \
              --num_sanity_val_steps 0

      for ENCODER_TYPE in egnn tfn
      do
          python scripts/probe.py \
              --project_name probe_ablation \
              --proteins_path data/torchdrug/${CONCEPT}_proteins.pt \
              --embeddings_path data/torchdrug/${CONCEPT}_esm2_t33_650M_UR50D.pt \
              --save_dir results/torchdrug \
              --concepts_dir data/torchdrug \
              --concept $CONCEPT \
              --embedding_method baseline \
              --encoder_type $ENCODER_TYPE \
              --encoder_num_layers 3 \
              --encoder_hidden_dim 16 \
              --predictor_num_layers 1 \
              --predictor_hidden_dim 100 \
              --batch_size 10 \
              --max_neighbors 24 \
              --split_path data/torchdrug/${CONCEPT}_split_to_pdb_id_30.json \
              --num_sanity_val_steps 0
      done
done

# Geometry
for CONCEPT in residue_sasa residue_distances
do
    python scripts/probe.py \
        --project_name probe_ablation \
        --proteins_path data/pdb_single_chain_protein_30_identity/proteins.pt \
        --embeddings_path data/pdb_single_chain_protein_30_identity/embeddings/esm2_t33_650M_UR50D.pt \
        --save_dir results/pdb_single_chain_protein_30_identity \
        --concepts_dir data/pdb_single_chain_protein_30_identity/concepts \
        --concept $CONCEPT \
        --embedding_method plm \
        --encoder_type mlp \
        --encoder_num_layers 0 \
        --encoder_hidden_dim 100 \
        --predictor_num_layers 1 \
        --predictor_hidden_dim 100 \
        --batch_size 100

    for ENCODER_TYPE in egnn tfn
    do
        python scripts/probe.py \
            --project_name probe_ablation \
            --proteins_path data/pdb_single_chain_protein_30_identity/proteins.pt \
            --embeddings_path data/pdb_single_chain_protein_30_identity/embeddings/esm2_t33_650M_UR50D.pt \
            --save_dir results/pdb_single_chain_protein_30_identity \
            --concepts_dir data/pdb_single_chain_protein_30_identity/concepts \
            --concept $CONCEPT \
            --embedding_method baseline \
            --encoder_type $ENCODER_TYPE \
            --encoder_num_layers 3 \
            --encoder_hidden_dim 16 \
            --predictor_num_layers 1 \
            --predictor_hidden_dim 100 \
            --batch_size 10 \
            --max_neighbors 24
    done
done

echo "Done"
exit 0
