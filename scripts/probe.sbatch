#!/bin/bash
#
#SBATCH --job-name=probe_penultimate
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --array=0-175
#SBATCH --partition=jamesz
#SBATCH --output=output/probe_penultimate/%a.out
#SBATCH --error=error/probe_penultimate/%a.err
#SBATCH --time=168:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=swansonk@stanford.edu

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory="$SLURM_SUBMIT_DIR

source /home/users/swansonk/.bashrc

conda activate 3d_protein_probing

cd /oak/stanford/groups/jamesz/swansonk/3d_protein_probing

PROJECT_NAME=probe_penultimate

experiments=()

# Protein function
for CONCEPT in enzyme_commission gene_ontology
do
    # Sequence
    for EMBEDDING_METHOD in baseline plm one
    do
        no_interaction="python scripts/probe.py \
            --project_name $PROJECT_NAME \
            --proteins_path data/torchdrug/${CONCEPT}_proteins.pt \
            --embeddings_path data/torchdrug/${CONCEPT}_esm2_t33_650M_UR50D.pt \
            --save_dir results/torchdrug \
            --concepts_dir data/torchdrug \
            --concept $CONCEPT \
            --embedding_method $EMBEDDING_METHOD \
            --encoder_type mlp \
            --encoder_num_layers 0 \
            --encoder_hidden_dim 100 \
            --predictor_num_layers 2 \
            --predictor_hidden_dim 100 \
            --batch_size 100 \
            --split_path data/torchdrug/${CONCEPT}_split_to_pdb_id_30.json \
            --run_id_number $SLURM_ARRAY_TASK_ID \
            --num_sanity_val_steps 0"
        interaction="${no_interaction} --interaction_model transformer"
        experiments+=("$no_interaction" "$interaction")
    done

    # Structure
    for EMBEDDING_METHOD in baseline plm one
    do
        for ENCODER_TYPE in egnn tfn ipa
        do
            no_interaction="python scripts/probe.py \
                --project_name $PROJECT_NAME \
                --proteins_path data/torchdrug/${CONCEPT}_proteins.pt \
                --embeddings_path data/torchdrug/${CONCEPT}_esm2_t33_650M_UR50D.pt \
                --save_dir results/torchdrug \
                --concepts_dir data/torchdrug \
                --concept $CONCEPT \
                --embedding_method $EMBEDDING_METHOD \
                --encoder_type $ENCODER_TYPE \
                --encoder_num_layers 3 \
                --encoder_hidden_dim 16 \
                --predictor_num_layers 2 \
                --predictor_hidden_dim 100 \
                --batch_size 10 \
                --max_neighbors 24 \
                --split_path data/torchdrug/${CONCEPT}_split_to_pdb_id_30.json \
                --run_id_number $SLURM_ARRAY_TASK_ID \
                --num_sanity_val_steps 0"
            interaction="${no_interaction} --interaction_model transformer"
            experiments+=("$no_interaction" "$interaction")
        done
    done
done

# Geometry
for CONCEPT in residue_sasa secondary_structure bond_angles dihedral_angles residue_distances residue_distances_by_residue residue_contacts residue_contacts_by_residue residue_locations
do
    # Sequence
    for EMBEDDING_METHOD in plm baseline one
    do
        no_interaction="python scripts/probe.py \
            --project_name $PROJECT_NAME \
            --proteins_path data/pdb_single_chain_protein_30_identity/proteins.pt \
            --embeddings_path data/pdb_single_chain_protein_30_identity/embeddings/esm2_t33_650M_UR50D.pt \
            --save_dir results/pdb_single_chain_protein_30_identity \
            --concepts_dir data/pdb_single_chain_protein_30_identity/concepts \
            --concept $CONCEPT \
            --embedding_method $EMBEDDING_METHOD \
            --encoder_type mlp \
            --encoder_num_layers 0 \
            --encoder_hidden_dim 100 \
            --predictor_num_layers 2 \
            --predictor_hidden_dim 100 \
            --batch_size 100 \
            --run_id_number $SLURM_ARRAY_TASK_ID"
        interaction="${no_interaction} --interaction_model transformer"
        experiments+=("$no_interaction" "$interaction")
    done

    # Structure
    for EMBEDDING_METHOD in baseline plm one
    do
        for ENCODER_TYPE in egnn tfn ipa
        do
            no_interaction="python scripts/probe.py \
                --project_name $PROJECT_NAME \
                --proteins_path data/pdb_single_chain_protein_30_identity/proteins.pt \
                --embeddings_path data/pdb_single_chain_protein_30_identity/embeddings/esm2_t33_650M_UR50D.pt \
                --save_dir results/pdb_single_chain_protein_30_identity \
                --concepts_dir data/pdb_single_chain_protein_30_identity/concepts \
                --concept $CONCEPT \
                --embedding_method $EMBEDDING_METHOD \
                --encoder_type $ENCODER_TYPE \
                --encoder_num_layers 3 \
                --encoder_hidden_dim 16 \
                --predictor_num_layers 2 \
                --predictor_hidden_dim 100 \
                --batch_size 10 \
                --max_neighbors 24 \
                --run_id_number $SLURM_ARRAY_TASK_ID"
            interaction="${no_interaction} --interaction_model transformer"
            experiments+=("$no_interaction" "$interaction")
        done
    done
done

command=${experiments[SLURM_ARRAY_TASK_ID]}
echo "Task ID = $SLURM_ARRAY_TASK_ID"
echo "Number of experiments = ${#experiments[@]}"
echo $command
eval "$command"

echo "Done"
exit 0

