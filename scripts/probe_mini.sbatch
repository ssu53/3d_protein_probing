#!/bin/bash
#
#SBATCH --job-name=probe
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:RTX_2080Ti|GPU_SKU:TITAN_V|GPU_SKU:TITAN_Xp
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-0%1
#SBATCH --partition=jamesz
#SBATCH --output=/home/groups/jamesz/shiye/3d_protein_probing/slurm_output/probe/%x.%j.%a.out
#SBATCH --error=/home/groups/jamesz/shiye/3d_protein_probing/slurm_error/probe/%x.%j.%a.err
#SBATCH --time=3:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shiye@stanford.edu

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory="$SLURM_SUBMIT_DIR

source /home/users/shiye/.bashrc

conda activate 3d_protein_probing

cd /home/groups/jamesz/shiye/3d_protein_probing

PROJECT_NAME=probe

experiments=()


# Geometry
for CONCEPT in residue_neighb_distances_5
do

    # Structure
    for EMBEDDING_METHOD in baseline-basic
    do
        for ENCODER_TYPE in egnn
        do
            no_interaction="python scripts/probe.py \
                --project_name $PROJECT_NAME \
                --proteins_path data/pdb_single_chain_protein_30_identity/proteins.pt \
                --embeddings_path data/pdb_single_chain_protein_30_identity/embeddings/esm2_t33_650M_UR50D.pt \
                --save_dir results/pdb_single_chain_protein_30_identity/basic_embed \
                --concepts_dir data/pdb_single_chain_protein_30_identity/concepts \
                --concept $CONCEPT \
                --embedding_method $EMBEDDING_METHOD \
                --encoder_type $ENCODER_TYPE \
                --encoder_num_layers 3 \
                --encoder_hidden_dim 16 \
                --predictor_num_layers 2 \
                --predictor_hidden_dim 100 \
                --batch_size 8 \
                --max_epochs 100 \
                --max_neighbors 24 \
                --run_id_number $SLURM_ARRAY_TASK_ID \
                --entity ssu53"
            interaction="${no_interaction} --interaction_model transformer"
            experiments+=("$no_interaction" "$interaction")
        done
    done
done

command=${experiments[SLURM_ARRAY_TASK_ID]}
echo "Task ID = $SLURM_ARRAY_TASK_ID"
echo "Number of experiments = ${#experiments[@]}"
echo "$command"
eval "$command"

echo "Done"
exit 0
