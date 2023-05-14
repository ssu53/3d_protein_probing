#!/bin/bash
#
#SBATCH --job-name=probe
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --array=0-179
#SBATCH --time=168:00:00
#SBATCH --partition=jamesz
#SBATCH --output=output/probe/%a.out
#SBATCH --error=error/probe/%a.err
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

experiments=()

for SPLIT_SEED in 0 1 2
do
    # Downstream sequence
    for CONCEPT in solubility
    do
        for EMBEDDING_METHOD in baseline plm
        do
            experiments+=("python scripts/probe.py \
                --project_name probing-crossval \
                --proteins_path data/downstream_tasks/computational/${CONCEPT}_proteins.pt \
                --embeddings_path data/downstream_tasks/computational/${CONCEPT}_esm2_t33_650M_UR50D.pt \
                --save_dir results/downstream_tasks/computational \
                --concepts_dir data/downstream_tasks/computational \
                --concept $CONCEPT \
                --embedding_method $EMBEDDING_METHOD \
                --encoder_type mlp \
                --encoder_num_layers 0 \
                --encoder_hidden_dim 100 \
                --predictor_num_layers 2 \
                --predictor_hidden_dim 100 \
                --batch_size 100 \
                --split_seed $SPLIT_SEED")
        done
    done

    # Downstream structure
    for CONCEPT in solubility
    do
        for EMBEDDING_METHOD in baseline plm
        do
            for ENCODER_TYPE in egnn tfn
            do
                experiments+=("python scripts/probe.py \
                    --project_name probing-crossval \
                    --proteins_path data/downstream_tasks/computational/${CONCEPT}_proteins.pt \
                    --embeddings_path data/downstream_tasks/computational/${CONCEPT}_esm2_t33_650M_UR50D.pt \
                    --save_dir results/downstream_tasks/computational \
                    --concepts_dir data/downstream_tasks/computational \
                    --concept $CONCEPT \
                    --embedding_method $EMBEDDING_METHOD \
                    --encoder_type $ENCODER_TYPE \
                    --encoder_num_layers 3 \
                    --encoder_hidden_dim 16 \
                    --predictor_num_layers 2 \
                    --predictor_hidden_dim 100 \
                    --batch_size 16 \
                    --max_neighbors 24 \
                    --split_seed $SPLIT_SEED")
            done
        done
    done

    # Concepts sequence
    for CONCEPT in residue_sasa secondary_structure bond_angles dihedral_angles residue_distances residue_distances_by_residue residue_contacts residue_contacts_by_residue residue_locations
    do
        for EMBEDDING_METHOD in baseline plm
        do
            experiments+=("python scripts/probe.py \
                --project_name probing-crossval \
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
                --split_seed $SPLIT_SEED")
        done
    done

    # Concepts structure
    for CONCEPT in residue_sasa secondary_structure bond_angles dihedral_angles residue_distances residue_distances_by_residue residue_contacts residue_contacts_by_residue residue_locations
    do
        for EMBEDDING_METHOD in baseline plm
        do
            for ENCODER_TYPE in egnn tfn
            do
                experiments+=("python scripts/probe.py \
                    --project_name probing-crossval \
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
                    --batch_size 16 \
                    --max_neighbors 24 \
                    --split_seed $SPLIT_SEED")
            done
        done
    done
done

command=${experiments[SLURM_ARRAY_TASK_ID]}
echo $command
eval "$command"

echo "Done"
exit 0
