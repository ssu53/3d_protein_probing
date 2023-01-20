"""Parses PDB files and saves coordinates and sequence in PyTorch format while removing invalid structures."""
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from Bio.PDB import PDBParser, Structure
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    ids_path: Path  # Path to a TXT file containing PDB IDs.
    pdb_dir: Path  # Path to a directory containing PDB structures.
    save_dir: Path  # Path to a directory where PyTorch files with structures will be saved.

    def process_args(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)


def clean_pdb_structure(structure: Structure) -> None:
    """Clean the PDB structure by removing empty chains and hetero residues.

    :param structure: The PDB structure to clean.
    """
    for model in structure:
        residues_to_remove, chains_to_remove = [], []

        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':
                    residues_to_remove.append((chain.id, residue.id))

            if len(chain) == 0:
                chains_to_remove.append(chain.id)

        for chain_id, residue_id in residues_to_remove:
            model[chain_id].detach_child(residue_id)

        for chain in chains_to_remove:
            model.detach_child(chain)


def convert_pdb_to_pytorch(pdb_id: str, pdb_dir: Path, save_dir: Path) -> bool:
    """Parses PDB file and saves coordinates and sequence in PyTorch format while removing invalid structures.

    :param pdb_id: The PDB ID of the protein structure.
    :param pdb_dir: The directory containing the PDB structures.
    :param save_dir: The directory where PyTorch file with coordinates and sequence will be saved.
    :return: Whether the PDB file was successfully converted.
    """
    # Get PDB file path
    pdb_path = pdb_dir / pdb_id[1:3].lower() / f'pdb{pdb_id.lower()}.ent'

    # Check if PDB file exists
    if not pdb_path.exists():
        return False

    # Parse PDB structure
    structure = PDBParser().get_structure(id=pdb_id, file=pdb_path)

    # Clean the PDB structure (remove empty chains and hetero residues)
    clean_pdb_structure(structure)

    # Get residue coordinates
    residues = [residue for residue in structure.get_residues()]
    residue_coords = [residue['CA'].get_coord() for residue in residues]

    # Get the residue indices and ensure none are missing
    residue_indices = [residue.get_id()[1] for residue in residues]
    if residue_indices != list(range(min(residue_indices), max(residue_indices) + 1)):
        return False

    # Get sequence from structure residues
    structure_sequence = ''.join(residue.get_resname() for residue in residues)

    # Parse PDB sequence
    records = list(SeqIO.parse('pdb2go8.ent', 'pdb-seqres'))

    # Ensure only one sequence record
    if len(records) != 1:
        return False

    # Get sequence
    sequence = str(records[0].seq)

    # Ensure the structure's sequence matches a subsequence of the full PDB sequence
    if structure_sequence not in sequence:
        return False

    # Get the start and end indices of the structure's sequence in the full PDB sequence
    start_index = sequence.index(structure_sequence)
    end_index = start_index + len(structure_sequence)

    # Save PyTorch file
    torch.save({
        'residue_coords': torch.FloatTensor(np.array(residue_coords)),
        'sequence': sequence,
        'start_index': start_index,
        'end_index': end_index
    }, save_dir / f'{pdb_id}.pt')


def pdb_to_pytorch(args: Args) -> None:
    """Parses PDB files and saves coordinates and sequence in PyTorch format while removing invalid structures."""
    # Load PDB IDs
    with open(args.ids_path) as f:
        pdb_ids = f.read().strip().split(',')

    print(f'Loaded {len(pdb_ids):,} PDB IDs')

    # Check which PDB IDs have structures
    with Pool() as pool:
        convert_pdb_to_pytorch_fn = partial(convert_pdb_to_pytorch, pdb_dir=args.pdb_dir, save_dir=args.save_dir)
        convert_success = list(
            tqdm(pool.imap(convert_pdb_to_pytorch_fn, pdb_ids), total=len(pdb_ids))
        )

    print(f'Converted {sum(convert_success):,} PDB files successfully')


if __name__ == '__main__':
    pdb_to_pytorch(Args().parse_args())
