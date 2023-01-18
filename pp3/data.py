"""Data classes and functions."""
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser, Structure
from tqdm import tqdm

from pp3.probes import get_probe


PDB_PARSER = PDBParser()


class ProteinStructure:
    """A protein structure."""
    def __init__(self, pdb_structure: Structure) -> None:
        self.pdb_structure = pdb_structure
        self.probes = {}

        # Clean PDB structure
        self.clean_pdb_structure(pdb_structure=self.pdb_structure)

    def add_probe(self, probe: str) -> None:
        """Add a probe to the protein structure.

        :param probe: The name of the probe.
        """
        self.probes[probe] = get_probe(probe)(self.pdb_structure)

    @staticmethod
    def clean_pdb_structure(pdb_structure: Structure) -> None:
        """Clean the PDB structure, including removing hetero residues.

        :param pdb_structure: The PDB structure to clean.
        """
        for model in pdb_structure:
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

    @classmethod
    def from_file(cls, pdb_id: str, pdb_dir: Path) -> 'ProteinStructure':
        """Load a protein structure from a file.

        :param pdb_id: The PDB ID of the protein structure.
        :param pdb_dir: The directory containing the PDB structures.
        :return: The loaded protein structure.
        """
        pdb_structure = PDB_PARSER.get_structure(
            id=pdb_id,
            file=pdb_dir / pdb_id[1:3].lower() / f'pdb{pdb_id.lower()}.ent'
        )

        structure = cls(pdb_structure=pdb_structure)

        return structure


class ProteinDataset:
    """A dataset of protein structures."""
    def __init__(self, structures: list[ProteinStructure]) -> None:
        self.structures = structures

    def add_probe(self, probe: str) -> None:
        """Add a probe to the protein structures in the dataset.

        :param probe: The name of the probe.
        """
        for structure in tqdm(self.structures):
            structure.add_probe(probe)

    def __len__(self) -> int:
        """Get the number of protein structures in the dataset."""
        return len(self.structures)

    @classmethod
    def from_file(cls, pdb_ids_path: Path, pdb_dir: Path) -> 'ProteinDataset':
        """Load a dataset of protein structures from a file.

        :param pdb_ids_path: Path to a CSV file containing PDB IDs.
        :param pdb_dir: Path to a directory containing PDB structures.
        :return: The loaded dataset of protein structures.
        """
        pdb_ids = pd.read_csv(pdb_ids_path)['pdb_id']
        structures = [
            ProteinStructure.from_file(pdb_id=pdb_id, pdb_dir=pdb_dir)
            for pdb_id in tqdm(pdb_ids)
        ]
        protein_dataset = cls(structures=structures)

        return protein_dataset
