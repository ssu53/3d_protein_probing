"""Probe a model for 3D geometric concepts."""
from pathlib import Path

from tap import Tap

from pp3.data import ProteinDataset
from pp3.probes import PROBE_REGISTRY


class Args(Tap):
    pdb_ids_path: Path  # Path to a CSV file containing PDB IDs.
    pdb_dir: Path  # Path to a directory containing PDB structures.
    probes: list[str]  # Names of the probes to use.

    def configure(self) -> None:
        self.add_argument('--probes', choices=sorted(PROBE_REGISTRY))


def probe(
        pdb_ids_path: Path,
        pdb_dir: Path,
        probes: list[str]
) -> None:
    """Probe a model for 3D geometric concepts.

    :param pdb_ids_path: Path to a CSV file containing PDB IDs.
    :param pdb_dir: Path to a directory containing PDB structures.
    :param probes: Names of the probes to use.
    """
    dataset = ProteinDataset.from_file(pdb_ids_path=pdb_ids_path, pdb_dir=pdb_dir)
    dataset.add_probe('residue_pair_distances')


if __name__ == '__main__':
    probe(**Args().parse_args().as_dict())
