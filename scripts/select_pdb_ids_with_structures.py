"""Select the PDB IDs from a list for which we have structures."""
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    ids_path: Path  # Path to a TXT file containing PDB IDs.
    pdb_dir: Path  # Path to a directory containing PDB structures.
    save_path: Path  # Path to a CSV file where PDB IDs with structures will be saved.

    def process_args(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)


def check_for_pdb_structure(pdb_id: str, pdb_dir: Path) -> bool:
    """Check if a PDB structure exists for a PDB ID."""
    return (pdb_dir / pdb_id[1:3].lower() / f'pdb{pdb_id.lower()}.ent').exists()


def select_pbd_ids_with_structures(args: Args) -> None:
    """Select the PDB IDs from a list for which we have structures."""
    # Load PDB IDs
    with open(args.ids_path) as f:
        pdb_ids = f.read().strip().split(',')

    print(f'Loaded {len(pdb_ids):,} PDB IDs')

    # Check which PDB IDs have structures
    with Pool() as pool:
        check_for_pdb_structure_fn = partial(check_for_pdb_structure, pdb_dir=args.pdb_dir)
        pdb_id_has_structure = list(
            tqdm(pool.imap(check_for_pdb_structure_fn, pdb_ids), total=len(pdb_ids))
        )

    # Select PDB IDs for which we have structures
    pdb_ids_with_structures = [pdb_id for pdb_id, has_structure in zip(pdb_ids, pdb_id_has_structure) if has_structure]

    print(f'Selected {len(pdb_ids_with_structures):,} PDB IDs with structures')

    # Save data
    data = pd.DataFrame({'pdb_id': pdb_ids_with_structures})
    data.to_csv(args.save_path, index=False)


if __name__ == '__main__':
    select_pbd_ids_with_structures(Args().parse_args())
