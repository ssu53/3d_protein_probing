"""Select the PDB IDs from a list for which we have structures."""
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


def select_pbd_ids_with_structures(args: Args) -> None:
    """Select the PDB IDs from a list for which we have structures."""
    # Load PDB IDs
    with open(args.ids_path) as f:
        pdb_ids = f.read().strip().split(',')

    print(f'Loaded {len(pdb_ids):,} PDB IDs')

    # Select PDB IDs for which we have structures
    pdb_ids = [
        pdb_id
        for pdb_id in tqdm(pdb_ids)
        if (args.pdb_dir / pdb_id[1:3].lower() / f'pdb{pdb_id.lower()}.ent').exists()
    ]

    print(f'Selected {len(pdb_ids):,} PDB IDs with structures')

    # Save data
    data = pd.DataFrame({'pdb_id': pdb_ids})
    data.to_csv(args.save_path, index=False)


if __name__ == '__main__':
    select_pbd_ids_with_structures(Args().parse_args())
