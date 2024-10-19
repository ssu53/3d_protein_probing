import numpy as np

from numpy.linalg import norm
from Bio.PDB import PDBParser

from pp3.utils.constants import AA_3_TO_1
AA_3_TO_1['UNK'] = 'X'

DISTANCE_ALPHA_BETA = 1.5336


def approx_c_beta_position(c_alpha, n, c_carboxyl):
    """
    Approximate C beta position,
    from C alpha, N and C positions,
    assuming the four ligands of the C alpha
    form a regular tetrahedron.
    """
    v1 = c_carboxyl - c_alpha
    v1 = v1 / norm(v1)
    v2 = n - c_alpha
    v2 = v2 / norm(v2)

    b1 = v2 + 1/3 * v1
    b2 = np.cross(v1, b1)

    u1 = b1/norm(b1)
    u2 = b2/norm(b2)

    # direction from c_alpha to c_beta
    v4 = -1/3 * v1 + np.sqrt(8)/3 * norm(v1) * (-1/2 * u1 - np.sqrt(3)/2 * u2)

    return c_alpha + DISTANCE_ALPHA_BETA * v4  # c_beta


def get_atom_coordinates(chain, verbose=False, full_backbone=False, raise_error=True):
    """Get CA/CB coordinates from list of biopython residues.

    C betas from GLY are approximated.

    chain: list of residues
    full_backbone: in addition return C and N coords
    return: np.array (n_residues x 6)
    """

    chain = list(chain)
    n_res = len(chain)

    # coordinates of C alpha and C beta
    n_cols = 6 if not full_backbone else 12
    # CA, CB(, N, C)
    coords = np.full((n_res, n_cols), np.nan, dtype=np.float32)
    is_hetatm_mask = np.full((n_res,), False, dtype=bool)

    for i, res in enumerate(chain):

        is_HETATM = len(res.id[0].strip())
        if is_HETATM:
            is_hetatm_mask[i] = True
            continue  # skip HETATMs

        ca_atoms = [atom for atom in res if atom.name == 'CA']
        if len(ca_atoms) != 1:
            if verbose:
                print(f'No CA found [{i}] {chain.full_id}')
            if raise_error: 
                raise ValueError
        else:
            coords[i, 0:3] = ca_atoms[0].coord

        cb_atoms = [atom for atom in res if atom.name == 'CB']
        if res.resname != 'GLY' and cb_atoms:
            if len(cb_atoms) == 1:
                coords[i, 3:6] = cb_atoms[0].coord
            else:
                if verbose:
                    print(f'No CB found [{i}] {chain.full_id}')
                if raise_error:
                    raise ValueError

        else:  # approx CB position
            n_atoms = [atom for atom in res if atom.name == 'N']
            co_atoms = [atom for atom in res if atom.name == 'C']
            if len(ca_atoms) != 1 or len(n_atoms) != 1 or len(co_atoms) != 1:
                if verbose:
                    print(f'Failed to approx CB ({ca_atoms}, {n_atoms}, {co_atoms})')
                if raise_error:
                    raise ValueError
            else:
                cb_coord = approx_c_beta_position(
                        ca_atoms[0].coord,
                        n_atoms[0].coord,
                        co_atoms[0].coord)
                coords[i, 3:6] = cb_coord

        if full_backbone:
            n_atoms = [atom for atom in res if atom.name == 'N']
            co_atoms = [atom for atom in res if atom.name == 'C']
            if len(n_atoms) != 1 or len(co_atoms) != 1:
                if raise_error:
                    raise ValueError
                pass
            else:
                coords[i, 6:9] = n_atoms[0].coord
                coords[i, 9:12] = co_atoms[0].coord

    # Collapse chain to be continuous around the HETATM entries
    coords = coords[~is_hetatm_mask, :]
    valid_mask = ~np.any(np.isnan(coords), axis=1)  # mask all rows containing NANs
    if raise_error:
        assert np.all(valid_mask), "structure is not everywhere valid"

    return coords, valid_mask



def get_coords_from_pdb(path, full_backbone=False):
    """
    Read pdb file and return CA + CB (+ N + C) coords.
    CB from GLY are approximated.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('None', path)

    model = structure[0]  # take only first model
    chain = list(model.get_chains())[0]  # take only first chain

    coords, valid_mask = get_atom_coordinates(list(chain.get_residues()),
            full_backbone=full_backbone)

    # # Remove HETATM entries
    # for residue in chain:
    #     residue_id = residue.id
    #     het_flag, _, _ = residue_id
    #     if het_flag != ' ':
    #         chain.detach_child(residue_id)

    # Get sequence
    sequence = []
    for residue in chain: 
        het_flag,_ ,_ = residue.id
        if het_flag != ' ':
            continue
        sequence.append(AA_3_TO_1[residue.resname])
    sequence = ''.join(sequence)

    return coords, valid_mask, sequence

