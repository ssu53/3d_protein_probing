"""Constant values used across multiple modules."""
import numpy as np
from biotite.sequence.align import SubstitutionMatrix
from biotite.structure.info import residue

# Mapping from amino acid three-letter codes to one-letter codes
AA_3_TO_1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
    'SEC': 'U', 'PYL': 'O'
}

# Set of canonical three letter amino acid codes
AA_3 = sorted(AA_3_TO_1)

# Set of canonical one letter amino acid codes
AA_1 = sorted(AA_3_TO_1.values())

# Get mapping from one letter amino acid code to index
AA_1_TO_INDEX = {aa: i for i, aa in enumerate(AA_1)}

# Mapping from secondary structure letters to indices
SS_LETTER_TO_INDEX = {
    'a': 0,  # alpha helix
    'b': 1,  # beta sheet
    'c': 2   # coil
}

# Canonical amino acid atom names (without hydrogen atoms)
AA_ATOM_NAMES = {
    aa: set(residue(aa).atom_name)
    for aa in AA_3_TO_1
}

# Backbone atom names
BACKBONE_ATOM_NAMES = {'N', 'CA', 'C'}

# BLOSUM62 AA to vector
BLOSUM62 = SubstitutionMatrix.std_protein_matrix()
BLOSUM62_AA_TO_VECTOR = {
    aa1: [BLOSUM62.get_score(aa1, aa2) for aa2 in BLOSUM62.get_alphabet2()]
    for aa1 in BLOSUM62.get_alphabet1()
}

# Maximum sequence length
MAX_SEQ_LEN = 512
