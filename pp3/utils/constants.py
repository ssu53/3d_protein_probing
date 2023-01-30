"""Constant values used across multiple modules."""
# Mapping from amino acid three-letter codes to one-letter codes
AA_3_TO_1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
}

# Mapping from secondary structure letters to indices
SS_LETTER_TO_INDEX = {
    'a': 0,  # alpha helix
    'b': 1,  # beta sheet
    'c': 2   # coil
}
