"""Contains baseline embeddings for proteins and residues."""
from collections import Counter

import torch

from pp3.utils.constants import AA_1, AA_1_TO_INDEX, BLOSUM62_AA_TO_VECTOR


def get_baseline_protein_embedding(sequence: str) -> torch.Tensor:
    """Get the baseline protein embedding from a protein sequence.

    Baseline protein embedding includes:
        - Protein length
        - Amino acid frequencies

    :param sequence: The amino acid sequence of a protein.
    :return: The embedding of the protein. (1, embedding_size)
    """
    # Get the length of the protein sequence
    protein_length = len(sequence)

    # Get amino acid counts
    aa_counts = Counter(sequence)

    # Get the amino acid frequencies of the protein
    aa_frequencies = [aa_counts[aa] / protein_length for aa in AA_1]

    # Combine features to create the protein embedding
    protein_embedding = torch.FloatTensor([
        protein_length,
        *aa_frequencies,  # length 22
    ])

    return protein_embedding


def get_baseline_residue_embedding_index(sequence: str, index: int) -> torch.Tensor:
    """Get the baseline residue embedding from a protein sequence and residue index.

    Baseline residue embedding includes:
        - One-hot encoding of the residue
        - Relative position of the residue in the protein sequence
        - Protein length
        - BLOSUM62 embedding of the residue

    :param sequence: The amino acid sequence of a protein.
    :param index: The index of the residue in the protein sequence.
    :return: The embedding of the residue.
    """
    # Get the length of the protein sequence
    protein_length = len(sequence)

    # Create a one-hot vector for the residue
    residue_one_hot = [0] * len(AA_1)
    residue_one_hot[AA_1_TO_INDEX[sequence[index]]] = 1

    # Compute the residue's relative position in the protein sequence
    residue_position = index / len(sequence)

    # Combine features to create the residue embedding
    residue_embedding = torch.FloatTensor([
        *residue_one_hot,  # length 22
        residue_position,
        protein_length,
        *BLOSUM62_AA_TO_VECTOR[sequence[index]]  # length 24
    ])

    return residue_embedding


def get_baseline_residue_embedding(sequence: str) -> torch.Tensor:
    """Get the baseline residue embeddings from a protein sequence.

    :param sequence: The amino acid sequence of a protein.
    :return: A tensor of residue embeddings. (num_residues, embedding_size)
    """
    # Get the residue embeddings
    residue_embeddings = torch.stack([
        get_baseline_residue_embedding_index(sequence=sequence, index=index)
        for index in range(len(sequence))
    ])

    return residue_embeddings
