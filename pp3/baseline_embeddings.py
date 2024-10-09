"""Contains baseline embeddings for proteins and residues."""
from collections import Counter

import torch

from pp3.utils.constants import AA_1, AA_1_TO_INDEX, BLOSUM62_AA_TO_VECTOR, MAX_SEQ_LEN


def get_baseline_residue_embedding_index(sequence: str, index: int, identify_residue: bool) -> torch.Tensor:
    """Get the baseline residue embedding from a protein sequence and residue index.

    Baseline residue embedding includes:
        - One-hot encoding of the residue (if identify_residue)
        - Relative position of the residue in the protein sequence
        - Protein length
        - BLOSUM62 embedding of the residue (if identify_residue)

    :param sequence: The amino acid sequence of a protein.
    :param index: The index of the residue in the protein sequence.
    :return: The embedding of the residue.
    """
    # Get the length of the protein sequence
    protein_length = len(sequence) / MAX_SEQ_LEN

    if identify_residue:
        # Create a one-hot vector for the residue
        residue_one_hot = [0] * len(AA_1)
        residue_one_hot[AA_1_TO_INDEX[sequence[index]]] = 1

    # Compute the residue's relative position in the protein sequence
    residue_position = index / len(sequence)

    if identify_residue:
        # Combine features to create the residue embedding
        residue_embedding = torch.FloatTensor([
            *residue_one_hot,  # length 22
            residue_position,
            protein_length,
            *BLOSUM62_AA_TO_VECTOR[sequence[index]]  # length 24
        ])
    else:
        # Combine features to create the residue embedding
        residue_embedding = torch.FloatTensor([
            residue_position,
            protein_length,
        ])

    return residue_embedding


def get_baseline_residue_embedding(sequence: str, identify_residue: bool) -> torch.Tensor:
    """Get the baseline residue embeddings from a protein sequence.

    :param sequence: The amino acid sequence of a protein.
    :return: A tensor of residue embeddings. (num_residues, embedding_size)
    """
    # Get the residue embeddings
    residue_embeddings = torch.stack([
        get_baseline_residue_embedding_index(sequence=sequence, index=index, identify_residue=identify_residue)
        for index in range(len(sequence))
    ])

    return residue_embeddings


def get_residue_tokens_embedding(sequence: str) -> torch.Tensor:
    """Get the residue tokens embeddings from a protein sequence.

    :param sequence: The amino acid sequence of a protein.
    :return: A tensor of residue tokens embeddings. (num_residues,)
    """
    # Get the residue tokens embeddings
    residue_embeddings = torch.tensor([
        AA_1_TO_INDEX[sequence[index]] + 1
        for index in range(len(sequence))
    ], dtype=torch.long)

    return residue_embeddings
