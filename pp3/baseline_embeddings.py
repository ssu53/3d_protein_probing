"""Contains baseline embeddings for proteins and residues."""
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from pp3.utils.constants import AA_1, AA_1_TO_INDEX
from pp3.utils.constants import BLOSUM62_AA_TO_VECTOR


def get_baseline_protein_embedding(sequence: str) -> torch.Tensor:
    """Get the baseline protein embedding from a protein sequence.

    Baseline protein embedding includes:
        - Protein length
        - Protein features from the ProteinAnalysis module

    :param sequence: The amino acid sequence of a protein.
    :return: The embedding of the protein.
    """
    # Analyze protein with biopython ProteinAnalysis module
    protein_analysis = ProteinAnalysis(sequence)

    # Get the amino acid composition of the protein
    aa_frequencies = protein_analysis.get_amino_acids_percent()

    # Create an amino acid frequency vector as the protein embedding
    aa_frequencies = [aa_frequencies.get(aa, 0) for aa in AA_1]

    # Combine features to create the protein embedding
    protein_embedding = torch.FloatTensor([
        len(sequence),
        *aa_frequencies,  # length 22
        protein_analysis.molecular_weight(),
        protein_analysis.aromaticity(),
        protein_analysis.instability_index(),
        protein_analysis.gravy(),
        protein_analysis.isoelectric_point(),
        protein_analysis.charge_at_pH(7.4),
        *protein_analysis.secondary_structure_fraction(),  # length 3
        *protein_analysis.molar_extinction_coefficient()  # length 2
    ])

    return protein_embedding


def get_baseline_protein_one_hot_mask() -> torch.Tensor:
    """Get a binary mask indicating which features in the protein baseline embedding are one-hot encoded."""
    length = get_baseline_residue_embedding_index(sequence='A', index=0).shape[0]
    one_hot_mask = torch.zeros(length, dtype=torch.bool)

    return one_hot_mask


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
    # Create a one-hot vector for the residue
    residue_one_hot = [0] * len(AA_1)
    residue_one_hot[AA_1_TO_INDEX[sequence[index]]] = 1

    # Compute the residue's relative position in the protein sequence
    residue_position = index / len(sequence)

    # Get the length of the protein sequence
    protein_length = len(sequence)

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
    :return: A tensor of residue embeddings.
    """
    # Get the residue embeddings
    residue_embeddings = torch.stack([
        get_baseline_residue_embedding_index(sequence=sequence, index=index)
        for index in range(len(sequence))
    ])

    return residue_embeddings


def get_baseline_residue_one_hot_mask() -> torch.Tensor:
    """Get a binary mask indicating which features in the residue baseline embedding are one-hot encoded."""
    length = get_baseline_residue_embedding_index(sequence='A', index=0).shape[0]
    one_hot_mask = torch.zeros(length, dtype=torch.bool)
    one_hot_mask[:len(AA_1)] = 1

    return one_hot_mask
