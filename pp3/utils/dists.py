import torch


def pairwise_dist(a, b):
    """Computes the pairwise euclidean distance between a and b."""
    return torch.cdist(a, b, p=2)
