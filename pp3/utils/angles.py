import torch


def angle(p1, p2, p3):
    """_summary_

    Parameters
    ----------
    p0 : _description_
    p1 : _description_
    p2 : _description_

    Returns
    -------
    _description_

    """
    # Compute vectors
    u1 = p1 - p2
    u2 = p3 - p2

    # Compute angle
    x = torch.norm(torch.cross(u1, u2, dim=-1), dim=-1)
    y = (u1 * u2).sum(dim=-1)
    return torch.atan2(x, y).squeeze(-1)


def dihedral_angle(p1, p2, p3, p4):
    """_summary_

    Parameters
    ----------
    p0 : _description_
    p1 : _description_
    p2 : _description_
    p3 : _description_

    Returns
    -------
    _description_

    """
    # Compute bond vectors
    u1 = p2 - p1
    u2 = p3 - p2
    u3 = p4 - p3

    # Compute dihedral angle
    b1 = torch.cross(u1, u2)
    b2 = torch.cross(u2, u3)
    x = torch.sum(torch.cross(b1, b2) * u2, dim=1, keepdim=True)
    u2_norm = torch.linalg.norm(u2, dim=1, keepdim=True)
    y = u2_norm * torch.sum(b1 * b2, dim=1, keepdim=True)
    return torch.arctan2(x, y).squeeze(-1)


def internal_coordinates(x):
    """_summary_

    Parameters
    ----------
    coords : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_

    """
    # Compute phi dihedral angle: C - N - CA - C
    phi = dihedral_angle(x[0:-1, 2], x[1:, 0], x[1:, 1], x[1:, 2])

    # Compute psi dihedral angle: N - CA - C - N
    psi = dihedral_angle(x[0:-1, 0], x[0:-1, 1], x[0:-1, 2], x[1:, 0])

    # Compute omega dihedral angle: CA - C - N - CA
    omega = dihedral_angle(x[0:-1, 1], x[0:-1, 2], x[1:, 0], x[1:, 1])

    # Compute bond angle: N - CA - C
    n_ca_c = angle(x[:, 0], x[:, 1], x[:, 2])

    # Compute bond angle: CA - C - N
    ca_c_n = angle(x[:-1, 1], x[:-1, 2], x[1:, 0])

    # Compute bond angle: C - N - CA
    c_n_ca = angle(x[:-1, 2], x[1:, 0], x[1:, 1])

    # Pad missing angles with zeros
    phi = torch.cat([[0], phi])
    psi = torch.cat([psi, [0]])
    omega = torch.cat([omega, [0]])
    ca_c_n = torch.cat([ca_c_n, [0]])
    c_n_ca = torch.cat([c_n_ca, [0]])

    # Stack with sidechainnet ordering
    internals = torch.stack([phi, psi, omega, n_ca_c, ca_c_n, c_n_ca], dim=1)
    return internals
