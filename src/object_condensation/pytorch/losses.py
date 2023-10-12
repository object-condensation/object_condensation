from __future__ import annotations

import torch
from torch import Tensor as T
from torch.nn.functional import relu


# todo: Replace mask with weight and also check how it should factor in repulsive/
#   background losses
@torch.compile
def condensation_loss(
    *,
    beta: T,
    x: T,
    object_id: T,
    mask: T,
    q_min: float,
    radius_threshold: float,
) -> dict[str, T]:
    """Condensation losses

    Args:
        beta: Condensation likelihoods
        x: Clustering coordinates
        object_id: Labels for objects. Objects with `object_id <= 0` are considered
            noise
        mask: Mask for attractive loss, e.g., to only attract hits for
        q_min: Minimal charge
        radius_threshold: Radius threshold for repulsive potential. In case of linear
            scarlarization of the multi objective losses, this is redundant and should
            be fixed to 1.

    Returns:
        Dictionary of scalar tensors.

        ``attractive``: Averaged over object, then averaged over all objects.
        ``repulsive``: Averaged like ``attractive``
        ``cl_peak``: Averaged over all objects
        ``cl_noise``
    """
    # x: n_nodes x n_outdim
    not_noise = object_id > 0
    unique_pids = torch.unique(object_id[not_noise])
    assert len(unique_pids) > 0, "No particles found, cannot evaluate loss"
    # n_nodes x n_pids
    # The nodes in every column correspond to the hits of a single particle and
    # should attract each other
    attractive_mask = object_id[:, None] == unique_pids[None, :]

    q = torch.arctanh(beta) ** 2 + q_min
    assert not torch.isnan(q).any(), "q contains NaNs"
    alphas = torch.argmax(q[:, None] * attractive_mask, dim=0)

    # n_pids x n_outdim
    x_alphas = x[alphas]
    # 1 x n_pids
    q_alphas = q[alphas][None, :]

    # n_nodes x n_pids
    dist = torch.cdist(x, x_alphas)

    # Attractive potential (n_nodes x n_pids)
    va = q[:, None] * attractive_mask * torch.square(dist) * q_alphas
    # Repulsive potential (n_nodes x n_pids)
    vr = q[:, None] * (~attractive_mask) * relu(radius_threshold - dist) * q_alphas

    cl_peak = torch.mean(1 - beta[alphas])
    cl_noise = torch.mean(beta[~not_noise])

    return {
        "attractive": torch.mean(torch.mean(va[mask], dim=0)),
        "repulsive": torch.mean(torch.mean(vr, dim=0)),
        "cl_peak": cl_peak,
        "cl_noise": cl_noise,
    }
