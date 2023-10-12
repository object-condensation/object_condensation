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
        ``cl_noise``: Averaged over all noise hits
    """
    # x: n_nodes x n_outdim
    not_noise = object_id > 0
    unique_oids = torch.unique(object_id[not_noise])
    assert len(unique_oids) > 0, "No particles found, cannot evaluate loss"
    # n_nodes x n_pids
    # The nodes in every column correspond to the hits of a single particle and
    # should attract each other
    attractive_mask = object_id[:, None] == unique_oids[None, :]

    q = torch.arctanh(beta) ** 2 + q_min
    assert not torch.isnan(q).any(), "q contains NaNs"
    alphas = torch.argmax(q[:, None] * attractive_mask, dim=0)

    # n_pids x n_outdim
    x_alphas = x[alphas]
    # 1 x n_pids
    q_alphas = q[alphas][None, :]

    # n_nodes x n_pids
    dist = torch.cdist(x, x_alphas)

    # It's important to directly do the .mean here so we don't keep these large
    # matrices in memory longer than we need them
    # Attractive potential (n_nodes x n_pids)
    va_matrix = q[:, None] * q_alphas * attractive_mask * torch.square(dist)
    va = torch.mean(torch.mean(va_matrix[mask], dim=0))
    # Repulsive potential (n_nodes x n_pids)
    vr_matrix = (
        q[:, None] * q_alphas * (~attractive_mask) * relu(radius_threshold - dist)
    )
    vr = torch.mean(torch.mean(vr_matrix, dim=0))

    peak = torch.mean(1 - beta[alphas])
    noise = torch.mean(beta[~not_noise])

    return {
        "attractive": va,
        "repulsive": vr,
        "peak": peak,
        "noise": noise,
    }
