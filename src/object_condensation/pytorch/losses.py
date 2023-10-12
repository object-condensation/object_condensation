from __future__ import annotations

import torch
from torch import Tensor as T
from torch.nn.functional import relu


# todo: Replace mask with weight and also check how it should factor in repulsive/
#   background losses
# @torch.compile
def condensation_loss(
    *,
    beta: T,
    x: T,
    object_id: T,
    mask: T,
    q_min: float,
    radius_threshold: float,
    noise_thld: int,
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
        noise_thld: Threshold for noise hits. Hits with ``object_id <= noise_thld``
            are considered to be noise

    Returns:
        Dictionary of scalar tensors.

        ``attractive``: Averaged over object, then averaged over all objects.
        ``repulsive``: Averaged like ``attractive``
        ``cl_peak``: Averaged over all objects
        ``cl_noise``: Averaged over all noise hits
    """
    # To protect against nan in divisions
    eps = 1e-9

    # x: n_nodes x n_outdim
    not_noise = object_id > noise_thld
    unique_oids = torch.unique(object_id[not_noise])
    assert len(unique_oids) > 0, "No particles found, cannot evaluate loss"
    # n_nodes x n_pids
    # The nodes in every column correspond to the hits of a single particle and
    # should attract each other
    attractive_mask = object_id[:, None] == unique_oids[None, :]

    q = torch.arctanh(beta) ** 2 + q_min
    assert not torch.isnan(q).any(), "q contains NaNs"
    # n_objs
    alphas = torch.argmax(q[:, None] * attractive_mask, dim=0)

    # _j means indexed by hits
    # _k means indexed by objects

    # n_objs x n_outdim
    x_k = x[alphas]
    # 1 x n_objs
    q_k = q[alphas][None, :]

    dist_j_k = torch.cdist(x, x_k)

    # Attractive potential/loss
    # todo: do I need the copy/new axis here or would it broadcast?
    v_att_j_k = q[:, None] * q_k * attractive_mask * torch.square(dist_j_k)
    # It's important to directly do the .mean here so we don't keep these large
    # matrices in memory longer than we need them
    # Attractive potential per object normalized over number of hits in object
    v_att_k = torch.sum(v_att_j_k[mask], dim=0) / (
        torch.sum(attractive_mask[mask], dim=0) + eps
    )
    v_att = torch.mean(v_att_k)

    # Repulsive potential/loss
    v_rep_j_k = (
        q[:, None] * q_k * (~attractive_mask) * relu(radius_threshold - dist_j_k)
    )
    v_rep_k = torch.sum(v_rep_j_k, dim=0) / (torch.sum(~attractive_mask, dim=0) + eps)
    v_rep = torch.mean(v_rep_k)

    l_coward = torch.mean(1 - beta[alphas])
    l_noise = torch.mean(beta[~not_noise])

    return {
        "attractive": v_att,
        "repulsive": v_rep,
        "coward": l_coward,
        "noise": l_noise,
    }
