from __future__ import annotations

import tensorflow as tf


def condensation_loss(
    *,
    q_min: float,
    object_id: tf.Tensor,
    beta: tf.Tensor,
    x: tf.Tensor,
    weights: tf.Tensor = None,
    noise_threshold: int = 0,
) -> dict[str, tf.Tensor]:
    """Condensation losses

    Args:
        beta: Condensation likelihoods
        x: Clustering coordinates
        object_id: Labels for objects. Objects with `object_id <= 0` are considered
            noise
        weights: Weights per hit, multiplied to attractive/repulsive potentials
        q_min: Minimal charge
        noise_threshold: Threshold for noise hits. Hits with ``object_id <= noise_thld``
            are considered to be noise

    Returns:
        Dictionary of scalar tensors.

        ``attractive``: Averaged over object, then averaged over all objects.
        ``repulsive``: Averaged like ``attractive``
        ``coward``: Averaged over all objects
        ``noise``: Averaged over all noise hits
    """
    if weights is None:
        weights = tf.ones_like(beta)
    q_min = tf.cast(q_min, tf.float32)
    object_id = tf.reshape(object_id, (-1,))
    beta = tf.cast(beta, tf.float32)
    x = tf.cast(x, tf.float32)
    weights = tf.cast(weights, tf.float32)

    not_noise = object_id > noise_threshold
    unique_oids, _ = tf.unique(object_id[not_noise])
    q = tf.cast(tf.math.atanh(beta) ** 2 + q_min, tf.float32)
    mask_att = tf.cast(object_id[:, None] == unique_oids[None, :], tf.float32)
    mask_rep = tf.cast(object_id[:, None] != unique_oids[None, :], tf.float32)
    alphas = tf.argmax(beta * mask_att, axis=0)
    beta_k = tf.gather(beta, alphas)
    q_k = tf.gather(q, alphas)
    x_k = tf.gather(x, alphas)

    dist_j_k = tf.norm(x[None, :, :] - x_k[:, None, :], axis=-1)

    v_att_k = tf.math.divide_no_nan(
        tf.reduce_sum(
            q_k
            * tf.transpose(weights)
            * tf.transpose(q)
            * tf.transpose(mask_att)
            * dist_j_k**2,
            axis=1,
        ),
        tf.reduce_sum(mask_att, axis=0) + 1e-9,
    )
    v_att = tf.math.divide_no_nan(
        tf.reduce_sum(v_att_k), tf.cast(tf.shape(unique_oids)[0], tf.float32)
    )

    v_rep_k = tf.math.divide_no_nan(
        tf.reduce_sum(
            q_k
            * tf.transpose(weights)
            * tf.transpose(q)
            * tf.transpose(mask_rep)
            * tf.math.maximum(0, 1.0 - dist_j_k),
            axis=1,
        ),
        tf.reduce_sum(mask_rep, axis=0) + 1e-9,
    )

    v_rep = tf.math.divide_no_nan(
        tf.reduce_sum(v_rep_k), tf.cast(tf.shape(unique_oids)[0], tf.float32)
    )

    coward_loss_k = 1.0 - beta_k
    coward_loss = tf.math.divide_no_nan(
        tf.reduce_sum(coward_loss_k),
        tf.cast(tf.shape(unique_oids)[0], tf.float32),
    )

    noise_loss = tf.math.divide_no_nan(
        tf.reduce_sum(beta[object_id <= noise_threshold]),
        tf.reduce_sum(tf.cast(object_id <= noise_threshold, tf.float32)),
    )

    return {
        "attractive": v_att,
        "repulsive": v_rep,
        "coward": coward_loss,
        "noise": noise_loss,
    }


def _condensation_loss_tiger(
    *,
    beta: tf.Tensor,
    x: tf.Tensor,
    object_id: tf.Tensor,
    weights: tf.Tensor,
    q_min: float,
    noise_threshold: int,
    max_n_rep: int,
) -> dict[str, tf.Tensor]:
    """Extracted function for torch compilation. See `condensation_loss_tiger` for
    docstring.
    """
    # To protect against nan in divisions
    eps = 1e-9
    # x: n_nodes x n_outdim
    not_noise = object_id > noise_threshold
    unique_oids = tf.unique(object_id[not_noise])
    assert len(unique_oids) > 0, "No particles found, cannot evaluate loss"
    # n_nodes x n_pids
    # The nodes in every column correspond to the hits of a single particle and
    # should attract each other
    attractive_mask = object_id.view(-1, 1) == unique_oids.view(1, -1)

    q = tf.math.atanh(beta) ** 2 + q_min
    assert not tf.math.reduce_any(tf.math.is_nan(q)), "q contains NaNs"
    # n_objs
    alphas = tf.argmax(q.view(-1, 1) * attractive_mask, dim=0)

    # _j means indexed by hits
    # _k means indexed by objects
    # n_objs x n_outdim
    x_k = x[alphas]
    # 1 x n_objs
    q_k = q[alphas].view(1, -1)

    dist_j_k = tf.norm(x[None, :, :] - x_k[:, None, :], axis=-1)

    qw_j_k = weights.view(-1, 1) * q.view(-1, 1) * q_k

    att_norm_k = (attractive_mask.sum(dim=0) + eps) * len(unique_oids)
    qw_att = (qw_j_k / att_norm_k)[attractive_mask]

    # Attractive potential/loss
    v_att = (qw_att * tf.square(dist_j_k[attractive_mask])).sum()

    repulsive_mask = (~attractive_mask) & (dist_j_k < 1)
    n_rep_k = (~attractive_mask).sum(dim=0)
    n_rep = repulsive_mask.sum()
    # Don't normalize to repulsive_mask, it includes the dist < 1 count,
    # (less points within the radius 1 ball should translate to lower loss)
    rep_norm = (n_rep_k + eps) * len(unique_oids)
    if n_rep > max_n_rep > 0:
        sampling_freq = max_n_rep / n_rep
        sampling_mask = (
            tf.random.uniform_like(repulsive_mask, dtype=tf.float16) < sampling_freq
        )
        repulsive_mask &= sampling_mask
        rep_norm *= sampling_freq
    qw_rep = (qw_j_k / rep_norm)[repulsive_mask]
    v_rep = (qw_rep * (1 - dist_j_k[repulsive_mask])).sum()

    l_coward = tf.reduce_mean(1 - beta[alphas])
    l_noise = tf.reduce_mean(beta[~not_noise])

    return {
        "attractive": v_att,
        "repulsive": v_rep,
        "coward": l_coward,
        "noise": l_noise,
        "n_rep": n_rep,
    }


def condensation_loss_tiger(
    *,
    beta: tf.Tensor,
    x: tf.Tensor,
    object_id: tf.Tensor,
    weights: tf.Tensor | None = None,
    q_min: float,
    noise_threshold: int = 0,
    max_n_rep: int = 0,
) -> dict[str, tf.Tensor]:
    """Condensation losses

    Args:
        beta: Condensation likelihoods
        x: Clustering coordinates
        object_id: Labels for objects.
        weights: Weights per hit, multiplied to attractive/repulsive potentials
        q_min: Minimal charge
        noise_threshold: Threshold for noise hits. Hits with ``object_id <= noise_thld``
            are considered to be noise
        max_n_rep: Maximum number of elements to consider for repulsive loss.
            Set to 0 to disable.
        torch_compile: Torch compile loss function. This is recommended, but might not
            work in older pytorch version or in cutting edge python.

    Returns:
        Dictionary of scalar tensors; see readme.
        `n_rep`: Number of repulsive elements (before sampling).
    """
    if weights is None:
        weights = tf.ones_like(beta)
    loss = _condensation_loss_tiger
    return loss(
        beta=beta,
        x=x,
        object_id=object_id,
        weights=weights,
        q_min=q_min,
        noise_threshold=noise_threshold,
        max_n_rep=max_n_rep,
    )
