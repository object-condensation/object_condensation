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
        ``cl_peak``: Averaged over all objects
        ``cl_noise``: Averaged over all noise hits
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
