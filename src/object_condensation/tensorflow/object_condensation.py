from __future__ import annotations

import tensorflow as tf


class ObjectCondensation(tf.keras.layers.Layer):
    """ """

    def __init__(
        self,
        q_min: float = 1.0,
        record_metrics: bool = False,
        active: bool = True,
        scale: float = 1.0,
        weight_attractive: float = 1.0,
        weight_repulsive: float = 1.0,
        weight_noise: float = 1.0,
        weight_beta: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.q_min = q_min
        self.record_metrics = record_metrics
        self.active = active
        self.scale = scale
        self.weight_attractive = weight_attractive
        self.weight_repulsive = weight_repulsive
        self.weight_noise = weight_noise
        self.weight_beta = weight_beta

    def get_config(self) -> dict:
        config = {
            "q_min": self.q_min,
            "active": self.active,
            "scale": self.scale,
            "weight_attractive": self.weight_attractive,
            "weight_repulsive": self.weight_repulsive,
            "weight_noise": self.weight_noise,
            "weight_beta": self.weight_beta,
            "record_metrics": self.record_metrics,
            "name": self.name,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

    def build(self, input_shape):
        super().build(input_shape)

    def _calculate_losses(self, vertex_ids, beta, ccoords):
        object_ids, _ = tf.unique(vertex_ids)
        q = tf.tanh(beta) ** 2 + self.q_min
        M = tf.cast(vertex_ids[:, None] == object_ids[None, :], tf.float32)
        Mnot = tf.cast(vertex_ids[:, None] != object_ids[None, :], tf.float32)
        cond_points = tf.argmax(beta * M, axis=0)
        beta_k = tf.gather(beta, cond_points)
        q_k = tf.gather(q, cond_points)
        x_k = tf.gather(ccoords, cond_points)
        d_j_k = tf.norm(ccoords[None, :, :] - x_k[:, None, :], axis=-1)

        V_att_k = tf.math.divide_no_nan(
            tf.reduce_sum(
                q_k[
                    :,
                ] * tf.transpose(
                    beta
                ) * tf.transpose(M) * d_j_k**2,
                axis=1,
            ),
            tf.reduce_sum(M, axis=0) + 1e-3,
        )

        V_rep_k = tf.math.divide_no_nan(
            tf.reduce_sum(
                q_k[:,]
                * tf.transpose(beta)
                * tf.transpose(Mnot)
                * tf.math.maximum(0, 1.0 - d_j_k),
                axis=1,
            ),
            tf.reduce_sum(Mnot, axis=0) + 1e-3,
        )

        L_beta_k = 1.0 - beta_k

        V_att = tf.divide_no_nan(
            V_att_k, tf.cast(tf.shape(object_ids)[0] - 1.0, tf.float32)
        )

        V_rep = tf.divide_no_nan(
            V_rep_k, tf.cast(tf.shape(object_ids)[0] - 1.0, tf.float32)
        )

        L_beta = tf.divide_no_nan(
            tf.reduce_sum(L_beta_k), tf.cast(tf.shape(object_ids)[0] - 1.0, tf.float32)
        )

        L_noise = tf.math.divide_no_nan(
            tf.reduce_sum(beta[vertex_ids == -1]),
            tf.reduce_sum(tf.cast(vertex_ids == -1, tf.float32)),
        )

        return {
            "V_att": V_att,
            "V_rep": V_rep,
            "L_beta": L_beta,
            "L_noise": L_noise,
        }


    def loss(self, inputs: list):
        assert (
            len(inputs) == 3
        ), "ObjectCondensation needs 3 inputs: vertex_ids, beta, ccoords"
        vertex_ids, beta, ccoords = inputs

        oc_loss_dict = self._calculate_potentials(vertex_ids, beta, ccoords)
        lossval = 0.0
        lossval += self.weight_attractive * oc_loss_dict["V_att"]
        lossval += self.weight_repulsive * oc_loss_dict["V_rep"]
        lossval += self.weight_beta * oc_loss_dict["L_beta"]
        loss += self.weight_noise * oc_loss_dict["L_noise"]
        lossval *= self.scale

        return lossval

    def call(self, inputs):
        lossval = tf.constant([0.0], dtype="float32")

        if self.active:
            lossval = self.loss(inputs)

            if not self.return_lossval:
                self.add_loss(lossval)

        self.add_prompt_metric(lossval, self.name + "_loss")

        return inputs[0]
