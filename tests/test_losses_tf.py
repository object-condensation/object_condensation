from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from loss_test_cases import CondensationMockData
from tensorflow import Tensor as T


@dataclass
class TFCondensationMockData:
    beta: T
    x: T
    object_id: T
    weights: T
    q_min: float
    radius_threshold: float

    @classmethod
    def from_numpy(cls, data: CondensationMockData) -> TFCondensationMockData:
        return cls(
            beta=tf.convert_to_tensor(data.beta),
            x=tf.convert_to_tensor(data.x),
            object_id=tf.convert_to_tensor(data.object_id),
            weights=tf.convert_to_tensor(data.weights),
            q_min=data.q_min,
            radius_threshold=data.radius_threshold,
        )
