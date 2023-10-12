from __future__ import annotations

from dataclasses import dataclass

import pytest
import tensorflow as tf
from tensorflow import Tensor as T

from object_condensation.tensorflow.losses import calculate_losses

from .loss_test_cases import CondensationMockData, test_cases


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


def tensor_to_python(dct: dict[str, T]):
    return {k: v.numpy().item() for k, v in dct.items()}


@pytest.mark.parametrize(("data", "expected"), test_cases)
def test_condensation_loss(data: CondensationMockData, expected: dict[str, float]):
    data = TFCondensationMockData.from_numpy(data)
    assert tensor_to_python(
        calculate_losses(
            beta=data.beta,
            x=data.x,
            object_id=data.object_id,
            weights=data.weights,
            q_min=data.q_min,
            noise_threshold=0,
        )
    ) == pytest.approx(expected)
