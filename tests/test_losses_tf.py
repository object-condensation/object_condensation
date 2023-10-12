from __future__ import annotations

from dataclasses import dataclass

import pytest
import tensorflow as tf
from tensorflow import Tensor as T

from object_condensation.pytorch.losses import condensation_loss

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


@pytest.mark.parametrize(("data", "expected"), test_cases)
def test_condensation_loss(data: CondensationMockData, expected: dict[str, float]):
    data = TFCondensationMockData.from_numpy(data)
    assert (
        condensation_loss(
            beta=data.beta,
            x=data.x,
            object_id=data.object_id,
            mask=data.weights,
            q_min=data.q_min,
        )
        == expected.item()
    )
