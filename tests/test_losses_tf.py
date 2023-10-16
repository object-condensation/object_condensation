from __future__ import annotations

from dataclasses import dataclass

import pytest
import tensorflow as tf
from tensorflow import Tensor as T

from object_condensation.tensorflow.losses import (
    condensation_loss,
    condensation_loss_tiger,
)

from .loss_test_cases import CondensationMockData, test_cases


@dataclass
class TFCondensationMockData:
    beta: T
    x: T
    object_id: T
    weights: T
    q_min: float

    @classmethod
    def from_numpy(cls, data: CondensationMockData) -> TFCondensationMockData:
        return cls(
            beta=tf.convert_to_tensor(data.beta),
            x=tf.convert_to_tensor(data.x),
            object_id=tf.convert_to_tensor(data.object_id),
            weights=tf.convert_to_tensor(data.weights),
            q_min=data.q_min,
        )


def tensor_to_python(dct: dict[str, T]):
    return {k: v.numpy().item() for k, v in dct.items()}


@pytest.mark.parametrize(("data", "expected"), test_cases)
def test_condensation_loss(data: CondensationMockData, expected: dict[str, float]):
    data = TFCondensationMockData.from_numpy(data)
    assert tensor_to_python(
        condensation_loss(
            beta=data.beta,
            x=data.x,
            object_id=data.object_id,
            weights=data.weights,
            q_min=data.q_min,
            noise_threshold=0,
        )
    ) == pytest.approx(expected)


@pytest.mark.parametrize(("data", "expected"), test_cases)
def test_condensation_loss_tiger(
    data: CondensationMockData,
    expected: dict[str, float],
):
    data = TFCondensationMockData.from_numpy(data)
    result = tensor_to_python(
        condensation_loss_tiger(
            beta=data.beta.squeeze(),
            x=data.x,
            object_id=data.object_id.squeeze(),
            weights=data.weights.squeeze(),
            q_min=data.q_min,
            noise_threshold=0,
            max_n_rep=1_000_000,
        )
    )
    assert result.pop("n_rep") == 220768
    assert result == pytest.approx(expected)
