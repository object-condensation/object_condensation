from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor as T

from object_condensation.pytorch.losses import condensation_loss

from .loss_test_cases import CondensationMockData, test_cases


@dataclass
class TorchCondensationMockData:
    beta: T
    x: T
    object_id: T
    weights: T
    q_min: float
    radius_threshold: float

    @classmethod
    def from_numpy(cls, data: CondensationMockData) -> TorchCondensationMockData:
        return cls(
            beta=torch.from_numpy(data.beta),
            x=torch.from_numpy(data.x),
            object_id=torch.from_numpy(data.object_id),
            weights=torch.from_numpy(data.weights),
            q_min=data.q_min,
            radius_threshold=data.radius_threshold,
        )


def tensor_to_python(dct: dict[str, T]):
    return {k: v.item() for k, v in dct.items()}


@pytest.mark.parametrize(("data", "expected"), test_cases)
def test_condensation_loss(data: CondensationMockData, expected: dict[str, float]):
    data = TorchCondensationMockData.from_numpy(data)
    assert tensor_to_python(
        condensation_loss(
            beta=data.beta.squeeze(),
            x=data.x,
            object_id=data.object_id.squeeze(),
            mask=data.weights.squeeze().bool(),
            q_min=data.q_min,
            radius_threshold=data.radius_threshold,
            noise_thld=0,
        )
    ) == pytest.approx(expected)
