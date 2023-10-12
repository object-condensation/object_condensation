from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor as T

from object_condensation.pytorch.losses import condensation_loss
from tests.loss_test_cases import CondensationMockData, test_cases


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


@pytest.mark.parametrize(("data", "expected"), test_cases)
def test_condensation_loss(data: CondensationMockData, expected: dict[str, float]):
    data = TorchCondensationMockData.from_numpy(data)
    assert (
        condensation_loss(
            beta=data.beta,
            x=data.x,
            object_id=data.object_id,
            mask=data.weights,
            q_min=data.q_min,
            radius_threshold=data.radius_threshold,
        )
        == expected.item()
    )
