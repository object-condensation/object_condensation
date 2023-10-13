from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import ndarray as A


@dataclass
class CondensationMockData:
    beta: A
    x: A
    object_id: A
    weights: A
    q_min: float


def generate_test_data(
    n_hits=1000, n_objects=250, n_cluster_coords=3, rng=None
) -> CondensationMockData:
    if rng is None:
        rng = np.random.default_rng(seed=0)
    return CondensationMockData(
        beta=rng.random((n_hits, 1)),
        x=rng.random((n_hits, n_cluster_coords)),
        object_id=rng.choice(np.arange(n_objects), size=n_hits).reshape(-1, 1),
        q_min=1.0,
        weights=np.ones((n_hits, 1)),
    )


test_cases: list[tuple[CondensationMockData, dict[str, float]]] = [
    (
        generate_test_data(),
        {
            "attractive": 1.7951122042912249,
            "repulsive": 1.9508766539017457,
            "coward": 0.2156792078639991,
            "noise": 0.77484477845007,
        },
    )
]
