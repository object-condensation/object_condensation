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
    radius_threshold: float


def generate_test_data(
    n_hits=1000, n_objects=250, n_cluster_coords=3, rng=None
) -> CondensationMockData:
    if rng is None:
        rng = np.random.default_rng(seed=0)
    return CondensationMockData(
        beta=rng.random(n_hits),
        x=rng.random((n_hits, n_cluster_coords)),
        object_id=rng.choice(np.arange(n_objects), size=n_hits),
        q_min=1.0,
        radius_threshold=1.0,
        weights=np.ones(n_hits),
    )


test_cases = list[tuple[CondensationMockData, dict[str, float]]]()
