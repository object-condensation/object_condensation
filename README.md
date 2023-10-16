# object_condensation

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/object-condensation/object_condensation/workflows/CI/badge.svg
[actions-link]:             https://github.com/object-condensation/object_condensation/actions
[pypi-link]:                https://pypi.org/project/object_condensation/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/object_condensation
[pypi-version]:             https://img.shields.io/pypi/v/object_condensation
[rtd-badge]:                https://readthedocs.org/projects/object_condensation/badge/?version=latest
[rtd-link]:                 https://object-condensation.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

The Object Condensation loss -
[developed by Jan Kieseler](https://arxiv.org/abs/2002.03605) - is now being
used by several groups in high energy physics for both track reconstruction and
shower reconstruction in calorimeters.

Several implementations of this idea already exist, but often they are
maintained by very few people. This repository aims to provide an easy to use
implementation for both the TensorFlow and PyTorch backend.

Existing Implementations:

- [cms-pepr](https://github.com/cms-pepr/HGCalML) [TensorFlow]
- [mlpf](https://github.com/selvaggi/mlpf/blob/main/README.md) [PyTorch]
- [gnn-tracking](https://github.com/gnn-tracking/gnn_tracking/tree/main)
  [PyTorch]

## Installation

```bash
python3 -m pip install -e 'object_condensation[pytorch]'
# or
python3 -m pip install -e 'object_condensation[tensorflow]'
```

### Development setup

For the development setup, clone this repository and also add the `dev` and
`testing` extra options, e.g.,

```bash
python3 -m pip install -e '.[pytorch,dev,testing]'
```

Please also install pre-commit:

```bash
python3 -m pip install pre-commit
pre-commit install  # in top-level directory of repository
```

## Conventions

## Implementations

> **Note** For a comparison of the performance of the different implementations,
> [see the docs][benchmark docs].

[benchmark docs]:
  https://object-condensation.readthedocs.io/en/latest/?badge=latest#benchmarks

### Default

`condensation_loss` is a straightforward implementation that is easy to read and
to verify. It is used as baseline.

- [pytorch API](https://object-condensation.readthedocs.io/en/latest/#object_condensation.pytorch.losses.condensation_loss)
- [tensorflow API](https://object-condensation.readthedocs.io/en/latest/#object_condensation.tensorflow.losses.condensation_loss)

### Tiger

`condensation_loss_tiger` saves memory by "masking instead of multiplying".
Consider the repulsive loss: It is an aggregation of potentials between
condensation points (CPs) and individual nodes. If these potentials are taken to
be hinge losses `relu(1-dist)`, then they vanish for most CP-node pairs
(assuming a sufficiently well-trained model).

Compare now the following two implementation strategies (where `dist` is the
CP-node distance matrix):

```python
# Simplified by assuming that all points belong to repulsive potential
# Strategy 1
v_rep = sum(relu(1 - dist))
# Strategy 2 (tiger)
rep_mask = dist < 1
v_rep = sum((1 - dist)[rep_mask])
```

In strategy 1, pytorch will keep all elements of `dist` in memory for
backpropagation (even though most of the `relu`-differentials will be 0). In
strategy 2 (because the mask is detached from the computational graph), the
number of elements to backpropagate with will be greatly reduced.

However, there is still one problem: What if our latent space collapses at some
point (or at the beginning of the training)? This would result in batches with a
greatly increased memory consumption, possibly crashing the run. To counter
this, an additional parameter `max_n_rep` is introduced. If the number of
repulsive pairs (`rep_mask.sum()` in the example above) exceeds `max_n_rep`,
then `rep_mask` will sample `max_n_rep` elements and upweight them by
`n_rep/max_n_rep`. To check for this, `condensation_loss_tiger` will return
`n_rep` in addition to the losses.

- [pytorch API](https://object-condensation.readthedocs.io/en/latest/#object_condensation.pytorch.losses.condensation_loss_tiger)
