# object_condensation

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/object-condensation/object_condensation/workflows/CI/badge.svg
[actions-link]:             https://github.com/object-condensation/object_condensation/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/object_condensation
[conda-link]:               https://github.com/conda-forge/object_condensation-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/object-condensation/object_condensation/discussions
[pypi-link]:                https://pypi.org/project/object_condensation/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/object_condensation
[pypi-version]:             https://img.shields.io/pypi/v/object_condensation
[rtd-badge]:                https://readthedocs.org/projects/object_condensation/badge/?version=latest
[rtd-link]:                 https://object_condensation.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

The Object Condensation loss - [developed by Jan Kieseler](https://arxiv.org/abs/2002.03605) - is now being used by several groups in high energy physics for both track reconstruction and shower reconstruction in calorimeters. 

Several implementations of this idea already exist, but often they are maintained by very few people. This repository aims to provide an easy to use imlementation for both the TensorFlow and PyTorch backend.  

Existing Implementations: 

* [cms-pepr](https://github.com/cms-pepr/HGCalML) [TensorFlow]
* [mlpf](https://github.com/selvaggi/mlpf/blob/main/README.md) [PyTorch]
* [gnn-tracking](https://github.com/gnn-tracking/gnn_tracking/tree/main) [PyTorch]
