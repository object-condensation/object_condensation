from __future__ import annotations

import importlib.metadata

import object_condensation as m


def test_version():
    assert importlib.metadata.version("object_condensation") == m.__version__
