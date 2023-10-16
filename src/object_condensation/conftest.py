from __future__ import annotations

import os
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def set_env() -> None:  # noqa: PT004
    """Set environment variables for tests"""
    if sys.platform == "darwin":
        _ = (
            "Workaround for https://github.com/kevlened/pytest-parallel/issues/93 "
            "resp https://bugs.python.org/issue30385"
        )
        print(_)  # noqa: T201
        os.environ["NO_PROXY"] = "*"
