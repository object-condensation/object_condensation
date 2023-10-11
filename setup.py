from pathlib import Path
from setuptools import find_packages, setup


with Path('requirements.txt').open() as f:
    requirements = f.read().splitlines()
with Path('requirements_tf.txt').open() as f:
    requirements_tf = f.read().splitlines()
with Path('requirements_pt.txt').open() as f:
    requirements_pt = f.read().splitlines()

setup(
    name="object_condensation",
    version="0.1",
    description="Your package description",
    author="Kilian Lieret, Philipp Zehetner",
    author_email="kilian.lieret@posteo.de, philipp.zehetner@cern.ch",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "tf": requirements_tf,
        "pt": requirements_pt,
    },
)

