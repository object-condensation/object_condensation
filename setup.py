from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="object_condensation",
    version="0.1",
    description="Your package description",
    author="Kilian Lieret, Philipp Zehetner",
    author_email="kilian.lieret@posteo.de, philipp.zehetner@cern.ch",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "tf": open("requirements_tf.txt").read().splitlines(),
        "pt": open("requirements_pt.txt").read().splitlines(),
    },
)

