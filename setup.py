import setuptools
from setuptools import setup

NAME = "tagger"
print('packages found', setuptools.find_packages(include=[NAME, NAME + ".*"]))

setup(
    name=NAME,
    packages=setuptools.find_packages(),
    description='Named Entity Recognition for meddoprof task',
    setup_requires=['pytest_runner'],
    python_requires='>=3.7.9'
)
