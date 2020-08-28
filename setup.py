import os
from setuptools import setup, find_packages


setup(
    name="autogradcupy",
    version="0.1",
    author="miguel myers",
    description=("autograd wrapper for CuPy"),
    keywords="machine learning, deep learning",
    packages=find_packages(),
    package_data={"": ["README.md", "LICENSE"]},
)
