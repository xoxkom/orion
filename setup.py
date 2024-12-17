# -*- encoding:utf-8 -*-
# author: xoxkom
# version: 0.1.0
# date: 2024.12.17

import os

from setuptools import setup, find_packages

def get_project_version():
    version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "VERSION")
    with open(version_file, "r") as f:
        return f.read().strip()

__version__ = get_project_version()

setup(
    name="orion",
    version=__version__,
    description="A Python library for deep learning",
    author="xoxkom",
    author_email="c2085579991@gmail.com",
    packages=find_packages(include=["orion", "orion.*"]),
    python_requires=">=3.9",
)
