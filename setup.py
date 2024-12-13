from setuptools import setup, find_packages

setup(
    name="orion",
    version="0.1.0",
    description="A Python library for deep learning",
    author="xoxkom",
    author_email="c2085579991@gmail.com",
    packages=find_packages(include=["orion", "orion.*"]),
    python_requires=">=3.9",
)
