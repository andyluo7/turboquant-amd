"""Setup script for TurboQuant."""
from setuptools import setup, find_packages

setup(
    name="turboquant",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "triton>=3.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "scipy>=1.10"],
        "bench": ["scipy>=1.10", "matplotlib>=3.7", "tabulate>=0.9"],
    },
)
