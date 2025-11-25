"""
Setup file for the Negotiation Platform
"""
from setuptools import setup, find_packages

setup(
    name="negotiation-platform",
    version="1.0.0",
    description="Compare LLMs in corporate negotiation situations under uncertainty.",
    author="Tom Ziegler",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "pydantic>=1.8.0",
        "pyyaml>=6.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "docs": [
            "sphinx>=4.0.0",
            "sphinx_rtd_theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
    },
)