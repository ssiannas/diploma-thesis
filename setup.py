"""
Setup script for PCML package.

This setup.py is mainly for backward compatibility and editable installs.
Modern packaging configuration is in pyproject.toml.
"""

from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="pcml",
        packages=find_packages(exclude=["tests", "tests.*", "scripts", "docs"]),
        include_package_data=True,
    )
