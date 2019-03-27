# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages


def filepath(fname):
    return os.path.join(os.path.dirname(__file__), fname)


exec(compile(open("lifelines/version.py").read(), "lifelines/version.py", "exec"))

with open("README.md") as f:
    long_description = f.read()

setup(
    name="lifelines",
    version=__version__,
    author="Cameron Davidson-Pilon",
    author_email="cam.davidson.pilon@gmail.com",
    description="Survival analysis in Python, including Kaplan Meier, Nelson Aalen and regression",
    license="MIT",
    keywords="survival analysis statistics data analysis",
    url="https://github.com/CamDavidsonPilon/lifelines",
    packages=find_packages(),
    python_requires=">=3.5",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=[
        "numpy>=1.6.0",
        "scipy>=1.0",
        "pandas>=0.23.0",
        "matplotlib>=3.0",
        "bottleneck>=1.0",
        "autograd>=1.2",
    ],
    package_data={"lifelines": ["../README.md", "../README.txt", "../LICENSE", "../MANIFEST.in", "datasets/*"]},
)
