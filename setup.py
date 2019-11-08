# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages


with open("reqs/base-requirements.txt") as f:
    requirements = f.read().splitlines()


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
    install_requires=requirements,
    package_data={"lifelines": ["../README.md", "../README.txt", "../LICENSE", "../MANIFEST.in", "datasets/*"]},
)
