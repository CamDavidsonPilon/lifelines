# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages

exec(compile(open("lifelines/version.py").read(), "lifelines/version.py", "exec"))

with open("reqs/base-requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

with open("README.md") as f:
    LONG_DESCRIPTION, LONG_DESC_TYPE = f.read(), "text/markdown"

NAME = "lifelines"
AUTHOR_NAME, AUTHOR_EMAIL = "Cameron Davidson-Pilon", "cam.davidson.pilon@gmail.com"
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering",
]
LICENSE = "MIT"
PACKAGE_DATA = {"lifelines": ["datasets/*"]}
DESCRIPTION = "Survival analysis in Python, including Kaplan Meier, Nelson Aalen and regression"
URL = "https://github.com/CamDavidsonPilon/lifelines"
PYTHON_REQ = ">=3.7, !=3.11.*"

setup(
    name=NAME,
    version=__version__,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    python_requires=PYTHON_REQ,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    classifiers=CLASSIFIERS,
    install_requires=REQUIREMENTS,
    packages=find_packages(exclude=["*.tests", "*.tests.*", "*benchmarks*"]),
    package_data=PACKAGE_DATA,
    include_package_data=False,
)
