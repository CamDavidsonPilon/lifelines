# -*- coding: utf-8 -*-
import os

from setuptools import setup


def filepath(fname):
    return os.path.join(os.path.dirname(__file__), fname)


exec(compile(open("lifelines/version.py").read(), "lifelines/version.py", "exec"))

with open("README.md") as f:
    long_description = f.read()

setup(
    name="lifelines",
    version=__version__,
    author="Cameron Davidson-Pilon, Jonas Kalderstam",
    author_email="cam.davidson.pilon@gmail.com",
    description="Survival analysis in Python, including Kaplan Meier, Nelson Aalen and regression",
    license="MIT",
    keywords="survival analysis statistics data analysis",
    url="https://github.com/CamDavidsonPilon/lifelines",
    packages=["lifelines", "lifelines.datasets", "lifelines.fitters", "lifelines.utils"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=["numpy", "scipy>=1.0", "pandas>=0.18", "matplotlib>=2.0"],
    package_data={
        "lifelines": ["../README.md", "../README.txt", "../LICENSE", "../MANIFEST.in", "datasets/*"]
    },
)
