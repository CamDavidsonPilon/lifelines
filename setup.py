import os
import sys

from numpy.distutils.core import setup, Extension


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Fortran extensions for statistics module
ext_fstat = Extension(name="lifelines._statistics",
                      sources=["lifelines/_statistics.f90"])


setup(
    name="lifelines",
    version="0.3.2.3",
    author="Cameron Davidson-Pilon",
    author_email="cam.davidson.pilon@gmail.com",
    description="Survival analysis in Python, including Kaplan Meier, Nelson Aalen and regression",
    license="MIT",
    keywords="survival analysis statistics data analysis",
    url="https://github.com/CamDavidsonPilon/lifelines",
    packages=['lifelines'],
    long_description=read('README.txt'),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas>=0.12",
    ],
    package_data={
        "lifelines": [
            "*.f90",
            "../README.md",
            "../README.txt",
            "../LICENSE",
            "../MANIFEST.in",
            "../*.ipynb",
            "../datasets/*",
        ]
    },
    ext_modules=[ext_fstat]
)
