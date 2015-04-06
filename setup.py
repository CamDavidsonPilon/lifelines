import os
import sys

# bdist_wheel requires setuptools
import setuptools
# use setup from numpy to build fortran codes
from numpy.distutils.core import setup, Extension


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Fortran extensions for extra speed (optional)
ext_fstat = Extension(name="lifelines._utils._cindex",
                      sources=["lifelines/_utils/_cindex.f90"])
exts = [ext_fstat]

exec(compile(open('lifelines/version.py').read(),
                  'lifelines/version.py', 'exec'))

done = False
# First try to build with the extension. In case of failure, do without.
for ext_modules in [exts, []]:
    if done:
        break

    try:
        setup(
            name="lifelines",
            version=__version__,
            author="Cameron Davidson-Pilon, Jonas Kalderstam",
            author_email="cam.davidson.pilon@gmail.com",
            description="Survival analysis in Python, including Kaplan Meier, Nelson Aalen and regression",
            license="MIT",
            keywords="survival analysis statistics data analysis",
            url="https://github.com/CamDavidsonPilon/lifelines",
            packages=['lifelines',
                      'lifelines.datasets',
                      'lifelines._utils',
                      'lifelines.tests'],
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
                "pandas>=0.14",
            ],
            package_data={
                "lifelines": [
                    "*.f90",
                    "../README.md",
                    "../README.txt",
                    "../LICENSE",
                    "../MANIFEST.in",
                    "../*.ipynb",
                    "datasets/*",
                ]
            },
            ext_modules=ext_modules
        )
        done = True
    except:
        print("Failed to build. Trying again without native extensions.")
