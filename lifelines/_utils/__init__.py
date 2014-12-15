'''
Make sure that concordance index is available, even if the Fortran code
has not been compiled.
'''


try:
    from ._cindex import concordance_index
except ImportError:
    print("Using pure Python version of concordance index. \n\
You can speed this up 100x by compiling the Fortran code with:\n\
>>> python setup.py build_ext --inplace")
    from .cindex import concordance_index
