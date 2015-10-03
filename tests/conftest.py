from __future__ import print_function
import numpy as np
import pytest


def pytest_runtest_setup(item):
    seed = np.random.randint(1000)
    print("Seed used in np.random.seed(): %d" % seed)
    np.random.seed(seed)


def pytest_addoption(parser):
    parser.addoption("--block", action="store", default=True,
                     help="Should plotting block or not.")


@pytest.fixture
def block(request):
    try:
        return request.config.getoption("--block") not in "False,false,no,0".split(",")
    except ValueError:
        return True
