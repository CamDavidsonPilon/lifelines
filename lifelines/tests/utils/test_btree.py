# -*- coding: utf-8 -*-
from __future__ import print_function

import pytest
import numpy as np

from lifelines.utils.btree import _BTree as BTree


def test_btree():
    t = BTree(np.arange(10))
    for i in range(10):
        assert t.rank(i) == (0, 0)

    assert len(t) == 0
    t.insert(5)
    t.insert(6)
    t.insert(6)
    t.insert(0)
    t.insert(9)
    assert len(t) == 5

    assert t.rank(0) == (0, 1)
    assert t.rank(0.5) == (1, 0)
    assert t.rank(4.5) == (1, 0)
    assert t.rank(5) == (1, 1)
    assert t.rank(5.5) == (2, 0)
    assert t.rank(6) == (2, 2)
    assert t.rank(6.5) == (4, 0)
    assert t.rank(8.5) == (4, 0)
    assert t.rank(9) == (4, 1)
    assert t.rank(9.5) == (5, 0)

    for i in range(1, 32):
        BTree(np.arange(i))

    with pytest.raises(ValueError):
        # This has to go last since it screws up the counts
        t.insert(5.5)
