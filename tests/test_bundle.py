import numpy as np

from processor.bundle import assign_indices


def test_assign_indices_empty():
    assert assign_indices([], []) == []


def test_assign_indices_continuous_then_unit(continuous_source):
    c0 = continuous_source(np.zeros(4, dtype=np.float32), id="c0")
    c1 = continuous_source(np.zeros(4, dtype=np.float32), id="c1")
    # assign_indices only pairs sources with indices, so opaque sentinels suffice.
    u0, u1 = object(), object()
    assert assign_indices([c0, c1], [u0, u1]) == [
        (0, c0),
        (1, c1),
        (2, u0),
        (3, u1),
    ]


def test_assign_indices_only_continuous(continuous_source):
    c0 = continuous_source(np.zeros(1, dtype=np.float32))
    assert assign_indices([c0], []) == [(0, c0)]


def test_assign_indices_only_units():
    u0 = object()
    assert assign_indices([], [u0]) == [(0, u0)]
