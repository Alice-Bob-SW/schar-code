import random

import numpy as np
import pytest
import qualtran.testing as qlt_testing
import sympy as sp
from qualtran import QUInt
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join

from scharcode.bloqs.operations.lookup import (
    Lookup,
    LookupUncompute,
    LookupUnwrapped,
    lookup_generalizer,
)
from scharcode.bloqs.operations.multi_target_cnot import generalize_CtrlXorK


def test_lookup_unwrapped_recomputed_table():
    bloq = LookupUnwrapped((1, 2, 4, 8), 2, 4)
    assert bloq._recomputed_table == [1, 3, 5, 15]


def test_lookup_wrapper_caches_unwrapped_bloq():
    bloq = Lookup((0, 1, 2, 3), 2, 2)
    assert bloq.lookup_unwrapped == LookupUnwrapped((0, 1, 2, 3), 2, 2)


@pytest.mark.parametrize("T,w,n", [((0, 1, 2, 3), 2, 2), ((0, 1, 3, 2), 2, 3)])
def test_lookup_decomposition(T, w, n):
    qlt_testing.assert_valid_bloq_decomposition(Lookup(T, w, n))


@pytest.mark.parametrize("w", range(2, 7))
def test_lookup_call_graph(w):
    bloq = LookupUnwrapped(tuple(range(2**w)), w, w)
    qlt_testing.assert_equivalent_bloq_counts(
        bloq, [ignore_split_join, ignore_alloc_free, generalize_CtrlXorK]
    )


def test_lookup_symbolic_call_graph():
    T, w, n = sp.symbols("T w n")
    Lookup(T, w, n).call_graph()


def test_lookup_classical_action():
    # `target` is RIGHT-only: the Lookup allocates it, so only `selection` is an input.
    bloq = Lookup(np.arange(16), 4, 4)
    qlt_testing.assert_consistent_classical_action(bloq, selection=range(2**4))


def test_lookup_output_values():
    table = (5, 0, 7, 2, 3, 6, 1, 4)
    bloq = Lookup(table, 3, 3)
    for x in range(8):
        assert bloq.call_classically(selection=x) == (x, table[x])


@pytest.mark.parametrize(
    "bad_kwargs", [{"T": (0, 1, 2), "w": 2, "n": 2}, {"T": (0, 4), "w": 1, "n": 2}]
)
def test_lookup_unwrapped_rejects_bad_tables(bad_kwargs):
    with pytest.raises(ValueError):
        LookupUnwrapped(**bad_kwargs)


def test_lookup_adjoint_round_trip():
    bloq = Lookup((0, 1, 2, 3), 2, 2)
    adj = bloq.adjoint()
    assert isinstance(adj, LookupUncompute)
    assert adj.adjoint() == bloq
    assert adj.call_classically(selection=2, target=2) == (2,)


def test_lookup_generalizer():
    T = sp.Symbol("T")
    bloq = Lookup((0, 1, 2, 3), 2, 2)
    assert lookup_generalizer(bloq) == Lookup(T, 2, 2)
    assert lookup_generalizer(bloq.adjoint()) == Lookup(T, 2, 2).adjoint()


@pytest.mark.slow
@pytest.mark.parametrize("w,seed", [(w, seed) for w in range(2, 7) for seed in range(3)])
def test_lookup_unwrapped_complete_classical_action(w, seed):
    rng = random.Random(seed)
    T = list(range(2**w))
    rng.shuffle(T)
    bloq = LookupUnwrapped(T, w, w)
    qlt_testing.assert_consistent_classical_action(
        bloq, selection=[QUInt(w).to_bits(i) for i in range(2**w)], target=[0]
    )
