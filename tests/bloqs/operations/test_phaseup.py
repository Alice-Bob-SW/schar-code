"""Phaseup is a cost-only placeholder: call graph and static costs only.

There is no phased simulation: the measured target bits and the table are not inputs, so the
phase correction it stands for cannot be checked here.
"""

import pytest
import sympy as sp
from qualtran import DecomposeNotImplementedError
from qualtran.bloqs.basic_gates import CZ, MeasureX
from qualtran.bloqs.mcmt import And
from qualtran.resource_counting import QubitCount, get_bloq_call_graph, get_cost_value

from scharcode.bloqs.operations.lookup import Lookup
from scharcode.bloqs.operations.phaseup import Phaseup

# w -> (And count, CZ count, qubit count), worked out by hand from Gidney 2025, A.3:
# h1 = floor(w/2), h2 = w - h1; And = 2^h1 + 2^h2 - w - 2; CZ = 2^h1 - 1; qubits = 2^h1 + 2^h2 - 2.
EXPECTED = {
    2: (0, 1, 2),
    3: (1, 1, 4),
    4: (2, 3, 6),
    5: (5, 3, 10),
    6: (8, 7, 14),
    8: (22, 15, 30),
}


@pytest.mark.parametrize("w", sorted(EXPECTED))
def test_call_graph(w):
    n_and, n_cz, _ = EXPECTED[w]
    _, sigma = get_bloq_call_graph(Phaseup(w))
    assert sigma.get(And(), 0) == n_and
    assert sigma.get(And().adjoint(), 0) == n_and
    assert sigma.get(CZ(), 0) == n_cz


@pytest.mark.parametrize("w", sorted(EXPECTED))
def test_static_qubit_count(w):
    assert get_cost_value(Phaseup(w), QubitCount()) == EXPECTED[w][2]


@pytest.mark.parametrize("w", sorted(EXPECTED))
def test_matches_lookup_uncompute_without_measurements(w):
    # Phaseup is LookupUncompute with the target's M_X moved earlier.
    _, phaseup = get_bloq_call_graph(Phaseup(w))
    _, uncompute = get_bloq_call_graph(Lookup(tuple(range(2**w)), w, w).adjoint())
    assert uncompute.pop(MeasureX()) == w
    assert {k: v for k, v in phaseup.items() if v} == {k: v for k, v in uncompute.items() if v}


def test_symbolic_costs():
    w = sp.Symbol("w", integer=True, positive=True)
    _, sigma = get_bloq_call_graph(Phaseup(w))
    h1 = sp.floor(w / 2)
    h2 = w - h1
    assert sp.simplify(sigma[And()] - (2**h1 + 2**h2 - w - 2)) == 0
    assert sp.simplify(sigma[CZ()] - (2**h1 - 1)) == 0
    assert sp.simplify(get_cost_value(Phaseup(w), QubitCount()) - (2**h1 + 2**h2 - 2)) == 0


def test_is_a_placeholder():
    # No decomposition, and the classical action leaves the selection unchanged.
    bloq = Phaseup(3)
    with pytest.raises(DecomposeNotImplementedError):
        bloq.decompose_bloq()
    assert all(bloq.call_classically(selection=x) == (x,) for x in range(8))
