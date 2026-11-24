import numpy as np
import pytest
import sympy as sp
from qualtran import QUInt
from qualtran.resource_counting import get_bloq_callee_counts
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.testing import assert_equivalent_bloq_counts, assert_valid_bloq_decomposition

from helpers import phased_run
from scharcode.bloqs.factoring._common import window_sizes
from scharcode.bloqs.factoring.immediate_mbu import ApproxModExp, ComputeExponent
from scharcode.bloqs.operations._utils import T
from scharcode.bloqs.operations.lookup import Lookup, lookup_generalizer
from scharcode.bloqs.operations.qq_adder import QQAdder
from scharcode.classical.rns import RnsParameters

GENERALIZERS = [ignore_split_join, ignore_alloc_free, lookup_generalizer]

# n_e = 5, w1 = 2: windows of 2, 2 and 1 bits from the least significant end, so table1 lists
# the 1-bit window first.
EXAMPLE_TABLE1 = ((5, 60), (0, 1, 3, 2), (1, 0, 2, 33))


def _example() -> ComputeExponent:
    return ComputeExponent(QUInt(5), QUInt(6), 2, EXAMPLE_TABLE1)


def _reference(bloq: ComputeExponent, e: int, s: int) -> int:
    """s + sum over windows of table1[window][address] mod 2^{|s|}, windows read from e."""
    sizes = window_sizes(bloq.dtype_e.bitsize, bloq.w1)  # least significant first
    total, shift = s, 0
    for j, size in enumerate(sizes):
        address = (e >> shift) & ((1 << size) - 1)
        total += bloq.table1[len(sizes) - 1 - j][address]
        shift += size
    return total % 2**bloq.dtype_s.bitsize


def test_window_metadata():
    bloq = _example()
    assert bloq.num_windows1 == 3
    assert bloq.last_w1 == 1
    assert bloq.signature.get_left("e").dtype == QUInt(5)
    assert bloq.signature.get_left("s").dtype == QUInt(6)


def test_on_classical_vals_matches_reference():
    bloq = _example()
    for e in range(2**5):
        for s in range(2**6):
            assert bloq.on_classical_vals(e=e, s=s) == {"e": e, "s": _reference(bloq, e, s)}


def test_decomposition_classical_action(rng):
    bloq = _example()
    assert_valid_bloq_decomposition(bloq)
    cbloq = bloq.decompose_bloq().flatten()
    for e in range(2**5):
        for s in (0, 1, 31, 63):
            out, phase = phased_run(cbloq, {"e": e, "s": s}, rng)
            assert out == {"e": e, "s": _reference(bloq, e, s)}
            assert phase == 1


def test_bloq_counts():
    assert_equivalent_bloq_counts(_example(), generalizer=GENERALIZERS)


def test_symbolic_call_graph():
    n_e, n_s, w1 = sp.symbols("n_e n_s w1", positive=True, integer=True)
    counts = dict(get_bloq_callee_counts(ComputeExponent(QUInt(n_e), QUInt(n_s), w1, T)))
    windows = sp.ceiling(n_e / w1)
    # Symbolic sizes assume full windows, so the last Lookup has the same key as the others.
    expected = {QQAdder(n_s): windows, Lookup(T, w1, n_s): windows}
    expected[Lookup(T, w1, n_s).adjoint()] = windows
    assert set(counts) == set(expected)
    for key, count in expected.items():
        assert sp.simplify(counts[key] - count) == 0, key


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(N=1022117, g=5, n_e=6, ell=9, kappa=6, w1=2, w3a=3, w3b=3, w4=3),
        dict(N=1000036000099, g=7, n_e=8, ell=10, kappa=8, w1=3, w3a=3, w3b=4, w4=3),
    ],
    ids=["full-windows", "partial-windows"],
)
def test_t1_rows_produce_s_i(kwargs):
    """With the tables of `from_parameters`, row i turns S_{i-1} into S_i.

    S_i = sum_k e_k D_{p_i,k}, without reduction modulo p_i - 1, and row |P| erases
    S_{|P|-1}, so s returns to 0.
    """
    kwargs = dict(kwargs)
    params = RnsParameters.search(kwargs.pop("N"), kwargs.pop("g"), **kwargs)
    table1 = ApproxModExp.from_parameters(params).table1
    rows = [
        ComputeExponent(QUInt(params.n_e), QUInt(params.s_bitsize), params.w1, t) for t in table1
    ]
    for e in range(2**params.n_e):
        bits = np.array([e >> k & 1 for k in range(params.n_e)])
        s = 0
        for i, row in enumerate(rows):
            s = row.on_classical_vals(e=e, s=s)["s"]
            expected = int(params.dlogs[i] @ bits) if i < params.num_primes else 0
            assert s == expected, (e, i)
