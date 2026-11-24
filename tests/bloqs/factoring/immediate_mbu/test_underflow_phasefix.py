import itertools

import sympy as sp
from qualtran import QUInt
from qualtran.bloqs.basic_gates import ZGate
from qualtran.resource_counting import get_bloq_callee_counts
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.testing import assert_equivalent_bloq_counts, assert_valid_bloq_decomposition

from helpers import phased_run
from scharcode.bloqs.factoring.immediate_mbu import UnderflowPhasefix
from scharcode.bloqs.operations._utils import T
from scharcode.bloqs.operations.half_compare import HalfCompare
from scharcode.bloqs.operations.lookup import Lookup, lookup_generalizer

GENERALIZERS = [ignore_split_join, ignore_alloc_free, lookup_generalizer]
TABLE = (1, 3, 4, 7)


def _example() -> UnderflowPhasefix:
    return UnderflowPhasefix(dtype_reg=QUInt(3), w=2, table=TABLE)


def test_metadata():
    bloq = _example()
    assert bloq.signature.get_left("address").dtype == QUInt(2)
    assert bloq.signature.get_left("reg").dtype == QUInt(3)


def test_basis_state_phase_is_y_ge_t():
    """phase -1 exactly when reg >= table[address], including at the boundary."""
    bloq = _example()
    for address, t in enumerate(TABLE):
        assert bloq.basis_state_phase(address=address, reg=t) == -1
        assert bloq.basis_state_phase(address=address, reg=t - 1) == 1


def test_decomposition_phase_and_values(rng):
    bloq = _example()
    assert_valid_bloq_decomposition(bloq)
    cbloq = bloq.decompose_bloq().flatten()
    for address, reg in itertools.product(range(4), range(8)):
        for _ in range(3):  # random M_X outcomes
            out, phase = phased_run(cbloq, {"address": address, "reg": reg}, rng)
            assert out == {"address": address, "reg": reg}
            assert phase == bloq.basis_state_phase(address=address, reg=reg), (address, reg)


def test_bloq_counts():
    assert_equivalent_bloq_counts(_example(), generalizer=GENERALIZERS)


def test_symbolic_call_graph():
    n, w = sp.symbols("n w", positive=True, integer=True)
    counts = dict(get_bloq_callee_counts(UnderflowPhasefix(QUInt(n), w, T)))
    compare = HalfCompare(QUInt(n), QUInt(n))
    assert counts == {
        Lookup(T, w, n): 1,
        Lookup(T, w, n).adjoint(): 1,
        compare: 1,
        compare.adjoint(): 1,
        ZGate(): 1,
    }
