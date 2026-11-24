import pytest
import qualtran.testing as qlt_testing
import sympy as sp
from qualtran import QUInt
from qualtran.resource_counting.generalizers import ignore_alloc_free

from scharcode.bloqs.operations.ctrl_add_constant import ControlledConstantAdd


def _example(add: bool = True) -> ControlledConstantAdd:
    return ControlledConstantAdd(dtype_target=QUInt(4), val=3, add=add)


def test_signature():
    bloq = _example()
    assert bloq.signature.get_left("target").dtype == QUInt(4)
    assert bloq.signature.get_left("ctrl").dtype.num_qubits == 1


@pytest.mark.parametrize("add", [True, False])
def test_classical_action(add: bool):
    bloq = _example(add=add)
    qlt_testing.assert_consistent_classical_action(
        bloq, target=range(2**bloq.dtype_target.bitsize), ctrl=range(2)
    )


@pytest.mark.parametrize(
    "add,ctrl,target_in,expected_target",
    [
        (True, 1, 14, 1),
        (False, 1, 1, 14),
        (True, 0, 14, 14),
        (False, 0, 1, 1),
    ],
)
def test_wraparound(add: bool, ctrl: int, target_in: int, expected_target: int):
    assert _example(add=add).call_classically(ctrl=ctrl, target=target_in) == (
        ctrl,
        expected_target,
    )


@pytest.mark.parametrize("add", [True, False])
def test_bloq_counts(add: bool):
    qlt_testing.assert_equivalent_bloq_counts(_example(add=add), ignore_alloc_free)


@pytest.mark.parametrize("add", [True, False])
def test_symbolic_call_graph(add: bool):
    n = sp.Symbol("n", integer=True, positive=True)
    val = sp.Symbol("val", integer=True)
    ControlledConstantAdd(dtype_target=QUInt(n), val=val, add=add).call_graph()
