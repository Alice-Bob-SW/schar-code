import numpy as np
import pytest
import qualtran.testing as qlt_testing
import sympy as sp
from helpers import left_parameter_ranges
from qualtran import BloqBuilder, QBit, QUInt, Register
from qualtran.bloqs.basic_gates import CNOT
from qualtran.resource_counting import get_bloq_call_graph

from scharcode.bloqs.operations.multi_target_cnot import (
    CtrlXorK,
    K,
    MultiTargetCNOT,
    generalize_CtrlXorK,
)

# ---- MultiTargetCNOT ----


@pytest.mark.parametrize("n", range(1, 5))
def test_mcx_classical_action(n):
    bloq = MultiTargetCNOT(n)
    for ctrl in (0, 1):
        for targets in left_parameter_ranges(bloq)["targets"]:
            ctrl_out, targets_out = bloq.call_classically(ctrl=ctrl, targets=targets)
            assert ctrl_out == ctrl
            np.testing.assert_array_equal(targets_out, targets ^ ctrl)


def _cnot_ladder(n: int):
    bb = BloqBuilder()
    ctrl = bb.add_register("ctrl", 1)
    targets = bb.add_register(Register("targets", QBit(), shape=(n,)))
    for j in range(n):
        ctrl, targets[j] = bb.add(CNOT(), ctrl=ctrl, target=targets[j])
    return bb.finalize(ctrl=ctrl, targets=targets)


@pytest.mark.parametrize("n", range(1, 5))
def test_mcx_tensor_equals_cnot_ladder(n):
    np.testing.assert_allclose(
        MultiTargetCNOT(n).tensor_contract(), _cnot_ladder(n).tensor_contract(), atol=1e-12
    )


def test_mcx_adjoint_is_self():
    assert MultiTargetCNOT(3).adjoint() == MultiTargetCNOT(3)


@pytest.mark.parametrize("n", [1, 4, sp.Symbol("n", integer=True, positive=True)])
def test_mcx_counts_one_cnot(n):
    _, sigma = get_bloq_call_graph(MultiTargetCNOT(n))
    assert sigma == {CNOT(): 1}


# ---- CtrlXorK ----


@pytest.mark.parametrize("n,k", [(3, 0), (3, 5), (4, 15), (4, 6)])
def test_ctrl_xork_classical_action(n, k):
    bloq = CtrlXorK(QUInt(n), k)
    qlt_testing.assert_consistent_classical_action(bloq, ctrl=range(2), x=range(2**n))
    for ctrl in (0, 1):
        for x in range(2**n):
            assert bloq.call_classically(ctrl=ctrl, x=x) == (ctrl, x ^ (k * ctrl))


def test_ctrl_xork_adjoint_is_self():
    assert CtrlXorK(QUInt(4), 6).adjoint() == CtrlXorK(QUInt(4), 6)


@pytest.mark.parametrize("n,k", [(3, 5), (4, 15), (4, 6)])
def test_ctrl_xork_bloq_counts(n, k):
    qlt_testing.assert_equivalent_bloq_counts(CtrlXorK(QUInt(n), k))


def test_ctrl_xork_symbolic_call_graph():
    n = sp.Symbol("n", integer=True, positive=True)
    _, sigma = get_bloq_call_graph(CtrlXorK(QUInt(n), K))
    assert sigma == {CNOT(): 1}


def test_generalize_ctrl_xork():
    assert generalize_CtrlXorK(CtrlXorK(QUInt(4), 6)) == CtrlXorK(QUInt(4), K)
    assert generalize_CtrlXorK(CNOT()) == CNOT()
