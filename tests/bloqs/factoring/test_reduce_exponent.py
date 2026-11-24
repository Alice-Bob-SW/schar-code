import pytest
import sympy as sp
from qualtran import QUInt
from qualtran.resource_counting import QubitCount, get_bloq_callee_counts, get_cost_value
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.testing import assert_equivalent_bloq_counts, assert_valid_bloq_decomposition

from scharcode.bloqs.factoring._common import l_p, p_i
from scharcode.bloqs.factoring.reduce_exponent import (
    ReduceExponent,
    ReduceExponentAdjoint,
    ReduceStep,
)
from scharcode.bloqs.operations.multi_target_cnot import CtrlXorK, K
from scharcode.bloqs.operations.qc_adder import QCAdder
from scharcode.bloqs.operations.qq_adder import QQAdder

PRIMES = [7, 11, 13, 17, 23]
LOG_N_E = [1, 2, 3]
GENERALIZERS = [ignore_split_join, ignore_alloc_free]


def _step(p: int) -> ReduceStep:
    return ReduceStep(QUInt(p.bit_length() + 1), p)


def _reduce(p: int, log_n_e: int) -> ReduceExponent:
    return ReduceExponent(QUInt(p.bit_length() + log_n_e), p)


# ---- ReduceStep ------------------------------------------------------------------------------


@pytest.mark.parametrize("p", PRIMES)
def test_reduce_step_classical_action(p):
    """Lemma `efficient_modular_reduction`: |x> -> |x mod b>|1_{x<b}>, b = p_i - 1."""
    bloq = _step(p)
    cbloq = bloq.decompose_bloq()
    b = p - 1
    for x in range(2 * b):
        expected = (x % b, int(x < b))
        assert bloq.call_classically(window=x) == expected
        assert cbloq.call_classically(window=x) == expected


@pytest.mark.parametrize("p", PRIMES)
def test_reduce_step_adjoint_round_trip(p):
    bloq = _step(p)
    adjoint = bloq.adjoint().decompose_bloq()
    for x in range(2 * (p - 1)):
        window, quotient_qubit = bloq.call_classically(window=x)
        assert adjoint.call_classically(window=window, quotient_qubit=quotient_qubit) == (x,)


@pytest.mark.parametrize("p", PRIMES)
def test_reduce_step_decomposition(p):
    bloq = _step(p)
    assert_valid_bloq_decomposition(bloq)
    assert_equivalent_bloq_counts(bloq, generalizer=GENERALIZERS)


def test_reduce_step_symbolic_call_graph():
    bloq = ReduceStep(QUInt(l_p + 1), p_i)
    assert get_bloq_callee_counts(bloq) == [
        (QCAdder(l_p + 1, p_i - 1).adjoint(), 1),
        (CtrlXorK(QUInt(l_p), K), 2),
        (QQAdder(l_p), 1),
    ]


@pytest.mark.parametrize("p", PRIMES)
def test_reduce_step_qubit_count(p):
    """u, y_t, the auxiliary register and the l_p - 1 ancillas of the Q-Q Adder: 3 l_p."""
    assert get_cost_value(_step(p), QubitCount()) == 3 * p.bit_length()


def test_reduce_step_rejects_input_out_of_range():
    with pytest.raises(ValueError, match="x < 2"):
        _step(11).call_classically(window=20)


@pytest.mark.parametrize(
    "bitsize, p, match",
    [(4, 11, "l_p \\+ 1"), (6, 11, "l_p \\+ 1"), (5, 15, "odd prime"), (3, 3, "at least 3")],
)
def test_reduce_step_validation(bitsize, p, match):
    with pytest.raises(ValueError, match=match):
        ReduceStep(QUInt(bitsize), p)


# ---- ReduceExponent and ReduceExponent† -------------------------------------------------------


@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("log_n_e", LOG_N_E)
def test_reduce_exponent_classical_action(p, log_n_e):
    """|S_i> -> |S_i mod (p_i - 1)>|ā>, for every S_i < 2^log(n_e) (p_i - 1)."""
    bloq = _reduce(p, log_n_e)
    cbloq = bloq.decompose_bloq()
    b = p - 1
    for s in range(2**log_n_e * b):
        a = s // b
        expected = (s % b, (2**log_n_e - 1) - a)
        assert bloq.call_classically(s=s) == expected
        assert cbloq.call_classically(s=s) == expected


@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("log_n_e", LOG_N_E)
def test_reduce_exponent_adjoint_restores_s(p, log_n_e):
    bloq = _reduce(p, log_n_e)
    adjoint = bloq.adjoint()
    assert isinstance(adjoint, ReduceExponentAdjoint)
    assert adjoint.adjoint() == bloq
    cadjoint = adjoint.decompose_bloq()
    for s in range(2**log_n_e * (p - 1)):
        reduced, quotient = bloq.call_classically(s=s)
        assert adjoint.call_classically(s=reduced, quotient=quotient) == (s,)
        assert cadjoint.call_classically(s=reduced, quotient=quotient) == (s,)


@pytest.mark.parametrize("p", [11, 17])
@pytest.mark.parametrize("log_n_e", LOG_N_E)
def test_reduce_exponent_decomposition(p, log_n_e):
    """The call graph is the decomposition, log(n_e) steps on (l_p + 1)-qubit windows."""
    for bloq in (_reduce(p, log_n_e), _reduce(p, log_n_e).adjoint()):
        assert_valid_bloq_decomposition(bloq)
        assert_equivalent_bloq_counts(bloq, generalizer=GENERALIZERS)


def test_reduce_exponent_symbolic_call_graph():
    """Table `dlog_modular_reduction_operation_count`, for one prime."""
    log_n_e = sp.Symbol("log_n_e", positive=True, integer=True)
    bloq = ReduceExponent(QUInt(l_p + log_n_e), p_i)
    step = ReduceStep(QUInt(l_p + 1), p_i)
    assert bloq.log_n_e == log_n_e
    assert get_bloq_callee_counts(bloq) == [(step, log_n_e)]
    assert get_bloq_callee_counts(bloq.adjoint()) == [(step.adjoint(), log_n_e)]


@pytest.mark.parametrize("p", PRIMES)
@pytest.mark.parametrize("log_n_e", LOG_N_E)
def test_reduce_exponent_qubit_count(p, log_n_e):
    bloq = _reduce(p, log_n_e)
    expected = 3 * p.bit_length() + log_n_e - 1
    assert get_cost_value(bloq, QubitCount()) == expected
    assert get_cost_value(bloq.decompose_bloq(), QubitCount()) == expected
    assert get_cost_value(bloq.adjoint(), QubitCount()) == expected


def test_reduce_exponent_rejects_s_out_of_range():
    with pytest.raises(ValueError, match="S_i <"):
        _reduce(11, 2).call_classically(s=40)


def test_reduce_exponent_needs_a_quotient_register():
    """no reduction step when s has only l_p qubits."""
    with pytest.raises(ValueError, match="more than l_p"):
        ReduceExponent(QUInt(4), 11)
