import pytest
import qualtran.testing as qlt_testing
import sympy as sp
from helpers import assert_consistent_symbolic_qubit_count, left_parameter_ranges

from scharcode.bloqs.operations.qq_adder import QQAdder, QQAdderSub, QQAdderUnwrapped

N_RANGE = range(3, 6)  # n = 6 makes the exhaustive 4^n sweeps take minutes


@pytest.mark.parametrize("n", N_RANGE)
@pytest.mark.parametrize("bloq_cls", [QQAdder, QQAdderSub])
def test_classical_action(bloq_cls, n):
    bloq = bloq_cls(n)
    qlt_testing.assert_consistent_classical_action(bloq, **left_parameter_ranges(bloq))


@pytest.mark.parametrize("n", N_RANGE)
def test_unwrapped_classical_action(n):
    bloq = QQAdderUnwrapped(n)
    qlt_testing.assert_consistent_classical_action(bloq, **left_parameter_ranges(bloq))


@pytest.mark.parametrize(
    "bloq_cls,expected",
    [(QQAdder, lambda i, t, n: (t + i) % 2**n), (QQAdderSub, lambda i, t, n: (t - i) % 2**n)],
    ids=["add", "sub"],
)
def test_output_values(bloq_cls, expected):
    # Pins the direction of the subtraction (t - i, not i - t).
    n = 4
    for i in range(2**n):
        for t in range(2**n):
            assert bloq_cls(n).call_classically(i=i, t=t) == (i, expected(i, t, n))


@pytest.mark.parametrize("n", [3, 5])
def test_adjoint(n):
    assert QQAdder(n).adjoint() == QQAdderSub(n)
    assert QQAdderSub(n).adjoint() == QQAdder(n)


@pytest.mark.parametrize("n", N_RANGE)
def test_unwrapped_bloq_counts(n):
    # Only the unwrapped adder: the wrappers inline it with `add_from`, but their call graph
    # counts it as one bloq so that symbolic counting doesn't recurse into it.
    qlt_testing.assert_equivalent_bloq_counts(QQAdderUnwrapped(n))


@pytest.mark.parametrize("bloq_cls", [QQAdder, QQAdderSub, QQAdderUnwrapped])
def test_symbolic_call_graph(bloq_cls):
    n = sp.Symbol("n", integer=True, positive=True)
    bloq_cls(n).call_graph()


@pytest.mark.parametrize("bloq_cls", [QQAdder, QQAdderSub, QQAdderUnwrapped])
def test_symbolic_qubit_count(bloq_cls):
    assert_consistent_symbolic_qubit_count(bloq_cls, n=range(3, 10))
