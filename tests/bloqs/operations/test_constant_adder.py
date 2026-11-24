import pytest
import qualtran.testing as qlt_testing
from helpers import left_parameter_ranges
from qualtran.resource_counting.generalizers import ignore_alloc_free

from scharcode.bloqs.operations._classical_ctrl import FreeClassVal, InitClassVal
from scharcode.bloqs.operations.qc_adder import QCAdder, QCAdderUnwrapped

N_RANGE = range(4, 6)  # QCAdder needs n > 3; n = 6 makes the exhaustive sweeps take minutes


def _ignore_classical_loads(bloq):
    # InitClassVal / FreeClassVal only load the classical constant i: no quantum cost.
    if isinstance(bloq, (InitClassVal, FreeClassVal)):
        return None
    return bloq


@pytest.mark.parametrize("n", N_RANGE)
def test_classical_action(n):
    for i in range(2**n):
        qlt_testing.assert_consistent_classical_action(QCAdder(n, i), t=range(2**n))


def test_output_values():
    n = 5
    for i in range(2**n):
        for t in range(2**n):
            assert QCAdder(n, i).call_classically(t=t) == ((t + i) % 2**n,)


@pytest.mark.parametrize("n", N_RANGE)
def test_unwrapped_classical_action(n):
    bloq = QCAdderUnwrapped(n)
    qlt_testing.assert_consistent_classical_action(bloq, **left_parameter_ranges(bloq))


@pytest.mark.parametrize("n", N_RANGE)
def test_adjoint_subtracts(n):
    for i in (0, 1, 2**n - 1, 2 ** (n - 1) + 3):
        adj = QCAdder(n, i).adjoint()
        for t in range(2**n):
            assert adj.call_classically(t=t) == ((t - i) % 2**n,)


@pytest.mark.parametrize("n", N_RANGE)
def test_bloq_counts(n):
    generalizers = [ignore_alloc_free, _ignore_classical_loads]
    qlt_testing.assert_equivalent_bloq_counts(QCAdder(n, 5), generalizers)
    qlt_testing.assert_equivalent_bloq_counts(QCAdderUnwrapped(n), generalizers)


def test_symbolic_call_graph():
    import sympy as sp

    n, i = sp.symbols("n i", integer=True, positive=True)
    QCAdder(n, i).call_graph()
    QCAdderUnwrapped(n).call_graph()
