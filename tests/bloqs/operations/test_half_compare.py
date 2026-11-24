import pytest
import qualtran.testing as qlt_testing
import sympy as sp
from helpers import assert_consistent_symbolic_qubit_count
from qualtran import QUInt

from scharcode.bloqs.operations.half_compare import HalfCompare, HalfCompareUnwrapped


def _compare_target(bloq: HalfCompare, i: int, t: int) -> int:
    _, _, target, _ = bloq.call_classically(i=i, t=t)
    return int(target)


@pytest.mark.parametrize("n", range(2, 6))
def test_target_is_t_geq_i_plus_one(n):
    # t one bit wider than i: t - i cannot overflow, so the flag is exact.
    bloq = HalfCompare(QUInt(n), QUInt(n + 1))
    for i in range(2**n):
        for t in range(2 ** (n + 1)):
            assert _compare_target(bloq, i, t) == int(t >= i), f"{i=} {t=}"


@pytest.mark.parametrize("bitsize_i,bitsize_t", [(3, 3), (3, 4), (4, 5)])
def test_classical_action(bitsize_i, bitsize_t):
    # The closed-form on_classical_vals matches the decomposition on all four outputs.
    # i and t are NOT restored (that is the uncompute half's job).
    qlt_testing.assert_consistent_classical_action(
        HalfCompare(QUInt(bitsize_i), QUInt(bitsize_t)),
        i=range(2**bitsize_i),
        t=range(2**bitsize_t),
    )


@pytest.mark.parametrize(
    "bloq",
    [
        HalfCompare(QUInt(3), QUInt(3)),
        HalfCompare(QUInt(3), QUInt(4)),
        HalfCompareUnwrapped(3, 3),
        HalfCompareUnwrapped(3, 4),
    ],
    ids=["equal", "plus-one", "unwrapped-equal", "unwrapped-plus-one"],
)
def test_bloq_counts(bloq):
    qlt_testing.assert_equivalent_bloq_counts(bloq)


@pytest.mark.parametrize("bloq_cls", [HalfCompare, HalfCompareUnwrapped])
def test_symbolic_call_graph(bloq_cls):
    n = sp.Symbol("n", integer=True, positive=True)
    if bloq_cls is HalfCompare:
        bloq_cls(QUInt(n), QUInt(n + 1)).call_graph()
    else:
        bloq_cls(n, n + 1).call_graph()


def test_symbolic_qubit_count():
    n = sp.Symbol("n", integer=True, positive=True)
    assert_consistent_symbolic_qubit_count(
        HalfCompare,
        arg_specs={
            "dtype_i": (QUInt, {n: range(3, 7)}),
            "dtype_t": (lambda n: QUInt(n + 1), {n: range(3, 7)}),
        },
    )
