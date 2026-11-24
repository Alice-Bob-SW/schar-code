import itertools

import pytest
import sympy as sp
from qualtran import QUInt
from qualtran.bloqs.basic_gates import MeasureX
from qualtran.resource_counting import get_bloq_callee_counts
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join


from helpers import assert_valid_decomposition, phased_run
from scharcode.bloqs.factoring._common import underflow_phasefix_generalizer, window_sizes
from scharcode.bloqs.factoring.immediate_mbu import AccumulateResidue, UnderflowPhasefix
from scharcode.bloqs.operations._utils import T
from scharcode.bloqs.operations.lookup import lookup_generalizer

GENERALIZERS = [
    ignore_split_join,
    ignore_alloc_free,
    lookup_generalizer,
    underflow_phasefix_generalizer,
]


def _example() -> AccumulateResidue:
    """4-bit r in two full windows of 2 bits, M = 13."""
    return AccumulateResidue(QUInt(4), QUInt(4), 2, 13, ((0, 1, 2, 3), (0, 2, 4, 6)))


def _non_dividing() -> AccumulateResidue:
    """5-bit r: windows of 1, 2 and 2 bits, most significant first."""
    return AccumulateResidue(QUInt(5), QUInt(4), 2, 13, ((0, 3), (0, 1, 5, 7), (0, 2, 6, 8)))


EXAMPLES = {"dividing": _example, "non-dividing": _non_dividing}


def _addresses(bloq: AccumulateResidue, r: int) -> list[int]:
    """Window addresses of r, most significant window first."""
    sizes = window_sizes(bloq.dtype_r.bitsize, bloq.w4)[::-1]
    out, remaining = [], bloq.dtype_r.bitsize
    for size in sizes:
        remaining -= size
        out.append((r >> remaining) & ((1 << size) - 1))
    return out


def _reference(bloq: AccumulateResidue, r: int, acc: int) -> int:
    """The arithmetic of Fig. 32, written out independently of the bloq."""
    kappa = bloq.dtype_acc.bitsize
    for j, address in enumerate(_addresses(bloq, r)):
        acc = (acc - bloq.table4[j][address]) % 2 ** (kappa + 1)
        if acc >> kappa:
            acc = (acc + bloq.trunc_modulus) % 2**kappa
    return acc


def test_metadata():
    bloq = _example()
    assert bloq.num_windows4 == 2
    assert bloq.trunc_modulus == 13
    assert bloq.signature.get_left("r").dtype == QUInt(4)
    assert bloq.signature.get_left("acc").dtype == QUInt(4)


def test_non_dividing_metadata():
    bloq = _non_dividing()
    assert bloq.num_windows4 == 3
    assert bloq.last_w4 == 1
    assert tuple(len(t) for t in bloq.table4) == (2, 4, 4)


@pytest.mark.parametrize(
    "r, expected",
    [
        (0b00000, [0, 0, 0]),
        (0b00001, [0, 0, 1]),
        (0b00100, [0, 1, 0]),
        (0b01111, [0, 3, 3]),
        (0b10000, [1, 0, 0]),
        (0b10101, [1, 1, 1]),
        (0b11111, [1, 3, 3]),
    ],
)
def test_non_dividing_window_addresses(r, expected):
    assert _addresses(_non_dividing(), r) == expected


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_on_classical_vals(name):
    """On acc < M the accumulator becomes (acc - sum_j V_j) mod M."""
    bloq = EXAMPLES[name]()
    m = bloq.trunc_modulus
    for r, acc in itertools.product(range(2**bloq.dtype_r.bitsize), range(2**4)):
        out = bloq.on_classical_vals(r=r, acc=acc)
        assert out == {"r": r, "acc": _reference(bloq, r, acc)}
        if acc < m:
            loaded = sum(bloq.table4[j][a] for j, a in enumerate(_addresses(bloq, r)))
            assert out["acc"] == (acc - loaded) % m


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_decomposition_phase_and_values(name, rng):
    """Phase 1 and the classical action, under random M_X outcomes, for every acc < M."""
    bloq = EXAMPLES[name]()
    assert_valid_decomposition(bloq)
    cbloq = bloq.decompose_bloq().flatten()
    for r, acc in itertools.product(range(2**bloq.dtype_r.bitsize), range(bloq.trunc_modulus)):
        for _ in range(3):
            out, phase = phased_run(cbloq, {"r": r, "acc": acc}, rng)
            assert out == {"r": r, "acc": _reference(bloq, r, acc)}
            assert phase == 1, (r, acc)


@pytest.mark.slow
def test_non_dividing_all_wraparound_branches():
    """Every combination of the wrap-around M_X outcomes, one per window.

    The wrap-around measurements are the `MeasureX` instances of the top-level decomposition
    (the others sit inside `LookupUncompute` leaves). Their outcomes are fixed by
    `fixed_random_vals`, keyed by the instance index of that decomposition.
    """
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    bloq = _non_dividing()
    cbloq = bloq.decompose_bloq()
    measurements = [b.i for b in cbloq.bloq_instances if b.bloq_is(MeasureX)]
    assert len(measurements) == bloq.num_windows4
    for outcomes in itertools.product((0, 1), repeat=len(measurements)):
        fixed = dict(zip(measurements, outcomes, strict=True))
        for r, acc in [(0, 0), (1, 0), (0b00100, 12), (0b01111, 2), (0b10101, 3), (31, 12)]:
            out, phase = do_phased_classical_simulation(
                cbloq, {"r": r, "acc": acc}, fixed_random_vals=fixed
            )
            assert {k: int(v) for k, v in out.items()} == {"r": r, "acc": _reference(bloq, r, acc)}
            assert phase == 1, (outcomes, r, acc)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_bloq_counts(name):
    """The call graph equals the decomposition, with UnderflowPhasefix counted 1/2."""
    bloq = EXAMPLES[name]()
    from_graph = dict(get_bloq_callee_counts(bloq, generalizer=GENERALIZERS))
    from_decomp = dict(get_bloq_callee_counts(bloq.decompose_bloq(), generalizer=GENERALIZERS))
    for key in from_graph.keys() | from_decomp.keys():
        factor = sp.Rational(1, 2) if isinstance(key, UnderflowPhasefix) else 1
        assert from_graph.get(key, 0) == factor * from_decomp.get(key, 0), key


def test_symbolic_call_graph():
    """The expected UnderflowPhasefix count is ceil(n/w4)/2, exact."""
    n, kappa, w4, modulus = sp.symbols("n kappa w4 N", positive=True, integer=True)
    bloq = AccumulateResidue(QUInt(n), QUInt(kappa), w4, modulus, T)
    counts = dict(get_bloq_callee_counts(bloq))
    phasefix = UnderflowPhasefix(QUInt(kappa), w4, T)
    assert sp.simplify(counts[phasefix] - sp.ceiling(n / w4) / 2) == 0
    assert not any(isinstance(c, float) or sp.sympify(c).has(sp.Float) for c in counts.values())
