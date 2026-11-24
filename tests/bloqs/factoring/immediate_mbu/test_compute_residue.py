import attrs
import pytest
import sympy as sp
from qualtran import QAny, QBit, QUInt
from qualtran.bloqs.basic_gates import Discard, MeasureX
from qualtran.bloqs.bookkeeping import Free
from qualtran.resource_counting import get_bloq_callee_counts
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.testing import assert_equivalent_bloq_counts

from helpers import assert_valid_decomposition as assert_valid_bloq_decomposition
from helpers import phased_run
from scharcode.bloqs.factoring._common import l_p, p_i, underflow_phasefix_generalizer
from scharcode.bloqs.operations._utils import T
from scharcode.bloqs.operations.ctrl_add_constant import ControlledConstantAdd
from scharcode.bloqs.operations.lookup import Lookup, lookup_generalizer
from scharcode.bloqs.operations.qq_adder import QQAdder
from scharcode.bloqs.factoring.immediate_mbu.compute_residue import (
    ComputeResidue,
    ComputeResidueAdjoint,
    ModMulStep,
)
from scharcode.bloqs.factoring.immediate_mbu.underflow_phasefix import UnderflowPhasefix


def mod_mul_step_generalizer(bloq):
    """A ModMulStep with the symbolic table T, as the call graphs count it."""
    if isinstance(bloq, ModMulStep):
        return attrs.evolve(bloq, table3_jk=T)
    return bloq


GENERALIZERS = [ignore_split_join, ignore_alloc_free, lookup_generalizer, mod_mul_step_generalizer]

# (l_p, w3a, w3b). The comments give the outer windows j_a and which top windows are partial.
CASES = [
    (4, 2, 2),  # no outer window
    (6, 2, 3),  # j_a = 2, all windows full
    (8, 2, 2),  # j_a = 2, 3, all windows full
    (12, 5, 5),  # j_a = 2, partial x_a and y_b (2 bits)
    (11, 3, 4),  # j_a = 2, 3, partial x_a (2 bits) and y_b (3 bits)
    (20, 3, 3),  # paper parameters: j_a = 2..6, partial x_a and y_b (2 bits)
]

w3a_sym, w3b_sym, g_sym = sp.symbols("w3a w3b g", positive=True, integer=True)


def _bloq(ell: int, w3a: int, w3b: int) -> ComputeResidue:
    p = int(sp.prevprime(2**ell))
    return ComputeResidue(QUInt(ell), w3a, w3b, 3, p)


# ---- ModMulStep ------------------------------------------------------------------------------


def test_mod_mul_step_symbolic_call_graph():
    bloq = ModMulStep(w3a_sym, w3b_sym, T, p_i)
    lookup = Lookup(T, w3a_sym + w3b_sym, l_p + 1)
    assert dict(get_bloq_callee_counts(bloq)) == {
        lookup: 1,
        lookup.adjoint(): 1,
        QQAdder(l_p + 1).adjoint(): 1,
        ControlledConstantAdd(QAny(l_p), p_i): 1,
    }


@pytest.mark.parametrize("ell,w3a,w3b", CASES)
def test_mod_mul_step_decomposition(ell, w3a, w3b):
    """Every window pair, including the partial top windows."""
    residue = _bloq(ell, w3a, w3b)
    for j_a in residue.layout_window.outer_windows:
        for j_b in range(residue.layout_window.num_w3b):
            step = residue.mod_mul_step(j_a, j_b)
            assert_valid_bloq_decomposition(step)
            assert_equivalent_bloq_counts(step, generalizer=GENERALIZERS)


# ---- ComputeResidue --------------------------------------------------------------------------


def test_compute_residue_symbolic_call_graph():
    bloq = ComputeResidue(QUInt(l_p), w3a_sym, w3b_sym, g_sym, p_i)
    counts = dict(get_bloq_callee_counts(bloq))
    step = ModMulStep(w3a_sym, w3b_sym, T, p_i)
    free_underflow, free_helper = Free(QBit(), dirty=True), Free(QUInt(l_p), dirty=True)
    assert set(counts) == {Lookup(T, 2 * w3a_sym, l_p), step, free_underflow, free_helper}
    assert counts[Lookup(T, 2 * w3a_sym, l_p)] == 1
    outer = sp.ceiling(l_p / w3a_sym) - 2
    expected = outer * sp.ceiling(l_p / w3b_sym)
    assert sp.simplify(counts[step] - expected) == 0
    assert sp.simplify(counts[free_underflow] - expected) == 0
    assert sp.simplify(counts[free_helper] - outer) == 0


@pytest.mark.parametrize("ell,w3a,w3b", CASES)
def test_compute_residue_decomposition(ell, w3a, w3b):
    bloq = _bloq(ell, w3a, w3b)
    assert_valid_bloq_decomposition(bloq)
    assert_equivalent_bloq_counts(bloq, generalizer=GENERALIZERS)


@pytest.mark.parametrize("ell,w3a,w3b", CASES)
def test_compute_residue_steps_per_window_size(ell, w3a, w3b):
    """With full windows, the concrete call graph has the symbolic number of ModMulSteps."""
    bloq = _bloq(ell, w3a, w3b)
    w = bloq.layout_window
    steps = {b: n for b, n in get_bloq_callee_counts(bloq) if isinstance(b, ModMulStep)}
    assert sum(steps.values()) == w.num_outer_steps * w.num_w3b
    sizes = {(b.w3a, b.w3b) for b in steps}
    assert sizes == {(w.w3a_sizes[j_a], s) for j_a in w.outer_windows for s in w.w3b_sizes}


# ---- ComputeResidue† -------------------------------------------------------------------------


def _adjoint(ell: int, w3a: int, w3b: int) -> ComputeResidueAdjoint:
    p = int(sp.prevprime(2**ell))
    return ComputeResidueAdjoint(QUInt(ell), w3a, w3b, 3, p)


def test_compute_residue_adjoint_symbolic_call_graph():
    bloq = ComputeResidueAdjoint(QUInt(l_p), w3a_sym, w3b_sym, g_sym, p_i)
    counts = dict(get_bloq_callee_counts(bloq))
    pairs = (sp.ceiling(l_p / w3a_sym) - 2) * sp.ceiling(l_p / w3b_sym)
    expected = {
        Lookup(T, 2 * w3a_sym, l_p).adjoint(): 1,
        ModMulStep(w3a_sym, w3b_sym, T, p_i): 2 * pairs,
        MeasureX(): 2 * pairs,
        Discard(): 2 * pairs,
        # expected-count convention: 1/2 per step
        UnderflowPhasefix(QUInt(l_p), w3a_sym + w3b_sym, T): pairs,
    }
    assert set(counts) == set(expected)
    for bloq_key, count in expected.items():
        assert sp.simplify(counts[bloq_key] - count) == 0, bloq_key


@pytest.mark.parametrize("ell,w3a,w3b", CASES)
def test_compute_residue_adjoint_decomposition(ell, w3a, w3b):
    """The call graph matches the decomposition, except the UnderflowPhasefix count.

    The decomposition holds one classically controlled UnderflowPhasefix per step, and the call
    graph counts it 1/2 per step (expected-count convention).
    """
    bloq = _adjoint(ell, w3a, w3b)
    assert_valid_bloq_decomposition(bloq)
    generalizer = [*GENERALIZERS, underflow_phasefix_generalizer]
    from_graph = dict(get_bloq_callee_counts(bloq, generalizer=generalizer))
    from_decomp = dict(get_bloq_callee_counts(bloq.decompose_bloq(), generalizer=generalizer))
    for key in from_graph.keys() | from_decomp.keys():
        factor = 2 if isinstance(key, UnderflowPhasefix) else 1
        assert from_decomp.get(key, 0) == factor * from_graph.get(key, 0), key


# ---- classical action and phase (ported from mod_exp_test.py) --------------------------------

# (p, g, w3a, w3b) from the source tests. 133 = 7 * 19 is not prime, and `ResidueTables`
# refuses it, so the last case uses the prime 131 instead.
PORTED_CASES = [
    (13, 3, 2, 2),
    (61, 2, 2, 3),
    (61, 5, 3, 2),
    (61, 2, 4, 3),  # ell % w3a != 0
    (61, 2, 2, 4),  # ell % w3b != 0
    (131, 3, 4, 8),
]


def test_metadata():
    bloq = ComputeResidue(QUInt(6), 2, 3, 3, 61)
    assert bloq.l_p == 6
    assert bloq.layout_window.num_w3a == 3
    assert bloq.layout_window.num_w3b == 2
    assert bloq.signature.get_left("s").dtype == QUInt(6)
    assert bloq.signature.get_right("r").dtype == QUInt(6)


def _check_forward(bloq: ComputeResidue, exponents, rng):
    """Values only: the forward decomposition is not phase-exact."""
    cbloq = bloq.decompose_bloq().flatten()
    for s in exponents:
        out, _ = phased_run(cbloq, {"s": s}, rng)
        assert out == {"s": s, "r": pow(bloq.g, s, bloq.p)}, s
        assert bloq.on_classical_vals(s=s) == out


def _check_adjoint(bloq: ComputeResidueAdjoint, exponents, rng, trials: int = 2):
    """Values and phase 1, under random M_X outcomes of the underflow qubits."""
    cbloq = bloq.decompose_bloq().flatten()
    for s in exponents:
        for _ in range(trials):
            out, phase = phased_run(cbloq, {"s": s, "r": pow(bloq.g, s, bloq.p)}, rng)
            assert out == {"s": s}, s
            assert phase == 1, s


@pytest.mark.slow
@pytest.mark.parametrize("p, g, w3a, w3b", PORTED_CASES)
def test_compute_residue_classical_action(p, g, w3a, w3b, rng):
    bloq = ComputeResidue(QUInt(p.bit_length()), w3a, w3b, g, p)
    _check_forward(bloq, range(2 ** p.bit_length()), rng)


@pytest.mark.slow
@pytest.mark.parametrize("p, g, w3a, w3b", PORTED_CASES)
def test_compute_residue_adjoint_phase(p, g, w3a, w3b, rng):
    bloq = ComputeResidueAdjoint(QUInt(p.bit_length()), w3a, w3b, g, p)
    _check_adjoint(bloq, range(2 ** p.bit_length()), rng)


# ---- Decomposition-level tests with several outer windows -------------------------------


def _sample(ell: int, rng, size: int) -> list[int]:
    return [0, 1, 2**ell - 1, *(int(s) for s in rng.integers(0, 2**ell, size))]


def test_d1_partial_top_window(rng):
    """ell = 7, w3a = 2: four exponent windows, the top one of 1 bit (ell % w3a != 0)."""
    p = int(sp.prevprime(2**7))
    exponents = _sample(7, rng, 20)
    _check_forward(ComputeResidue(QUInt(7), 2, 3, 3, p), exponents, rng)
    _check_adjoint(ComputeResidueAdjoint(QUInt(7), 2, 3, 3, p), exponents, rng, trials=1)


@pytest.mark.slow
@pytest.mark.parametrize(
    "ell, w3a, w3b",
    [
        (12, 2, 2),  # six exponent windows, j_a = 2..5
        (12, 2, 5),  # same, with a partial top product window
        (11, 3, 4),  # four exponent windows, partial top one (2 bits)
    ],
)
def test_d1_many_windows(ell, w3a, w3b, rng):
    p = int(sp.prevprime(2**ell))
    exponents = _sample(ell, rng, 40)
    _check_forward(ComputeResidue(QUInt(ell), w3a, w3b, 5, p), exponents, rng)
    _check_adjoint(ComputeResidueAdjoint(QUInt(ell), w3a, w3b, 5, p), exponents, rng, trials=1)
