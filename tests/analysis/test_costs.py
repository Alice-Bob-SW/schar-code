"""Operation counts of the delayed-MBU oracle against the tables of Appendix B.

The expected values are transcribed from `schar_paper/tex_files/04_logical-implementation.tex`:
Tables `computing_dlog_operation_count`, `dlog_modular_reduction_operation_count`,
`modexp_operation_count`, `unmodexp_operation_count`, `accumulation_operation_count` and
`operation_count_recap` give the adders, comparisons, Lookups, Phaseups and
UnderflowPhasefixes; Table `logical_qubit_count` gives the CX and M_X.

Measurements of the code that the paper tables do not list are added to the expected
counts through `PAPER_OMITS`.
"""

import importlib.util
from pathlib import Path

import pytest
import sympy as sp
from qualtran import QUInt

from scharcode.analysis.costs import (
    COMPARISON,
    CX,
    LOOKUP,
    M_X,
    PHASEUP,
    QC_ADDER,
    QQ_ADDER,
    UNDERFLOW_PHASEFIX,
    PaperOp,
    kappa,
    l_p,
    log_n_e,
    n_e,
    num_primes,
    paper_op_counts,
    subroutines,
    w1,
    w3,
    w4,
)
from scharcode.bloqs.operations._utils import T
from scharcode.bloqs.operations.half_compare import HalfCompare
from scharcode.bloqs.operations.lookup import Lookup
from scharcode.bloqs.operations.qq_adder import QQAdder

ROOT = Path(__file__).resolve().parents[2]

# Shorthands for the paper's notation.
P, ell, log = num_primes, l_p, log_n_e
c1 = sp.ceiling(n_e / w1)
c3 = sp.ceiling(ell / w3)
c4 = sp.ceiling(ell / w4)
inner = (c3 - 2) * c3  # ModMulSteps of one ComputeResidue, (ceil(l/w3) - 2) ceil(l/w3)

TABLE_CATEGORIES = (QQ_ADDER, QC_ADDER, COMPARISON, LOOKUP, PHASEUP, UNDERFLOW_PHASEFIX)
"""Categories listed by the operation-count tables (they leave out CX and M_X)."""


def qq(n, sub=False):
    return PaperOp(QQ_ADDER, (n,), subtraction=sub)


def qc_sub(n):
    return PaperOp(QC_ADDER, (n,), subtraction=True)


def lookup(w, n):
    return PaperOp(LOOKUP, (w, n))


def phaseup(w):
    return PaperOp(PHASEUP, (w,))


def upf(w, n):
    return PaperOp(UNDERFLOW_PHASEFIX, (w, n))


def comparison(n):
    return PaperOp(COMPARISON, (n,))


def mx(n):
    return PaperOp(M_X, (n,))


def cx(n):
    return PaperOp(CX, (n,))


def restrict(counts, categories):
    return {op: n for op, n in counts.items() if op.category in categories}


def assert_counts_equal(code, paper):
    """Same operations, and every count equal as a sympy expression."""
    assert set(code) == set(paper), (
        f"only in code: {set(code) - set(paper)}; only in paper: {set(paper) - set(code)}"
    )
    for op in paper:
        assert sp.simplify(code[op] - paper[op]) == 0, f"{op}: code {code[op]}, paper {paper[op]}"


@pytest.fixture(scope="module")
def subs():
    return subroutines("delayed_mbu")


@pytest.fixture(scope="module")
def script():
    """`scripts/make_paper_outputs.py`, which builds the table parts and the recap rows."""
    path = ROOT / "scripts" / "make_paper_outputs.py"
    spec = importlib.util.spec_from_file_location("make_paper_outputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- counting conventions on single primitives -------------------------------------------


def test_lookup_counts_its_target_measurement():
    assert paper_op_counts(Lookup(T, 3, 7)) == {lookup(3, 7): 1, mx(7): 1}
    assert paper_op_counts(Lookup(T, 3, 7).adjoint()) == {phaseup(3): 1}


def test_adder_and_subtraction():
    assert paper_op_counts(QQAdder(5)) == {qq(5): 1}
    assert paper_op_counts(QQAdder(5).adjoint()) == {qq(5, sub=True): 1}


def test_comparison_as_adder():
    cmp = HalfCompare(QUInt(6), QUInt(6))
    assert paper_op_counts(cmp) == {comparison(6): 1}
    assert paper_op_counts(cmp, comparison_as_adder=True) == {qq(7): 1}
    assert paper_op_counts(cmp.adjoint()) == {}


# ---- per-subroutine tables ---------------------------------------------------------------


def test_computing_dlog_operation_count(subs):
    code = restrict(
        {**subs["ComputeExponent"].counts(), **subs["Last Phaseup"].counts()}, TABLE_CATEGORIES
    )
    assert_counts_equal(
        code,
        {
            qq(ell + log): (P + 1) * c1,
            lookup(w1, ell + log): (P + 1) * c1,
            phaseup(w1): c1,
        },
    )


def test_dlog_modular_reduction_operation_count(subs):
    code = restrict(subs["ReduceExponent"].counts(), TABLE_CATEGORIES)
    assert_counts_equal(code, {qq(ell): P * log, qc_sub(ell + 1): P * log})


def test_modexp_operation_count(subs):
    code = restrict(subs["ComputeResidue"].counts(), TABLE_CATEGORIES)
    assert_counts_equal(
        code,
        {
            lookup(2 * w3, ell): P,
            qq(ell + 1, sub=True): P * inner,
            qq(ell): P * inner,
            lookup(2 * w3, ell + 1): P * inner,
        },
    )


def test_unmodexp_operation_count(subs, script):
    sub = subs["ComputeResidue†"]
    code = restrict(sub.counts(expand_underflow_phasefix=False), TABLE_CATEGORIES)
    assert_counts_equal(
        code,
        {
            # the final unloading and the 2 P inner Phaseups have the same address size
            phaseup(2 * w3): P + 2 * P * inner,
            qq(ell + 1, sub=True): 2 * P * inner,
            qq(ell): 2 * P * inner,
            lookup(2 * w3, ell + 1): 2 * P * inner,
            upf(2 * w3, ell): P * inner,
        },
    )
    inside = restrict(script.underflow_phasefix_part(sub), TABLE_CATEGORIES)
    assert_counts_equal(inside, {lookup(2 * w3, ell): P * inner, comparison(ell): P * inner})


def test_accumulation_operation_count(subs, script):
    sub = subs["AccumulateResidue"]
    code = restrict(sub.counts(expand_underflow_phasefix=False), TABLE_CATEGORIES)
    assert_counts_equal(
        code,
        {
            qq(kappa + 1, sub=True): P * c4,
            qq(kappa): P * c4,
            lookup(w4, kappa + 1): P * c4,
            phaseup(w4): P * c4,
            upf(w4, kappa): P * c4 / 2,
        },
    )
    inside = restrict(script.underflow_phasefix_part(sub), TABLE_CATEGORIES)
    assert_counts_equal(inside, {lookup(w4, kappa): P * c4 / 2, comparison(kappa): P * c4 / 2})


def test_expected_counts_are_exact_rationals(subs):
    """The 1/2 of a classically controlled UnderflowPhasefix is a sympy Rational."""
    n = subs["AccumulateResidue"].counts(expand_underflow_phasefix=False)[upf(w4, kappa)]
    values = {P: 3, ell: 20, w4: 5}
    assert sp.sympify(n).subs(values) == sp.Rational(6, 1)
    assert not sp.sympify(n).has(sp.Float)


# ---- Table operation_count_recap ---------------------------------------------------------

R = sp.Rational
RECAP = {
    # label: (register size, address size, Q-Q Adder, Q-C Adder, Lookup, Phaseup) per call
    "ComputeExponent": (ell + log, w1, 1, 0, 1, 0),
    "ReduceExponent": (ell + 1, None, 1, 1, 0, 0),
    "ComputeResidue (initialization)": (ell, 2 * w3, 0, 0, 1, 0),
    "ComputeResidue (outer loop)": (ell + 1, 2 * w3, 2, 0, 1, 0),
    "AccumulateResidue": (kappa + 1, w4, R(5, 2), 0, R(3, 2), 1),
    "ComputeResidue† (outer loop)": (ell + 1, 2 * w3, R(5, 2), 0, R(3, 2), 1),
    # The paper gives register size l, the unloaded register; the code has no adder or
    # Lookup target in this row, so the script leaves it empty.
    "ComputeResidue† (unloading)": (None, 2 * w3, 0, 0, 0, 1),
    "ReduceExponent†": (ell + 1, None, 1, 1, 0, 0),
    "Last Phaseup (ComputeExponent)": (None, w1, 0, 0, 0, 1),
}
RECAP_CALLS = {
    "ComputeExponent": (P + 1) * c1,
    "ReduceExponent": P * log,
    "ComputeResidue (initialization)": P,
    "ComputeResidue (outer loop)": P * inner,
    "AccumulateResidue": P * c4,
    "ComputeResidue† (outer loop)": 2 * P * inner,
    "ComputeResidue† (unloading)": P,
    "ReduceExponent†": P * log,
    "Last Phaseup (ComputeExponent)": c1,
}


@pytest.fixture(scope="module")
def recap(script):
    return {row["subroutine"]: row for row in script.recap_rows("delayed_mbu")}


def test_recap_has_the_rows_of_the_paper(recap):
    assert list(recap) == list(RECAP)


@pytest.mark.parametrize("label", list(RECAP))
def test_operation_count_recap(recap, label):
    row = recap[label]
    register, address, *per_call = RECAP[label]
    assert sp.simplify(row["calls"] - RECAP_CALLS[label]) == 0
    for got, want in ((row["register size"], register), (row["address size"], address)):
        if want is None:
            assert got is None
        else:
            assert sp.simplify(got - want) == 0
    for category, want in zip((QQ_ADDER, QC_ADDER, LOOKUP, PHASEUP), per_call, strict=True):
        got = row[category]
        assert got == want, f"{category}: code {got}, paper {want}"
        assert not sp.sympify(got).has(sp.Float)  # exact 1/2, not 0.5


# ---- CX and M_X, Table logical_qubit_count -----------------------------------------------

QUBIT_TABLE = {
    "ComputeExponent": {mx(ell + log): (P + 1) * c1},
    "ReduceExponent": {cx(ell): 2 * P * log},
    "ComputeResidue": {
        mx(ell + 1): P * inner,
        cx(ell): 2 * P * inner,
        mx(1): P * inner,
        mx(ell): P * (c3 - 2),
    },
    "AccumulateResidue": {
        mx(kappa + 1): P * c4,
        cx(kappa): 2 * P * c4,
        mx(kappa): P * c4 / 2,
    },
    "ComputeResidue†": {
        mx(ell + 1): 2 * P * inner,
        cx(ell): 4 * P * inner,
        mx(ell): P * inner,
    },
    "ReduceExponent†": {cx(ell): 2 * P * log},
    "Last Phaseup": {},
}

PAPER_OMITS = {  # measurements in the code not listed in the paper tables
    "ComputeResidue": {mx(ell): P},  # init Lookup target
    "ComputeResidue†": {mx(1): 2 * P * inner},  # underflow MeasureX
    "AccumulateResidue": {mx(1): P * c4},  # wraparound MeasureX
}
"""Exact counts, so that a change in these measurements still fails the test. Each is one
measurement per prime or of a single qubit, negligible next to the M_X the paper lists."""


@pytest.mark.parametrize("name", list(QUBIT_TABLE))
def test_logical_qubit_count_cx_and_mx(subs, name):
    code = restrict(subs[name].counts(), (CX, M_X))
    expected = dict(QUBIT_TABLE[name])
    for op, n in PAPER_OMITS.get(name, {}).items():
        expected[op] = expected.get(op, 0) + n
    assert_counts_equal(code, expected)


def test_paper_omits_only_measurements():
    assert {op.category for omits in PAPER_OMITS.values() for op in omits} == {M_X}


def test_symbolic_subroutines_resolve_in_both_versions():
    for version in ("delayed_mbu", "immediate_mbu"):
        for sub in subroutines(version).values():
            assert sub.counts()
