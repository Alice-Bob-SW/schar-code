"""Operation counts of the oracle in the categories of the Appendix B tables.

`paper_op_counts` walks the call graph of a bloq down to the primitives of the paper and
counts them by category and size:

=====================  ==========================================  ===========================
Category               Bloqs                                       Sizes
=====================  ==========================================  ===========================
Q-Q Adder              `QQAdder`, `QQAdderSub` (subtraction)       (n,)
Q-C Adder              `QCAdder`, `QCAdder.adjoint()` (subtraction) (n,)
Comparison             `HalfCompare`                               (n,), the compared size
Lookup                 `Lookup`                                    (address, target)
Phaseup                `Phaseup`, `Lookup.adjoint()`               (address,)
UnderflowPhasefix      `UnderflowPhasefix`, only if not expanded   (address, comparison)
M_X                    see below                                   (qubits,)
CX                     `CtrlXorK`, `MultiTargetCNOT`               (targets,)
=====================  ==========================================  ===========================

Conventions, which follow the text of Appendix B ("Logical cost breakdown"):

- A comparison is a `HalfCompare` and its adjoint. Only the forward half is counted; the
  adjoint is its measurement-based uncomputation. With `comparison_as_adder=True`, a
  Comparison(n) is counted as a Q-Q Adder(n + 1), as Table `operation_count_recap` does.
- Every Lookup target is measured in X once, before the Phaseup that uncomputes it, so each
  Lookup(w, n) also counts one M_X(n). In the immediate version this M_X is inside
  `Lookup.adjoint()`, which is therefore counted as a Phaseup only. In the delayed version
  the code frees the target dirty instead of measuring it (the deferred Phaseups are
  cost-only placeholders), and this rule restores the M_X that the paper counts.
- A single-qubit `MeasureX` (an underflow or wraparound qubit) counts as M_X(1).
- A dirty `Free(dtype)` in a call graph counts as M_X(dtype.num_qubits): it stands for an X
  measurement that the code does not perform (the underflow qubit and the helper register of
  ComputeResidue). Call graphs list such a free only where no other rule already counts that
  measurement: the Lookup targets freed dirty in the delayed version are counted by the Lookup rule
  above and stay out of the call graphs.
- `UnderflowPhasefix` is classically controlled by an M_X outcome. Its counts are expected
  counts, 1/2 per call site, kept exact as a `sympy.Rational`.
- Bookkeeping bloqs, `Discard` and single-qubit Paulis are not counted.

`subroutines` gives symbolic instances of the subroutines of `ApproxModExp`, in the symbols
of Table `parameter_definition`, and `RSA2048` the values of Table `algo_parameter_value`.
"""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module

import sympy as sp
from qualtran import Adjoint, Bloq, QUInt
from qualtran.bloqs.basic_gates import Discard, DiscardQ, MeasureX, XGate, ZGate
from qualtran.bloqs.bookkeeping import Allocate, Cast, Free, Join, Partition, Split
from qualtran.resource_counting import SympySymbolAllocator, get_bloq_callee_counts
from qualtran.symbolics import SymbolicInt

from scharcode.bloqs.factoring._common import l_p, num_primes, p_i
from scharcode.bloqs.factoring.delayed_mbu import UnderflowPhasefix as _UpfDelayed
from scharcode.bloqs.factoring.immediate_mbu import UnderflowPhasefix as _UpfImmediate
from scharcode.bloqs.factoring.reduce_exponent import ReduceExponent
from scharcode.bloqs.operations._utils import T
from scharcode.bloqs.operations.half_compare import HalfCompare
from scharcode.bloqs.operations.lookup import LookupAbstract, LookupUncompute
from scharcode.bloqs.operations.multi_target_cnot import CtrlXorK, MultiTargetCNOT
from scharcode.bloqs.operations.phaseup import Phaseup
from scharcode.bloqs.operations.qc_adder import QCAdder
from scharcode.bloqs.operations.qq_adder import QQAdder, QQAdderSub

# ---- categories ------------------------------------------------------------------------

QQ_ADDER = "Q-Q Adder"
QC_ADDER = "Q-C Adder"
COMPARISON = "Comparison"
LOOKUP = "Lookup"
PHASEUP = "Phaseup"
UNDERFLOW_PHASEFIX = "UnderflowPhasefix"
M_X = "M_X"
CX = "CX"
CATEGORIES = (QQ_ADDER, QC_ADDER, COMPARISON, LOOKUP, PHASEUP, UNDERFLOW_PHASEFIX, M_X, CX)
"""The categories of `PaperOp`, in the order of the paper tables."""

SIZE_NAMES = {
    QQ_ADDER: ("size",),
    QC_ADDER: ("size",),
    COMPARISON: ("size",),
    LOOKUP: ("address", "target"),
    PHASEUP: ("address",),
    UNDERFLOW_PHASEFIX: ("address", "comparison"),
    M_X: ("qubits",),
    CX: ("targets",),
}
"""Meaning of each entry of `PaperOp.sizes`, per category."""


@dataclass(frozen=True)
class PaperOp:
    """One operation of the paper tables, such as Lookup(w1, l + log n_e).

    Attributes
    ----------
    category : str
        One of `CATEGORIES`.
    sizes : tuple[SymbolicInt, ...]
        The sizes named by `SIZE_NAMES[category]`.
    subtraction : bool
        True for an adder run backwards, written "(subtraction)" or Adder† in the paper.
    """

    category: str
    sizes: tuple[SymbolicInt, ...]
    subtraction: bool = False

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category {self.category!r}")
        if len(self.sizes) != len(SIZE_NAMES[self.category]):
            raise ValueError(f"{self.category} takes sizes {SIZE_NAMES[self.category]}")

    @property
    def name(self) -> str:
        """Category, with "(subtraction)" for adders run backwards."""
        return f"{self.category} (subtraction)" if self.subtraction else self.category

    def size_str(self, printer: Callable[[sp.Expr], str] = str) -> str:
        """Sizes as in the paper tables, "address = w_1, target = l_p + log_n_e" or "l_p"."""
        names = SIZE_NAMES[self.category]
        values = [printer(sp.sympify(s)) for s in self.sizes]
        if names == ("size",) or names == ("qubits",) or names == ("targets",):
            return values[0]
        return ", ".join(f"{n} = {v}" for n, v in zip(names, values, strict=True))

    def sort_key(self) -> tuple:
        return (CATEGORIES.index(self.category), self.subtraction, str(self.sizes))


# ---- the walk --------------------------------------------------------------------------

_NOT_COUNTED = (
    Allocate,
    Free,
    Partition,
    Split,
    Join,
    Cast,
    Discard,
    DiscardQ,
    XGate,
    ZGate,
)
"""Leaves that are not paper operations: bookkeeping, discarded bits and single-qubit Paulis."""


def _primitive(
    bloq: Bloq, comparison_as_adder: bool, expand_underflow_phasefix: bool
) -> Counter | None:
    """The paper operations of `bloq` if it is a primitive, else None (walk its callees)."""
    if isinstance(bloq, Free) and bloq.dirty:
        return Counter({PaperOp(M_X, (bloq.dtype.num_qubits,)): 1})
    if isinstance(bloq, _NOT_COUNTED):
        return Counter()
    if isinstance(bloq, Adjoint):
        sub = bloq.subbloq
        if isinstance(sub, HalfCompare):
            return Counter()  # uncomputation half of a comparison counted at its forward half
        if isinstance(sub, QCAdder):
            return Counter({PaperOp(QC_ADDER, (sub.n,), subtraction=True): 1})
        if isinstance(sub, QQAdder):
            return Counter({PaperOp(QQ_ADDER, (sub.n,), subtraction=True): 1})
        if isinstance(sub, (CtrlXorK, MultiTargetCNOT)):
            return _primitive(sub, comparison_as_adder, expand_underflow_phasefix)
        return None
    if isinstance(bloq, QQAdderSub):
        return Counter({PaperOp(QQ_ADDER, (bloq.n,), subtraction=True): 1})
    if isinstance(bloq, QQAdder):
        return Counter({PaperOp(QQ_ADDER, (bloq.n,)): 1})
    if isinstance(bloq, QCAdder):
        return Counter({PaperOp(QC_ADDER, (bloq.n,)): 1})
    if isinstance(bloq, HalfCompare):
        n = bloq.dtype_t.bitsize
        if comparison_as_adder:
            return Counter({PaperOp(QQ_ADDER, (n + 1,)): 1})
        return Counter({PaperOp(COMPARISON, (n,)): 1})
    if isinstance(bloq, LookupAbstract):
        return Counter({PaperOp(LOOKUP, (bloq.w, bloq.n)): 1, PaperOp(M_X, (bloq.n,)): 1})
    if isinstance(bloq, LookupUncompute):
        return Counter({PaperOp(PHASEUP, (bloq.w,)): 1})  # its M_X is counted at the Lookup
    if isinstance(bloq, Phaseup):
        return Counter({PaperOp(PHASEUP, (bloq.w,)): 1})
    if isinstance(bloq, MeasureX):
        return Counter({PaperOp(M_X, (1,)): 1})
    if isinstance(bloq, CtrlXorK):
        return Counter({PaperOp(CX, (bloq.dtype.num_qubits,)): 1})
    if isinstance(bloq, MultiTargetCNOT):
        return Counter({PaperOp(CX, (bloq.bitsize,)): 1})
    if not expand_underflow_phasefix and isinstance(bloq, (_UpfDelayed, _UpfImmediate)):
        return Counter({PaperOp(UNDERFLOW_PHASEFIX, (bloq.w, bloq.dtype_reg.bitsize)): 1})
    return None


def paper_op_counts(
    bloq: Bloq,
    *,
    comparison_as_adder: bool = False,
    expand_underflow_phasefix: bool = True,
) -> dict[PaperOp, SymbolicInt]:
    """Count the paper operations in `bloq`, following its call graph down to the primitives.

    Parameters
    ----------
    bloq : Bloq
        Any bloq of the oracle, symbolic or concrete. A primitive counts itself.
    comparison_as_adder : bool
        Count each Comparison(n) as a Q-Q Adder(n + 1), as Table `operation_count_recap` does.
    expand_underflow_phasefix : bool
        If False, stop at `UnderflowPhasefix` and count it as one operation, as the tables
        `unmodexp_operation_count` and `accumulation_operation_count` list it.

    Returns
    -------
    dict[PaperOp, SymbolicInt]
        Number of calls of each operation, sorted by category. Counts are sympy expressions
        when `bloq` is symbolic, and can be expected counts (multiples of 1/2).

    Raises
    ------
    ValueError
        If the walk reaches a leaf that is neither a paper operation nor in `_NOT_COUNTED`.
    """
    ssa = SympySymbolAllocator()
    memo: dict[Bloq, Counter] = {}

    def walk(b: Bloq) -> Counter:
        if b in memo:
            return memo[b]
        counts = _primitive(b, comparison_as_adder, expand_underflow_phasefix)
        if counts is None:
            callees = get_bloq_callee_counts(b, ssa=ssa)
            if not callees:
                raise ValueError(f"{b!r} is a leaf with no paper category")
            counts = Counter()
            for callee, n in callees:
                for op, k in walk(callee).items():
                    counts[op] += n * k
        memo[b] = counts
        return counts

    counts = walk(bloq)
    return {op: counts[op] for op in sorted(counts, key=PaperOp.sort_key) if counts[op] != 0}


def add_counts(*terms: tuple[SymbolicInt, Mapping[PaperOp, SymbolicInt]]) -> dict:
    """Sum of `multiplier * counts` over the (multiplier, counts) pairs, sorted by category."""
    total: Counter = Counter()
    for multiplier, counts in terms:
        for op, n in counts.items():
            total[op] += multiplier * n
    return {op: total[op] for op in sorted(total, key=PaperOp.sort_key) if total[op] != 0}


# ---- symbolic instances ----------------------------------------------------------------

n_e = sp.Symbol("n_e", positive=True, integer=True)
"""Size of the exponent register."""
log_n_e = sp.Symbol("log_n_e", positive=True, integer=True)
"""Size of the quotient register, log(n_e) = ceil(log2 n_e) in the paper."""
kappa = sp.Symbol("kappa", positive=True, integer=True)
"""Size of the accumulation register."""
w1 = sp.Symbol("w_1", positive=True, integer=True)
"""Window of ComputeExponent."""
w3 = sp.Symbol("w_3", positive=True, integer=True)
"""Window of ComputeResidue and its adjoint, w3 = w3a = w3b."""
w4 = sp.Symbol("w_4", positive=True, integer=True)
"""Window of AccumulateResidue."""
_g = sp.Symbol("g_p", positive=True, integer=True)
_N = sp.Symbol("N", positive=True, integer=True)

SYMBOLS = (num_primes, l_p, n_e, log_n_e, kappa, w1, w3, w4)
"""The parameters of the symbolic counts: |P|, l, n_e, log(n_e), kappa, w1, w3, w4."""

LATEX_NAMES = {num_primes: "|P|", l_p: r"\ell", log_n_e: r"\log(n_e)"}
"""`symbol_names` for `sympy.latex`, to print the symbols as the paper does."""

RSA2048 = {
    num_primes: 22658,
    l_p: 20,
    n_e: 1280,
    log_n_e: (1280 - 1).bit_length(),
    kappa: 33,
    w1: 6,
    w3: 3,
    w4: 5,
}
"""The RSA-2048 instance, Table `algo_parameter_value` (`05-5_final_count.tex`).

log(n_e) = ceil(log2 n_e) = 11, as in `RnsParameters.s_bitsize`.
"""


def latex(expr: SymbolicInt) -> str:
    """LaTeX of a count or size, with the symbols of the paper."""
    return sp.latex(sp.sympify(expr), symbol_names=LATEX_NAMES)


def evaluate(expr: SymbolicInt, values: Mapping[sp.Symbol, int] = RSA2048) -> sp.Expr:
    """`expr` at the given parameter values (RSA-2048 by default)."""
    return sp.nsimplify(sp.sympify(expr).subs(values))


@dataclass(frozen=True)
class Subroutine:
    """One subroutine of `ApproxModExp`, with the number of times the oracle calls it.

    Attributes
    ----------
    name : str
        Name used in the paper ("ComputeResidue†" for an adjoint).
    bloq : Bloq
        Symbolic instance.
    calls : SymbolicInt
        Calls per oracle, |P| or |P| + 1, as in `ApproxModExp.build_call_graph`.
    """

    name: str
    bloq: Bloq
    calls: SymbolicInt

    def counts(self, **kwargs) -> dict[PaperOp, SymbolicInt]:
        """`paper_op_counts` of all its calls in one oracle."""
        return add_counts((self.calls, paper_op_counts(self.bloq, **kwargs)))


VERSIONS = ("delayed_mbu", "immediate_mbu")
"""`delayed_mbu` is the version of Appendix B; `immediate_mbu` uncomputes each Lookup at once."""


def subroutines(version: str = "delayed_mbu") -> dict[str, Subroutine]:
    """Symbolic instances of the subroutines of `ApproxModExp`, in the order it runs them.

    Sizes are those of Algorithm `alg:approx-modexp`: the exponent has n_e qubits, S_i has
    l + log(n_e) before reduction and l after, the accumulator kappa, and w3a = w3b = w3.
    Symbolic windows are all full, as in the tables of Appendix B.

    The delayed version also has "Last Phaseup": the ceil(n_e / w1) Phaseup(w1) that
    `ApproxModExp` applies at the end for the Lookups of ComputeExponent.
    """
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}, got {version!r}")
    mod = import_module(f"scharcode.bloqs.factoring.{version}")
    dtype_s = QUInt(l_p + log_n_e)
    dtype_r = QUInt(l_p)
    reduce = ReduceExponent(dtype_s, p_i)
    subs = [
        Subroutine(
            "ComputeExponent", mod.ComputeExponent(QUInt(n_e), dtype_s, w1, T), num_primes + 1
        ),
        Subroutine("ReduceExponent", reduce, num_primes),
        Subroutine("ComputeResidue", mod.ComputeResidue(dtype_r, w3, w3, _g, p_i), num_primes),
        Subroutine(
            "AccumulateResidue",
            mod.AccumulateResidue(dtype_r, QUInt(kappa), w4, _N, T),
            num_primes,
        ),
        Subroutine(
            "ComputeResidue†", mod.ComputeResidueAdjoint(dtype_r, w3, w3, _g, p_i), num_primes
        ),
        Subroutine("ReduceExponent†", reduce.adjoint(), num_primes),
    ]
    if version == "delayed_mbu":
        subs.append(Subroutine("Last Phaseup", Phaseup(w1), sp.ceiling(n_e / w1)))
    return {s.name: s for s in subs}
