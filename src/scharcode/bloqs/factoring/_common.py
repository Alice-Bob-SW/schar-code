"""Helpers shared by the factoring bloqs of both versions (immediate and delayed MBU).

Windows are indexed from the least significant end, like the paper's j_a and j_b. Only the
most significant window can be partial.

A qualtran `Partition` gives its first register the most significant bits. The partitions
below therefore list their registers from the top window down, and the `split_*` / `join_*`
helpers hide that order: they return and take windows keyed by their paper index.

The module also holds the Lookup tables of ComputeResidue and ComputeResidue†
(`ResidueTables`), the symbols that the factoring bloqs share (`l_p`, `num_primes`, `p_i`)
and the generalizer of UnderflowPhasefix for call-graph tests.
"""

from collections.abc import Mapping
from functools import cached_property

import attrs
import sympy as sp
from qualtran import Bloq, BloqBuilder, QAny, QBit, Register, Soquet
from qualtran.bloqs.bookkeeping import Partition
from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled
from qualtran.symbolics import SymbolicInt, is_symbolic
from sympy.ntheory import isprime

from scharcode.bloqs.operations._utils import T
from scharcode.classical.rns import double_window_table

# Symbols for the sizes of symbolic call graphs, shared by every factoring bloq.
l_p = sp.Symbol("l_p", positive=True, integer=True)
"""Bit length of every RNS prime, written ell in the paper."""
num_primes = sp.Symbol("num_primes", positive=True, integer=True)
"""Number of RNS primes, written |P| in the paper."""
p_i = sp.Symbol("p_i", positive=True, integer=True)
"""One RNS prime, the modulus of ComputeResidue and ReduceExponent."""


def num_windows(bits: SymbolicInt, w: SymbolicInt) -> SymbolicInt:
    """Number of windows of w bits needed to cover `bits` bits, ceil(bits / w)."""
    if is_symbolic(bits, w):
        return sp.ceiling(sp.sympify(bits) / w)
    return -(-bits // w)


def last_window(bits: SymbolicInt, w: SymbolicInt) -> SymbolicInt:
    """Size of the most significant window when `bits` bits are cut into windows of w bits.

    It is w when w divides `bits`. Symbolic sizes return w: like the operation-count tables
    of Appendix B, symbolic call graphs assume that every window is full.
    """
    if is_symbolic(bits, w):
        return w
    return bits - (num_windows(bits, w) - 1) * w


def window_sizes(bits: int, w: int) -> tuple[int, ...]:
    """Sizes of the windows of w bits covering `bits` bits, least significant window first."""
    n = num_windows(bits, w)
    return (w,) * (n - 1) + (last_window(bits, w),)


def address_partition(w3a: SymbolicInt, w3b: SymbolicInt) -> Partition:
    """Join an exponent window x_a and a product window y_b into one Lookup address.

    Partition: `x` (w3a + w3b qubits) -> `x_a` (w3a), `y_b` (w3b); use `.adjoint()` to join.
    """
    regs = (Register("x_a", QAny(w3a)), Register("y_b", QAny(w3b)))
    return Partition(w3a + w3b, regs)


def helper_partition(l_p: SymbolicInt) -> Partition:
    """Split the (l_p + 1)-qubit register of a modular addition into u and h.

    The underflow qubit u is the most significant bit, the sign bit
    of h - (p_i - T) in the paragraph "The modular addition" of Appendix B.
    Partition: `x` (l_p + 1 qubits) -> `u` (1), `h` (l_p); use `.adjoint()` to join.
    """
    regs = (Register("u", QBit()), Register("h", QAny(l_p)))
    return Partition(l_p + 1, regs)


@attrs.frozen
class ResidueWindows:
    """Window layout of ComputeResidue and ComputeResidue†.

    Parameters
    ----------
    l_p : SymbolicInt
        Bit length of the RNS prime p_i, and so of the registers `s` (exponent S_i) and `r`
        (partial product).
    w3a : SymbolicInt
        Window size over the exponent (outer loop).
    w3b : SymbolicInt
        Window size over the partial product (inner loop).
    """

    l_p: SymbolicInt
    w3a: SymbolicInt
    w3b: SymbolicInt

    def __attrs_post_init__(self):
        for name in ("l_p", "w3a", "w3b"):
            value = getattr(self, name)
            if not is_symbolic(value) and value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")

    @cached_property
    def num_w3a(self) -> SymbolicInt:
        """Number of exponent windows, ceil(l_p / w3a)."""
        return num_windows(self.l_p, self.w3a)

    @cached_property
    def num_w3b(self) -> SymbolicInt:
        """Number of partial-product windows, ceil(l_p / w3b)."""
        return num_windows(self.l_p, self.w3b)

    @cached_property
    def last_w3a(self) -> SymbolicInt:
        """Size of the most significant exponent window."""
        return last_window(self.l_p, self.w3a)

    @cached_property
    def last_w3b(self) -> SymbolicInt:
        """Size of the most significant partial-product window."""
        return last_window(self.l_p, self.w3b)

    @cached_property
    def num_outer_steps(self) -> SymbolicInt:
        """Iterations of the outer loop, ceil(l_p / w3a) - 2 (none if l_p <= 2 * w3a)."""
        if is_symbolic(self.num_w3a):
            return self.num_w3a - 2
        return max(self.num_w3a - 2, 0)

    @cached_property
    def init_bitsize(self) -> SymbolicInt:
        """Address size of the initial Lookup, the 2 * w3a low bits of S_i (at most l_p)."""
        if is_symbolic(self.l_p, self.w3a):
            return 2 * self.w3a
        return min(2 * self.w3a, self.l_p)

    @cached_property
    def w3a_sizes(self) -> tuple[int, ...]:
        """Size of exponent window j_a, for j_a = 0, ..., ceil(l_p / w3a) - 1."""
        self._require_concrete("w3a_sizes")
        return window_sizes(self.l_p, self.w3a)

    @cached_property
    def w3b_sizes(self) -> tuple[int, ...]:
        """Size of partial-product window j_b, for j_b = 0, ..., ceil(l_p / w3b) - 1.

        Replaces `L_w3b`, which listed the same sizes most significant window first.
        """
        self._require_concrete("w3b_sizes")
        return window_sizes(self.l_p, self.w3b)

    @cached_property
    def outer_windows(self) -> range:
        """The exponent windows j_a visited by the outer loop, 2, ..., ceil(l_p / w3a) - 1."""
        self._require_concrete("outer_windows")
        return range(2, self.num_w3a)

    @cached_property
    def exponent_partition(self) -> Partition:
        """Split `s` into the initial Lookup address and the outer-loop windows.

        Partition: `x` (l_p qubits) -> `x_a_<j_a>` for each j_a in
        `outer_windows`, from the top window down, then `s_init` (the `init_bitsize` low bits).
        """
        self._require_concrete("exponent_partition")
        regs = [Register(f"x_a_{j_a}", QAny(self.w3a_sizes[j_a])) for j_a in self.outer_windows]
        regs = regs[::-1] + [Register("s_init", QAny(self.init_bitsize))]
        return Partition(self.l_p, tuple(regs))

    @cached_property
    def product_partition(self) -> Partition:
        """Split a partial-product register (`r` or `h`) into its windows y_b.

        Partition: `x` (l_p qubits) -> `y_b_<j_b>` for each j_b, from
        the top window down.
        """
        self._require_concrete("product_partition")
        regs = [Register(f"y_b_{j_b}", QAny(size)) for j_b, size in enumerate(self.w3b_sizes)]
        return Partition(self.l_p, tuple(regs[::-1]))

    def split_exponent(self, bb: BloqBuilder, s: Soquet) -> tuple[Soquet, dict[int, Soquet]]:
        """Split `s` into the initial Lookup address and the windows x_a, keyed by j_a."""
        out = bb.add_d(self.exponent_partition, x=s)
        x_a = {j_a: out[f"x_a_{j_a}"] for j_a in self.outer_windows}
        return out["s_init"], x_a

    def join_exponent(self, bb: BloqBuilder, s_init: Soquet, x_a: Mapping[int, Soquet]) -> Soquet:
        """Inverse of `split_exponent`."""
        windows = {f"x_a_{j_a}": x_a[j_a] for j_a in self.outer_windows}
        return bb.add(self.exponent_partition.adjoint(), s_init=s_init, **windows)

    def split_product(self, bb: BloqBuilder, r: Soquet) -> list[Soquet]:
        """Split a partial-product register into its windows; element j_b is y_b."""
        out = bb.add_d(self.product_partition, x=r)
        return [out[f"y_b_{j_b}"] for j_b in range(self.num_w3b)]

    def join_product(self, bb: BloqBuilder, y_b: list[Soquet]) -> Soquet:
        """Inverse of `split_product`."""
        windows = {f"y_b_{j_b}": soq for j_b, soq in enumerate(y_b)}
        return bb.add(self.product_partition.adjoint(), **windows)

    def _require_concrete(self, what: str) -> None:
        if is_symbolic(self.l_p, self.w3a, self.w3b):
            raise ValueError(f"{what} needs concrete l_p, w3a and w3b")


@attrs.frozen
class ResidueTables:
    """Lookup tables of ComputeResidue and ComputeResidue† for one prime p = p_i.

    Every table is a flat tuple, ready to be passed to `Lookup`. A table of window pair
    (j_a, j_b) is addressed by x_a * 2^{b} + y_b, where x_a is window j_a of S_i (a bits) and
    y_b is window j_b of the partial product (b bits), with a = `w3a_sizes[j_a]` and
    b = `w3b_sizes[j_b]` of `windows`. This is the address that `address_partition(a, b)`
    joins. Window j_a has offset J_{j_a} = j_a * w3a, and window j_b has offset
    J_{j_b} = j_b * w3b, both from the least significant end.

    Each step of an inner loop is a modular addition. To add a constant C in [0, p) to the
    helper register h, the step loads V = (p - C) mod p, in [0, p), subtracts it on l_p + 1
    bits and adds p back on underflow (paragraph "The modular addition" of Appendix B). The
    underflow bit u is 1 exactly when the new h is at least p - V, so UnderflowPhasefix
    compares h with the complement p - V, in (0, p] (paragraph "Phase corrections"):

    ===============================  ======  ====================  =========================
    Step                             Adds C  Lookup loads V        UnderflowPhasefix table
    ===============================  ======  ====================  =========================
    ComputeResidue, inner loop       T3      `p_minus_t3`          `complement(p_minus_t3)`
    ComputeResidue†, 1st inner loop  T3inv   `p_minus_t3_inv`      `complement(p_minus_t3_inv)`
    ComputeResidue†, 2nd inner loop  -T3     `t3`                  `complement(t3)`
    ===============================  ======  ====================  =========================

    The initial Lookup of ComputeResidue, and the final unloading in ComputeResidue†, use
    `alpha2`.

    Parameters
    ----------
    p : int
        The prime p_i. Its bit length is l_p.
    g_p : int
        The generator g_{p_i}. Any g_p prime to p gives consistent tables.
    w3a : int
        Window size over the exponent (outer loop).
    w3b : int
        Window size over the partial product (inner loop).
    """

    p: int
    g_p: int
    w3a: int
    w3b: int

    def __attrs_post_init__(self):
        if is_symbolic(self.p, self.g_p, self.w3a, self.w3b):
            raise ValueError("ResidueTables needs concrete p, g_p, w3a and w3b")
        if not isprime(self.p):
            raise ValueError(f"p must be prime, got {self.p}")
        if self.g_p % self.p == 0:
            raise ValueError(f"g_p must be prime to p, got g_p = {self.g_p}, p = {self.p}")

    @cached_property
    def windows(self) -> ResidueWindows:
        """Window layout for l_p = bit length of p."""
        return ResidueWindows(self.p.bit_length(), self.w3a, self.w3b)

    @cached_property
    def alpha2(self) -> tuple[int, ...]:
        """Initial Lookup: entry x is g_p^x mod p, for x < 2^{init_bitsize}.

        Addressed by the `init_bitsize` low bits of S_i (windows 0 and 1), it loads
        alpha_2 = g_p^{S_{i,0:J_2}} mod p (paragraph "Initialization" of Appendix B).
        """
        return tuple(pow(self.g_p, x, self.p) for x in range(2**self.windows.init_bitsize))

    def t3(self, j_a: int, j_b: int) -> tuple[int, ...]:
        """T3 of Equation `double_window`.

        Entry [x_a, y_b] is y_b 2^{J_{j_b}} g_p^{x_a 2^{J_{j_a}}} mod p. ComputeResidue† loads it
        in its second inner loop, which subtracts T3.
        """
        return self._double_window(j_a, j_b, inverse=False)

    def t3_inv(self, j_a: int, j_b: int) -> tuple[int, ...]:
        """T3inv of ComputeResidue†.

        Entry [x_a, y_b] is y_b 2^{J_{j_b}} g_p^{-x_a 2^{J_{j_a}}} mod p.
        """
        return self._double_window(j_a, j_b, inverse=True)

    def p_minus_t3(self, j_a: int, j_b: int) -> tuple[int, ...]:
        """(p - T3) mod p, loaded by the inner loop of ComputeResidue."""
        return tuple((self.p - t) % self.p for t in self.t3(j_a, j_b))

    def p_minus_t3_inv(self, j_a: int, j_b: int) -> tuple[int, ...]:
        """(p - T3inv) mod p, loaded by the first inner loop of ComputeResidue†."""
        return tuple((self.p - t) % self.p for t in self.t3_inv(j_a, j_b))

    def complement(self, loaded: tuple[int, ...]) -> tuple[int, ...]:
        """p - V for each entry V of a loaded table: the table of the matching UnderflowPhasefix."""
        return tuple(self.p - v for v in loaded)

    def _double_window(self, j_a: int, j_b: int, inverse: bool) -> tuple[int, ...]:
        w = self.windows
        if not (0 <= j_a < w.num_w3a and 0 <= j_b < w.num_w3b):
            raise IndexError(
                f"window pair ({j_a}, {j_b}) outside [0, {w.num_w3a}) x [0, {w.num_w3b})"
            )
        table = double_window_table(
            self.p,
            self.g_p,
            x_shift=j_a * self.w3a,
            y_shift=j_b * self.w3b,
            x_bits=w.w3a_sizes[j_a],
            y_bits=w.w3b_sizes[j_b],
            inverse=inverse,
        )
        return tuple(int(t) for t in table.ravel())


def underflow_phasefix_generalizer(bloq: Bloq) -> Bloq:
    """Call-graph generalizer: an UnderflowPhasefix, uncontrolled, with the symbolic table T.

    UnderflowPhasefix is applied under a classical control. Call graphs count it as an
    uncontrolled UnderflowPhasefix whose table is the symbol T, so this generalizer removes
    the `ClassicallyControlled` wrapper and replaces the table field `table` by T. Any other bloq
    is returned unchanged.

    Both versions define a class named `UnderflowPhasefix`, and both import this module, so
    the class is recognised by name rather than imported here.
    """
    sub = bloq.subbloq if isinstance(bloq, ClassicallyControlled) else bloq
    if type(sub).__name__ == "UnderflowPhasefix" and attrs.has(type(sub)):
        return attrs.evolve(sub, table=T)
    return bloq
