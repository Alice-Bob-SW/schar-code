import attrs
from functools import cached_property
from collections import Counter
import sympy as sp

from qualtran import (
    Bloq,
    BloqBuilder,
    Signature,
    Register,
    Soquet,
)
from qualtran import QUInt
from qualtran.resource_counting import (
    SympySymbolAllocator,
    BloqCountDictT,
)
from qualtran.symbolics import SymbolicInt, is_symbolic

from scharcode.bloqs.factoring._common import l_p, window_sizes
from scharcode.classical.rns import RnsParameters
from scharcode.bloqs.operations._utils import tuple_or_symbolic, T
from scharcode.bloqs.operations.phaseup import Phaseup
from scharcode.bloqs.factoring.reduce_exponent import ReduceExponent, ReduceExponentAdjoint
from scharcode.bloqs.factoring.delayed_mbu.compute_exponent import ComputeExponent
from scharcode.bloqs.factoring.delayed_mbu.compute_residue import (
    ComputeResidue,
    ComputeResidueAdjoint,
)
from scharcode.bloqs.factoring.delayed_mbu.accumulate_residue import AccumulateResidue


def get_generalizer_subbloq_approx_mod_exp(primes):
    g = sp.Symbol("g")

    def reduceexponent_generalizer(bloq):
        if isinstance(bloq, ReduceExponent) or isinstance(bloq, ReduceExponentAdjoint):
            return ReduceExponent(bloq.dtype_s, int(primes[0]))
        return bloq

    def computeresidue_generalizer(bloq):
        if isinstance(bloq, ComputeResidue):
            return ComputeResidue(bloq.dtype_s, bloq.w3a, bloq.w3b, g, int(primes[0]))
        return bloq

    def computeresidueadjoint_generalizer(bloq):
        if isinstance(bloq, ComputeResidueAdjoint):
            return ComputeResidueAdjoint(bloq.dtype_s, bloq.w3a, bloq.w3b, g, int(primes[0]))
        return bloq

    return reduceexponent_generalizer, computeresidue_generalizer, computeresidueadjoint_generalizer


def computeexponent_generalizer(bloq):
    if isinstance(bloq, ComputeExponent):
        return ComputeExponent(bloq.dtype_e, bloq.dtype_s, bloq.w1, T)
    return bloq


def accumulateresidue_generalizer(bloq):
    if isinstance(bloq, AccumulateResidue):
        return AccumulateResidue(bloq.dtype_r, bloq.dtype_acc, bloq.w4, bloq.modulus, T)
    return bloq


def _msb_first_windows(tables, bits: int, w: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Convert per-prime window tables of `classical.rns` to the layout of the bloqs.

    `tables[i]` has shape (ceil(bits/w), 2^w), least significant window first, and its
    partial last window is padded to 2^w entries. ComputeExponent and AccumulateResidue
    index their windows most significant first (window 0 is the partial one), and a Lookup
    on a window of b bits needs exactly 2^b entries. So the window order is reversed and
    each window is cut to the first 2^b entries, the addresses whose padding bits are 0.
    """
    sizes = window_sizes(bits, w)
    return tuple(
        tuple(
            tuple(int(x) for x in row[j][: 2**size]) for j, size in reversed(list(enumerate(sizes)))
        )
        for row in tables
    )


@attrs.frozen
class ApproxModExp(Bloq):
    """
    Implement an approximate of |e> -> |e>|x^e%N> using RNS and the Chinese theoreme

    Parameters
    ----------
    generators : tuple[SymbolicInt]
        List of all value of the RNS to exponentiate
    primes : tuple[SymbolicInt]
        List of all value prime number of the RNS
    N : SymbolicInt
        modulus of the accumulated modular exponentiation, in RSA, it is the number to factor
    dtype_exp : QUInt
        type of the exponent register
    dtype_s : QUInt
        size for register holding s before the reduction
    dtype_acc : QUInt
        type of the accumulator containing the result of the modular exponentiation
    w1 : SymbolicInt
        window size for ComputeExponent
    w3a : SymbolicInt
        window size for ComputeResidue
    w3b : SymbolicInt
        window size for ComputeResidue
    w4 : SymbolicInt
        window size for AccumulateResidue
    table1 : tuple[tuple[SymbolicInt]]
        precomputed table to offset by, one per prime plus one, windows most significant first
    table4 : tuple[tuple[SymbolicInt]]
        precomputed table,
        table4[i,j,k] = -(((U_i << (j*w4)) * k) mod L mod N) >> dropped_bits) mod trunc,
        windows most significant first

    `from_parameters` builds an instance, and both tables, from `classical.rns.RnsParameters`.

    Attributes
    ----------
    exp : dtype_exp
        register holding the exponent by which we exponentiate
    result : dtype_acc
        the accumulator, kappa qubits. It is an input, the caller prepares it
        in the mask state |σ⟩ as in the paper. On a basis state acc < M, the oracle maps it to
        (acc + Ṽ(e)) mod M, with M = N >> (log N - kappa) and Ṽ(e)
        given by `RnsParameters.approx_top_bits`.

    ref: Fig 26 [Shor's algorithm compiled on cat qubits: 22 593 qubits for RSA-2048 in 3.14 days]
    (https://arxiv.org/abs/2610.37839)
    """

    generators: tuple[SymbolicInt] = attrs.field(converter=tuple_or_symbolic)
    primes: tuple[SymbolicInt] = attrs.field(converter=tuple_or_symbolic)
    N: SymbolicInt
    dtype_exp: QUInt
    dtype_s: QUInt
    dtype_acc: QUInt
    w1: SymbolicInt
    w3a: SymbolicInt
    w3b: SymbolicInt
    w4: SymbolicInt
    table1: tuple[SymbolicInt] = attrs.field(converter=tuple_or_symbolic)
    table4: tuple[SymbolicInt] = attrs.field(converter=tuple_or_symbolic)

    def __attrs_post_init__(self):
        if not (is_symbolic(self.generators) or is_symbolic(self.primes)):
            if len(self.generators) != len(self.primes):
                raise ValueError(
                    f"generators and primes must be of same length but have \
                        size {len(self.generators)} and {len(self.primes)}"
                )

    @classmethod
    def from_parameters(cls, params: RnsParameters) -> "ApproxModExp":
        """The oracle of one RNS instance, with its tables computed by `classical.rns`.

        Register sizes: `exp` has n_e qubits, the s before reduction l_p + log n_e and `result`
        kappa. table1 is `params.t1`; table4 is (M - T4) mod M, the value that
        AccumulateResidue loads and subtracts (`params.t4` docstring). Both are converted to
        the window layout of the bloqs by `_msb_first_windows`.
        """
        loaded_t4 = (params.M - params.t4) % params.M
        return cls(
            generators=params.generators,
            primes=params.primes,
            N=params.N,
            dtype_exp=QUInt(params.n_e),
            dtype_s=QUInt(params.s_bitsize),
            dtype_acc=QUInt(params.kappa),
            w1=params.w1,
            w3a=params.w3a,
            w3b=params.w3b,
            w4=params.w4,
            table1=_msb_first_windows(params.t1, params.n_e, params.w1),
            table4=_msb_first_windows(loaded_t4, params.ell, params.w4),
        )

    @cached_property
    def signature(self) -> "Signature":
        return Signature(
            [
                Register("exp", self.dtype_exp),
                Register("result", self.dtype_acc),
            ]
        )

    @cached_property
    def computeexp_property(self):
        return ComputeExponent(self.dtype_exp, self.dtype_s, self.w1, self.table1[0])

    def build_composite_bloq(self, bb: BloqBuilder, exp: Soquet, result: Soquet):
        s = bb.allocate(self.dtype_s.bitsize, self.dtype_s)
        for i in range(len(self.primes)):
            p = int(self.primes[i])
            g = int(self.generators[i])
            compress_dtype = QUInt(p.bit_length())
            exp, s = bb.add(
                ComputeExponent(self.dtype_exp, self.dtype_s, self.w1, self.table1[i]),
                e=exp,
                s=s,
            )

            s, quotient = bb.add(ReduceExponent(self.dtype_s, p), s=s)

            s, r = bb.add(
                ComputeResidue(compress_dtype, self.w3a, self.w3b, g, p),
                s=s,
            )

            r, result = bb.add(
                AccumulateResidue(compress_dtype, self.dtype_acc, self.w4, self.N, self.table4[i]),
                r=r,
                acc=result,
            )

            s = bb.add(
                ComputeResidueAdjoint(compress_dtype, self.w3a, self.w3b, g, p),
                s=s,
                r=r,
            )

            s = bb.add(
                ReduceExponent(self.dtype_s, p).adjoint(),
                s=s,
                quotient=quotient,
            )

        exp, s = bb.add(
            ComputeExponent(self.dtype_exp, self.dtype_s, self.w1, self.table1[len(self.primes)]),
            e=exp,
            s=s,
        )

        bb.free(s)

        exp_last, exp_chunk = bb.add(self.computeexp_property._e_split, x=exp)
        for j in range(len(exp_chunk)):
            exp_chunk[j] = bb.add(Phaseup(self.w1), selection=exp_chunk[j])
        exp_last = bb.add(Phaseup(self.computeexp_property.last_w1), selection=exp_last)
        exp = bb.add(
            self.computeexp_property._e_split.adjoint(),
            address_last=exp_last,
            address=exp_chunk,
        )

        return {"exp": exp, "result": result}

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> BloqCountDictT:
        if is_symbolic(self.primes):
            count_p = sp.Symbol("primes")
            compress_dtype = QUInt(l_p)
            P = sp.Symbol("P")
        else:
            count_p = len(self.primes)
            compress_dtype = QUInt(int(self.primes[0]).bit_length())
            P = int(self.primes[0])
        g = sp.Symbol("g")
        counts = Counter()
        counts[ComputeExponent(self.dtype_exp, self.dtype_s, self.w1, T)] += count_p + 1
        counts[ReduceExponent(self.dtype_s, P)] += count_p
        counts[ComputeResidue(compress_dtype, self.w3a, self.w3b, g, P)] += count_p
        counts[AccumulateResidue(compress_dtype, self.dtype_acc, self.w4, self.N, T)] += count_p
        counts[ComputeResidueAdjoint(compress_dtype, self.w3a, self.w3b, g, P)] += count_p
        counts[ReduceExponent(self.dtype_s, P).adjoint()] += count_p
        counts[Phaseup(self.w1)] += self.computeexp_property.num_windows1 - 1
        counts[Phaseup(self.computeexp_property.last_w1)] += 1
        return counts
