import attrs
from collections import Counter
from functools import cached_property

from qualtran import (
    Bloq,
    BloqBuilder,
    Signature,
    Register,
    Soquet,
)
from qualtran import QUInt, QAny
from qualtran.bloqs.bookkeeping import Partition
from qualtran.resource_counting import (
    SympySymbolAllocator,
    BloqCountDictT,
    QubitCount,
)
from qualtran.symbolics import SymbolicInt

from scharcode.bloqs.operations.lookup import Lookup
from scharcode.bloqs.operations._utils import tuple_or_symbolic, T
from scharcode.bloqs.operations.qq_adder import QQAdder
from scharcode.bloqs.factoring._common import last_window, num_windows


@attrs.frozen
class ComputeExponent(Bloq):
    """
    Offsets s to the discrete log of the next residue.
    Note that this version does not delay and merge phase correction, and thus uses
    (|P|+1)·⌈n_e/w1⌉ Phaseups instead of ⌈n_e/w1⌉.

    Parameters
    ----------
    dtype_e : QUInt
        dtype of the exponent register
    dtype_s : QUInt
        dtype of the register holding the s to offset
    w1 : SymbolicInt
        window size for this method
    table1 : tuple[tuple[SymbolicInt]]
        precomputed table


    Attributes
    ----------
    e : dtype_e
        register holding the exponent, here it serves as the address of the offset value
    s : dtype_s
        register holding the s to offset

    ref: Fig. 27 [Shor's algorithm compiled on cat qubits: 22 593 qubits for RSA-2048 in 3.14 days]
    (https://arxiv.org/abs/2610.37839)
    """

    dtype_e: QUInt
    dtype_s: QUInt
    w1: "SymbolicInt"
    table1: tuple[SymbolicInt] = attrs.field(converter=tuple_or_symbolic)

    @cached_property
    def signature(self) -> "Signature":
        return Signature([Register("e", self.dtype_e), Register("s", self.dtype_s)])

    @property
    def num_windows1(self) -> int:
        return num_windows(self.dtype_e.bitsize, self.w1)

    @property
    def last_w1(self) -> int:
        return last_window(self.dtype_e.bitsize, self.w1)

    def qrom(self, table, w) -> Lookup:
        return Lookup(table, w, self.dtype_s.bitsize)

    @cached_property
    def _e_split(self) -> Partition:
        regs = (
            Register("address_last", QAny(self.last_w1)),
            Register("address", QAny(self.w1), shape=(self.num_windows1 - 1,)),
        )
        return Partition(self.dtype_e.bitsize, regs)

    def build_composite_bloq(self, bb: BloqBuilder, e: Soquet, s: Soquet):
        address_last, address = bb.add(self._e_split, x=e)
        for j in range(1, self.num_windows1):
            address[j - 1], offset = bb.add(
                self.qrom(self.table1[j], self.w1), selection=address[j - 1]
            )
            offset, s = bb.add(QQAdder(self.dtype_s.bitsize), i=offset, t=s)
            bb.free(offset, dirty=True)  # Should measure in X basis + phase correction pushed later

        # last iteration is done separately to deal with potentially smaller window size
        address_last, offset = bb.add(
            self.qrom(self.table1[0], self.last_w1),
            selection=address_last,
        )
        offset, s = bb.add(QQAdder(self.dtype_s.bitsize), i=offset, t=s)
        address_last = bb.add(
            self.qrom(self.table1[0], self.last_w1).adjoint(),
            selection=address_last,
            target=offset,
        )

        e = bb.add(
            self._e_split.adjoint(),
            address_last=address_last,
            address=address,
        )
        return {"e": e, "s": s}

    def on_classical_vals(self, **vals):
        e, s = vals["e"], vals["s"]
        e_bin = QUInt(self.dtype_e.bitsize).to_bits(e)
        modulus = 1 << self.dtype_s.bitsize
        for j in range(self.num_windows1 - 1):
            address_bin = e_bin[self.last_w1 + j * self.w1 : self.last_w1 + (j + 1) * self.w1]
            address = QUInt(self.w1).from_bits(address_bin)
            s += self.table1[j + 1][address]
        address_bin = e_bin[: self.last_w1]
        address = QUInt(self.last_w1).from_bits(address_bin)
        s += self.table1[0][address]
        s = s % modulus
        return {"e": e, "s": s}

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> BloqCountDictT:
        counts = Counter()
        counts[QQAdder(self.dtype_s.bitsize)] += self.num_windows1
        counts[self.qrom(T, self.w1)] += self.num_windows1 - 1
        counts[self.qrom(T, self.last_w1)] += 1
        return counts

    def my_static_costs(self, cost_key):
        if isinstance(cost_key, QubitCount):
            return 3 * self.dtype_s.bitsize + self.dtype_e.bitsize
        return NotImplemented
