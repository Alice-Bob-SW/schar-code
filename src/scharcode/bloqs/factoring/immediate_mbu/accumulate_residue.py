import attrs
from functools import cached_property
from collections import Counter
import numpy as np
import sympy as sp

from qualtran import (
    Bloq,
    BloqBuilder,
    Signature,
    Register,
    Soquet,
    CtrlSpec,
)
from qualtran import QUInt, QAny, CBit
from qualtran.bloqs.basic_gates import MeasureX, Discard
from qualtran.bloqs.bookkeeping import Partition
from qualtran.resource_counting import (
    SympySymbolAllocator,
    BloqCountDictT,
    QubitCount,
)
from qualtran.symbolics import SymbolicInt, is_symbolic

from scharcode.bloqs.operations.lookup import Lookup
from scharcode.bloqs.operations._utils import tuple_or_symbolic, T
from scharcode.bloqs.operations.qq_adder import QQAdder
from scharcode.bloqs.operations.ctrl_add_constant import ControlledConstantAdd
from scharcode.bloqs.factoring.immediate_mbu.underflow_phasefix import UnderflowPhasefix
from scharcode.bloqs.factoring._common import last_window, num_windows, helper_partition


@attrs.frozen
class AccumulateResidue(Bloq):
    """
    Implement an Approximate of |r>|acc> -> |r>|acc + r_i*U_i mod truncated_modulus>
    with U_i = (L // p_i) * inverse(L // p_i mod p_i), and L the product of all p_j

    Parameters
    ----------
    dtype_r : QUInt
        dtype of the register holding the residue to add to the accumulator
    dtype_acc : QUInt
        dtype of the register holding the accumulator, its size also define the truncated_modulus
    w4 : SymbolicInt
        window size for this method
    modulus : SymbolicInt
        here, modulus of the accumulated modular exponentiation, in RSA, it is the number to factor
    table4 : tuple[tuple[SymbolicInt]]
        precomputed table,
        table4[i,j,k] = -(((U_i << (j*w4)) * k) mod L mod N) >> dropped_bits) mod trunc

    Attributes
    ----------
    r : dtype_r
        register holding the residue to add to the accumulator
    acc : dtype_acc
        register holding the accumulator

    ref: Fig. 32 [Shor's algorithm compiled on cat qubits: 22 593 qubits for RSA-2048 in 3.14 days]
    (https://arxiv.org/abs/2610.37839)
    """

    dtype_r: QUInt
    dtype_acc: QUInt
    w4: "SymbolicInt"
    modulus: "SymbolicInt"
    table4: tuple[SymbolicInt] = attrs.field(converter=tuple_or_symbolic)

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("r", self.dtype_r),
                Register("acc", self.dtype_acc),
            ]
        )

    @cached_property
    def kappa(self) -> int:
        return self.dtype_acc.bitsize

    @cached_property
    def trunc_modulus(self):
        if is_symbolic(self.modulus):
            return sp.Symbol("f")
        dropped_bits = max(0, self.modulus.bit_length() - self.kappa)
        return self.modulus >> dropped_bits

    @cached_property
    def num_windows4(self) -> int:
        return num_windows(self.dtype_r.bitsize, self.w4)

    @cached_property
    def last_w4(self) -> int:
        return last_window(self.dtype_r.bitsize, self.w4)

    @cached_property
    def _r_split(self) -> Partition:
        regs = (
            Register("r_last", QAny(self.last_w4)),
            Register("r_chunk", QAny(self.w4), shape=(self.num_windows4 - 1,)),
        )
        return Partition(self.dtype_r.bitsize, regs)

    @cached_property
    def table4b(self):
        all_table = []
        for table in self.table4:
            new_table = []
            for t in table:
                new_table.append(self.trunc_modulus - t)
            all_table.append(new_table)
        return all_table

    def lookup4_add_acc(self, table, w4_loc) -> Lookup:
        return Lookup(table, w4_loc, self.kappa + 1)

    def build_composite_bloq(self, bb: BloqBuilder, r: Soquet, acc: Soquet):
        r_last, r_chunk = bb.add(self._r_split, x=r)
        r_chunk = np.concatenate([[r_last], r_chunk])
        for j in range(self.num_windows4):
            if j == 0:
                w4_loc = self.last_w4
            else:
                w4_loc = self.w4

            r_chunk[j], target_lookup4a = bb.add(
                self.lookup4_add_acc(self.table4[j], w4_loc), selection=r_chunk[j]
            )

            wrap_around = bb.allocate(1)
            acc = bb.add(helper_partition(self.kappa).adjoint(), u=wrap_around, h=acc)
            target_lookup4a, acc = bb.add(
                QQAdder(self.kappa + 1).adjoint(),
                i=target_lookup4a,
                t=acc,
            )

            r_chunk[j] = bb.add(
                self.lookup4_add_acc(self.table4[j], w4_loc).adjoint(),
                selection=r_chunk[j],
                target=target_lookup4a,
            )
            wrap_around, acc = bb.add(helper_partition(self.kappa), x=acc)
            wrap_around, acc = bb.add(
                ControlledConstantAdd(self.dtype_acc, self.trunc_modulus),
                ctrl=wrap_around,
                target=acc,
            )
            meas_wrap = bb.add(MeasureX(), q=wrap_around)

            if_meas_wrap = CtrlSpec(CBit())
            meas_wrap, r_chunk[j], acc = bb.add(
                UnderflowPhasefix(self.dtype_acc, w4_loc, self.table4b[j]).controlled(if_meas_wrap),
                ctrl=meas_wrap,
                address=r_chunk[j],
                reg=acc,
            )

            bb.add(Discard(), c=meas_wrap)

        r = bb.add(
            self._r_split.adjoint(),
            r_last=r_chunk[0],
            r_chunk=r_chunk[1:],
        )
        return {"r": r, "acc": acc}

    def on_classical_vals(self, **vals):
        r, acc = vals["r"], vals["acc"]
        r_bin = self.dtype_r.to_bits(r)
        start_ind = 0
        for j in range(self.num_windows4):
            if j == 0:
                w4_loc = self.last_w4
            else:
                w4_loc = self.w4

            address_bin = r_bin[start_ind : start_ind + w4_loc]
            start_ind += w4_loc
            address = self.dtype_r.from_bits(address_bin)
            acc = (acc - self.table4[j][address]) % 2 ** (self.kappa + 1)
            if acc >> self.kappa:
                acc = (acc + self.trunc_modulus) % 2 ** (self.kappa)
        return {"r": r, "acc": acc}

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> BloqCountDictT:
        counts = Counter()
        counts[self.lookup4_add_acc(T, self.w4)] += self.num_windows4 - 1
        counts[self.lookup4_add_acc(T, self.w4).adjoint()] += self.num_windows4 - 1
        counts[QQAdder(self.kappa + 1).adjoint()] += self.num_windows4
        counts[ControlledConstantAdd(self.dtype_acc, self.trunc_modulus)] += self.num_windows4
        counts[MeasureX()] += self.num_windows4
        counts[Discard()] += self.num_windows4
        # The UnderflowPhasefix counts are *expected* counts, they are classically controlled
        # and happen on average half of the time
        counts[UnderflowPhasefix(self.dtype_acc, self.w4, T)] += (
            self.num_windows4 - 1
        ) * sp.Rational(1, 2)

        # Counting the Lookup and UnderflowPhasefix when self.dtype_r.bitsize%w4 != 0
        counts[self.lookup4_add_acc(T, self.last_w4)] += 1
        counts[self.lookup4_add_acc(T, self.last_w4).adjoint()] += 1
        counts[UnderflowPhasefix(self.dtype_acc, self.last_w4, T)] += sp.Rational(1, 2)  # expected
        return counts

    # Qubitcount from decomposition is wrong in qualtran here,
    # for some reason it initialize all qrom target first (and underflow qubit)
    def my_static_costs(self, cost_key):
        if isinstance(cost_key, QubitCount):
            return self.dtype_r.bitsize + 3 * self.kappa
        return NotImplemented
