from functools import cached_property

import attrs
import numpy as np

from qualtran import Bloq, BloqBuilder, Signature, Register, SoquetT
from qualtran import QBit, QUInt, CBit
from qualtran.bloqs.bookkeeping import Partition, Free, Allocate
from qualtran.bloqs.basic_gates import CNOT, Toffoli
from qualtran.resource_counting import (
    QubitCount,
    SympySymbolAllocator,
    BloqCountDictT,
    CostKey,
)
from qualtran.symbolics import SymbolicInt

from scharcode.bloqs.operations._classical_ctrl import (
    Mixed_toffoli,
    Classicaly_CNOT,
    InitClassVal,
)
from scharcode.bloqs.operations.maj_classoutplace import (
    MAJClassOutPlace,
    UMAClassOutPlace,
)
from scharcode.bloqs.operations._utils import check_bigger_than


@attrs.frozen
class QCAdderUnwrapped(Bloq):
    """Quantum - Classical Adder, modulo 2^n.

    Parameters
    ----------
    n : SymbolicInt
        Size of the register for i

    Attributes
    ----------
    i : CBit((self.n,))
        Array of QBit holding i

    t : QBit((self.n,))
        Array of QBit holding t, holds the result of i + t at the end.

    Ref: See Fig.19 of [Gouzien et al.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.131.040602)
    """

    n: SymbolicInt

    def __attrs_post_init__(self):
        if not check_bigger_than(self.n, 3):
            raise ValueError(f"n must be greater than 3, got {self.n}.")

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("i", CBit(), shape=(self.n,)),
                Register("t", QBit(), shape=(self.n,)),
            ]
        )

    @cached_property
    def maj_class_outplace(self) -> MAJClassOutPlace:
        return MAJClassOutPlace()

    @cached_property
    def uma_class_outplace(self) -> UMAClassOutPlace:
        return UMAClassOutPlace()

    def build_composite_bloq(
        self, bb: BloqBuilder, *, i: SoquetT, t: SoquetT
    ) -> dict[str, SoquetT]:
        # Careful, qualtran is big indian
        all_carry = np.empty(self.n, dtype=object)

        # Least significant bit MAJ.
        all_carry[-1] = bb.allocate(1, QBit())
        i[-1], t[-1], all_carry[-1] = bb.add(
            Mixed_toffoli(), ctrl2=i[-1], ctrl=t[-1], target=all_carry[-1]
        )

        # All the MAJ.
        for j in reversed(range(2, self.n - 1)):
            all_carry[j + 1], i[j], t[j], all_carry[j] = bb.add(
                self.maj_class_outplace, c=all_carry[j + 1], i=i[j], t=t[j]
            )

        # Most significant bits handling.
        i[1], t[1] = bb.add(Classicaly_CNOT(), ctrl=i[1], q=t[1])
        i[1], t[0] = bb.add(Classicaly_CNOT(), ctrl=i[1], q=t[0])
        i[1], t[1], t[0] = bb.add(Mixed_toffoli(), ctrl2=i[1], ctrl=t[1], target=t[0])
        (all_carry[2], t[1]), t[0] = bb.add(Toffoli(), ctrl=(all_carry[2], t[1]), target=t[0])
        i[0], t[0] = bb.add(Classicaly_CNOT(), ctrl=i[0], q=t[0])
        all_carry[2], t[1] = bb.add(CNOT(), ctrl=all_carry[2], target=t[1])

        # UMAs
        for j in range(2, self.n - 1):
            all_carry[j + 1], i[j], t[j] = bb.add(
                self.uma_class_outplace,
                c=all_carry[j + 1],
                i=i[j],
                t=t[j],
                c2=all_carry[j],
            )

        # Least significant bit UMA.
        i[-1], t[-1], all_carry[-1] = bb.add(
            Mixed_toffoli(), ctrl2=i[-1], ctrl=t[-1], target=all_carry[-1]
        )
        bb.free(all_carry[-1])
        i[-1], t[-1] = bb.add(Classicaly_CNOT(), ctrl=i[-1], q=t[-1])

        return {"i": i, "t": t}

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> BloqCountDictT:
        return {
            self.maj_class_outplace: self.n - 3,
            self.uma_class_outplace: self.n - 3,
            Classicaly_CNOT(): 4,
            Mixed_toffoli(): 3,
            CNOT(): 1,
            Toffoli(): 1,
            Allocate(QBit()): 1,
            Free(QBit()): 1,
        }

    def my_static_costs(self, cost_key: CostKey):
        if isinstance(cost_key, QubitCount):
            return 2 * self.n - 2
        return NotImplemented


@attrs.frozen
class QCAdder(Bloq):
    r"""Quantum - Classical Adder, modulo 2^n.

    Implements:
    $$\ket{t} -> \ket{i + t \mod 2^n}$$
    With i some integer

    In the paper QCAdder are counted as QQAdder

    Parameters
    ----------
    n : SymbolicInt
        Size of the register for x
    i : SymbolicInt
        added number

    Attributes
    ----------
    t : QUInt(self.n) -> QUInt(self.n)
        QUInt holding t, holds the result of i + t mod 2^n at the end.

    Ref: See Fig.19 of [Gouzien et al.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.131.040602)
    """

    n: SymbolicInt
    i: SymbolicInt

    def __attrs_post_init__(self):
        if not check_bigger_than(self.n, 3):
            raise ValueError(f"n must be greater than 3, got {self.n}.")

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register("t", QUInt(self.n))])

    @cached_property
    def _t_split(self) -> Partition:
        regs = (Register("t", QBit(), shape=(self.n,)),)
        return Partition(self.n, regs)

    @cached_property
    def _init_i(self) -> InitClassVal:
        return InitClassVal(bitsize=self.n, val=self.i)

    @cached_property
    def adder_unwrapped(self) -> QCAdderUnwrapped:
        return QCAdderUnwrapped(self.n)

    def build_composite_bloq(self, bb: BloqBuilder, *, t: SoquetT) -> dict[str, SoquetT]:
        t = bb.add(self._t_split, x=t)
        i = bb.add(self._init_i)

        i, t_out = bb.add(self.adder_unwrapped, i=i, t=t)

        t_out = bb.add(self._t_split.adjoint(), t=t_out)
        bb.add(self._init_i.adjoint(), x=i)
        return {"t": t_out}

    def on_classical_vals(self, t):
        return {"t": (self.i + t) % 2**self.n}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        return {
            self._t_split: 1,
            self._t_split.adjoint(): 1,
            self.adder_unwrapped: 1,
        }

    def my_static_costs(self, cost_key: CostKey):
        if isinstance(cost_key, QubitCount):
            return 2 * self.n - 2
        return NotImplemented
