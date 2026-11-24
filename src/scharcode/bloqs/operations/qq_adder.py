from functools import cached_property
import attrs
import numpy as np

from qualtran import Bloq, BloqBuilder, Signature, Register, SoquetT
from qualtran import QBit, QUInt
from qualtran.bloqs.bookkeeping import Partition
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.mcmt import And
from qualtran.resource_counting import (
    QubitCount,
    SympySymbolAllocator,
    BloqCountDictT,
    CostKey,
)
from qualtran.symbolics import SymbolicInt

from scharcode.bloqs.operations._utils import check_bigger_than
from scharcode.bloqs.operations.maj_outplace import MAJOutPlace, UMAOutPlace


@attrs.frozen
class QQAdderAbstract(Bloq):
    """
    Abstract action of an adder circuit mod 2^n
    Circuit inheriting this class should implement |i>|t> -> |i>|t+i mod 2^n>

    This Bloq can be used as an "auto-wrapper" for QQAdder by using this class as the parent
    and overriding adder_unwrapped with the adder you want to wrap

    Parameters
    ----------
    n : SymbolicInt
        Size of the register for i

    Attributes
    ----------
    i : QUInt(self.n)
        Array of QBit holding i

    t : QUInt(self.n)
        Array of QBit holding t, holds the result of i + t mod 2^n at the end.
    """

    n: SymbolicInt

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("i", QUInt(self.n)),
                Register("t", QUInt(self.n)),
            ]
        )

    @cached_property
    def _i_split(self) -> Partition:
        regs = (Register("i", QBit(), shape=(self.n,)),)
        return Partition(self.n, regs)

    @cached_property
    def _t_split(self) -> Partition:
        regs = (Register("t", QBit(), shape=(self.n,)),)
        return Partition(self.n, regs)

    @cached_property
    def adder_unwrapped(self):
        raise NotImplementedError(f"{self} must overwrite this with its own adder_unwrapped")

    def build_composite_bloq(
        self, bb: BloqBuilder, *, i: SoquetT, t: SoquetT
    ) -> dict[str, SoquetT]:
        i = bb.add(self._i_split, x=i)
        t = bb.add(self._t_split, x=t)
        i, t = bb.add_from(self.adder_unwrapped, i=i, t=t)
        i = bb.add(self._i_split.adjoint(), i=i)
        t = bb.add(self._t_split.adjoint(), t=t)
        return {"i": i, "t": t}

    def on_classical_vals(self, **vals):
        i, t = vals["i"], vals["t"]
        return {"i": i, "t": (i + t) % 2**self.n}

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> BloqCountDictT:
        return {
            self._i_split: 1,
            self._i_split.adjoint(): 1,
            self._t_split: 1,
            self._t_split.adjoint(): 1,
            self.adder_unwrapped: 1,
        }


@attrs.frozen
class QQAdderUnwrapped(Bloq):
    """
    Quantum - Quantum Adder
    Implements |i>|t> -> |i>|i+t mod 2^n>

    Reimplemented since Qualtran's version does not use MultiTargetCNOT

    Parameters
    ----------
    n : SymbolicInt
        Size of the register for i

    Attributes
    ----------
    i : QBit((self.n,))
        Array of QBit holding i

    t : QBit((self.n,))
        Array of QBit holding t, holds the result of t + i mod 2^n at the end.

    Ref: The circuit implemented as in Fig.1 of [Gidney](https://quantum-journal.org/papers/q-2018-06-18-74/)
    Note that the resource estimation uses the logical implementation of the algorithm at the level
    of operations.
    Then the physical implementation of those operations are used to obtain the resource estimate.
    Thus the exact logical implementation of the Adder does not affect the number in the resource
    estimate.
    """

    n: SymbolicInt

    def __attrs_post_init__(self):
        if not check_bigger_than(self.n, 2):
            raise ValueError(f"n must be greater than 2, got {self.n}.")

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("i", QBit(), shape=(self.n,)),
                Register("t", QBit(), shape=(self.n,)),
            ]
        )

    @cached_property
    def maj_outplace(self) -> MAJOutPlace:
        return MAJOutPlace()

    @cached_property
    def uma_outplace(self) -> UMAOutPlace:
        return UMAOutPlace()

    def build_composite_bloq(
        self, bb: BloqBuilder, *, i: SoquetT, t: SoquetT
    ) -> dict[str, SoquetT]:
        # Careful, qualtran is big indian
        all_carry = np.empty(self.n - 1, dtype=object)

        # Least significant bit MAJ.
        (i[-1], t[-1]), all_carry[-1] = bb.add(And(), ctrl=(i[-1], t[-1]))

        # All the MAJ.
        for j in reversed(range(1, self.n - 1)):
            all_carry[j], i[j], t[j], all_carry[j - 1] = bb.add(
                self.maj_outplace, c=all_carry[j], i=i[j], t=t[j]
            )

        # Most significant bit handling.
        all_carry[0], t[0] = bb.add(CNOT(), ctrl=all_carry[0], target=t[0])
        i[0], t[0] = bb.add(CNOT(), ctrl=i[0], target=t[0])

        # UMAs , with special name for the last one.
        for j in range(1, self.n - 1):
            all_carry[j], i[j], t[j] = bb.add(
                self.uma_outplace, c=all_carry[j], i=i[j], t=t[j], c2=all_carry[j - 1]
            )

        # Least significant bit UMA.
        i[-1], t[-1] = bb.add(And().adjoint(), ctrl=(i[-1], t[-1]), target=all_carry[-1])
        i[-1], t[-1] = bb.add(CNOT(), ctrl=i[-1], target=t[-1])

        return {"i": i, "t": t}

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> BloqCountDictT:
        return {
            self.maj_outplace: self.n - 2,
            self.uma_outplace: self.n - 2,
            And(): 1,
            And().adjoint(): 1,
            CNOT(): 3,
        }

    def my_static_costs(self, cost_key: CostKey):
        if isinstance(cost_key, QubitCount):
            return 3 * self.n - 1
        return NotImplemented


@attrs.frozen
class QQAdder(QQAdderAbstract):
    """
    Quantum - Quantum Adder
    Implements |i>|t> -> |i>|i+t mod 2^n>

    Reimplemented since Qualtran's version does not use MultiTargetCNOT

    Parameters
    ----------
    n : SymbolicInt
        Size of the register for i

    Attributes
    ----------
    i : QUInt(self.n)
        Array of QBit holding i

    t : QUInt(self.n)
        Array of QBit holding t, holds the result of i + t mod 2^n at the end.
    """

    @cached_property
    def adder_unwrapped(self):
        return QQAdderUnwrapped(self.n)

    def adjoint(self):
        return QQAdderSub(self.n)

    def my_static_costs(self, cost_key: CostKey):
        if isinstance(cost_key, QubitCount):
            return 3 * self.n - 1
        return NotImplemented


@attrs.frozen
class QQAdderSub(QQAdderAbstract):
    """
    Quantum - Quantum Adder (substraction)
    Implements |i>|t> -> |i>|t-i mod 2^n>

    Wrapper for the adjoint of QQAdder (speeds up call_graph/static_cost which could be obtain via
    decomposition otherwise)

    Parameters
    ----------
    n : SymbolicInt
        Size of the register for i

    Attributes
    ----------
    i : QUInt(self.n)
        Array of QBit holding i

    t : QUInt(self.n)
        Array of QBit holding t, holds the result of t - i mod 2^n at the end.
    """

    @cached_property
    def adder_unwrapped(self):
        return QQAdderUnwrapped(self.n).adjoint()

    def adjoint(self):
        return QQAdder(self.n)

    def on_classical_vals(self, **vals):
        i, t = vals["i"], vals["t"]
        return {"i": i, "t": (t - i) % 2**self.n}

    def my_static_costs(self, cost_key: CostKey):
        if isinstance(cost_key, QubitCount):
            return 3 * self.n - 1
        return NotImplemented
