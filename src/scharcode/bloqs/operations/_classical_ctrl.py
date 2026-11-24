import attrs
import numpy as np
from functools import cached_property
import quimb.tensor as qtn

from qualtran import Bloq, Side, Register, Signature, SoquetT, BloqBuilder
from qualtran import CBit, QBit
from qualtran import CtrlSpec
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.symbolics import is_symbolic, SymbolicInt
from qualtran.resource_counting import SympySymbolAllocator, BloqCountDictT


@attrs.frozen
class Classicaly_CNOT(Bloq):
    """Classically controlled NOT.

    Usage: `ctrl, q = bb.add(classicaly_CNOT(), ctrl=ctrl, q=q)`."""

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register("ctrl", CBit()), Register("q", QBit())])

    def build_composite_bloq(
        self, bb: BloqBuilder, *, ctrl: SoquetT, q: SoquetT
    ) -> dict[str, SoquetT]:
        ctrl, q = bb.add(XGate().controlled(CtrlSpec(qdtypes=CBit(), cvs=1)), ctrl=ctrl, q=q)
        return {"ctrl": ctrl, "q": q}

    def adjoint(self) -> Bloq:
        return self


@attrs.frozen
class Mixed_toffoli(Bloq):
    """Classically controlled CNOT.

    Usage:
    ```
    c_ctrl, q_ctrl, targ = bb.add(
        mixed_toffoli(), ctrl2=c_ctrl, ctrl=q_ctrl, target=targ)
    ```
    """

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("ctrl2", CBit()),
                Register("ctrl", QBit()),
                Register("target", QBit()),
            ]
        )

    def build_composite_bloq(
        self, bb: BloqBuilder, *, ctrl2: SoquetT, ctrl: SoquetT, target: SoquetT
    ) -> dict[str, SoquetT]:
        ctrl2, ctrl, target = bb.add(
            CNOT().controlled(CtrlSpec(qdtypes=CBit(), cvs=1)),
            ctrl2=ctrl2,
            ctrl=ctrl,
            target=target,
        )
        return {"ctrl2": ctrl2, "ctrl": ctrl, "target": target}

    def adjoint(self) -> Bloq:
        return self


@attrs.frozen
class InitClassVal(Bloq):
    """Init register of n classical bits with value from b (big-ended)."""

    # Should have decomposed in the previous definitions. Treat all that as experimental.
    bitsize: SymbolicInt
    val: SymbolicInt

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("x", CBit(), shape=(self.bitsize,), side=Side.RIGHT),
            ]
        )

    def adjoint(self) -> Bloq:
        return FreeClassVal(self.bitsize, self.val)

    def my_tensors(self, incoming, outgoing):
        # We need concrete sizes to build an explicit tensor.
        if is_symbolic(self.bitsize) or is_symbolic(self.val):
            raise ValueError(
                f"{self} requires concrete (non-symbolic) bitsize and val for my_tensors()."
            )

        bits = np.array(
            list(reversed([(self.val >> i) & 1 for i in range(self.bitsize)])),
            dtype=np.int8,
        )

        # Ket tensor |bits⟩: one-hot with outgoing indices.
        if self.bitsize == 0:
            data = np.array(1.0 + 0.0j)
            outgoing_inds = []
        else:
            data = np.zeros((2,) * self.bitsize, dtype=np.complex128)
            data[tuple(int(b) for b in bits)] = 1.0 + 0.0j
            outgoing_inds = [(outgoing["x"][j], 0) for j in range(self.bitsize)]

        return [qtn.Tensor(data=data, inds=outgoing_inds, tags=[str(self)])]

    def on_classical_vals(self):
        return {
            "x": np.array(
                list(reversed([(self.val >> i) & 1 for i in range(self.bitsize)])),
                dtype=np.uint8,
            )
        }

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        return {}


@attrs.frozen
class FreeClassVal(Bloq):
    """
    Free register of "bitsize" classical bits, can be call be calling the adjoint of InitClassVal
    """

    bitsize: SymbolicInt
    val: SymbolicInt

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("x", CBit(), shape=(self.bitsize,), side=Side.LEFT),
            ]
        )

    def adjoint(self) -> Bloq:
        return InitClassVal(self.bitsize, self.val)

    def on_classical_vals(self, x=None):
        return {}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        return {}

    def my_tensors(self, incoming, outgoing):
        if is_symbolic(self.bitsize) or is_symbolic(self.val):
            raise ValueError(
                f"{self} requires concrete (non-symbolic) bitsize and val for my_tensors()."
            )

        bits = np.array(
            list(reversed([(self.val >> i) & 1 for i in range(self.bitsize)])),
            dtype=np.int8,
        )

        # Bra tensor ⟨bits|: same one-hot, but indices are incoming now.
        if self.bitsize == 0:
            data = np.array(1.0 + 0.0j)
            incoming_inds = []
        else:
            data = np.zeros((2,) * self.bitsize, dtype=np.complex128)
            data[tuple(int(b) for b in bits)] = 1.0 + 0.0j
            incoming_inds = [(incoming["x"][j], 0) for j in range(self.bitsize)]

        return [qtn.Tensor(data=data.conj(), inds=incoming_inds, tags=[str(self)])]
