import itertools
import attrs
from functools import cached_property
from typing import Optional
from collections.abc import Sequence
import numpy as np
import sympy as sp
from attrs import frozen
import quimb.tensor as qtn

from qualtran import (
    Bloq,
    BloqBuilder,
    CompositeBloq,
    DecomposeTypeError,
    Register,
    Signature,
    QBit,
    QDType,
    Soquet,
    SoquetT,
    Controlled,
)
from qualtran.bloqs.basic_gates import CNOT
from qualtran.drawing import Circle, ModPlus, Text, WireSymbol, TextBox
from qualtran.symbolics import SymbolicInt, is_symbolic
from qualtran.resource_counting import (
    SympySymbolAllocator,
    BloqCountDictT,
)
from qualtran.simulation.classical_sim import ClassicalValT

COPY = [1, 0, 0, 0, 0, 0, 0, 1]
COPY = np.array(COPY, dtype=np.complex128).reshape((2, 2, 2))

XOR = np.array(list(itertools.product([0, 1], repeat=3)))
XOR = 1 - np.sum(XOR, axis=1) % 2
XOR = XOR.reshape((2, 2, 2)).astype(np.complex128)


@frozen
class MultiTargetCNOT(Bloq):
    r"""Implements single control, multi-target $C[X^{\otimes n}]$ gate.

    Implements $|0><0| I + |1><1| X^{\otimes n}$ using a single CNOT gate (using lattice surgery).
    """

    bitsize: "SymbolicInt"

    @cached_property
    def signature(self) -> "Signature":
        return Signature(
            [
                Register("ctrl", QBit()),
                Register("targets", QBit(), shape=(self.bitsize,)),
            ]
        )

    def decompose_bloq(self) -> "CompositeBloq":
        raise DecomposeTypeError(f"{self} is atomic")

    def adjoint(self) -> "Bloq":
        return self

    def my_tensors(self, incoming, outgoing):

        internal = qtn.rand_uuid()
        copy = [
            qtn.Tensor(
                data=COPY,
                inds=[(incoming["ctrl"], 0), (outgoing["ctrl"], 0), internal],
                tags=["COPY"],
            )
        ]
        xors = [
            qtn.Tensor(
                data=XOR,
                inds=[
                    (incoming["targets"][j], 0),
                    (outgoing["targets"][j], 0),
                    internal,
                ],
                tags=["XOR"],
            )
            for j in range(self.bitsize)
        ]
        return copy + xors

    def on_classical_vals(self, **vals) -> dict[str, "ClassicalValT"]:
        ctrl, targets = vals["ctrl"], vals["targets"]
        targets = (np.ones(self.bitsize, dtype=np.int_) ^ targets) * ctrl + targets * (1 - ctrl)
        return {"ctrl": ctrl, "targets": targets}

    def wire_symbol(self, reg: Register | None, idx: tuple[int, ...] = tuple()) -> "WireSymbol":
        if reg is None:
            return Text("")
        if reg.name == "ctrl":
            return Circle(filled=True)
        elif reg.name == "targets":
            return ModPlus()
        raise ValueError(f"Unknown wire symbol register name: {reg.name}")

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> BloqCountDictT:
        return {CNOT(): 1}


K = sp.Symbol("K")


def generalize_CtrlXorK(b):
    if isinstance(b, CtrlXorK):
        return CtrlXorK(b.dtype, K)
    if isinstance(b, Controlled) and isinstance(b.subbloq, CtrlXorK):
        return CtrlXorK(b.subbloq.dtype, K).controlled()
    return b


@attrs.frozen
class CtrlXorK(Bloq):
    r"""Maps |ctrl>|x> to |ctrl>|x \oplus ctrl.k> for a constant k.

    Parameters
    ----------
        dtype: Data type of the input register `x`.
        k: The classical integer value to be XOR-ed to x.

    Attributes
    ----------
        ctrl : Control qubit
        x: A quantum register of type `self.dtype` (see above).
    """

    dtype: QDType
    k: SymbolicInt

    @cached_property
    def signature(self) -> "Signature":
        return Signature([Register("ctrl", QBit()), Register("x", self.dtype)])

    @cached_property
    def bitsize(self) -> SymbolicInt:
        return self.dtype.num_qubits + 1

    def is_symbolic(self):
        return is_symbolic(self.k, self.dtype)

    def adjoint(self) -> "CtrlXorK":
        return self

    @cached_property
    def _bits_k(self) -> Sequence[int]:
        return self.dtype.to_bits(self.k)

    def build_composite_bloq(  # type: ignore[override]
        self, bb: "BloqBuilder", ctrl: SoquetT, x: "Soquet"
    ) -> dict[str, "SoquetT"]:
        if self.is_symbolic():
            raise DecomposeTypeError(f"Cannot decompose symbolic {self}")

        xs = bb.split(x)
        ind_to_CNOT = []
        for i, bit in enumerate(self._bits_k):  # Store every bit which needs to be X
            if bit == 1:
                ind_to_CNOT.append(i)
        if ind_to_CNOT:
            ctrl, xs[ind_to_CNOT] = bb.add(
                MultiTargetCNOT(len(ind_to_CNOT)), ctrl=ctrl, targets=xs[ind_to_CNOT]
            )

        x = bb.join(xs, dtype=self.dtype)

        return {"ctrl": ctrl, "x": x}

    def on_classical_vals(self, **vals) -> dict[str, "ClassicalValT"]:
        ctrl, x = vals["ctrl"], vals["x"]
        if isinstance(self.k, sp.Expr):
            raise ValueError(f"Cannot classically simulate with symbolic {self.k=}")

        k: ClassicalValT = self.k
        if isinstance(x, np.integer):
            k = np.array(k, dtype=x.dtype)[()]
        return {"ctrl": ctrl, "x": x ^ k * ctrl}

    def wire_symbol(
        self, reg: Optional["Register"], idx: tuple[int, ...] = tuple()
    ) -> "WireSymbol":
        if reg is None:
            return Text("")
        elif reg.name == "ctrl":
            return TextBox("ctrl")
        else:
            return TextBox(f"⊕{self.k}.ctrl")

    def __str__(self):
        return f"ctrl_XorK({self.k})"

    def build_call_graph(self, ssa):
        if (
            self.is_symbolic()
        ):  # if the bloq is symbolic we assume for the call graph that k =2**n-1
            return {MultiTargetCNOT(self.dtype.num_qubits): 1}
        return super().build_call_graph(ssa)
