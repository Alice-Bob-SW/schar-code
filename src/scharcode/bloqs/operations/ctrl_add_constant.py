import attrs
from functools import cached_property

from qualtran import (
    Bloq,
    BloqBuilder,
    Signature,
    Register,
    Soquet,
)
from qualtran import QUInt, QAny, QBit
from qualtran.symbolics import SymbolicInt

from scharcode.bloqs.operations.multi_target_cnot import CtrlXorK
from scharcode.bloqs.operations.qq_adder import QQAdder


@attrs.frozen
class ControlledConstantAdd(Bloq):
    """
    Correspond to loading a temporary register with a MultiTargetCNOT and adding that register
    to another. The temporary register is then uncomputed by repeation the MultiTargetCNOT.

    Parameters
    ----------
    dtype_target : QUInt | QAny
        dtype of the register holding exponent
    val : SymbolicInt
        value to potentially load
    add : bool
        wether to add or substract val to the target register


    Attributes
    ----------
    ctrl : QBit()
        control qubit, control the loading of val
    target : dtype_target
        register holding the final result

    Ref: Fig 28 (top) last 3 operations
    [Shor's algorithm compiled on cat qubits: 22 593 qubits for RSA-2048 in 3.14 days], (https://arxiv.org/abs/2610.37839)
    """

    dtype_target: QAny | QUInt = attrs.field()
    val: "SymbolicInt"
    add: bool = True

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register("ctrl", QBit()), Register("target", self.dtype_target)])

    @cached_property
    def operation(self):
        if self.add:
            return QQAdder(self.dtype_target.bitsize)
        else:
            return QQAdder(self.dtype_target.bitsize).adjoint()

    def build_composite_bloq(self, bb: BloqBuilder, ctrl: Soquet, target: Soquet):
        temp_reg = bb.allocate(self.dtype_target.bitsize, self.dtype_target)
        ctrl, temp_reg = bb.add(CtrlXorK(self.dtype_target, self.val), ctrl=ctrl, x=temp_reg)
        temp_reg, target = bb.add(self.operation, i=temp_reg, t=target)
        ctrl, temp_reg = bb.add(CtrlXorK(self.dtype_target, self.val), ctrl=ctrl, x=temp_reg)
        bb.free(temp_reg)
        return {"ctrl": ctrl, "target": target}

    def on_classical_vals(self, **vals):
        ctrl, target = vals["ctrl"], vals["target"]
        if self.add:
            target = (target + self.val * ctrl) % 2 ** (self.dtype_target.bitsize)
        else:
            target = (target - self.val * ctrl) % 2 ** (self.dtype_target.bitsize)
        return {"ctrl": ctrl, "target": target}

    def build_call_graph(self, ssa):
        return {CtrlXorK(self.dtype_target, self.val): 2, self.operation: 1}
