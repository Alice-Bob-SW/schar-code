import attrs
from functools import cached_property
from collections import Counter

from qualtran import (
    Bloq,
    BloqBuilder,
    Signature,
    Register,
    Soquet,
)
from qualtran import QUInt
from qualtran.bloqs.basic_gates import ZGate
from qualtran.symbolics import SymbolicInt

from scharcode.bloqs.operations.half_compare import HalfCompare
from scharcode.bloqs.operations.lookup import Lookup
from scharcode.bloqs.operations._utils import tuple_or_symbolic, T


@attrs.frozen
class UnderflowPhasefix(Bloq):
    """
    Implement the phase fixing for measurement in the X basis of the underflow qubit

    Parameters
    ----------
    dtype_reg : QUInt
        dtype of the register which needs phase correction
    w : SymbolicInt
        window size for this method
    table : tuple[SymbolicInt]
        precomputed table to compare to reg for the phasefix

    Attributes
    ----------
    address : QUInt(self.w)
        register holding the address for the comparison table
    reg : dtype_reg
        register which needs phase correction

    ref: Fig. 31 [Shor's algorithm compiled on cat qubits: 22 593 qubits for RSA-2048 in 3.14 days]
    (https://arxiv.org/abs/2610.37839)
    """

    dtype_reg: QUInt
    w: "SymbolicInt"
    table: tuple[SymbolicInt] = attrs.field(converter=tuple_or_symbolic)

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("address", QUInt(self.w)),
                Register("reg", self.dtype_reg),
            ]
        )

    def lookup_phase_fix(self, table) -> Lookup:
        return Lookup(table, self.w, self.dtype_reg.bitsize)

    def build_composite_bloq(self, bb: BloqBuilder, address: Soquet, reg: Soquet):
        address, target_lookup = bb.add(self.lookup_phase_fix(self.table), selection=address)
        target_lookup, reg, comp_res, junk = bb.add(
            HalfCompare(self.dtype_reg, self.dtype_reg), i=target_lookup, t=reg
        )

        comp_res = bb.add(ZGate(), q=comp_res)

        target_lookup, reg = bb.add(
            HalfCompare(self.dtype_reg, self.dtype_reg).adjoint(),
            i=target_lookup,
            t=reg,
            target=comp_res,
            junk=junk,
        )
        # ? needed because no deffered phase correction in the version
        address = bb.add(
            self.lookup_phase_fix(self.table).adjoint(),
            selection=address,
            target=target_lookup,
        )
        return {"address": address, "reg": reg}

    def on_classical_vals(self, **vals):
        return vals

    def basis_state_phase(self, **vals):
        address, reg = vals["address"], vals["reg"]
        if reg >= self.table[address]:
            return -1
        return 1

    def build_call_graph(self, ssa):
        counts = Counter()
        counts[self.lookup_phase_fix(T)] += 1
        counts[self.lookup_phase_fix(T).adjoint()] += 1
        counts[HalfCompare(self.dtype_reg, self.dtype_reg)] += 1
        counts[HalfCompare(self.dtype_reg, self.dtype_reg).adjoint()] += 1
        counts[ZGate()] += 1
        return counts
