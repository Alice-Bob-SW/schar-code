import attrs
from functools import cached_property

from qualtran import Register, Bloq, Signature, QAny
from qualtran.bloqs.basic_gates import CZ
from qualtran.bloqs.mcmt.and_bloq import And
from qualtran.symbolics import SymbolicInt
from qualtran.resource_counting import QubitCount


@attrs.frozen
class Phaseup(Bloq):
    """
    Resource efficient uncompute of a lookup table.
    This version assumes the target register was measured earlier and all phase fix needed
    as well as potential additional phase fix are pushed here.
    This bloq should thus have measurement result has input, but handling this and the computation
    of the resulting phase fix in the current version of qualtran would be too painful.
    Therefore, this bloq does not implement a build_decompose_bloq, but gives all resource estimate
    and the classical action.

    #! This bloq does not have any significant physical action
    #! and only serve as a placeholder for ressource estimation.

    Parameters
    ----------
    w : SymbolicInt
        Size of the selection register

    Attributes
    ----------
        selection: QBit((self.w,))

    ref: Gidney [How to factor 2048 bit RSA integers with less than a million noisy qubits, A.3, https://arxiv.org/abs/2505.15917]
    """

    w: SymbolicInt

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register("selection", QAny(self.w))])

    def on_classical_vals(self, selection):
        return {"selection": selection}

    def build_call_graph(self, ssa):
        h1 = self.w // 2
        h2 = self.w - h1
        return {
            And(): 2**h1 + 2**h2 - h1 - h2 - 2,
            And().adjoint(): 2**h1 + 2**h2 - h1 - h2 - 2,
            CZ(): 2**h1 - 1,
            # We don't count 1 qubit gate, should be 2**w - 1
        }

    def my_static_costs(self, cost_key):
        if isinstance(cost_key, QubitCount):
            h1 = self.w // 2
            h2 = self.w - h1
            return 2**h1 + 2**h2 - 2
        return NotImplemented
