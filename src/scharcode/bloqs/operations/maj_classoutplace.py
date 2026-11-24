from functools import cached_property
import attrs

from qualtran import Bloq, BloqBuilder, Signature, Register, SoquetT, Side
from qualtran import QBit, CBit
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.basic_gates import CNOT

from scharcode.bloqs.operations._classical_ctrl import Classicaly_CNOT, Mixed_toffoli


@attrs.frozen
class MAJClassOutPlace(Bloq):
    """Out of place semi-classical MAJ.

    Attributes
    ----------
        c : QBit()
            Previous carry
        i : CBit()
            Cbit of the added number
        t : QBit()
            Qubit of the number on which we add
        c2: QBit()
            Qubit() containing the new carry.
    """

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("c", QBit()),
                Register("i", CBit()),
                Register("t", QBit()),
                Register("c2", QBit(), side=Side.RIGHT),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, *, c, i, t) -> dict[str, SoquetT]:
        i, t = bb.add(Classicaly_CNOT(), ctrl=i, q=t)
        (c, t), c2 = bb.add(And(), ctrl=(c, t))
        i, t = bb.add(Classicaly_CNOT(), ctrl=i, q=t)
        i, t, c2 = bb.add(Mixed_toffoli(), ctrl2=i, ctrl=t, target=c2)
        return {"c": c, "i": i, "t": t, "c2": c2}


@attrs.frozen
class UMAClassOutPlace(Bloq):
    """Out of place semi-classical UMA.

    Attributes
    ----------
        c : QBit()
            Previous carry
        i : CBit()
            Cbit of the added number
        t : QBit()
            Qubit of the number on which we add
        c2: QBit()
            Qubit() containing the new carry.
    """

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("c", QBit()),
                Register("i", CBit()),
                Register("t", QBit()),
                Register("c2", QBit(), side=Side.LEFT),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, *, c, i, t, c2) -> dict[str, SoquetT]:
        i, t, c2 = bb.add(Mixed_toffoli(), ctrl2=i, ctrl=t, target=c2)
        i, t = bb.add(Classicaly_CNOT(), ctrl=i, q=t)
        c, t = bb.add(And().adjoint(), ctrl=(c, t), target=c2)
        c, t = bb.add(CNOT(), ctrl=c, target=t)
        return {"c": c, "i": i, "t": t}
