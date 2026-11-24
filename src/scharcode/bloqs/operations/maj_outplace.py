from functools import cached_property
import attrs

from qualtran import Bloq, BloqBuilder, Signature, Register, SoquetT, Side
from qualtran import QBit
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.basic_gates import CNOT

from scharcode.bloqs.operations.multi_target_cnot import MultiTargetCNOT


@attrs.frozen
class MAJOutPlace(Bloq):
    """Out of place MAJ, as in Gidney's article.

    Attributes
    ----------
        c : QBit()
            Previous carry
        i : QBit()
            Qubit of the added number
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
                Register("i", QBit()),
                Register("t", QBit()),
                Register("c2", QBit(), side=Side.RIGHT),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, *, c, i, t) -> dict[str, SoquetT]:
        c, (i, t) = bb.add(MultiTargetCNOT(2), ctrl=c, targets=(i, t))
        (i, t), c2 = bb.add(And(), ctrl=(i, t))
        c, c2 = bb.add(CNOT(), ctrl=c, target=c2)
        return {"c": c, "i": i, "t": t, "c2": c2}


@attrs.frozen
class UMAOutPlace(Bloq):
    """Out of place UMA, as in Gidney's article.

    Attributes
    ----------
        c : QBit()
            Previous carry
        i : QBit()
            Qubit of the added number
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
                Register("i", QBit()),
                Register("t", QBit()),
                Register("c2", QBit(), side=Side.LEFT),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, *, c, i, t, c2) -> dict[str, SoquetT]:
        c, c2 = bb.add(CNOT(), ctrl=c, target=c2)
        i, t = bb.add(And().adjoint(), ctrl=(i, t), target=c2)
        c, i = bb.add(CNOT(), ctrl=c, target=i)
        i, t = bb.add(CNOT(), ctrl=i, target=t)
        return {"c": c, "i": i, "t": t}
