from functools import cached_property
import attrs
import numpy as np

from qualtran import (
    Bloq,
    BloqBuilder,
    Signature,
    Register,
    Side,
    SoquetT,
    Soquet,
)
from qualtran import QBit, QUInt, QInt
from qualtran.bloqs.bookkeeping import Partition
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.bloqs.mcmt import And
from qualtran.resource_counting import (
    QubitCount,
    CostKey,
)
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics import SymbolicInt

from scharcode.bloqs.operations._utils import check_equal


@attrs.frozen
class _downBloq(Bloq):
    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("old_carry", QBit()),
                Register("i_j", QBit()),
                Register("t_j", QBit()),
                Register("new_carry", QBit(), side=Side.RIGHT),
            ]
        )

    def build_composite_bloq(
        self, bb: BloqBuilder, *, old_carry: Soquet, i_j: Soquet, t_j: Soquet
    ) -> dict[str, SoquetT]:
        i_j, t_j = bb.add(CNOT(), ctrl=i_j, target=t_j)
        old_carry, i_j = bb.add(CNOT(), ctrl=old_carry, target=i_j)
        [i_j, t_j], new_carry = bb.add(And(), ctrl=[i_j, t_j])
        old_carry, new_carry = bb.add(CNOT(), ctrl=old_carry, target=new_carry)

        return {"old_carry": old_carry, "i_j": i_j, "t_j": t_j, "new_carry": new_carry}

    def build_call_graph(self, ssa):
        return {CNOT(): 3, And(): 1}


@attrs.frozen
class HalfCompareUnwrapped(Bloq):
    """Unwrapped version of the first half of CompareAnd, see HalfCompare for more details

    Parameters
    ----------
    bitsize_i: SymbolicInt
        Bitsize of the register i

    bitsize_t: SymbolicInt
        Bitsize of the register t

    Attributes
    ----------
    i : QBit() (bitsize_i,)
        Register holding i
    t : QBit() (bitsize_t,)
        Register holding t
    target : QBit()
        Target holding the result of i >= t
    junk : QBit() (bitsize_t - 1,)
        Register holding carries, to be uncomputed

    Note
    ----
    To do a controlled version, which we will not do here,
    you just have to control the CNOT targeting the target register

    """

    bitsize_i: SymbolicInt
    bitsize_t: SymbolicInt

    def __attrs_post_init__(self):
        if not (
            check_equal(self.bitsize_i, self.bitsize_t)
            or check_equal(self.bitsize_i, self.bitsize_t - 1)
        ):
            raise ValueError(
                f"{self.bitsize_i=} should be the same size as {self.bitsize_t=} or 1 bit smaller"
            )

        if check_equal(self.bitsize_t, 1, ret_bool_symb=False):
            raise NotImplementedError(f"The bloq doesn't accept yet {self.bitsize_t=}")

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("i", QBit(), shape=(self.bitsize_i,)),
                Register("t", QBit(), shape=(self.bitsize_t,)),
                Register("target", QBit(), side=Side.RIGHT),
                Register("junk", QBit(), shape=(self.bitsize_t - 1,), side=Side.RIGHT),
            ]
        )

    # TODO, do the case when n=1
    def _caseOne(self):
        pass

    def _last_connexion_normal(
        self,
        bb: BloqBuilder,
        *,
        i: SoquetT,
        t: SoquetT,
        carries: SoquetT,
    ):
        carries[1], i[0], t[0], target = bb.add(
            _downBloq(), old_carry=carries[1], i_j=i[0], t_j=t[0]
        )
        target = bb.add(XGate(), q=target)

        return i, t, target, carries

    def _last_connexion_imbalenced(self, bb: BloqBuilder, *, t: SoquetT, carries: SoquetT):
        (t[0], carries[1]), target = bb.add(And(cv1=0, cv2=1), ctrl=(t[0], carries[1]))
        target = bb.add(XGate(), q=target)

        return t, target, carries

    def build_composite_bloq(
        self, bb: BloqBuilder, *, i: SoquetT, t: SoquetT
    ) -> dict[str, SoquetT]:
        # Carries 0 will never be used, be we do so to have easier indexing
        # -> carries[j] is the carry of i[j], t[j]
        carries = np.empty(self.bitsize_t, dtype=object)

        # If i is smaller we pad it with a None so that we keep the same indexing
        if self.bitsize_i == self.bitsize_t - 1:
            i = np.concatenate(([None], i))

        # First step is a bit different
        i[-1], t[-1] = bb.add(CNOT(), ctrl=i[-1], target=t[-1])
        [i[-1], t[-1]], carries[-1] = bb.add(And(), ctrl=[i[-1], t[-1]])

        # Going down
        for j in reversed(range(1, self.bitsize_t - 1)):
            carries[j + 1], i[j], t[j], carries[j] = bb.add(
                _downBloq(), old_carry=carries[j + 1], i_j=i[j], t_j=t[j]
            )

        # last connexion
        if self.bitsize_i == self.bitsize_t:
            i, t, target, carries = self._last_connexion_normal(bb, i=i, t=t, carries=carries)
        elif self.bitsize_i + 1 == self.bitsize_t:
            t, target, carries = self._last_connexion_imbalenced(bb, t=t, carries=carries)
        else:
            raise NotImplementedError()

        carries_out = carries[1:]

        if self.bitsize_i == self.bitsize_t - 1:
            i = i[1:]

        return {"i": i, "t": t, "target": target, "junk": carries_out}

    def build_call_graph(self, ssa):
        base = {
            CNOT(): 1,
            And(): 1,
            _downBloq(): self.bitsize_t - 2,
        }

        if check_equal(self.bitsize_i, self.bitsize_t, ret_bool_symb=True):
            base[_downBloq()] += 1
            base[XGate()] = 1

        if check_equal(self.bitsize_i, self.bitsize_t - 1, ret_bool_symb=False):
            base[And(cv1=0, cv2=1)] = 1
            base[XGate()] = 1

        return base


@attrs.frozen
class HalfCompare(Bloq):
    """
    Compare two integer registers

    This bloq computes whether `t >= i` and writes that comparison into
    `target` (XOR update)

    #! error with QInt

    Parameters
    ----------
    dtype_i : QInt | QUInt
        Type of input register `i`.
        Supported configurations are:

        - `dtype_i = QInt` and `dtype_t = QInt` with
          `dtype_i.bitsize == dtype_t.bitsize`.
        - `dtype_i = QUInt` and `dtype_t = QInt | QUInt` with
          `dtype_t.bitsize == dtype_i.bitsize` or
          `dtype_t.bitsize == dtype_i.bitsize + 1`.

    dtype_t : QInt | QUInt
        Type of input register `t`.

    Registers
    ---------
    i : dtype_i
        Input register `i`. On output it holds `i ^ (junk << 1)` (truncated to its bitsize),
        until the uncompute half restores it.
    t : dtype_t
        Input register `t`. On output it holds `t ^ i`, until the uncompute half restores it.
    target : QBit, Side.RIGHT
        Fresh qubit holding `[t >= i]`.
    junk : QUInt(bitsize_t - 1,), Side.RIGHT
        Borrows of `t - i`: bit `p` (from the LSB) is `[t mod 2^(p+1) < i mod 2^(p+1)]`.
        To be uncomputed.

    Notes
    -----
    `target` is the complement of the borrow out of the top bit of `t - i`, computed on the
    unsigned bit patterns of the registers. It is therefore exactly `[t >= i]` for unsigned
    inputs, whether the bitsizes are equal or `t` is one bit wider.

    """

    dtype_i: QInt | QUInt
    dtype_t: QInt | QUInt

    def __attrs_post_init__(self):
        i_is_qint = isinstance(self.dtype_i, QInt)
        i_is_quint = isinstance(self.dtype_i, QUInt)
        t_is_qint = isinstance(self.dtype_t, QInt)
        t_is_quint = isinstance(self.dtype_t, QUInt)

        if not (i_is_qint or i_is_quint):
            raise ValueError("Only QInt and QUInt types are supported for dtype_i.")
        if not (t_is_qint or t_is_quint):
            raise ValueError("Only QInt and QUInt types are supported for dtype_t.")

        if i_is_qint:
            if not t_is_qint:
                raise ValueError("When dtype_i is QInt, dtype_t must be QInt.")
            if not check_equal(self.dtype_t.bitsize, self.dtype_i.bitsize):
                raise ValueError(
                    f"{self.dtype_t.bitsize=} should be the same size as {self.dtype_i.bitsize=}"
                    " when dtype_i is QInt."
                )
            return

        if not (
            check_equal(self.dtype_t.bitsize, self.dtype_i.bitsize)
            or check_equal(self.dtype_t.bitsize, self.dtype_i.bitsize + 1)
        ):
            raise ValueError(
                f"For QUInt dtype_i, {self.dtype_t.bitsize=} should be equal or one bit bigger than"
                f"{self.dtype_i.bitsize=}."
            )

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("i", self.dtype_i),
                Register("t", self.dtype_t),
                Register("target", QBit(), side=Side.RIGHT),
                Register("junk", QUInt(self.dtype_t.bitsize - 1), side=Side.RIGHT),
            ]
        )

    @property
    def part_t(self) -> Partition:
        regs = [Register("t_split", QBit(), (self.dtype_t.bitsize,))]
        return Partition(n=self.dtype_t.bitsize, regs=regs)

    @property
    def part_i(self) -> Partition:
        regs = [Register("i_split", QBit(), (self.dtype_i.bitsize,))]
        return Partition(n=self.dtype_i.bitsize, regs=regs)

    @property
    def part_carries(self) -> Partition:
        regs = [Register("carries_split", QBit(), (self.dtype_t.bitsize - 1,))]
        return Partition(n=self.dtype_t.bitsize - 1, regs=regs)

    @property
    def _compAndHalf(self):
        return HalfCompareUnwrapped(self.dtype_i.bitsize, self.dtype_t.bitsize)

    def build_composite_bloq(self, bb: BloqBuilder, *, i: Soquet, t: Soquet):
        i_split = bb.add(self.part_i, x=i)
        t_split = bb.add(self.part_t, x=t)

        i_split, t_split, target, carries_split = bb.add_from(
            self._compAndHalf, i=i_split, t=t_split
        )

        i = bb.add(self.part_i.adjoint(), i_split=i_split)
        t = bb.add(self.part_t.adjoint(), t_split=t_split)
        carries = bb.add(self.part_carries.adjoint(), carries_split=carries_split)

        return {"i": i, "t": t, "target": target, "junk": carries}

    def on_classical_vals(self, i, t) -> dict[str, ClassicalValT]:
        # Closed form of the decomposition, on bit patterns: the circuit is unsigned whatever
        # the dtype (hence the QInt caveat above).
        n_i, n_t = int(self.dtype_i.bitsize), int(self.dtype_t.bitsize)
        i_u = QUInt(n_i).from_bits(self.dtype_i.to_bits(i))
        t_u = QUInt(n_t).from_bits(self.dtype_t.to_bits(t))
        # borrows[p]: borrow out of bit p of t - i (the carry each _downBloq computes)
        borrows = [int(t_u % 2 ** (p + 1) < i_u % 2 ** (p + 1)) for p in range(n_t)]
        junk = sum(b << p for p, b in enumerate(borrows[:-1]))
        # Each _downBloq XORs the incoming borrow into i_j and i_j into t_j.
        i_out = i_u ^ ((junk << 1) % 2**n_i)
        t_out = t_u ^ i_u
        return {
            "i": self.dtype_i.from_bits(QUInt(n_i).to_bits(i_out)),
            "t": self.dtype_t.from_bits(QUInt(n_t).to_bits(t_out)),
            "target": 1 - borrows[-1],
            "junk": junk,
        }

    def build_call_graph(self, ssa):
        return {
            self.part_i: 1,
            self.part_i.adjoint(): 1,
            self.part_t: 1,
            self.part_t.adjoint(): 1,
            self.part_carries.adjoint(): 1,
            **self._compAndHalf.build_call_graph(ssa),
        }

    def my_static_costs(self, cost_key: CostKey):
        if isinstance(cost_key, QubitCount):
            return self.dtype_i.bitsize + 2 * self.dtype_t.bitsize
        return NotImplemented
