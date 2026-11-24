from functools import cached_property

import attrs
import numpy as np
from qualtran import Bloq, BloqBuilder, QAny, QBit, QUInt, Register, Side, Signature, Soquet
from qualtran.bloqs.bookkeeping import Partition
from qualtran.resource_counting import BloqCountDictT, QubitCount, SympySymbolAllocator
from qualtran.symbolics import SymbolicInt, is_symbolic
from sympy.ntheory import isprime

from scharcode.bloqs.factoring._common import helper_partition, l_p
from scharcode.bloqs.operations.multi_target_cnot import CtrlXorK, K
from scharcode.bloqs.operations.qc_adder import QCAdder
from scharcode.bloqs.operations.qq_adder import QQAdder


def _bit_length(p_i: SymbolicInt) -> SymbolicInt:
    """l_p, the bit length of p_i, or the symbol `l_p` when p_i is symbolic."""
    if is_symbolic(p_i):
        return l_p
    return int(p_i).bit_length()


@attrs.frozen
class ReduceStep(Bloq):
    r"""One reduction step, Figure `compress_dlog` (top).

    Implements |x> -> |x mod (p_i - 1)>|1_{x < p_i - 1}>, for 0 <= x < 2 (p_i - 1), with the
    construction of Lemma B.1 (b = p_i - 1, n = l_p):

    Parameters
    ----------
    dtype_window : QUInt
        dtype of the window to reduce, QUInt(l_p + 1).
    p_i : SymbolicInt
        The RNS prime. Its bit length is l_p.

    Attributes
    ----------
    window : QUInt(l_p + 1) -> QUInt(l_p)
        Holds x on input and x mod (p_i - 1) on output.
    quotient_qubit : QBit (RIGHT)
        The bit 1_{x < p_i - 1}, the complement of the quotient bit of this step.
    """

    dtype_window: QUInt
    p_i: SymbolicInt

    def __attrs_post_init__(self):
        if is_symbolic(self.p_i):
            return
        if not (self.p_i >= 3 and isprime(self.p_i)):
            raise ValueError(f"p_i must be an odd prime, got {self.p_i}")
        if self.l_p < 3:
            raise ValueError(f"l_p must be at least 3 (adder sizes), got {self.l_p}")
        if not is_symbolic(self.dtype_window.bitsize) and self.dtype_window.bitsize != self.l_p + 1:
            raise ValueError(
                f"the window must have l_p + 1 = {self.l_p + 1} qubits, "
                f"got {self.dtype_window.bitsize}"
            )

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("window", self.dtype_window, side=Side.LEFT),
                Register("window", QUInt(self.dtype_window.bitsize - 1), side=Side.RIGHT),
                Register("quotient_qubit", QBit(), side=Side.RIGHT),
            ]
        )

    @cached_property
    def l_p(self) -> SymbolicInt:
        return _bit_length(self.p_i)

    @cached_property
    def b(self) -> SymbolicInt:
        return self.p_i - 1

    def build_composite_bloq(self, bb: BloqBuilder, *, window: Soquet) -> dict[str, Soquet]:
        # |x> -> |x - b> = |1_{x<b}>|y_t>
        window = bb.add(QCAdder(self.l_p + 1, self.b).adjoint(), t=window)
        u, y_t = bb.add(helper_partition(self.l_p), x=window)

        # |0> -> |b 1_{x<b}>, then |y_t> -> |y_t + b 1_{x<b}>, then |b 1_{x<b}> -> |0>
        aux = bb.allocate(dtype=QUInt(self.l_p))
        u, aux = bb.add(CtrlXorK(QUInt(self.l_p), self.b), ctrl=u, x=aux)
        aux, y_t = bb.add(QQAdder(self.l_p), i=aux, t=y_t)
        u, aux = bb.add(CtrlXorK(QUInt(self.l_p), self.b), ctrl=u, x=aux)
        bb.free(aux)

        return {"window": y_t, "quotient_qubit": u}

    def on_classical_vals(self, window) -> dict[str, int]:
        x = int(window)
        if not x < 2 * self.b:
            raise ValueError(f"ReduceStep needs x < 2 (p_i - 1) = {2 * self.b}, got {x}")
        return {"window": x % self.b, "quotient_qubit": int(x < self.b)}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        ctrl_load_b = (
            CtrlXorK(QUInt(self.l_p), K)
            if is_symbolic(self.p_i)
            else CtrlXorK(QUInt(self.l_p), self.b)
        )
        return {QCAdder(self.l_p + 1, self.b).adjoint(): 1, ctrl_load_b: 2, QQAdder(self.l_p): 1}

    def my_static_costs(self, cost_key):
        if isinstance(cost_key, QubitCount):
            # During the Q-Q Adder: u, y_t (l_p), the auxiliary register (l_p) and the l_p - 1
            # ancillas of the adder.
            return 3 * self.l_p
        return NotImplemented


@attrs.frozen
class ReduceExponent(Bloq):
    r"""Reversible Euclidean division of S_i by p_i - 1, Figure 28 (bottom).

    Implements |S_i> -> |S_i mod (p_i - 1)>|ā>, where a = floor(S_i / (p_i - 1)) and ā is its
    bitwise complement on log(n_e) bits. It needs S_i < 2^{log(n_e)} (p_i - 1), which holds for
    the S_i produced by ComputeExponent.

    It applies log(n_e) `ReduceStep` bloqs.

    Parameters
    ----------
    dtype_s : QUInt
        dtype of the register holding S_i, QUInt(l_p + log(n_e)).
    p_i : SymbolicInt
        The RNS prime. Its bit length is l_p.

    Attributes
    ----------
    s : QUInt(l_p + log(n_e)) -> QUInt(l_p)
        Holds S_i on input and S_i mod (p_i - 1) on output.
    quotient : QUInt(log(n_e)) (RIGHT)
        The quotient register, holding ā.

    ref: Fig. 28 [Shor's algorithm compiled on cat qubits:
    22 593 qubits for RSA-2048 in 3.14 days](https://arxiv.org/abs/2610.37839)
    """

    dtype_s: QUInt
    p_i: SymbolicInt

    def __attrs_post_init__(self):
        if not is_symbolic(self.log_n_e) and self.log_n_e < 1:
            raise ValueError(
                f"s must have more than l_p = {self.l_p} qubits, got {self.dtype_s.bitsize}"
            )
        _ = self.reduce_step  # runs the checks of ReduceStep

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("s", self.dtype_s, side=Side.LEFT),
                Register("s", QUInt(self.l_p), side=Side.RIGHT),
                Register("quotient", QUInt(self.log_n_e), side=Side.RIGHT),
            ]
        )

    @cached_property
    def l_p(self) -> SymbolicInt:
        return _bit_length(self.p_i)

    @cached_property
    def log_n_e(self) -> SymbolicInt:
        """Size of the quotient register, and number of reduction steps."""
        return self.dtype_s.bitsize - self.l_p

    @cached_property
    def reduce_step(self) -> ReduceStep:
        return ReduceStep(QUInt(self.l_p + 1), self.p_i)

    @cached_property
    def s_partition(self) -> Partition:
        """Split S_i into its l_p most significant qubits and its log(n_e) low qubits.

        Partition: `x` (l_p + log(n_e) qubits) -> `s_high` (l_p), `s_low` (log(n_e) qubits).
        Element k of `s_low` is bit log(n_e) - 1 - k of S_i, the bit entering the window at
        step k.
        """
        regs = (
            Register("s_high", QAny(self.l_p)),
            Register("s_low", QBit(), shape=(self.log_n_e,)),
        )
        return Partition(self.dtype_s.bitsize, regs)

    @cached_property
    def window_partition(self) -> Partition:
        """Split a window into the remainder (l_p top qubits) and the incoming bit (bottom).

        Partition: `x` (l_p + 1 qubits) -> `r` (l_p), `bit` (1)
        """
        regs = (Register("r", QAny(self.l_p)), Register("bit", QBit()))
        return Partition(self.l_p + 1, regs)

    @cached_property
    def quotient_partition(self) -> Partition:
        """Split the quotient register into its qubits, element k is the qubit of step k."""
        regs = (Register("q", QBit(), shape=(self.log_n_e,)),)
        return Partition(self.log_n_e, regs)

    def build_composite_bloq(self, bb: BloqBuilder, *, s: Soquet) -> dict[str, Soquet]:
        r, s_low = bb.add(self.s_partition, x=s)
        quotient = []
        for k in range(self.log_n_e):
            window = bb.add(self.window_partition.adjoint(), r=r, bit=s_low[k])
            r, q_k = bb.add(self.reduce_step, window=window)
            quotient.append(q_k)
        quotient = bb.add(self.quotient_partition.adjoint(), q=np.array(quotient))
        return {"s": r, "quotient": quotient}

    def on_classical_vals(self, s) -> dict[str, int]:
        s = int(s)
        b = self.p_i - 1
        if not s < 2**self.log_n_e * b:
            raise ValueError(
                f"ReduceExponent needs S_i < 2^log(n_e) (p_i - 1) = {2**self.log_n_e * b}, got {s}"
            )
        a = s // b
        return {"s": s % b, "quotient": (2**self.log_n_e - 1) - a}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        return {self.reduce_step: self.log_n_e}

    def my_static_costs(self, cost_key):
        if isinstance(cost_key, QubitCount):
            # The register of S_i, plus what a ReduceStep uses beyond its window.
            return self.dtype_s.bitsize + 3 * self.l_p - (self.l_p + 1)
        return NotImplemented

    def adjoint(self) -> "ReduceExponentAdjoint":
        return ReduceExponentAdjoint(self.dtype_s, self.p_i)


@attrs.frozen
class ReduceExponentAdjoint(Bloq):
    r"""ReduceExponent†: |S_i mod (p_i - 1)>|ā> -> |S_i>.

    Runs the `ReduceStep`s of `ReduceExponent` in reverse order, each one adjoint. Step k
    takes the remainder and the quotient qubit of step k, and gives back the window, whose
    bottom qubit is bit log(n_e) - 1 - k of S_i.

    Parameters
    ----------
    dtype_s : QUInt
        dtype of the register holding S_i, QUInt(l_p + log(n_e)).
    p_i : SymbolicInt
        The RNS prime. Its bit length is l_p.

    Attributes
    ----------
    s : QUInt(l_p) -> QUInt(l_p + log(n_e))
        Holds S_i mod (p_i - 1) on input and S_i on output.
    quotient : QUInt(log(n_e)) (LEFT)
        The quotient register, holding ā.
    """

    dtype_s: QUInt
    p_i: SymbolicInt

    @cached_property
    def forward(self) -> ReduceExponent:
        return ReduceExponent(self.dtype_s, self.p_i)

    @cached_property
    def signature(self) -> Signature:
        return self.forward.signature.adjoint()

    @cached_property
    def l_p(self) -> SymbolicInt:
        return self.forward.l_p

    @cached_property
    def log_n_e(self) -> SymbolicInt:
        return self.forward.log_n_e

    def build_composite_bloq(
        self, bb: BloqBuilder, *, s: Soquet, quotient: Soquet
    ) -> dict[str, Soquet]:
        fwd = self.forward
        q = bb.add(fwd.quotient_partition, x=quotient)
        r = s
        s_low = np.empty(self.log_n_e, dtype=object)
        for k in reversed(range(self.log_n_e)):
            window = bb.add(fwd.reduce_step.adjoint(), window=r, quotient_qubit=q[k])
            r, s_low[k] = bb.add(fwd.window_partition, x=window)
        s = bb.add(fwd.s_partition.adjoint(), s_high=r, s_low=s_low)
        return {"s": s}

    def on_classical_vals(self, s, quotient) -> dict[str, int]:
        a = (2**self.log_n_e - 1) - int(quotient)
        return {"s": int(s) + a * (self.p_i - 1)}

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        return {self.forward.reduce_step.adjoint(): self.log_n_e}

    def my_static_costs(self, cost_key):
        return self.forward.my_static_costs(cost_key)

    def adjoint(self) -> ReduceExponent:
        return self.forward
