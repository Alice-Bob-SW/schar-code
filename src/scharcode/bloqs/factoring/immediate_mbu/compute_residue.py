import attrs
from functools import cached_property
from collections import Counter

from qualtran import Bloq, BloqBuilder, Signature, Register, Soquet, Side, CtrlSpec
from qualtran import QUInt, QAny, QBit, CBit
from qualtran.bloqs.basic_gates import MeasureX, Discard
from qualtran.bloqs.bookkeeping import Free
from qualtran.resource_counting import (
    SympySymbolAllocator,
    BloqCountDictT,
    QubitCount,
)
from qualtran.symbolics import SymbolicInt, is_symbolic

from scharcode.bloqs.operations.lookup import Lookup
from scharcode.bloqs.operations._utils import tuple_or_symbolic, T
from scharcode.bloqs.operations.qq_adder import QQAdder
from scharcode.bloqs.operations.ctrl_add_constant import ControlledConstantAdd
from scharcode.bloqs.factoring.immediate_mbu.underflow_phasefix import UnderflowPhasefix
from scharcode.bloqs.factoring._common import (
    address_partition,
    helper_partition,
    ResidueWindows,
    ResidueTables,
    l_p,
)


@attrs.frozen
class ComputeResidue(Bloq):
    """
    Implement |s>|0> -> |s>|g^s%p>
    Subroutine ComputeResidue

    Parameters
    ----------
    dtype_s : QUInt
        dtype of the register holding exponent
    w3a : SymbolicInt
        window size for this method
    w3b : SymbolicInt
        window size for this method
    g : SymbolicInt
        base of the modular exponentiation
    p : SymbolicInt
        prime number here used as the modulus of the modular exponentiation


    Attributes
    ----------
    s : dtype_s
        register holding the exponent, should be of size l_p (since it was just compressed)
    r : QUInt(self.l_p), side.RIGHT
        register holding the residue

    ref: Fig. 29 [Shor's algorithm compiled on cat qubits: 22 593 qubits for RSA-2048 in 3.14 days]
    (https://arxiv.org/abs/2610.37839)
    """

    dtype_s: QUInt
    w3a: "SymbolicInt"
    w3b: "SymbolicInt"
    g: "SymbolicInt"
    p: "SymbolicInt"

    def __attrs_post_init__(self):
        if not (
            is_symbolic(self.p)
            or is_symbolic(self.w3a)
            or is_symbolic(self.w3b)
            or is_symbolic(self.dtype_s.bitsize)
        ):
            if self.l_p != self.dtype_s.bitsize:
                raise ValueError(
                    f"s should have the same size has p but is {self.dtype_s.bitsize} != {self.l_p}"
                )

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("s", self.dtype_s),
                Register("r", QUInt(self.l_p), side=Side.RIGHT),
            ]
        )

    @cached_property
    def l_p(self):
        if is_symbolic(self.p):
            return l_p
        return self.p.bit_length()

    @cached_property
    def layout_window(self):
        layout = ResidueWindows(self.l_p, self.w3a, self.w3b)
        return layout

    @cached_property
    def tables3(self):
        tables = ResidueTables(self.p, self.g, self.w3a, self.w3b)
        return tables

    @cached_property
    def is_concrete(self) -> bool:
        """True when the tables can be computed (`ResidueTables` needs concrete p, g, w3a, w3b)."""
        return not is_symbolic(self.p, self.g, self.w3a, self.w3b)

    @cached_property
    def lookup3_ini(self):
        if not self.is_concrete:
            return Lookup(T, self.layout_window.init_bitsize, self.l_p)
        return Lookup(self.tables3.alpha2, self.layout_window.init_bitsize, self.l_p)

    def mod_mul_step(self, j_a: int, j_b: int) -> "ModMulStep":
        """ModMulStep of window pair (j_a, j_b)."""
        return ModMulStep(
            self.layout_window.w3a_sizes[j_a],
            self.layout_window.w3b_sizes[j_b],
            self.tables3.p_minus_t3(j_a, j_b),
            self.p,
        )

    def build_composite_bloq(self, bb: BloqBuilder, s: Soquet):
        s_init, s_chunk = self.layout_window.split_exponent(bb, s)
        # skipping first 2 iterations by initializing with a direct lookup
        s_init, r = bb.add(self.lookup3_ini, selection=s_init)

        for j_a in self.layout_window.outer_windows:
            helper = bb.allocate(self.l_p, QUInt(self.l_p))
            r_chunk = self.layout_window.split_product(bb, r)
            for j_b in range(self.layout_window.num_w3b):
                s_chunk[j_a], r_chunk[j_b], helper, underflow = bb.add(
                    self.mod_mul_step(j_a, j_b),
                    s_chunk=s_chunk[j_a],
                    r_chunk=r_chunk[j_b],
                    helper=helper,
                )
                # The underflow qubit should be measured in the X basis and
                # its phase corrected by an UnderflowPhasefix. It is freed instead, so this
                # decomposition is not phase-exact.
                bb.free(underflow, dirty=True)

            r = self.layout_window.join_product(bb, r_chunk)
            r, helper = helper, r

            # uncompute helper by measuring in X basis and deferring phase measurement to adjoint
            bb.free(
                helper, dirty=True
            )  #! should be MeasureX + phase correction in ComputeResidueAdjoint, would require
            #! keeping track of all measurement result in CBit, so we free instead.

        s = self.layout_window.join_exponent(bb, s_init, s_chunk)
        return {"s": s, "r": r}

    def on_classical_vals(self, **vals):
        s = vals["s"]
        r = pow(int(self.g), s, self.p)
        return {"s": s, "r": r}

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> BloqCountDictT:
        w = self.layout_window
        counts = Counter()
        counts[Lookup(T, w.init_bitsize, self.l_p)] += 1
        if is_symbolic(self.l_p, self.w3a, self.w3b):
            # for symbolic cases, we assume every window is full.
            counts[ModMulStep(self.w3a, self.w3b, T, self.p)] += w.num_outer_steps * w.num_w3b
        else:
            # The top windows can be partial.
            for j_a in w.outer_windows:
                for size_b in w.w3b_sizes:
                    counts[ModMulStep(w.w3a_sizes[j_a], size_b, T, self.p)] += 1
        # Dirty frees standing for the X measurements of the underflow qubits and helper registers
        counts[Free(QBit(), dirty=True)] += w.num_outer_steps * w.num_w3b
        counts[Free(QUInt(self.l_p), dirty=True)] += w.num_outer_steps
        return Counter({k: c for k, c in counts.items() if c != 0})

    def my_static_costs(self, cost_key):
        if isinstance(cost_key, QubitCount):
            return 5 * self.l_p + 2
        return NotImplemented


@attrs.frozen
class ModMulStep(Bloq):
    """
    One iteration of ComputeResidue inner loop (inner loop of Fig. 29)

    Parameters
    ----------
    w3a : SymbolicInt
        window size, here size of the s_chunk register
    w3b : SymbolicInt
        window size, here size of the r_chunk register
    table3_jk : tuple[SymbolicInt]
        table to use for this iteration
    p : SymbolicInt
        prime number here used as the modulus of the modular exponentiation


    Attributes
    ----------
    s_chunk : QAny(self.w3a)
        register holding the current exponent window
    r_chunk : QAny(self.w3b)
        register holding the current residue window
    helper : QAny(self.l_p)
    u : QBit(), side.RIGHT
        underflow qubit of the modular addition. The caller uncomputes it: ComputeResidue
        frees it, ComputeResidue† measures it and applies an UnderflowPhasefix.

    ref: Fig. 29 [Shor's algorithm compiled on cat qubits: 22 593 qubits for RSA-2048 in 3.14 days]
    (https://arxiv.org/abs/2610.37839)
    """

    w3a: "SymbolicInt"
    w3b: "SymbolicInt"
    table3_jk: tuple[SymbolicInt] = attrs.field(converter=tuple_or_symbolic)
    p: "SymbolicInt"

    @cached_property
    def signature(self) -> "Signature":
        return Signature(
            [
                Register("s_chunk", QUInt(self.w3a)),
                Register("r_chunk", QUInt(self.w3b)),
                Register("helper", QUInt(self.l_p)),
                Register("u", QBit(), side=Side.RIGHT),
            ]
        )

    @cached_property
    def l_p(self):
        if is_symbolic(self.p):
            return l_p
        return self.p.bit_length()

    @cached_property
    def lookup3_windowed_modmul(self):
        if is_symbolic(self.p) or is_symbolic(self.w3a):
            return Lookup(T, self.w3a + self.w3b, self.l_p + 1)
        return Lookup(self.table3_jk, self.w3a + self.w3b, self.l_p + 1)

    def build_composite_bloq(
        self, bb: BloqBuilder, s_chunk: Soquet, r_chunk: Soquet, helper: Soquet
    ):
        address_windowed_modmul = bb.add(
            address_partition(self.w3a, self.w3b).adjoint(),
            x_a=s_chunk,
            y_b=r_chunk,
        )
        address_windowed_modmul, target_windowed_modmul = bb.add(
            self.lookup3_windowed_modmul,
            selection=address_windowed_modmul,
        )
        underflow = bb.allocate(1, QBit())
        helper = bb.add(helper_partition(self.l_p).adjoint(), u=underflow, h=helper)
        target_windowed_modmul, helper = bb.add(
            QQAdder(self.l_p + 1).adjoint(),
            i=target_windowed_modmul,
            t=helper,
        )
        # ? lookup.adjoint() here because no delayed phase correction in this version
        address_windowed_modmul = bb.add(
            self.lookup3_windowed_modmul.adjoint(),
            selection=address_windowed_modmul,
            target=target_windowed_modmul,
        )
        s_chunk, r_chunk = bb.add(
            address_partition(self.w3a, self.w3b),
            x=address_windowed_modmul,
        )
        underflow, helper = bb.add(helper_partition(self.l_p), x=helper)
        underflow, helper = bb.add(
            ControlledConstantAdd(QAny(self.l_p), self.p),
            ctrl=underflow,
            target=helper,
        )
        # The underflow qubit is returned: the caller uncomputes it.
        return {
            "s_chunk": s_chunk,
            "r_chunk": r_chunk,
            "helper": helper,
            "u": underflow,
        }

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> BloqCountDictT:
        # does not include partitions, allocations and frees. The table is generalized to T.
        lookup = Lookup(T, self.w3a + self.w3b, self.l_p + 1)
        return {
            lookup: 1,
            lookup.adjoint(): 1,
            QQAdder(self.l_p + 1).adjoint(): 1,
            ControlledConstantAdd(QAny(self.l_p), self.p): 1,
        }


@attrs.frozen
class ComputeResidueAdjoint(Bloq):
    """
    Implement |s>|g^s%p> -> |s>|0>

    Parameters
    ----------
    dtype_s : QUInt
        dtype of the register holding exponent
    w3a : SymbolicInt
        window size for this method
    w3b : SymbolicInt
        window size for this method
    g : SymbolicInt
        base of the modular exponentiation
    p : SymbolicInt
        prime number here used as the modulus of the modular exponentiation


    Attributes
    ----------
    s : dtype_s
        register holding the exponent, should be of size l_p (since it was just compressed)
    r : QUInt(self.l_p), side.LEFT
        register holding the residue

    ref: Fig. 30 [Shor's algorithm compiled on cat qubits: 22 593 qubits for RSA-2048 in 3.14 days] (https://arxiv.org/abs/2610.37839)
    """

    dtype_s: QUInt
    w3a: "SymbolicInt"
    w3b: "SymbolicInt"
    g: "SymbolicInt"
    p: "SymbolicInt"

    def __attrs_post_init__(self):
        if not (
            is_symbolic(self.p)
            or is_symbolic(self.w3a)
            or is_symbolic(self.w3b)
            or is_symbolic(self.dtype_s.bitsize)
        ):
            if self.l_p != self.dtype_s.bitsize:
                raise ValueError(
                    f"s should have the same size has p but is {self.dtype_s.bitsize} != {self.l_p}"
                )

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("s", self.dtype_s),
                Register("r", QUInt(self.l_p), side=Side.LEFT),
            ]
        )

    @cached_property
    def l_p(self):
        if is_symbolic(self.p):
            return l_p
        return self.p.bit_length()

    @cached_property
    def is_concrete(self) -> bool:
        """True when the tables can be computed (`ResidueTables` needs concrete p, g, w3a, w3b)."""
        return not is_symbolic(self.p, self.g, self.w3a, self.w3b)

    @cached_property
    def lookup3_ini(self):
        if not self.is_concrete:
            return Lookup(T, self.layout_window.init_bitsize, self.l_p)
        return Lookup(self.tables3.alpha2, self.layout_window.init_bitsize, self.l_p)

    @cached_property
    def layout_window(self):
        layout = ResidueWindows(self.l_p, self.w3a, self.w3b)
        return layout

    @cached_property
    def tables3(self):
        tables = ResidueTables(self.p, self.g, self.w3a, self.w3b)
        return tables

    def mod_mul_step(self, j_a: int, j_b: int, table) -> "ModMulStep":
        """ModMulStep of window pair (j_a, j_b) for a given table."""
        return ModMulStep(
            self.layout_window.w3a_sizes[j_a],
            self.layout_window.w3b_sizes[j_b],
            table(j_a, j_b),
            self.p,
        )

    def uncompute_underflow(
        self,
        bb: BloqBuilder,
        j_a: int,
        j_b: int,
        loaded,
        x_a: Soquet,
        y_b: Soquet,
        helper: Soquet,
        u: Soquet,
    ) -> tuple[Soquet, Soquet, Soquet]:
        """Measure the underflow qubit u of step (j_a, j_b) in the X basis
        and apply the appropriate phase correction.

        `loaded` is the table method given to `mod_mul_step` for this step, e.g.
        `self.tables3.p_minus_t3_inv`. If the measurement gives 1, an UnderflowPhasefix
        compares the helper register with p - V, V being the loaded value (see `ResidueTables`).
        Returns the windows x_a, y_b and the helper register.
        """
        size_a = self.layout_window.w3a_sizes[j_a]
        size_b = self.layout_window.w3b_sizes[j_b]
        phasefix = UnderflowPhasefix(
            QUInt(self.l_p), size_a + size_b, self.tables3.complement(loaded(j_a, j_b))
        )
        meas_u = bb.add(MeasureX(), q=u)
        address = bb.add(address_partition(size_a, size_b).adjoint(), x_a=x_a, y_b=y_b)
        meas_u, address, helper = bb.add(
            phasefix.controlled(CtrlSpec(CBit())), ctrl=meas_u, address=address, reg=helper
        )
        bb.add(Discard(), c=meas_u)
        x_a, y_b = bb.add(address_partition(size_a, size_b), x=address)
        return x_a, y_b, helper

    def build_composite_bloq(self, bb: BloqBuilder, s: Soquet, r: Soquet):
        s_init, s_chunk = self.layout_window.split_exponent(bb, s)
        helper = bb.allocate(self.l_p, QUInt(self.l_p))

        for j_a in reversed(self.layout_window.outer_windows):
            r_chunk = self.layout_window.split_product(bb, r)
            for j_b in range(self.layout_window.num_w3b):
                s_chunk[j_a], r_chunk[j_b], helper, underflow = bb.add(
                    self.mod_mul_step(j_a, j_b, self.tables3.p_minus_t3_inv),
                    s_chunk=s_chunk[j_a],
                    r_chunk=r_chunk[j_b],
                    helper=helper,
                )
                s_chunk[j_a], r_chunk[j_b], helper = self.uncompute_underflow(
                    bb,
                    j_a,
                    j_b,
                    self.tables3.p_minus_t3_inv,
                    s_chunk[j_a],
                    r_chunk[j_b],
                    helper,
                    underflow,
                )

            r = self.layout_window.join_product(bb, r_chunk)
            r, helper = helper, r
            r_chunk = self.layout_window.split_product(bb, r)

            for j_b in reversed(range(self.layout_window.num_w3b)):
                s_chunk[j_a], r_chunk[j_b], helper, underflow = bb.add(
                    self.mod_mul_step(j_a, j_b, self.tables3.t3),
                    s_chunk=s_chunk[j_a],
                    r_chunk=r_chunk[j_b],
                    helper=helper,
                )
                s_chunk[j_a], r_chunk[j_b], helper = self.uncompute_underflow(
                    bb,
                    j_a,
                    j_b,
                    self.tables3.t3,
                    s_chunk[j_a],
                    r_chunk[j_b],
                    helper,
                    underflow,
                )
            r = self.layout_window.join_product(bb, r_chunk)
        bb.free(helper)

        # uncomputing the initialisation with the first lookup
        s_init = bb.add(self.lookup3_ini.adjoint(), selection=s_init, target=r)

        s = self.layout_window.join_exponent(bb, s_init, s_chunk)
        return {"s": s}

    def on_classical_vals(self, **vals):
        s = vals["s"]
        return {"s": s}

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> BloqCountDictT:
        w = self.layout_window
        counts = Counter()
        counts[Lookup(T, w.init_bitsize, self.l_p).adjoint()] += 1
        num_steps = 2 * w.num_outer_steps * w.num_w3b  # two inner loops per outer iteration
        if is_symbolic(self.l_p, self.w3a, self.w3b):
            # for symbolic cases, we assume every window is full.
            counts[ModMulStep(self.w3a, self.w3b, T, self.p)] += num_steps
            counts[UnderflowPhasefix(QUInt(self.l_p), self.w3a + self.w3b, T)] += (
                w.num_outer_steps * w.num_w3b
            )
        else:
            # The top windows can be partial.
            for j_a in w.outer_windows:
                size_a = w.w3a_sizes[j_a]
                for size_b in w.w3b_sizes:
                    # per inner loop iteration, 1 ModMulStep and 1/2 UnderflowPhasefix
                    # (u=1 half of the time on average)
                    counts[ModMulStep(size_a, size_b, T, self.p)] += 2
                    counts[UnderflowPhasefix(QUInt(self.l_p), size_a + size_b, T)] += 1
        counts[MeasureX()] += num_steps
        counts[Discard()] += num_steps
        return Counter({k: c for k, c in counts.items() if c != 0})

    def my_static_costs(self, cost_key):
        if isinstance(cost_key, QubitCount):
            return 5 * self.l_p + 2
        return NotImplemented
