from abc import ABC, abstractmethod
from collections import Counter
from functools import cached_property
from collections.abc import Iterable
import attrs
import numpy as np
import quimb.tensor as qtn

from qualtran import BloqBuilder, SoquetT, Bloq, Signature, Register, Side, CtrlSpec
from qualtran import QAny, QBit
from qualtran.bloqs.arithmetic import XorK
from qualtran.bloqs.basic_gates import MeasureX, CZ
from qualtran.bloqs.bookkeeping import Partition
from qualtran.bloqs.mcmt import And
from qualtran.symbolics import is_symbolic, SymbolicInt
from qualtran.resource_counting import QubitCount

from scharcode.bloqs.operations.multi_target_cnot import CtrlXorK, K
from scharcode.bloqs.operations._utils import tuple_or_symbolic, T


def lookup_generalizer(b):
    if isinstance(b, Lookup):
        return Lookup(T, b.w, b.n)
    if isinstance(b, LookupUncompute):
        return Lookup(T, b.w, b.n).adjoint()
    return b


@attrs.frozen
class LookupAbstract(Bloq, ABC):
    """lookup abstract action as a circuit

    We can use this Bloq as an "auto-wrapper" for lookup by using this class as the parent
    and overriding lookup_unwrapped with the lookup you want to wrap.

    Parameters
    ----------
    T : Tuple[SymbolicInt]
        Table to load using the lookup, converted into tuple if not symbolic
    w : SymbolicInt
        Size of the selection register
    n : SymbolicInt
        Size of the target register


    Attributes
    ----------
    selection : QAny(self.w)
        Address register
    target : QAny(self.n), Side.RIGHT
        Register that will be xored.
    """

    T: tuple[SymbolicInt] = attrs.field(converter=tuple_or_symbolic)
    w: SymbolicInt
    n: SymbolicInt
    control_val: int | None = attrs.field(
        default=None, validator=attrs.validators.optional(attrs.validators.in_([0, 1]))
    )

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                *self.control_register,
                Register("selection", QAny(self.w)),
                Register("target", QAny(self.n), side=Side.RIGHT),
            ]
        )

    @property
    def control_register(self) -> Iterable[Register]:
        return [] if self.control_val is None else [Register("ctrl", QBit())]

    @cached_property
    def _selection_split(self) -> Partition:
        regs = (Register("selection", QBit(), shape=(self.w,)),)
        return Partition(self.w, regs)

    @cached_property
    @abstractmethod
    def lookup_unwrapped(self) -> Bloq:
        """unwrapped lookup to define in inheriting class"""

    def build_composite_bloq(self, bb: BloqBuilder, *, selection: SoquetT, **soqs):  # type: ignore[override]
        target: SoquetT = bb.allocate(self.n, QAny(self.n))
        selection = bb.add(self._selection_split, x=selection)
        if self.control_val is None:
            selection, target = bb.add_from(
                self.lookup_unwrapped, selection=selection, target=target
            )
        else:
            soqs["ctrl"], selection, target = bb.add_from(
                self.lookup_unwrapped.controlled(CtrlSpec(cvs=self.control_val)),
                ctrl=soqs["ctrl"],
                selection=selection,
                target=target,
            )
        selection = bb.add(self._selection_split.adjoint(), selection=selection)
        return {"selection": selection, "target": target} | soqs

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec"):
        from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv

        if ctrl_spec != CtrlSpec():
            return super().get_ctrl_system(ctrl_spec=ctrl_spec)

        return get_ctrl_system_1bit_cv(
            self,
            ctrl_spec,
            current_ctrl_bit=self.control_val,
            get_ctrl_bloq_and_ctrl_reg_name=lambda cv: (attrs.evolve(self, control_val=cv), "ctrl"),
        )

    def adjoint(self):
        return LookupUncompute(self, self.w, self.n)

    def build_call_graph(self, ssa):
        if self.control_val is None:
            counts = self.lookup_unwrapped.build_call_graph(ssa)
        else:
            counts = self.lookup_unwrapped.controlled(
                CtrlSpec(cvs=self.control_val)
            ).build_call_graph(ssa)

        counts[self._selection_split] = 1
        counts[self._selection_split.adjoint()] = 1
        return counts

    def on_classical_vals(self, **vals):
        selection = vals["selection"]
        if self.control_val is None:
            return {"selection": selection, "target": self.T[selection]}
        return {
            "ctrl": vals["ctrl"],
            "selection": selection,
            "target": self.T[selection] * (vals["ctrl"] ^ self.control_val ^ 1),
        }


def _get_indices_two(monome):
    """Get the indices of the two monome terms."""
    if not sum(monome) == 2:
        raise ValueError("Must have exactly 2 non-zero terms.")
    return np.where(monome)[0].tolist()


def _recursive_monomials_building(mon, list_mon=None):
    """
    Compute all possible monomials such that new_mon&mon == mon and len(new_mon) == len(mon).
    We need this function to go through monomials in a specific order.
    To generate all monomials of weight >=2 of a size n one should call this function with
    mon = [0]*n, and leave list_mon=None.

    Parameters
    ----------
    mon : list[int]
        monome upon which all possible monomials containing it are built
    list_mon : list[mon], optional
        list of already found monomials, should be set to None when starting initiating

    Returns
    -------
    list_mon:
        list of all monomials ordered such that for each monome, monomials requiring them for
        construction are next in the list.
    """
    if list_mon is None:
        list_mon = []
    n = len(mon)
    if mon not in list_mon:
        if sum(mon) >= 2:
            list_mon.append(mon)
        for i in reversed(range(n)):
            if mon[i] == 0:
                new_mon = mon.copy()
                new_mon[i] = 1
                _recursive_monomials_building(new_mon, list_mon)
    return list_mon


def _get_indices_address_mon(monome, list_mon) -> tuple[int, list]:
    """Get the indices of the variable to multiply and the previous monome."""
    if not sum(monome) > 2:
        raise ValueError("Must have more than 2 non-zero terms.")
    indices = np.where(monome)[0].tolist()
    reduced = monome.copy()
    reduced[indices[0]] = 0
    return indices[0], list_mon.index(reduced)


@attrs.frozen
class LookupUnwrapped(Bloq):
    """
    Address corresponding to a single bit are loaded using only a MultiTargetCNOT, this gains w + 1
    Toffoli, but it requires the actually loaded table to be recomputed from the original table to
    account for scenario where multiple addresses are loaded (and T[0] which is loaded by default).

    #!This lookup logical implementation is not exactly the same implementation as presented in
    #!https://arxiv.org/abs/2610.37840 which does not change the physical resource estimation
    #!as the implementation resource estimation goes directly from Lookup to physical count.

    This lookup only differs in the order of the term in the unary iteration, which does not allow
    to replace CNOT corrections of And gate injection by changing the classical data loaded.

    Parameters
    ----------
    T : Tuple[SymbolicInt]
        Table to load using the lookup, converted into tuple if not symbolic
    w : SymbolicInt
        Size of the selection register
    n : SymbolicInt
        Size of the target register

    Attributes
    ----------
    selection : QBit((self.w,))
        Array of QBit holding x
    target : QAny(self.n)
        Register that will be xored.

    """

    T: tuple[SymbolicInt] = attrs.field(converter=tuple_or_symbolic)
    w: SymbolicInt
    n: SymbolicInt

    def __attrs_post_init__(self):
        if is_symbolic(self.T, self.w, self.n):
            return

        if self.w < 0:
            raise ValueError(f"w must be non-negative, got {self.w}.")
        if self.n < 0:
            raise ValueError(f"n must be non-negative, got {self.n}.")

        if len(self.T) != 2**self.w:
            raise ValueError(f"expected len(T) == 2**w; got len(T)={len(self.T)} and w={self.w}.")

        for i, val in enumerate(self.T):
            if is_symbolic(val):
                continue
            if not 0 <= int(val) < 2**self.n:
                raise ValueError(
                    f"T[{i}]={val} does not fit in an unsigned {self.n}-bit target register."
                )

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("selection", QBit(), shape=(self.w,)),
                Register("target", QAny(self.n)),
            ]
        )

    @cached_property
    def _recomputed_table(self):
        """
        Recompute the table to load to account for the fact that certain address are loaded when
        they shouldn't for example we always load table[0] and thus need to xor it with all other
        table value

        Returns
        -------
        new_table: Tuple[SymbolicInt]
        """
        new_table = []
        for i in range(len(self.T)):
            new_table.append(self.T[i])
            for j in range(i):
                if i & j == j:
                    new_table[i] ^= new_table[j]
        return new_table

    def _compute_monome(
        self,
        bb: BloqBuilder,
        selection: SoquetT,
        monomial_soqs: SoquetT,
        mon: list,
        i: int,
        monomial_bits: list,
        monomial_uncompute_stack: list,
    ):
        assert sum(mon) >= 2
        if sum(mon) == 2:
            selection_idx, factor_idx = _get_indices_two(mon)
            (
                (selection[selection_idx], selection[factor_idx]),
                monomial_soqs[i],
            ) = bb.add(
                And(),
                ctrl=(selection[selection_idx], selection[factor_idx]),
            )
        else:
            selection_idx, factor_idx = _get_indices_address_mon(mon, monomial_bits)
            (
                (selection[selection_idx], monomial_soqs[factor_idx]),
                monomial_soqs[i],
            ) = bb.add(
                And(),
                ctrl=(selection[selection_idx], monomial_soqs[factor_idx]),
            )
        monomial_uncompute_stack.append(
            (selection_idx, factor_idx, i, QAny(len(selection)).from_bits(mon))
        )
        return selection, monomial_soqs

    # uncompte a monome given the 2 terms of its product (selection_idx,factor_idx)
    def _uncompute_monome(
        self,
        bb: BloqBuilder,
        selection: SoquetT,
        monomial_soqs: SoquetT,
        mon: list,
        factor_indices: tuple[int, int, int],
    ):
        assert sum(mon) >= 2
        selection_idx, factor_idx, i = factor_indices
        if sum(mon) == 2:
            (selection[selection_idx], selection[factor_idx]) = bb.add(
                And().adjoint(),
                ctrl=(selection[selection_idx], selection[factor_idx]),
                target=monomial_soqs[i],
            )
        else:
            (selection[selection_idx], monomial_soqs[factor_idx]) = bb.add(
                And().adjoint(),
                ctrl=(selection[selection_idx], monomial_soqs[factor_idx]),
                target=monomial_soqs[i],
            )
        return selection, monomial_soqs

    def _uncompute_useless_monomials(
        self,
        bb: BloqBuilder,
        selection: SoquetT,
        monomial_soqs: SoquetT,
        monomial_bits: list,
        monomial_addrs: list,
        monomial_uncompute_stack: list,
        i: int,
    ):
        """
        Uncompute all monomials that are no longer needed.
        A monome can be safely uncomputed once it is no longer a subword of the next monomial to
        build.

        Parameters
        ----------
        monomial_uncompute_stack : list
            list of all currently live monomial
        """
        if i == len(monomial_bits) - 1:
            selection_idx, factor_idx, k, _ = monomial_uncompute_stack.pop()
            selection, monomial_soqs = self._uncompute_monome(
                bb,
                selection,
                monomial_soqs,
                monomial_bits[k],
                (selection_idx, factor_idx, k),
            )
        else:
            while (
                monomial_uncompute_stack != []
                and monomial_uncompute_stack[-1][-1] & monomial_addrs[i + 1]
                != monomial_uncompute_stack[-1][-1]
            ):
                selection_idx, factor_idx, k, _ = monomial_uncompute_stack.pop()
                selection, monomial_soqs = self._uncompute_monome(
                    bb,
                    selection,
                    monomial_soqs,
                    monomial_bits[k],
                    (selection_idx, factor_idx, k),
                )
        return selection, monomial_soqs, monomial_uncompute_stack

    def build_composite_bloq(  # type: ignore[override]
        self, bb: BloqBuilder, *, selection, target
    ) -> dict[str, "SoquetT"]:
        target = bb.add(XorK(QAny(self.n), self._recomputed_table[0]), x=target)
        selection = selection[::-1]
        # 1st, for each linear term, we load the associate table value
        for i in range(self.w):
            selection[i], target = bb.add(
                CtrlXorK(QAny(self.n), self._recomputed_table[2**i]),
                ctrl=selection[i],
                x=target,
            )
        selection = selection[::-1]
        monomial_bits = _recursive_monomials_building([0] * self.w)
        monomial_addrs = [QAny(self.w).from_bits(mon) for mon in monomial_bits]
        # list of currently live monomials that remains to be uncomputed
        monomial_uncompute_stack: list[tuple[int, int, int, int]] = []
        monomial_soqs = np.empty(2**self.w - self.w - 1, dtype=object)

        # We generate monomials as needed, in a clever order so that we never need to compute more
        # then one monome to get the next one
        for i, mon in enumerate(monomial_bits):
            selection, monomial_soqs = self._compute_monome(
                bb,
                selection,
                monomial_soqs,
                mon,
                i,
                monomial_bits,
                monomial_uncompute_stack,
            )
            # Load their associated table values
            monomial_soqs[i], target = bb.add(
                CtrlXorK(QAny(self.n), self._recomputed_table[monomial_addrs[i]]),
                ctrl=monomial_soqs[i],
                x=target,
            )
            # These monomials are then uncomputed, only if they aren't needed for futur monomials
            selection, monomial_soqs, monomial_uncompute_stack = self._uncompute_useless_monomials(
                bb,
                selection,
                monomial_soqs,
                monomial_bits,
                monomial_addrs,
                monomial_uncompute_stack,
                i,
            )

        return {"selection": selection, "target": target}

    def build_call_graph(self, ssa):
        if is_symbolic(self.T):
            T0 = K
        else:
            T0 = self._recomputed_table[0]
        counts = Counter()

        counts[And()] += 2**self.w - self.w - 1
        counts[And().adjoint()] += 2**self.w - self.w - 1
        counts[XorK(QAny(self.n), T0)] += 1
        counts[CtrlXorK(QAny(self.n), K)] += 2**self.w - 1
        if counts[And()] == 0:
            del counts[And()]
            del counts[And().adjoint()]
        return counts

    def my_static_costs(self, cost_key):
        if isinstance(cost_key, QubitCount):
            return 2 * self.w - 1 + self.n
        return NotImplemented


@attrs.frozen
class Lookup(LookupAbstract):
    """
    Wrapper for LookupUnwrapped.
    The target register here is right sided to be consistent with the Phaseup.

    Parameters
    ----------
    T : Tuple[SymbolicInt]
        Table to load using the lookup, converted into tuple if not symbolic
    w : SymbolicInt
        Size of the selection register
    n : SymbolicInt
        Size of the target register


    Attributes
    ----------
    selection : QAny(self.w)
        Address register
    target : QAny(self.n), Side.RIGHT
        Register that will be xored.
    """

    @cached_property
    def lookup_unwrapped(self):
        return LookupUnwrapped(self.T, self.w, self.n)


@attrs.frozen
class LookupUncompute(Bloq):
    """
    Resource efficient uncompute of a lookup table.
    We assume the size of the target register to be big enough to avoid needing auxiliary qubits.
    (ie if w=selection_bitsize and n=target_bitsize, 2*sqrt(2**w)< n)
    This does not implement a build_decompose_bloq as implementing it would require the tracking of
    all measured out QBit into CBit, to be then use to compute the appropriate correction table.
    However it gives all resource estimate and classical/quantum action.

    This bloq implements:
    |selection>|target> -> |selection>
    the method which should be implemented in build_decompose_bloq goes as follows
    Input: |selection>|target>
    measure |target> in the X basis
    apply phase correction to |selection> so that we output |selection>
    Output: |selection>

    Parameters
    ----------
    lookup : Bloq
        Table lookup bloq to uncompute
    w : SymbolicInt
        Size of the selection register
    n : SymbolicInt
        Size of the target register

    Attributes
    ----------
        selection: QBit((self.w,))
        target: QAny(self.n), side=Side.LEFT

    ref: Gidney [How to factor 2048 bit RSA integers with less than a million noisy qubits, A.3, https://arxiv.org/abs/2505.15917]
    """

    lookup: Bloq
    w: SymbolicInt
    n: SymbolicInt

    @cached_property
    def signature(self) -> Signature:
        L_reg = []
        for reg in self.lookup.signature:
            if reg.side == Side.RIGHT:
                reg_adj = Register(reg.name, reg.dtype, reg.shape, Side.LEFT)
                L_reg.append(reg_adj)
            else:
                L_reg.append(reg)
        return Signature(L_reg)

    def on_classical_vals(self, selection, target):
        return {"selection": selection}

    def adjoint(self) -> Bloq:
        return self.lookup

    def build_call_graph(self, ssa):
        h1 = self.w // 2
        h2 = self.w - h1
        return {
            And(): 2**h1 + 2**h2 - h1 - h2 - 2,
            And().adjoint(): 2**h1 + 2**h2 - h1 - h2 - 2,
            CZ(): 2**h1 - 1,
            MeasureX(): self.n,
            # We don't count 1 qubit gate, should be 2**w - 1 (measurement being counted separately)
        }

    def my_tensors(self, incoming, outgoing):
        # We obtain the tensor by getting the tensor of the Lookup we are uncomputing and
        # taking the transpose
        def _reg_bit_inds(reg: Register, cxn_t):
            if reg.shape:
                return [(cxn_t[idx], j) for idx in reg.all_idxs() for j in range(reg.bitsize)]
            return [(cxn_t, j) for j in range(reg.bitsize)]

        out_inds = []
        for reg in self.signature.rights():
            out_inds.extend(_reg_bit_inds(reg, outgoing[reg.name]))

        in_inds = []
        for reg in self.signature.lefts():
            in_inds.extend(_reg_bit_inds(reg, incoming[reg.name]))

        # lookup should have right-only target registers, so uncompute corresponds to
        # the adjoint isometry with matrix U^\dagger.
        dense = self.lookup.tensor_contract().conj().T
        dense = dense.reshape((2,) * (len(out_inds) + len(in_inds)))
        return [qtn.Tensor(data=dense, inds=out_inds + in_inds, tags=[str(self)])]
