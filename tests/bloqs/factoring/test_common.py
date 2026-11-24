"""Window layout shared by ComputeResidue and ComputeResidue†.

The partitions are checked through their classical action: window j of a value must hold
bits j*w, ..., j*w + w - 1 of it, the paper's convention (offsets J_j = j*w).
"""

import itertools
import random
from functools import cached_property

import attrs
import pytest
import sympy as sp
from qualtran import Bloq, BloqBuilder, CBit, CtrlSpec, QUInt, Signature
from sympy.ntheory import primerange, primitive_root

from scharcode.bloqs.factoring._common import (
    ResidueTables,
    ResidueWindows,
    address_partition,
    helper_partition,
    l_p,
    last_window,
    num_windows,
    underflow_phasefix_generalizer,
    window_sizes,
)
from scharcode.bloqs.operations._utils import T
from scharcode.classical.rns import RnsParameters

# (ell, w3a, w3b): full windows, partial top windows, ell <= 2*w3a (no outer loop),
# and ell = 12, w3a = 2 (several outer windows, which pins the window order).
CASES = [(8, 2, 2), (12, 2, 2), (12, 2, 3), (11, 3, 4), (10, 3, 3), (5, 3, 2), (6, 3, 6), (2, 4, 1)]


def _window(value: int, offset: int, size: int) -> int:
    return (value >> offset) & ((1 << size) - 1)


@pytest.mark.parametrize(
    "bits, w, expected", [(12, 2, (2,) * 6), (11, 3, (3, 3, 3, 2)), (5, 8, (5,)), (1, 1, (1,))]
)
def test_window_sizes(bits, w, expected):
    assert window_sizes(bits, w) == expected
    assert num_windows(bits, w) == len(expected)
    assert last_window(bits, w) == expected[-1]


def test_symbolic_windows():
    w3a, w3b = sp.symbols("w3a w3b", positive=True, integer=True)
    windows = ResidueWindows(l_p, w3a, w3b)
    assert windows.num_w3a == sp.ceiling(l_p / w3a)
    assert windows.num_w3b == sp.ceiling(l_p / w3b)
    assert windows.last_w3a == w3a
    assert windows.last_w3b == w3b
    assert windows.num_outer_steps == sp.ceiling(l_p / w3a) - 2
    assert windows.init_bitsize == 2 * w3a
    with pytest.raises(ValueError, match="concrete"):
        _ = windows.exponent_partition


@pytest.mark.parametrize("field", ["l_p", "w3a", "w3b"])
def test_rejects_empty_windows(field):
    kwargs = {"l_p": 8, "w3a": 2, "w3b": 2, field: 0}
    with pytest.raises(ValueError, match=field):
        ResidueWindows(**kwargs)


@pytest.mark.parametrize("ell, w3a, w3b", CASES)
def test_sizes_cover_registers(ell, w3a, w3b):
    windows = ResidueWindows(ell, w3a, w3b)
    assert sum(windows.w3b_sizes) == ell
    assert len(windows.w3b_sizes) == windows.num_w3b
    outer = sum(windows.w3a_sizes[j_a] for j_a in windows.outer_windows)
    assert windows.init_bitsize + outer == ell
    assert len(windows.outer_windows) == windows.num_outer_steps


@pytest.mark.parametrize("ell, w3a, w3b", CASES)
def test_exponent_partition_layout(ell, w3a, w3b):
    windows = ResidueWindows(ell, w3a, w3b)
    for s in (0, 1, (1 << ell) - 1, 0b1011_0110_1101 % (1 << ell)):
        out = windows.exponent_partition.call_classically(x=s)
        names = [reg.name for reg in windows.exponent_partition.regs]
        vals = dict(zip(names, out, strict=True))
        assert vals["s_init"] == _window(s, 0, windows.init_bitsize)
        for j_a in windows.outer_windows:
            assert vals[f"x_a_{j_a}"] == _window(s, j_a * w3a, windows.w3a_sizes[j_a])


@pytest.mark.parametrize("ell, w3a, w3b", CASES)
def test_product_partition_layout(ell, w3a, w3b):
    windows = ResidueWindows(ell, w3a, w3b)
    for r in (0, 1, (1 << ell) - 1, 0b0110_1011_1001 % (1 << ell)):
        out = windows.product_partition.call_classically(x=r)
        names = [reg.name for reg in windows.product_partition.regs]
        vals = dict(zip(names, out, strict=True))
        for j_b, size in enumerate(windows.w3b_sizes):
            assert vals[f"y_b_{j_b}"] == _window(r, j_b * w3b, size)


@pytest.mark.parametrize("w3a, w3b", [(2, 2), (3, 2), (1, 4)])
def test_address_partition_matches_table_order(w3a, w3b):
    """The T3 tables are indexed [x_a, y_b]; the flat Lookup address must be x_a*2^w3b + y_b."""
    join = address_partition(w3a, w3b).adjoint()
    for x_a, y_b in itertools.product(range(2**w3a), range(2**w3b)):
        (address,) = join.call_classically(x_a=x_a, y_b=y_b)
        assert address == x_a * 2**w3b + y_b


def test_helper_partition_puts_underflow_on_top():
    ell = 5
    for value in (0, 7, 1 << ell, (1 << (ell + 1)) - 1):
        u, h = helper_partition(ell).call_classically(x=value)
        assert u == value >> ell
        assert h == value & ((1 << ell) - 1)


@pytest.mark.parametrize("ell, w3a, w3b", CASES)
def test_split_join_round_trip(ell, w3a, w3b):
    windows = ResidueWindows(ell, w3a, w3b)
    bb = BloqBuilder()
    s = bb.add_register("s", QUInt(ell))
    r = bb.add_register("r", QUInt(ell))
    s_init, x_a = windows.split_exponent(bb, s)
    assert sorted(x_a) == list(windows.outer_windows)
    y_b = windows.split_product(bb, r)
    assert len(y_b) == windows.num_w3b
    r = windows.join_product(bb, y_b)
    s = windows.join_exponent(bb, s_init, x_a)
    cbloq = bb.finalize(s=s, r=r)

    for s_val, r_val in [
        (0, 0),
        (1, (1 << ell) - 1),
        (0b1010_0101_1100 % (1 << ell), 3 % (1 << ell)),
    ]:
        assert cbloq.call_classically(s=s_val, r=r_val) == (s_val, r_val)


# ---- T3 tables ---------------------------------------------------------------------------


def _largest_prime(bits: int) -> int:
    return list(primerange(2 ** (bits - 1), 2**bits))[-1]


def _tables(ell: int, w3a: int, w3b: int) -> ResidueTables:
    p = _largest_prime(ell)
    return ResidueTables(p, int(primitive_root(p)), w3a, w3b)


def _exponents(ell: int) -> list[int]:
    if ell <= 10:
        return list(range(2**ell))
    return random.Random(ell).sample(range(2**ell), 300)


def _mod_add(h: int, loaded: int, threshold: int, p: int) -> int:
    """One step of the modular addition: h - V on l_p + 1 bits, then + p on underflow.

    Also checks the comparison of UnderflowPhasefix, the underflow bit must equal
    [new h >= threshold] (paragraph "Phase corrections" of Appendix B).
    """
    assert 0 <= h < p and 0 <= loaded < p and 0 < threshold <= p
    h -= loaded
    underflow = h < 0
    if underflow:
        h += p
    assert underflow == (h >= threshold)
    return h


def _step(tables, loaded_table, x_a, y_b, size_b, h):
    address = x_a * 2**size_b + y_b
    loaded = loaded_table[address]
    return _mod_add(h, loaded, tables.complement(loaded_table)[address], tables.p)


@pytest.mark.parametrize("ell, w3a, w3b", CASES)
def test_t3_tables_match_definition(ell, w3a, w3b):
    tables = _tables(ell, w3a, w3b)
    p, g_p, w = tables.p, tables.g_p, tables.windows
    g_inv = pow(g_p, -1, p)
    assert tables.alpha2 == tuple(pow(g_p, x, p) for x in range(2**w.init_bitsize))
    for j_a, j_b in itertools.product(range(w.num_w3a), range(w.num_w3b)):
        size_a, size_b = w.w3a_sizes[j_a], w.w3b_sizes[j_b]
        t3, t3_inv = tables.t3(j_a, j_b), tables.t3_inv(j_a, j_b)
        assert len(t3) == len(t3_inv) == 2 ** (size_a + size_b)
        for x_a, y_b in itertools.product(range(2**size_a), range(2**size_b)):
            address = x_a * 2**size_b + y_b
            unit = y_b * 2 ** (j_b * w3b)
            assert t3[address] == unit * pow(g_p, x_a * 2 ** (j_a * w3a), p) % p
            assert t3_inv[address] == unit * pow(g_inv, x_a * 2 ** (j_a * w3a), p) % p
        for loaded in (tables.p_minus_t3(j_a, j_b), tables.p_minus_t3_inv(j_a, j_b), t3):
            assert all(0 <= v < p for v in loaded)
            assert all(v + c == p for v, c in zip(loaded, tables.complement(loaded), strict=True))
        assert all((v + t) % p == 0 for v, t in zip(tables.p_minus_t3(j_a, j_b), t3, strict=True))


@pytest.mark.parametrize("ell, w3a, w3b", [(8, 2, 2), (12, 2, 3), (11, 3, 4), (5, 3, 2)])
def test_t3_tables_match_rns_parameters(ell, w3a, w3b):
    """The Lookup tables are the RNS tables of `classical.rns`, cut to the window sizes."""
    primes = list(primerange(2 ** (ell - 1), 2**ell))[-2:]
    params = RnsParameters(
        N=1_000_003,
        g=2,
        n_e=4,
        ell=ell,
        kappa=5,
        primes=primes,
        generators=[int(primitive_root(p)) for p in primes],
        w1=2,
        w3a=w3a,
        w3b=w3b,
        w4=2,
    )
    for i, (p, g_p) in enumerate(zip(params.primes, params.generators, strict=True)):
        tables = ResidueTables(p, g_p, w3a, w3b)
        w = tables.windows
        assert tables.alpha2 == tuple(params.alpha2_table[i].tolist())
        for j_a, j_b in itertools.product(range(w.num_w3a), range(w.num_w3b)):
            rows, cols = 2 ** w.w3a_sizes[j_a], 2 ** w.w3b_sizes[j_b]
            for ours, theirs in [(tables.t3, params.t3), (tables.t3_inv, params.t3_inv)]:
                expected = theirs[i, j_a, j_b, :rows, :cols].ravel().tolist()
                assert ours(j_a, j_b) == tuple(expected)


@pytest.mark.parametrize("ell, w3a, w3b", CASES)
def test_t3_tables_compute_and_erase_residue(ell, w3a, w3b):
    """Classical run of ComputeResidue then ComputeResidue†, with the windows of `_common`.

    ComputeResidue must give g_p^s mod p. Each iteration of ComputeResidue† must turn
    alpha_{j_a+1} into alpha_{j_a} and clear the register that held alpha_{j_a+1}, so that
    alpha_2 is left for the final unloading. Every step also checks UnderflowPhasefix.
    """
    tables = _tables(ell, w3a, w3b)
    p, g_p, w = tables.p, tables.g_p, tables.windows

    def window(value, j, width, size):
        return (value >> (j * width)) & ((1 << size) - 1)

    for s in _exponents(ell):
        s_init = s & ((1 << w.init_bitsize) - 1)
        x_a = {j_a: window(s, j_a, w3a, w.w3a_sizes[j_a]) for j_a in w.outer_windows}

        alpha = tables.alpha2[s_init]
        for j_a in w.outer_windows:
            h = 0
            for j_b, size_b in enumerate(w.w3b_sizes):
                y_b = window(alpha, j_b, w3b, size_b)
                h = _step(tables, tables.p_minus_t3(j_a, j_b), x_a[j_a], y_b, size_b, h)
            alpha = h  # role swap; alpha_{j_a} is measured out
        assert alpha == pow(g_p, s, p)

        for j_a in reversed(w.outer_windows):
            previous = 0
            for j_b, size_b in enumerate(w.w3b_sizes):
                y_b = window(alpha, j_b, w3b, size_b)
                loaded = tables.p_minus_t3_inv(j_a, j_b)
                previous = _step(tables, loaded, x_a[j_a], y_b, size_b, previous)
            for j_b, size_b in enumerate(w.w3b_sizes):
                y_b = window(previous, j_b, w3b, size_b)
                alpha = _step(tables, tables.t3(j_a, j_b), x_a[j_a], y_b, size_b, alpha)
            assert alpha == 0
            alpha = previous
        assert alpha == tables.alpha2[s_init] == pow(g_p, s_init, p)


def test_t3_tables_validation():
    with pytest.raises(ValueError, match="prime"):
        ResidueTables(15, 2, 2, 2)
    with pytest.raises(ValueError, match="prime to p"):
        ResidueTables(13, 26, 2, 2)
    with pytest.raises(ValueError, match="concrete"):
        ResidueTables(sp.Symbol("p"), 2, 2, 2)
    with pytest.raises(IndexError):
        _tables(8, 2, 2).t3(4, 0)


# ---- UnderflowPhasefix generalizer -------------------------------------------------------


@attrs.frozen
class UnderflowPhasefix(Bloq):
    """Stand-in with the table field of the ported class (`table`)."""

    n: int
    table: tuple

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(y=self.n)


def test_underflow_phasefix_generalizer():
    bloq = UnderflowPhasefix(3, (1, 2, 3))
    assert underflow_phasefix_generalizer(bloq) == UnderflowPhasefix(3, T)
    controlled = bloq.controlled(CtrlSpec(CBit()))
    assert underflow_phasefix_generalizer(controlled) == UnderflowPhasefix(3, T)


def test_underflow_phasefix_generalizer_leaves_other_bloqs():
    other = QUInt(3)  # not a bloq at all
    assert underflow_phasefix_generalizer(other) is other
    controlled_other = address_partition(2, 2).controlled(CtrlSpec(CBit()))
    assert underflow_phasefix_generalizer(controlled_other) is controlled_other
