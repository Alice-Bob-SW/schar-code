"""Shared test helpers (importable as `helpers` through `pythonpath` in pyproject.toml)."""

import itertools
from collections.abc import Callable, Iterable
from typing import Any

import attrs
import numpy as np
import sympy as sp
from qualtran import Bloq, QUInt
from qualtran.resource_counting import QubitCount, get_cost_value
from qualtran.simulation.classical_sim import do_phased_classical_simulation


# Small `RnsParameters.search` instances, keyed by name: N and g, then the size and window
# keyword arguments. Small enough for exhaustive classical checks.
SMALL_INSTANCES = {
    # N = 1009 * 1013. Every window is full.
    "full-windows": dict(N=1022117, g=5, n_e=6, ell=9, kappa=6, w1=2, w3a=3, w3b=3, w4=3),
    # N = 1000003 * 1000033. n_e % w1, ell % w3a, ell % w3b and ell % w4 are all nonzero, so
    # every table has a partial last window.
    "partial-windows": dict(N=1000036000099, g=7, n_e=8, ell=10, kappa=8, w1=3, w3a=3, w3b=4, w4=3),
}


# (exponent, accumulator) pairs from ten runs of the oracle with the tables of the RSA-100
# reference fixture, recorded with Gidney's code (see `data/rsa100_reference/PROVENANCE.md`).
RSA100_REFERENCE_OUTPUTS = [
    (1600, 925495),
    (7014, 930091),
    (353, 535514),
    (3447, 103731),
    (4779, 486218),
    (5081, 204031),
    (7980, 141719),
    (3322, 674042),
    (1124, 633541),
    (4958, 33942),
]


def _is_classical(dtype) -> bool:
    return dtype.num_qubits == 0 and getattr(dtype, "num_cbits", 0) > 0


def assert_valid_decomposition(bloq: Bloq):
    """
    `qualtran.testing.assert_valid_bloq_decomposition`, accepting classical-bit wires.
    """
    from qualtran import LeftDangle, RightDangle, Side
    from qualtran.testing import (
        assert_connections_compatible,
        assert_registers_match_dangling,
        assert_registers_match_parent,
        assert_soquets_belong_to_registers,
        assert_soquets_used_exactly_once,
    )
    from qualtran._infra.data_types import check_dtypes_consistent

    cbloq = assert_registers_match_parent(bloq)
    classical = [
        c
        for c in cbloq.connections
        if _is_classical(c.left.reg.dtype) or _is_classical(c.right.reg.dtype)
    ]
    quantum = tuple(c for c in cbloq.connections if c not in classical)
    assert_connections_compatible(attrs.evolve(cbloq, connections=quantum))
    for cxn in classical:
        left, right = cxn.left.reg, cxn.right.reg
        assert _is_classical(left.dtype) and _is_classical(right.dtype), cxn
        assert check_dtypes_consistent(left.dtype, right.dtype), cxn
        assert left.side & (Side.LEFT if cxn.left.binst is LeftDangle else Side.RIGHT), cxn
        assert right.side & (Side.RIGHT if cxn.right.binst is RightDangle else Side.LEFT), cxn
    assert_registers_match_dangling(cbloq)
    assert_soquets_belong_to_registers(cbloq)
    assert_soquets_used_exactly_once(cbloq)
    return cbloq


def phased_run(
    cbloq: Bloq, vals: dict[str, int], rng: np.random.Generator
) -> tuple[dict[str, int], complex]:
    """Phased classical simulation with random M_X outcomes drawn from `rng`.

    Returns the output values as Python ints and the phase. Simulate a flattened
    decomposition: simulating the bloq itself uses its `on_classical_vals` and gives phase 1
    without running any gate. Leaves that do not decompose keep their own action:
    `LookupUncompute` contributes no phase and does not check its target.
    """
    out, phase = do_phased_classical_simulation(cbloq, vals, rng=rng)
    return {k: int(v) for k, v in out.items()}, complex(phase)


def bit_array_domain(bitsize: int) -> list[np.ndarray]:
    """All values of a `QBit()` array register of length `bitsize`, as big-endian bit arrays."""
    return [np.array(QUInt(bitsize).to_bits(x), dtype=np.uint8) for x in range(2**bitsize)]


def left_parameter_ranges(bloq: Bloq) -> dict[str, Any]:
    """Full classical domain of each LEFT register, including 1-D arrays of bits."""
    ranges: dict[str, Any] = {}
    for reg in bloq.signature.lefts():
        if reg.shape:
            assert len(reg.shape) == 1, f"{reg.name} has a multi-dimensional shape"
            ranges[reg.name] = bit_array_domain(reg.shape[0])
        else:
            ranges[reg.name] = reg.dtype.get_classical_domain()
    return ranges


def assert_consistent_symbolic_qubit_count(
    bloq_cls: type[Bloq],
    arg_specs: dict[str, tuple[type | Callable, dict[sp.Expr, Iterable[int]]]] | None = None,
    **variable_ranges: Iterable[int],
) -> None:
    r"""Assert that the symbolic qubit count matches the decomposition over a parameter sweep.

    Builds `bloq_cls` with SymPy symbols to get a symbolic `QubitCount`, then, for each
    numeric assignment, checks that substituting it equals the `QubitCount` of the numeric
    bloq's decomposition.

    Parameters
    ----------
    bloq_cls
        Bloq class (not instance); its `init=True` attrs fields are the constructor parameters.
    arg_specs
        `{field: (builder, {symbol: range, ...})}`. `builder` is called with the symbols (or
        their values) in the insertion order of the inner dict, e.g.
        `{"dtype_i": (QUInt, {n: range(3, 7)})}`.
    **variable_ranges
        Simple mode, used when `arg_specs` is None: one range per constructor field, e.g.
        `n=range(3, 10)`.

    Raises
    ------
    TypeError
        If both or neither of `arg_specs` and `variable_ranges` are given.
    ValueError
        If the given names don't match the constructor fields.
    AssertionError
        If any assignment yields a mismatch.
    """

    def _id(x: Any) -> Any:
        return x

    init_field_names = [f.name for f in attrs.fields(bloq_cls) if f.init]
    if (arg_specs is None) == (not variable_ranges):
        raise TypeError("Give either arg_specs or variable_ranges, not both or neither.")
    if arg_specs is None:
        if set(init_field_names) != set(variable_ranges):
            raise ValueError(
                f"variable_ranges keys {set(variable_ranges)} != fields {set(init_field_names)}"
            )
        arg_specs = {
            name: (_id, {sp.Symbol(name): variable_ranges[name]}) for name in init_field_names
        }
    elif set(init_field_names) != set(arg_specs):
        raise ValueError(f"arg_specs keys {set(arg_specs)} != fields {set(init_field_names)}")

    symbolic_kwargs: dict[str, Any] = {}
    field_symbols: dict[str, list[sp.Expr]] = {}
    all_symbols: list[sp.Expr] = []
    all_ranges: list[Iterable[int]] = []
    for name in init_field_names:
        builder, symbol_ranges = arg_specs[name]
        syms = list(symbol_ranges)
        field_symbols[name] = syms
        symbolic_kwargs[name] = builder(*syms)
        all_symbols.extend(syms)
        all_ranges.extend(symbol_ranges[s] for s in syms)

    symbolic_bloq = bloq_cls(**symbolic_kwargs)
    qubit_count_symbolic = sp.sympify(get_cost_value(symbolic_bloq, QubitCount()))

    for vals in itertools.product(*all_ranges):
        subs = dict(zip(all_symbols, vals, strict=True))
        numeric_kwargs = {
            name: arg_specs[name][0](*(subs[s] for s in field_symbols[name]))
            for name in init_field_names
        }
        numeric_bloq = bloq_cls(**numeric_kwargs)
        expected = get_cost_value(numeric_bloq.decompose_bloq(), QubitCount())
        np.testing.assert_equal(
            qubit_count_symbolic.subs(subs),
            expected,
            err_msg=f"{symbolic_bloq=} {subs=}",
        )
