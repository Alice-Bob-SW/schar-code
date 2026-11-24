import numpy as np
import pytest
import sympy as sp
from qualtran import QUInt
from qualtran.resource_counting import get_bloq_callee_counts
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.testing import assert_equivalent_bloq_counts, assert_valid_bloq_decomposition

from helpers import RSA100_REFERENCE_OUTPUTS, SMALL_INSTANCES

from scharcode.bloqs.factoring._common import window_sizes
from scharcode.bloqs.factoring.immediate_mbu import ApproxModExp
from scharcode.bloqs.factoring.immediate_mbu.approx_mod_exp import (
    accumulateresidue_generalizer,
    computeexponent_generalizer,
    get_generalizer_subbloq_approx_mod_exp,
)
from scharcode.bloqs.operations._utils import T
from scharcode.classical.rns import RnsParameters


@pytest.fixture(scope="module", params=sorted(SMALL_INSTANCES))
def params(request) -> RnsParameters:
    kwargs = dict(SMALL_INSTANCES[request.param])
    return RnsParameters.search(kwargs.pop("N"), kwargs.pop("g"), **kwargs)


@pytest.fixture(scope="module")
def bloq(params) -> ApproxModExp:
    return ApproxModExp.from_parameters(params)


def test_from_parameters_sizes(params, bloq):
    assert bloq.primes == params.primes
    assert bloq.generators == params.generators
    assert bloq.N == params.N
    assert bloq.dtype_exp == QUInt(params.n_e)
    assert bloq.dtype_s == QUInt(params.s_bitsize)
    assert bloq.dtype_acc == QUInt(params.kappa)
    assert (bloq.w1, bloq.w3a, bloq.w3b, bloq.w4) == (
        params.w1,
        params.w3a,
        params.w3b,
        params.w4,
    )


def test_from_parameters_table_layout(params, bloq):
    """Windows most significant first, each with 2^(its size) entries."""
    w1_sizes = window_sizes(params.n_e, params.w1)[::-1]
    w4_sizes = window_sizes(params.ell, params.w4)[::-1]
    assert len(bloq.table1) == params.num_primes + 1
    assert len(bloq.table4) == params.num_primes
    for row in bloq.table1:
        assert [len(window) for window in row] == [2**b for b in w1_sizes]
    for row in bloq.table4:
        assert [len(window) for window in row] == [2**b for b in w4_sizes]
    # table4 holds the loaded values (M - T4) mod M; window 0 is the top window of T4.
    top = len(w4_sizes) - 1
    for i in range(params.num_primes):
        expected = [(params.M - int(t)) % params.M for t in params.t4[i, top, : 2 ** w4_sizes[0]]]
        assert list(bloq.table4[i][0]) == expected


def test_valid_decomposition(bloq):
    assert_valid_bloq_decomposition(bloq)


def test_adds_approx_top_bits(params, bloq, rng):
    """output = input + Ṽ(e) mod M, for accumulator inputs in [0, M)."""
    cbloq = bloq.decompose_bloq()
    exponents = [0, 2**params.n_e - 1, *(int(e) for e in rng.integers(0, 2**params.n_e, 6))]
    accs = [0, params.M - 1, *(int(a) for a in rng.integers(0, params.M, 3))]
    for e in exponents:
        for acc in accs:
            exp_out, acc_out = cbloq.call_classically(exp=e, result=acc)
            assert int(exp_out) == e
            assert int(acc_out) == (acc + params.approx_top_bits(e)) % params.M, (e, acc)


# ---- ported from approx_mod_exp_test.py ------------------------------------------------------


def test_rejects_mismatched_rns_lengths():
    with pytest.raises(ValueError, match="generators and primes must be of same length"):
        ApproxModExp(
            generators=(2, 3),
            primes=(5,),
            N=77,
            dtype_exp=QUInt(4),
            dtype_s=QUInt(5),
            dtype_acc=QUInt(4),
            w1=2,
            w3a=2,
            w3b=2,
            w4=2,
            table1=((0,), (0,)),
            table4=((0,),),
        )


def test_bloq_counts(params, bloq):
    """The call graph equals the decomposition, with the sub-bloqs generalized.

    Every sub-bloq is reduced to its shape: tables become T, the generator of ComputeResidue
    becomes the symbol g and the prime becomes the first prime (all primes have ell bits).
    """
    reduce_gen, residue_gen, residue_adj_gen = get_generalizer_subbloq_approx_mod_exp(params.primes)
    generalizer = [
        ignore_split_join,
        ignore_alloc_free,
        reduce_gen,
        residue_gen,
        residue_adj_gen,
        computeexponent_generalizer,
        accumulateresidue_generalizer,
    ]
    assert_equivalent_bloq_counts(bloq, generalizer=generalizer)


def test_symbolic_call_graph():
    n_e, n_s, kappa, w1, w3a, w3b, w4, N = sp.symbols(
        "n_e n_s kappa w1 w3a w3b w4 N", positive=True, integer=True
    )
    bloq = ApproxModExp(
        generators=sp.Symbol("generators"),
        primes=sp.Symbol("primes"),
        N=N,
        dtype_exp=QUInt(n_e),
        dtype_s=QUInt(n_s),
        dtype_acc=QUInt(kappa),
        w1=w1,
        w3a=w3a,
        w3b=w3b,
        w4=w4,
        table1=T,
        table4=T,
    )
    counts = {type(b).__name__: c for b, c in get_bloq_callee_counts(bloq)}
    count_p = sp.Symbol("primes")
    assert counts["ComputeExponent"] == count_p + 1
    for name in ("ComputeResidue", "AccumulateResidue", "ComputeResidueAdjoint"):
        assert counts[name] == count_p
    assert counts["ReduceExponent"] == count_p
    assert counts["ReduceExponentAdjoint"] == count_p


@pytest.mark.slow
def test_rsa100_reference_outputs(rsa100_reference):
    """Regression against the recorded runs of the RSA-100 reference fixture.

    The oracle is built from the fixture's own tables flipped to the most-significant-first window
    order, with the accumulator starting at 0.
    """
    meta, arrays = rsa100_reference.metadata, rsa100_reference.arrays
    bloq = ApproxModExp(
        generators=arrays["generators"],
        primes=arrays["periods"],
        N=meta["modulus"],
        dtype_exp=QUInt(meta["num_input_qubits"]),
        dtype_s=QUInt(meta["len_dlog_accumulator"]),
        dtype_acc=QUInt(meta["len_accumulator"]),
        w1=meta["window1"],
        w3a=meta["window3a"],
        w3b=meta["window3b"],
        w4=meta["window4"],
        table1=np.flip(arrays["table1"], 1),
        table4=np.flip(arrays["table4"], 1),
    )
    for e, expected in RSA100_REFERENCE_OUTPUTS:
        exp_out, acc_out = bloq.call_classically(exp=e, result=0)
        assert (int(exp_out), int(acc_out)) == (e, expected)
