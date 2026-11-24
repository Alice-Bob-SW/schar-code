import pytest

from helpers import phased_run
from scharcode.bloqs.factoring.immediate_mbu import ApproxModExp
from scharcode.classical.rns import RnsParameters, delta_N

pytestmark = pytest.mark.slow

INSTANCE = dict(N=1022117, g=5, n_e=6, ell=9, kappa=6, w1=2, w3a=3, w3b=3, w4=3)


@pytest.fixture(scope="module")
def params() -> RnsParameters:
    kwargs = dict(INSTANCE)
    params = RnsParameters.search(kwargs.pop("N"), kwargs.pop("g"), **kwargs)
    assert params.satisfies_size_condition and params.satisfies_deviation_condition
    return params


@pytest.fixture(scope="module")
def flat_oracle(params):
    return ApproxModExp.from_parameters(params).decompose_bloq().flatten()


def test_end_to_end(params, flat_oracle, rng):
    exponents = [0, 2**params.n_e - 1, *(int(e) for e in rng.integers(0, 2**params.n_e, 3))]
    accs = [0, params.M - 1, int(rng.integers(0, params.M))]
    for e in exponents:
        v_tilde = params.approx_top_bits(e)
        v = pow(params.g, e, params.N)
        for acc in accs:
            out, phase = phased_run(flat_oracle, {"exp": e, "result": acc}, rng)
            assert out["exp"] == e
            assert out["result"] == (acc + v_tilde) % params.M, (e, acc)
            added = (out["result"] - acc) % params.M
            assert delta_N(v - (added << params.kappa_bar), params.N) <= params.max_deviation
            assert phase == 1, (e, acc)
