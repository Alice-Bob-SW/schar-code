import numpy as np
import pytest
from sympy.ntheory import primitive_root

from helpers import RSA100_REFERENCE_OUTPUTS, SMALL_INSTANCES

from scharcode.classical.rns import RnsParameters, delta_N, discrete_logs, num_windows


def search(name: str, seed: int = 0) -> RnsParameters:
    kwargs = dict(SMALL_INSTANCES[name])
    return RnsParameters.search(kwargs.pop("N"), kwargs.pop("g"), seed=seed, **kwargs)


def windows(value: int, w: int, count: int) -> list[int]:
    """The `count` windows of w bits of `value`, least significant first."""
    return [value >> (j * w) & ((1 << w) - 1) for j in range(count)]


@pytest.fixture(scope="module", params=sorted(SMALL_INSTANCES))
def params(request) -> RnsParameters:
    return search(request.param)


@pytest.fixture
def exponents(params, rng) -> list[int]:
    """Random exponents, plus the two extremes."""
    top = 2**params.n_e - 1
    return [0, top, *(int(e) for e in rng.integers(0, top, size=10))]


# ---- helpers ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("z", "N", "expected"), [(0, 7, 0), (3, 7, 3), (4, 7, 3), (7, 7, 0), (-1, 7, 1), (22, 7, 1)]
)
def test_delta_N(z, N, expected):
    assert delta_N(z, N) == expected


@pytest.mark.parametrize("p", [3, 5, 1019, 4093])
def test_discrete_logs_invert_exponentiation(p):
    g_p = int(primitive_root(p))
    logs = discrete_logs(list(range(1, p)), g_p, p)
    assert all(0 <= d <= p - 2 for d in logs)
    assert [pow(g_p, d, p) for d in logs] == list(range(1, p))


def test_discrete_logs_reject_multiples_of_p():
    with pytest.raises(ValueError, match="divisible"):
        discrete_logs([2 * 1019], 2, 1019)


# ---- prime-set search and validation -------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name", sorted(SMALL_INSTANCES))
def test_search_meets_paper_conditions(name, seed):
    params = search(name, seed)
    assert params.satisfies_size_condition  # L >= N^{n_e}
    assert params.satisfies_deviation_condition  # Equation `dev`
    assert list(params.primes) == sorted(params.primes)
    assert all(p.bit_length() == params.ell for p in params.primes)
    assert params.generators == tuple(int(primitive_root(p)) for p in params.primes)


def test_search_is_reproducible():
    assert search("partial-windows", seed=5) == search("partial-windows", seed=5)


def test_search_at_rsa100_scale(rsa100_reference):
    """for the fixture's N and sizes, a set meeting both conditions needs more primes.

    With seed 0 the search returns 113 primes, against 60 in the fixture.
    """
    md = rsa100_reference.metadata
    params = RnsParameters.search(
        md["modulus"],
        md["generator"],
        n_e=md["num_input_qubits"],
        ell=md["rns_primes_bit_length"],
        kappa=md["len_accumulator"],
        w1=md["window1"],
        w3a=md["window3a"],
        w3b=md["window3b"],
        w4=md["window4"],
    )
    assert params.satisfies_size_condition
    assert params.satisfies_deviation_condition
    assert params.num_primes > len(rsa100_reference.arrays["periods"])


def test_search_rejects_too_few_primes():
    with pytest.raises(ValueError, match="L < N"):
        RnsParameters.search(1022117, 5, n_e=60, ell=9, kappa=6, w1=2, w3a=3, w3b=3, w4=3)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (dict(primes=(257, 257), generators=(3, 3)), "distinct"),
        (dict(primes=(257, 263), generators=(3,)), "one generator per prime"),
        (dict(primes=(257, 1031), generators=(3, 14)), "not an 9-bit prime"),
        (dict(primes=(257, 255), generators=(3, 3)), "not an 9-bit prime"),
        (dict(primes=(257,), generators=(4,)), "does not generate"),
        (dict(primes=(257,), generators=(257,)), "does not generate"),
        (dict(kappa=21), "kappa"),
        (dict(kappa=0), "kappa"),
        (dict(w4=0), "Window"),
        (dict(primes=(), generators=()), "at least one prime"),
    ],
)
def test_constructor_rejects_malformed_rns(change, message):
    kwargs = dict(
        N=1022117, g=5, n_e=6, ell=9, kappa=6, primes=(257,), generators=(3,), w1=2, w3a=3,
        w3b=3, w4=3,
    )  # fmt: skip
    kwargs.update(change)
    with pytest.raises(ValueError, match=message):
        RnsParameters(**kwargs)


def test_constructor_rejects_prime_dividing_some_power():
    # g = 257 * 3 makes M_0 = g divisible by 257.
    with pytest.raises(ValueError, match="divides some M_k"):
        RnsParameters(
            N=1022117, g=771, n_e=6, ell=9, kappa=6, primes=(257,), generators=(3,), w1=2,
            w3a=3, w3b=3, w4=3,
        )  # fmt: skip


# ---- RNS constants (Appendix A) -------------------------------------------------------------


def test_sizes(params):
    assert params.kappa_bar == params.N.bit_length() - params.kappa
    assert params.M == params.N // 2**params.kappa_bar
    assert params.s_bitsize == params.ell + (params.n_e - 1).bit_length()


def test_powers(params):
    assert params.powers == tuple(pow(params.g, 2**k, params.N) for k in range(params.n_e))


def test_dlogs(params):
    """D_{p_i,k} is the discrete log of M_k mod p_i, in [0, p_i - 2]."""
    for i, (p, g_p) in enumerate(zip(params.primes, params.generators, strict=True)):
        for k, m_k in enumerate(params.powers):
            d = int(params.dlogs[i, k])
            assert 0 <= d <= p - 2
            assert pow(g_p, d, p) == m_k % p


def test_crt_factors(params):
    """u_{p_j} = 1 mod p_j and 0 modulo every other prime."""
    for j, u in enumerate(params.crt_factors):
        assert [u % p for p in params.primes] == [int(i == j) for i in range(params.num_primes)]


def test_crt_identity(params, exponents):
    """Equation `rns-exact`: the residues determine V-bar(e), and V-bar(e) mod N = g^e mod N."""
    for e in exponents:
        v_bar = params.unreduced_product(e)
        residues = params.residues(e)
        assert residues == tuple(v_bar % p for p in params.primes)
        assert v_bar < params.L  # from the size condition
        crt = sum(r * u for r, u in zip(residues, params.crt_factors, strict=True)) % params.L
        assert crt == v_bar
        assert crt % params.N == pow(params.g, e, params.N)


def test_truncated_crt_factors(params):
    """C_{j,k} is (u_{p_j} 2^k mod L mod N) >> kappa_bar, to within one unit, modulo M."""
    unit = 2**params.kappa_bar
    for j, u in enumerate(params.crt_factors):
        for k in range(params.ell):
            c = int(params.truncated_crt_factors[j, k])
            assert 0 <= c < params.M
            assert delta_N((u << k) % params.L % params.N - c * unit, params.N) < unit


def test_exact_top_bits(params, exponents):
    for e in exponents:
        assert params.exact_top_bits(e) == pow(params.g, e, params.N) >> params.kappa_bar


@pytest.mark.parametrize(
    "method", ["unreduced_product", "residues", "exact_top_bits", "approx_top_bits"]
)
def test_references_reject_out_of_range_exponents(params, method):
    with pytest.raises(ValueError, match="exponent"):
        getattr(params, method)(2**params.n_e)


# ---- T1 (ComputeExponent) -------------------------------------------------------------------


def test_t1_shape(params):
    assert params.t1.shape == (
        params.num_primes + 1,
        num_windows(params.n_e, params.w1),
        2**params.w1,
    )
    assert params.t1.min() >= 0
    assert params.t1.max() < 2**params.s_bitsize


def test_t1_computes_and_erases_s(params, exponents):
    """Fig. 27: after row i, S holds S_i = sum_k e_k D_{p_i,k}; row |P| erases it."""
    modulus = 2**params.s_bitsize
    num_w1 = num_windows(params.n_e, params.w1)
    for e in exponents:
        address = windows(e, params.w1, num_w1)
        s = 0
        for i in range(params.num_primes + 1):
            s = (s + sum(int(params.t1[i, j, x]) for j, x in enumerate(address))) % modulus
            if i < params.num_primes:
                assert s == sum(int(params.dlogs[i, k]) for k in range(params.n_e) if e >> k & 1)
            else:
                assert s == 0


def test_s_meets_reduce_exponent_precondition(params):
    """Section ReduceExponent S_i < 2^{log n_e} (p_i - 1), for every e."""
    s_max = params.dlogs.sum(axis=1)  # S_i for e = 2^{n_e} - 1, the largest
    log_n_e = (params.n_e - 1).bit_length()
    assert all(int(s) < 2**log_n_e * (p - 1) for s, p in zip(s_max, params.primes, strict=True))


# ---- T3 (ComputeResidue) --------------------------------------------------------------------


def test_t3_shapes(params):
    shape = (
        params.num_primes,
        num_windows(params.ell, params.w3a),
        num_windows(params.ell, params.w3b),
        2**params.w3a,
        2**params.w3b,
    )
    assert params.t3.shape == shape
    assert params.t3_inv.shape == shape
    assert params.alpha2_table.shape == (params.num_primes, 2 ** (2 * params.w3a))


def test_t3_computes_and_erases_residue(params, rng):
    """Equation `double_window` and the initialization: alpha_2, then one product per window.

    Going up with T3 gives g_{p_i}^{S} mod p_i; going back down with T3_inv returns alpha_2.
    """
    num_w3a = num_windows(params.ell, params.w3a)
    num_w3b = num_windows(params.ell, params.w3b)
    for i, (p, g_p) in enumerate(zip(params.primes, params.generators, strict=True)):
        for s in [0, p - 2, *(int(x) for x in rng.integers(0, p - 1, size=3))]:
            x = windows(s, params.w3a, num_w3a)
            alpha2 = int(params.alpha2_table[i, s % 2 ** (2 * params.w3a)])
            assert alpha2 == pow(g_p, s % 2 ** (2 * params.w3a), p)
            alpha = alpha2
            for j_a in range(2, num_w3a):
                y = windows(alpha, params.w3b, num_w3b)
                alpha = sum(int(params.t3[i, j_a, j_b, x[j_a], y[j_b]]) for j_b in range(num_w3b))
                alpha %= p
            assert alpha == pow(g_p, s, p)
            for j_a in reversed(range(2, num_w3a)):
                y = windows(alpha, params.w3b, num_w3b)
                alpha = sum(
                    int(params.t3_inv[i, j_a, j_b, x[j_a], y[j_b]]) for j_b in range(num_w3b)
                )
                alpha %= p
            assert alpha == alpha2


# ---- T4 (AccumulateResidue) and V-tilde ----------------------------------------------------


def test_t4_shape(params):
    assert params.t4.shape == (
        params.num_primes,
        num_windows(params.ell, params.w4),
        2**params.w4,
    )
    assert params.t4.min() >= 0
    assert params.t4.max() < params.M


def test_t4_is_per_bit(params):
    """a one-bit entry is C_{i, j w4 + k} (0 past bit ell - 1) and entries add up bitwise."""
    for j in range(num_windows(params.ell, params.w4)):
        for k in range(params.w4):
            bit = j * params.w4 + k
            expected = params.truncated_crt_factors[:, bit] if bit < params.ell else 0
            assert (params.t4[:, j, 1 << k] == expected).all()
        for x in range(2**params.w4):
            one_bit = sum(params.t4[:, j, 1 << k] for k in range(params.w4) if x >> k & 1)
            assert (params.t4[:, j, x] == one_bit % params.M).all()


def test_t4_lookups_accumulate_v_tilde(params, exponents):
    """Fig. 32: adding T4 at the windows of each r_i gives V-tilde(e) modulo M."""
    num_w4 = num_windows(params.ell, params.w4)
    for e in exponents:
        acc = 0
        for i, r in enumerate(params.residues(e)):
            acc += sum(int(params.t4[i, j, x]) for j, x in enumerate(windows(r, params.w4, num_w4)))
        assert acc % params.M == params.approx_top_bits(e)


def test_v_tilde_within_max_deviation(params):
    """Equation 50 (deviation), with the explicit bound `max_deviation`, for every exponent."""
    for e in range(2**params.n_e):
        v = pow(params.g, e, params.N)
        assert delta_N(v - (params.approx_top_bits(e) << params.kappa_bar), params.N) <= (
            params.max_deviation
        )


# ---- cross-check with the RSA-100 reference fixture ---------------------------------


@pytest.fixture(scope="module")
def fixture_params(rsa100_reference) -> RnsParameters:
    md, arrays = rsa100_reference.metadata, rsa100_reference.arrays
    return RnsParameters(
        N=md["modulus"],
        g=md["generator"],
        n_e=md["num_input_qubits"],
        ell=md["rns_primes_bit_length"],
        kappa=md["len_accumulator"],
        primes=arrays["periods"],
        generators=arrays["generators"],
        w1=md["window1"],
        w3a=md["window3a"],
        w3b=md["window3b"],
        w4=md["window4"],
    )


def test_fixture_metadata(fixture_params, rsa100_reference):
    md = rsa100_reference.metadata
    assert fixture_params.kappa_bar == md["dropped_bits"]
    assert fixture_params.M == md["truncated_modulus"]
    assert fixture_params.s_bitsize == md["len_dlog_accumulator"]
    assert num_windows(fixture_params.n_e, fixture_params.w1) == md["num_windows1"]
    assert num_windows(fixture_params.ell, fixture_params.w4) == md["num_windows4"]


def test_fixture_conditions(fixture_params):
    """
    It meets L >= N^{ceil(n_e/w1)}, which is all that its own T1 needs: there, each window
    loads the log of a product already reduced modulo N, so V-bar has ceil(n_e/w1) factors.
    """
    p = fixture_params
    assert p.satisfies_deviation_condition
    assert not p.satisfies_size_condition
    assert p.L >= p.N ** num_windows(p.n_e, p.w1)


def test_fixture_t3(fixture_params, rsa100_reference):
    """T3 matches exactly: the fixture stores p - T3 (table3a), T3_inv (table3b), alpha_2 (table3c).

    Its address for (x_a, y_b) is x_a 2^{w3b} + y_b, the C order of the last two axes.
    """
    p = fixture_params
    arrays = rsa100_reference.arrays
    flat = (p.num_primes, *p.t3.shape[1:3], -1)
    primes = np.array(p.primes)[:, None, None, None]
    assert (arrays["table3a"] == (primes - p.t3.reshape(flat)) % primes).all()
    assert (arrays["table3b"] == p.t3_inv.reshape(flat)).all()
    assert (arrays["table3c"] == p.alpha2_table).all()


def test_fixture_t1_one_bit_entries(fixture_params, rsa100_reference):
    """T1 entries whose address has at most one bit set match the fixture.

    For such an address both constructions load D_{p_i,k} - D_{p_{i-1},k}. The other entries
    differ: the fixture loads the log of (g^{x 2^{j w1}} mod N) mod p_i, a per-window product
    reduced modulo N (Gidney's construction), where the paper sums the D_{p_i,k}. The
    fixture also keeps negative entries on 32 bits, so it is compared modulo 2^{s_bitsize}.
    """
    p = fixture_params
    fixture_t1 = rsa100_reference.arrays["table1"].astype("int64") % 2**p.s_bitsize
    for j in range(num_windows(p.n_e, p.w1)):
        for k in range(p.w1):
            if j * p.w1 + k < p.n_e:
                assert (p.t1[:, j, 1 << k] == fixture_t1[:, j, 1 << k]).all()
        assert (p.t1[:, j, 0] == fixture_t1[:, j, 0]).all()


def test_fixture_t4(fixture_params, rsa100_reference):
    """T4 matches the fixture at addresses with at most one bit set, and is close elsewhere.

    The fixture stores what the Lookup loads, (M - T4) mod M, with Gidney's per-window T4,
    ((u_{p_i} x 2^{j w4}) mod L mod N) >> kappa_bar. At one-bit addresses both are C_{i,k}.
    Elsewhere, in units of 2^{kappa_bar} and modulo N, both approximate the sum of the
    (u_{p_i} 2^{j w4 + k}) mod L over the m <= w4 set bits of x. The per-bit entry is within
    2m - 1 units of it (m truncations, m - 1 wraps of M); the per-window entry is within one
    unit after dropping t <= m - 1 copies of L, each of size Delta_N(L) modulo N.
    """
    p = fixture_params
    fixture_t4 = (p.M - rsa100_reference.arrays["table4"].astype("int64")) % p.M
    bound = (2 * p.w4 << p.kappa_bar) + (p.w4 - 1) * delta_N(p.L, p.N)
    for x in range(2**p.w4):
        if x.bit_count() <= 1:
            assert (p.t4[:, :, x] == fixture_t4[:, :, x]).all()
        else:
            for ours, theirs in zip(p.t4[:, :, x].flat, fixture_t4[:, :, x].flat, strict=True):
                assert delta_N(int(ours - theirs) << p.kappa_bar, p.N) < bound


@pytest.mark.parametrize(("e", "accumulator"), RSA100_REFERENCE_OUTPUTS)
def test_fixture_recorded_results_within_max_deviation(fixture_params, e, accumulator):
    p = fixture_params
    v = pow(p.g, e, p.N)
    assert delta_N(v - (accumulator << p.kappa_bar), p.N) <= p.max_deviation
