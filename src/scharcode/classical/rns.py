"""Classical precomputation for the RNS oracle of Appendix B.

Everything the quantum circuit loads from a table is computed here, in pure Python (with
numpy and sympy, no qualtran): the residue number system (RNS) of Appendix A, paragraph
"Residue arithmetic" (Section `sec:oracle-rns` of `03_algorithmic-overview.tex`), and the
tables of Appendix B (`04_logical-implementation.tex`).

Notation follows Table `parameter_definition`:

- N, g: the modulus and the base of the exponentiation V = g^e mod N.
- n_e: number of exponent bits; ell: bit length of every RNS prime.
- kappa: accumulator size; kappa_bar = n - kappa bits are dropped, with n the bit length of N.
- P = (p_0, ..., p_{|P|-1}): the primes, with product L; g_{p_i} generates Z_{p_i}^*.
- w1, w3a, w3b, w4: the windows of ComputeExponent, ComputeResidue (outer and inner loops) and
  AccumulateResidue.

Bits and windows are numbered from the least significant end, so window j of a value x is
bits j*w, ..., j*w + w - 1 of x. The last window may be partial; the tables are still
rectangular, and their entries for addresses that the partial window cannot hold are padding.

T1 and T4 follow the paper's per-bit definitions: an entry is the sum,
over the set bits of its address, of one constant per bit. Gidney's reference code, and so
the RSA-100 test fixture, instead reduces one product per window modulo N.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from sympy.ntheory import is_primitive_root, isprime, primerange, primitive_root


def delta_N(z: int, N: int) -> int:
    """Distance from z to the nearest multiple of N, written Delta_N(z) in Appendix A.

    It is the size of the centred residue of z, in [0, N/2].
    """
    r = z % N
    return min(r, N - r)


def num_windows(bits: int, w: int) -> int:
    """Number of windows of w bits needed to cover `bits` bits, ceil(bits / w)."""
    return -(-bits // w)


def discrete_logs(targets: list[int], g_p: int, p: int) -> list[int]:
    """Discrete logarithms base g_p modulo the prime p, by baby-step giant-step.

    One baby-step table of about sqrt(p) entries serves every target, so computing the n_e
    logarithms of one prime costs O(n_e * sqrt(p)) multiplications.

    Parameters
    ----------
    targets
        Values h, none of them divisible by p.
    g_p
        A generator of Z_p^*.
    p
        A prime.

    Returns
    -------
    list[int]
        For each h, the unique D in [0, p - 2] with g_p^D = h mod p.
    """
    m = math.isqrt(p - 1) + 1
    baby: dict[int, int] = {}
    x = 1
    for j in range(m):
        baby.setdefault(x, j)
        x = x * g_p % p
    giant = pow(g_p, -m, p)
    logs = []
    for h in targets:
        gamma = h % p
        if gamma == 0:
            raise ValueError(f"{h} is divisible by {p}: it has no discrete logarithm")
        for i in range(m):
            j = baby.get(gamma)
            if j is not None:
                logs.append((i * m + j) % (p - 1))
                break
            gamma = gamma * giant % p
        else:  # pragma: no cover - unreachable when g_p generates Z_p^*
            raise ValueError(f"{h} is not a power of {g_p} modulo {p}")
    return logs


def double_window_table(
    p: int,
    g_p: int,
    *,
    x_shift: int,
    y_shift: int,
    x_bits: int,
    y_bits: int,
    inverse: bool = False,
) -> np.ndarray:
    """One T3 (or T3_inv) table of Equation `double_window`, shape (2^{x_bits}, 2^{y_bits}).

    Entry [x_a, y_b] is y_b 2^{y_shift} g_p^{x_a 2^{x_shift}} mod p, or, if `inverse`,
    y_b 2^{y_shift} g_p^{-x_a 2^{x_shift}} mod p. For the windows j_a and j_b of
    ComputeResidue, x_shift = J_{j_a} = j_a w3a and y_shift = J_{j_b} = j_b w3b.

    Products are taken on Python integers, so the entries are exact for any p < 2^63.
    """
    g = pow(g_p, -1, p) if inverse else g_p % p
    unit = pow(2, y_shift, p)
    rows = []
    for x_a in range(2**x_bits):
        factor = pow(g, (x_a << x_shift) % (p - 1), p) * unit % p
        rows.append([factor * y_b % p for y_b in range(2**y_bits)])
    return np.array(rows, dtype=np.int64)


def _window_bits(w: int) -> np.ndarray:
    """Matrix of shape (2^w, w) whose row x holds the bits of x, least significant first."""
    return (np.arange(2**w)[:, None] >> np.arange(w)) & 1


@dataclass(frozen=True)
class RnsParameters:
    """One instance of the oracle: the modulus, the RNS and the window sizes.

    The constructor only checks that the RNS is well formed (see Raises). It does not check
    the two conditions that Appendix A puts on L, because the RSA-100 reference fixture (from
    Gidney's code) fails one of them; see `satisfies_size_condition` and
    `satisfies_deviation_condition`.
    `RnsParameters.search` builds instances that satisfy both.

    Parameters
    ----------
    N
        Integer to factor (or, in general, the modulus of the exponentiation).
    g
        Base of the exponentiation.
    n_e
        Number of bits of the exponent e.
    ell
        Bit length of every RNS prime: 2^(ell-1) < p < 2^ell.
    kappa
        Number of top bits kept by the approximation, between 1 and the bit length of N.
    primes
        The RNS primes p_i, in the order the oracle processes them.
    generators
        g_{p_i}, a generator of Z_{p_i}^* for each prime.
    w1, w3a, w3b, w4
        Window sizes of ComputeExponent, ComputeResidue (outer, inner) and AccumulateResidue.

    Raises
    ------
    ValueError
        If a size is out of range, a prime is repeated, is not prime or not ell bits long, a
        generator does not generate Z_p^*, or a prime divides some M_k, which then has no
        discrete logarithm.
    """

    N: int
    g: int
    n_e: int
    ell: int
    kappa: int
    primes: tuple[int, ...]
    generators: tuple[int, ...]
    w1: int
    w3a: int
    w3b: int
    w4: int

    def __post_init__(self) -> None:
        # Accept any sequence of integers (lists, numpy arrays) but store hashable tuples of
        # Python ints, so that big-integer arithmetic never overflows.
        object.__setattr__(self, "primes", tuple(int(p) for p in self.primes))
        object.__setattr__(self, "generators", tuple(int(x) for x in self.generators))
        if self.N < 2 or self.n_e < 1 or self.ell < 2:
            raise ValueError("Need N >= 2, n_e >= 1 and ell >= 2.")
        if not 1 <= self.kappa <= self.n:
            raise ValueError(f"kappa must lie in [1, {self.n}], the bit length of N.")
        if min(self.w1, self.w3a, self.w3b, self.w4) < 1:
            raise ValueError("Window sizes must be positive.")
        if not self.primes:
            raise ValueError("The RNS needs at least one prime.")
        if len(set(self.primes)) != len(self.primes):
            raise ValueError("The RNS primes must be distinct.")
        if len(self.generators) != len(self.primes):
            raise ValueError("Give one generator per prime.")
        for p, g_p in zip(self.primes, self.generators, strict=True):
            if p.bit_length() != self.ell or not isprime(p):
                raise ValueError(f"{p} is not an {self.ell}-bit prime.")
            if g_p % p == 0 or not is_primitive_root(g_p, p):
                raise ValueError(f"{g_p} does not generate Z_{p}^*.")
            if any(m_k % p == 0 for m_k in self.powers):
                raise ValueError(f"{p} divides some M_k = g^(2^k) mod N.")

    # ---- sizes --------------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Bit length of N, written log N in the paper."""
        return self.N.bit_length()

    @property
    def kappa_bar(self) -> int:
        """Number of low bits dropped by the truncation, kappa_bar = log N - kappa."""
        return self.n - self.kappa

    @property
    def M(self) -> int:
        """Accumulator modulus M = N >> kappa_bar = floor(N / 2^kappa_bar)."""
        return self.N >> self.kappa_bar

    @property
    def num_primes(self) -> int:
        """|P|, the number of RNS primes."""
        return len(self.primes)

    @cached_property
    def L(self) -> int:
        """Product of the RNS primes."""
        return math.prod(self.primes)

    @property
    def s_bitsize(self) -> int:
        """Size ell + log n_e of the register of S_i, with log n_e = ceil(log2 n_e)."""
        return self.ell + (self.n_e - 1).bit_length()

    # ---- conditions on L (Appendix A) ---------------------------------------------------

    @property
    def satisfies_size_condition(self) -> bool:
        """L >= N^{n_e}: the unreduced product of the M_k never wraps modulo L."""
        return self.L >= self.N**self.n_e

    @property
    def satisfies_deviation_condition(self) -> bool:
        """Delta_N(L) < N 2^{-kappa}, Equation `dev`."""
        return delta_N(self.L, self.N) << self.kappa < self.N

    # ---- RNS constants ------------------------------------------------------------------

    @cached_property
    def powers(self) -> tuple[int, ...]:
        """M_k = g^{2^k} mod N for k = 0, ..., n_e - 1."""
        out = [self.g % self.N]
        for _ in range(self.n_e - 1):
            out.append(out[-1] * out[-1] % self.N)
        return tuple(out)

    @cached_property
    def dlogs(self) -> np.ndarray:
        """D_{p_i,k}, the discrete log of M_k mod p_i base g_{p_i}, shape (|P|, n_e).

        Each D_{p_i,k} lies in [0, p_i - 2]. They are computed independently for each k:
        M_{k+1} = M_k^2 mod N is reduced modulo N, so D_{p_i,k+1} is not 2 D_{p_i,k}.
        """
        return np.array(
            [
                discrete_logs(list(self.powers), g_p, p)
                for p, g_p in zip(self.primes, self.generators, strict=True)
            ],
            dtype=np.int64,
        )

    @cached_property
    def crt_factors(self) -> tuple[int, ...]:
        """u_{p_j} = (L / p_j) * ((L / p_j)^{-1} mod p_j), the CRT contribution factors.

        u_{p_j} = 1 mod p_j and 0 modulo every other prime, so V-bar = (sum_j r_j u_{p_j})
        mod L (Equation `rns-exact`).
        """
        return tuple((self.L // p) * pow(self.L // p, -1, p) for p in self.primes)

    @cached_property
    def truncated_crt_factors(self) -> np.ndarray:
        """C_{j,k} = [((u_{p_j} 2^k) mod L mod N) >> kappa_bar] mod M, shape (|P|, ell).

        These are the constants of Equation `approximate_modular_exponentiation`.
        """
        return np.array(
            [
                [((u << k) % self.L % self.N >> self.kappa_bar) % self.M for k in range(self.ell)]
                for u in self.crt_factors
            ],
            dtype=np.int64,
        )

    # ---- tables -------------------------------------------------------------------------

    @cached_property
    def t1(self) -> np.ndarray:
        """T1 tables of ComputeExponent, shape (|P|+1, ceil(n_e/w1), 2^{w1}).

        Fig. `offsetdlog`: entry [i, j, x] is

            T1^{(i,j)}[x] = sum_{k < w1} x_k (D_{p_i, j w1 + k} - D_{p_{i-1}, j w1 + k}),

        with D_{p_{-1},k} = D_{p_{|P|},k} = 0, so row 0 computes S_0 and row |P| erases
        S_{|P|-1}. Bit k of the address x is exponent bit j w1 + k; in the partial last
        window, address bits at or above n_e are ignored. Entries are in two's complement on
        `s_bitsize` bits, that is, reduced modulo 2^{s_bitsize}.

        Window sums are not reduced modulo p_i - 1, as in the paper.
        """
        w1, n_e = self.w1, self.n_e
        num_w1 = num_windows(n_e, w1)
        padded = np.zeros((self.num_primes, num_w1 * w1), dtype=np.int64)
        padded[:, :n_e] = self.dlogs
        window_sums = padded.reshape(self.num_primes, num_w1, w1) @ _window_bits(w1).T
        zero = np.zeros((1, num_w1, 2**w1), dtype=np.int64)
        rows = np.concatenate([zero, window_sums, zero])
        return (rows[1:] - rows[:-1]) % (1 << self.s_bitsize)

    def _t3(self, inverse: bool) -> np.ndarray:
        """T3 or T3_inv, see `t3` and `t3_inv`."""
        w3a, w3b = self.w3a, self.w3b
        num_w3a, num_w3b = num_windows(self.ell, w3a), num_windows(self.ell, w3b)
        out = np.empty((self.num_primes, num_w3a, num_w3b, 2**w3a, 2**w3b), dtype=np.int64)
        for i, (p, g_p) in enumerate(zip(self.primes, self.generators, strict=True)):
            for j_a, j_b in itertools.product(range(num_w3a), range(num_w3b)):
                out[i, j_a, j_b] = double_window_table(
                    p,
                    g_p,
                    x_shift=j_a * w3a,
                    y_shift=j_b * w3b,
                    x_bits=w3a,
                    y_bits=w3b,
                    inverse=inverse,
                )
        return out

    @cached_property
    def t3(self) -> np.ndarray:
        """T3 tables of ComputeResidue, shape (|P|, ceil(ell/w3a), ceil(ell/w3b), 2^{w3a}, 2^{w3b}).

        Equation `double_window`: entry [i, j_a, j_b, x_a, y_b] is

            T3^{(i,j_a,j_b)}[x_a, y_b] = y_b 2^{j_b w3b} g_{p_i}^{x_a 2^{j_a w3a}} mod p_i.

        Window j_a of S_i is bits j_a w3a, ..., j_a w3a + w3a - 1, counted from the least
        significant end. The tables are rectangular: in a partial top window,
        the addresses that the window cannot hold are padding. The Lookup-ready tables, cut
        to the window sizes and with the complements the bloqs load, are built by
        `scharcode.bloqs.factoring._common.ResidueTables`.
        """
        return self._t3(inverse=False)

    @cached_property
    def t3_inv(self) -> np.ndarray:
        """T3_inv tables of ComputeResidue^dagger, same shape as `t3`.

        Entry [i, j_a, j_b, x_a, y_b] is y_b 2^{j_b w3b} g_{p_i}^{-x_a 2^{j_a w3a}} mod p_i
        (Fig. `unmodexp` and the equation that defines T_{3,inv}).
        """
        return self._t3(inverse=True)

    @cached_property
    def alpha2_table(self) -> np.ndarray:
        """Initial value of ComputeResidue, shape (|P|, 2^{min(2 w3a, ell)}).

        Entry [i, x] is g_{p_i}^x mod p_i, so the Lookup addressed by the 2 w3a low bits of
        S_i loads alpha_2 directly (paragraph "Initialization" of Appendix B). When
        ell < 2 w3a, the address is the whole ell-bit register.
        """
        init_bits = min(2 * self.w3a, self.ell)
        return np.array(
            [
                [pow(g_p, x, p) for x in range(2**init_bits)]
                for p, g_p in zip(self.primes, self.generators, strict=True)
            ],
            dtype=np.int64,
        )

    @cached_property
    def t4(self) -> np.ndarray:
        """T4 tables of AccumulateResidue, shape (|P|, ceil(ell/w4), 2^{w4}).

        Fig. `accumulation`: entry [i, j, x] is

            T4^{(i,j)}[x] = (sum_{k < w4} x_k C_{i, j w4 + k}) mod M,

        so the Lookups addressed by the windows of r_i add up to sum_k r_{i,k} C_{i,k}, the
        terms of Equation `approximate_modular_exponentiation` for p_i. Bit k of the address x
        is bit j w4 + k of r_i; in the partial last window, address bits at or above ell are
        ignored. Entries lie in [0, M).

        Each C_{i,k} is truncated on its own, as in the paper. The Lookup loads
        (M - T4) mod M, that is M - T4, or 0 when T4 = 0; building that complement is left to
        the bloqs, as for T3.
        """
        w4, ell = self.w4, self.ell
        num_w4 = num_windows(ell, w4)
        padded = np.zeros((self.num_primes, num_w4 * w4), dtype=np.int64)
        padded[:, :ell] = self.truncated_crt_factors
        return padded.reshape(self.num_primes, num_w4, w4) @ _window_bits(w4).T % self.M

    # ---- classical references -----------------------------------------------------------

    def _check_exponent(self, e: int) -> None:
        if not 0 <= e < 2**self.n_e:
            raise ValueError(f"The exponent must lie in [0, 2^{self.n_e}).")

    def unreduced_product(self, e: int) -> int:
        """V-bar(e) = prod_k M_k^{e_k}, without reduction modulo N (Appendix A).

        V-bar < N^{n_e}, so under `satisfies_size_condition` it is below L, and the CRT
        recovers it from the residues.
        """
        self._check_exponent(e)
        return math.prod(m_k for k, m_k in enumerate(self.powers) if e >> k & 1)

    def residues(self, e: int) -> tuple[int, ...]:
        """r_i = g_{p_i}^{S_i mod (p_i - 1)} mod p_i, with S_i = sum_k e_k D_{p_i,k}.

        This is the value ComputeExponent, ReduceExponent and ComputeResidue produce for the
        prime p_i (Algorithm `alg:approx-modexp`). It equals V-bar(e) mod p_i.
        """
        self._check_exponent(e)
        bits = np.array([e >> k & 1 for k in range(self.n_e)], dtype=np.int64)
        exponents = self.dlogs @ bits
        return tuple(
            pow(g_p, int(s) % (p - 1), p)
            for p, g_p, s in zip(self.primes, self.generators, exponents, strict=True)
        )

    def exact_top_bits(self, e: int) -> int:
        """(g^e mod N) >> kappa_bar, the value that the accumulator approximates."""
        self._check_exponent(e)
        return pow(self.g, e, self.N) >> self.kappa_bar

    def approx_top_bits(self, e: int) -> int:
        """V-tilde(e) = (sum_j sum_k r_{j,k} C_{j,k}) mod M, Equation
        `approximate_modular_exponentiation`.

        r_{j,k} is bit k of the residue r_j (see `residues`). This is what the oracle adds to
        its accumulator. It is within `max_deviation` of V(e) = g^e mod N, after scaling by
        2^{kappa_bar}, whenever V-bar(e) < L.
        """
        total = sum(
            int(c[k])
            for r, c in zip(self.residues(e), self.truncated_crt_factors, strict=True)
            for k in range(self.ell)
            if r >> k & 1
        )
        return total % self.M

    @property
    def max_deviation(self) -> int:
        """Bound on Delta_N(V - 2^{kappa_bar} V-tilde), Equation `deviation`.

        Equation `deviation` reads Delta_N(V - (V-tilde << kappa_bar)) <= (epsilon / 2) N
        with epsilon = O(|P| ell 2^{-kappa}). This property gives an explicit value for the
        right-hand side, |P| ell (Delta_N(L) + 2^{kappa_bar + 1}), so epsilon =
        2 * max_deviation / N. It holds whenever V-bar(e) < L.

        Derivation: V-tilde sums at most A = |P| ell truncated terms modulo M. Write N =
        2^{kappa_bar} M + nu, with 0 <= nu < 2^{kappa_bar}. Truncating a term c < N to
        (c >> kappa_bar) mod M moves 2^{kappa_bar} times it away from c by less than
        2^{kappa_bar}, modulo N. Each of the fewer than A wraps of the accumulator modulo M
        adds nu < 2^{kappa_bar}, modulo N. Each of the fewer than A wraps modulo L that the
        sum ignores adds L mod N, of size Delta_N(L) (Equation `rns-approx`).
        """
        return self.num_primes * self.ell * (delta_N(self.L, self.N) + (2 << self.kappa_bar))

    # ---- prime-set search ---------------------------------------------------------------

    @classmethod
    def search(
        cls,
        N: int,
        g: int,
        *,
        n_e: int,
        ell: int,
        kappa: int,
        w1: int,
        w3a: int,
        w3b: int,
        w4: int,
        seed: int = 0,
        max_attempts: int = 1000,
    ) -> RnsParameters:
        """Find ell-bit primes with L >= N^{n_e} and Delta_N(L) < N 2^{-kappa}.

        Appendix A takes L from a random search over prime sets, which succeeds under
        Gidney's Assumption 1. Each attempt shuffles the candidate primes, keeps the shortest
        prefix whose product reaches N^{n_e}, then tries every way of replacing its last two
        primes by two unused candidates until Equation `dev` holds. A random L meets `dev`
        with probability about 2^{1-kappa}, so the cost grows like 2^kappa: this is meant for
        test instances, not RSA-2048.

        Candidates are the ell-bit primes that divide no M_k. The primes are returned in
        increasing order, each with its smallest generator.

        Parameters
        ----------
        N, g, n_e, ell, kappa, w1, w3a, w3b, w4
            As in `RnsParameters`.
        seed
            Seed of the shuffles: the same arguments always give the same instance.
        max_attempts
            Number of shuffles to try before giving up.

        Raises
        ------
        ValueError
            If all ell-bit candidates together do not reach N^{n_e}.
        RuntimeError
            If no prime set meets Equation `dev` within `max_attempts` shuffles.
        """
        target = N**n_e
        powers = [g % N]
        for _ in range(n_e - 1):
            powers.append(powers[-1] ** 2 % N)
        pool = [p for p in primerange(2 ** (ell - 1) + 1, 2**ell) if all(m_k % p for m_k in powers)]
        if math.prod(pool) < target:
            raise ValueError(f"All {len(pool)} usable {ell}-bit primes give L < N^{n_e}.")

        rng = random.Random(seed)
        for _ in range(max_attempts):
            rng.shuffle(pool)
            size, product = 0, 1
            while product < target:
                product *= pool[size]
                size += 1
            base = pool[: max(size - 2, 0)]
            rest = pool[len(base) :]
            base_product = math.prod(base)
            base_mod_N = base_product % N
            need = -(-target // base_product)  # the replacements must multiply to >= need
            for swap in itertools.combinations(rest, size - len(base)):
                swap_product = math.prod(swap)
                if swap_product < need:
                    continue
                if delta_N(base_mod_N * swap_product, N) << kappa < N:
                    primes = tuple(sorted(base + list(swap)))
                    return cls(
                        N=N,
                        g=g,
                        n_e=n_e,
                        ell=ell,
                        kappa=kappa,
                        primes=primes,
                        generators=tuple(int(primitive_root(p)) for p in primes),
                        w1=w1,
                        w3a=w3a,
                        w3b=w3b,
                        w4=w4,
                    )
        raise RuntimeError(
            f"No prime set met Delta_N(L) < N 2^-{kappa} in {max_attempts} attempts."
        )
