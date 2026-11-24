# RSA-100 reference fixture

Reference data for a 100-bit instance of the RNS oracle, produced with the reference code of
C. Gidney, *How to factor 2048 bit RSA integers with less than a million noisy qubits*
(2025, [arXiv:2505.15917](https://arxiv.org/abs/2505.15917)).

The tests use it **only as a cross-check of the tables and as a regression reference**. 
End-to-end tests use instances from `scharcode.classical.rns.RnsParameters.search`,
which meet both conditions of Appendix A on L.

## Origin

| File | SHA-256 |
|---|---|
| `execution_fixture.npz` | `2d284d217843544708e6bc414a921f5c37f6e889b20888ceca375188720d28ee` |
| `metadata.json` | `cc13fa59c99c49c29248dfecb95f7a3496932ea5c991a2699c0ae9e7c1b0bb4e` |

## Parameters (`metadata.json`)

| Key | Paper symbol | Value |
|---|---|---|
| `modulus` | N (100 bits) | 1146337137660851397915791125349 |
| `generator` | g | 66 |
| `num_input_qubits` | n_e | 13 |
| `rns_primes_bit_length` | ℓ | 12 |
| `len_accumulator` | κ | 20 |
| `dropped_bits` | κ̄ = log N − κ | 80 |
| `truncated_modulus` | M = N ≫ κ̄ | 948227 |
| `len_dlog_accumulator` | ℓ + ⌈log₂ n_e⌉, size of S | 16 |
| `window1`, `window3a`, `window3b`, `window4` | w1, w3a, w3b, w4 | 2, 2, 2, 2 |
| `num_windows1`, `num_windows3a`, `num_windows3b`, `num_windows4` | ⌈n_e/w1⌉, ⌈ℓ/w3a⌉, ⌈ℓ/w3b⌉, ⌈ℓ/w4⌉ | 7, 6, 6, 6 |
| `mask_bits`, `min_wraparound_gap`, `num_shots` | (not used by the oracle) | 10, 20, 1 |

## Arrays (`execution_fixture.npz`)

Windows and address bits are numbered from the least significant end, as in
`scharcode.classical.rns`.

| Key | Shape, dtype | Content |
|---|---|---|
| `periods` | (60,) uint32 | the primes p_i, sorted, from 2503 to 4093 |
| `generators` | (60,) uint32 | g_{p_i}, a generator of ℤ_{p_i}^* (the smallest one for 33 of the 60 primes) |
| `table1` | (61, 7, 4) uint32 | T1[i, j, x] = log(W_i[j, x]) − log(W_{i−1}[j, x]), with W_i[j, x] = (g^{x·2^{j·w1}} mod N) mod p_i: the log of a per-window product reduced modulo N (Gidney's construction), not the paper's per-bit Σ_k x_k D_{p_i, j·w1+k} that `scharcode.classical.rns` uses. Negative entries are in two's complement on 32 bits. In the last window, which holds only bit 12 of e, addresses 2 and 3 use bit 13. |
| `table3a` | (60, 6, 6, 16) uint32 | p_i − T3[i, j_a, j_b, x_a, y_b] mod p_i, at address x_a·2^{w3b} + y_b |
| `table3b` | (60, 6, 6, 16) uint32 | T3_inv[i, j_a, j_b, x_a, y_b], same addressing |
| `table3c` | (60, 16) uint32 | α₂ table, g_{p_i}^x mod p_i |
| `table4` | (60, 6, 4) uint64 | (M − T4[i, j, x]) mod M, with Gidney's per-window T4 = (((u_{p_i}·x·2^{j·w4}) mod L mod N) ≫ κ̄) mod M, not the paper's per-bit T4 that `scharcode.classical.rns` uses |

All of the above was checked against `scharcode.classical.rns`, entry by entry (2026-11-10):

- T3, T3_inv and α₂ match exactly.
- T1 matches the per-window formula above on every entry, after reduction modulo 2^32. The
  paper's per-bit T1 agrees only at addresses with at most one bit set.
- T4 matches the per-window formula on all 1440 entries. The paper's per-bit T4,
  Σ_k x_k C_{i, j·w4+k} mod M, matches 1286 of them; the other 154 differ by ±1.

## Recorded results

Ten runs of the oracle with these tables. They are not stored in the npz: the tests read them
from `RSA100_REFERENCE_OUTPUTS` in `tests/helpers.py`.

| e | Accumulator Ṽ(e) | (g^e mod N) ≫ κ̄ | Difference |
|---:|---:|---:|---:|
| 1600 | 925495 | 925524 | −29 |
| 7014 | 930091 | 930119 | −28 |
| 353 | 535514 | 535536 | −22 |
| 3447 | 103731 | 103758 | −27 |
| 4779 | 486218 | 486238 | −20 |
| 5081 | 204031 | 204046 | −15 |
| 7980 | 141719 | 141740 | −21 |
| 3322 | 674042 | 674064 | −22 |
| 1124 | 633541 | 633570 | −29 |
| 4958 | 33942 | 33960 | −18 |

The explicit bound `RnsParameters.max_deviation` of Equation `deviation` allows 1444 units of
2^κ̄ for this instance. Recomputing Ṽ(e) from the tables reproduces all ten values with the
per-window T1 and T4. With the per-bit T4 it reproduces none of them, and with the per-bit T1
only five. `RnsParameters.approx_top_bits`, which uses the per-bit T1 and T4, lands
1 to 21 units below the recorded values, within the bound.

## Caveat: the condition L ≥ N^{n_e}

L, the product of the 60 primes, has 713.55 bits.

- **Paper condition, L ≥ N^{n_e} (Appendix A): not met.** It needs 13 × 99.85 = 1298.11 bits,
  about 109 twelve-bit primes. For this N and these sizes, `RnsParameters.search` with seed 0
  returns 113 primes.
- **L ≥ N^{⌈n_e/w1⌉} = N^7: met** (698.98 bits). This is all that the fixture's own T1 needs.
  Each window loads the log of a product already reduced modulo N, so the unreduced product
  has ⌈n_e/w1⌉ factors below N instead of n_e.
- **Equation `dev`, Δ_N(L) < N·2^{−κ}: met.** Δ_N(L) = N·2^{−27.33}.

So the fixture is a valid instance of Gidney's construction. With these 60 primes and the per-bit
tables, V-bar(e) < L holds for only 7629 of the 8192 exponents, and Ṽ(e) stays within
`max_deviation` for 7735. It holds for that many only because g = 66 makes M_0 to M_3 small
(6 to 48 bits). Tests that build an oracle from `scharcode.classical.rns` must not use this
prime set.
