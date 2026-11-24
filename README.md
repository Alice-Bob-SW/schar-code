# Shor's algorithm compiled on cat qubits: 22 593 qubits for RSA-2048 in 3.14 days — code

Code for *Shor's algorithm compiled on cat qubits: 22 593 qubits for RSA-2048 in 3.14 days*
(A. Pappalardo, D. Goudarzi et al., 2026), [arXiv:2610.37839](https://arxiv.org/abs/2610.37839).

This repository is a [Qualtran](https://github.com/quantumlib/Qualtran) implementation of the
logical approximate modular exponentiation oracle of Appendix B.
It regenerates the operation-count tables of Appendix B from the circuits' call graphs and
checks the circuits by classical simulation.

## Quick start

    git clone https://github.com/Alice-Bob-SW/schar-code.git && cd schar-code
    pixi run paper        # every Appendix B table into outputs/
    pixi run test         # all tests
    pixi run test-fast    # skip the slow tests, run in parallel

Run only some tables with `pixi run paper --only compute_residue accumulate_residue`.

Or with Docker only:

    docker build -t scharcode .
    docker run --rm -v "$PWD/outputs:/app/outputs" scharcode

## Two versions of the oracle

The oracle comes in two versions with the same subroutines and the same API:

- `scharcode.bloqs.factoring.delayed_mbu` is the version described in Appendix B. The
  measurement-based uncomputation of each Lookup is delayed, so that its Phaseups are merged.
  All the tables of `make_paper_outputs.py` describe this version, except `version_comparison`.
- `scharcode.bloqs.factoring.immediate_mbu` uncomputes each Lookup right after use, by a
  measurement in the X basis followed by a Phaseup. Its Lookup targets are measured instead of
  freed dirty, at the cost of more operations (see `version_comparison`).

In the delayed version, the Lookup targets are freed dirty instead of being measured, and the
merged Phaseups only carry their cost. Measuring them would require tracking every measurement
result up to the Phaseup that corrects it, which is possible but complex in the current version
of Qualtran and outside the scope of this code. The operation counts include these Phaseups,
but the classical simulation does not check the phase correctness of the delayed version. Some
classically controlled corrections, applied on half of the measurement outcomes on average, are
counted as expected counts (multiples of ½).

## Paper ↔ code

| Paper | Code |
| --- | --- |
| Algorithm 1, Fig. 26 | `ApproxModExp` |
| Fig. 27 | `ComputeExponent` |
| Lemma B.1, Fig. 28 | `ReduceExponent`, `ReduceExponentAdjoint` (`bloqs/factoring/reduce_exponent.py`) |
| Fig. 29 | `ComputeResidue` |
| Fig. 30 | `ComputeResidueAdjoint` |
| Fig. 31 | `UnderflowPhasefix` |
| Fig. 32 | `AccumulateResidue` |
| Appendix A, RNS parameters and tables | `scharcode.classical.rns.RnsParameters` |

Except for `ReduceExponent`, which both versions share, each class exists in
`delayed_mbu/` and in `immediate_mbu/`. The primitives (Q-Q and Q-C Adders, Comparison,
Lookup, Phaseup, CX) are in `scharcode.bloqs.operations`.

Each table of Appendix B is one job of `scripts/make_paper_outputs.py`. A job writes
`outputs/tables/<job>_symbolic.{csv,tex}` in the parameters of Table `parameter_definition`,
and `outputs/tables/<job>_rsa2048.{csv,tex}` at the values of Table `algo_parameter_value`.

| Paper table | Job |
| --- | --- |
| Table IX | `compute_exponent` |
| Table X | `reduce_exponent` |
| Table XI | `compute_residue` |
| Table XII | `compute_residue_adjoint` |
| Table XIII | `accumulate_residue` |
| Table XIV | `operation_count_recap` |
| Table XV | `logical_qubit_count` |
| — (immediate vs delayed version) | `version_comparison` |

The last job, `tables_pdf`, compiles all the tables into `outputs/figures/tables.pdf`.

## Layout

| Path | Content |
| --- | --- |
| `src/scharcode/bloqs/operations/` | Primitives |
| `src/scharcode/bloqs/factoring/` | Subroutines of the oracle, in `delayed_mbu/` and `immediate_mbu/` |
| `src/scharcode/classical/` | Classical precomputation: RNS parameters and lookup tables |
| `src/scharcode/analysis/` | Operation counts in the categories of the paper |
| `scripts/make_paper_outputs.py` | Paper tables |
| `tests/` | Tests; `tests/data/rsa100_reference/` holds a reference fixture |

## Acknowledgements

- C. Gidney, *How to factor 2048 bit RSA integers with less than a million noisy qubits*
  (2025, [arXiv:2505.15917](https://arxiv.org/abs/2505.15917)). The oracle builds on its
  construction; its reference code produced the RSA-100 fixture of the tests (see
  `tests/data/rsa100_reference/PROVENANCE.md`), and the Phaseup cost follows its Appendix A.3.
- C. Gidney, *Halving the cost of quantum addition*, Quantum 2, 74 (2018),
  [doi:10.22331/q-2018-06-18-74](https://doi.org/10.22331/q-2018-06-18-74), for the Q-Q Adder.
- C. Chevignard, P.-A. Fouque and A. Schrottenloher, *Reducing the Number of Qubits in Quantum
  Factoring*, CRYPTO 2025, [doi:10.1007/978-3-032-01878-6_13](https://doi.org/10.1007/978-3-032-01878-6_13),
  and its [code](https://gitlab.inria.fr/capsule/quantum-factoring-less-qubits).
- M. P. Harrigan et al., *Expressing and Analyzing Quantum Algorithms with Qualtran* (2024,
  [arXiv:2409.04643](https://arxiv.org/abs/2409.04643)).

## Citation

If you use this code, please cite the paper (see also `CITATION.cff`):

```bibtex
@misc{schar2026,
  author        = {Pappalardo, Axel and Goudarzi, Dahmun and Hertz, Victor and Jacinto, Hugo
                   and Kroell, Larissa and Schrottenloher, Andr{\'e} and Shanahan, Peter
                   and Gouzien, {\'E}lie and de La Vieuville, R{\'e}mi and Londe, Vivien},
  title         = {Shor's algorithm compiled on cat qubits: 22\,593 qubits for {RSA-2048}
                   in 3.14 days},
  year          = {2026},
  eprint        = {2610.37839},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2610.37839},
}
```

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE).
