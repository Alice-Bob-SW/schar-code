"""Bloqs of the RSA-2048 oracle of Appendix B.

- `operations`: the primitives of the paper (Q-Q and Q-C Adders, Comparison, Lookup,
  Phaseup, multi-target CNOT).
- `factoring`: the subroutines of the oracle, shared by both versions (`ReduceExponent`),
  then one subpackage per version: `delayed_mbu` (the paper's) and `immediate_mbu`.
"""
